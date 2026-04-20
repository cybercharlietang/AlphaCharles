"""Centralized metrics tracker with W&B + stdout sinks.

Usage:
    tracker = MetricsTracker(run_name="sl_warmstart", project="alphazero-chess",
                              wandb=True, stdout_every=50)
    tracker.log({"loss/total": 2.1, "loss/policy": 1.8}, step=1)
    tracker.log_gpu(step=1)                   # nvidia-smi-derived
    tracker.log_batch_stats(policy_logits, value_pred, policy_target, value_target,
                            legal_mask, step=1)

Only rank 0 logs when running under DDP. Safe to call from non-main ranks (no-ops).
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import torch


def _is_main() -> bool:
    return int(os.environ.get("RANK", 0)) == 0


class MetricsTracker:
    def __init__(
        self,
        run_name: str,
        project: str = "alphazero-chess",
        config: dict | None = None,
        wandb_enabled: bool = True,
        stdout_every: int = 50,
    ):
        self.run_name = run_name
        self.stdout_every = stdout_every
        self.wandb = None
        self._start_time = time.time()
        self._last_log_time = self._start_time
        self._last_log_step = 0

        if wandb_enabled and _is_main():
            try:
                import wandb
                self.wandb = wandb.init(
                    project=project, name=run_name, config=config or {},
                    reinit=True,
                )
            except Exception as e:
                print(f"[metrics] W&B init failed: {e}. Continuing without W&B.")
                self.wandb = None

    # ---- Core log ----------------------------------------------------------

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if not _is_main():
            return
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)

        if step % self.stdout_every == 0:
            now = time.time()
            steps_since = step - self._last_log_step
            dt = now - self._last_log_time
            rate = steps_since / dt if dt > 0 else 0.0
            self._last_log_time = now
            self._last_log_step = step

            # Print key metrics first (loss, lr), then optionals.
            priority = ["loss/total", "loss/policy", "loss/value", "train/lr",
                        "train/grad_norm", "policy/entropy", "value/corr"]
            bits = [f"step={step}"]
            for k in priority:
                if k in metrics:
                    bits.append(f"{k.split('/')[-1]}={metrics[k]:.3f}")
            bits.append(f"steps/s={rate:.1f}")
            print(" | ".join(bits))

    # ---- Batch-derived stats (entropy, value corr, etc.) -------------------

    def log_batch_stats(
        self,
        policy_logits: torch.Tensor,       # (B, 4672)
        value_pred: torch.Tensor,          # (B,)
        policy_target: torch.Tensor,       # (B, 4672)
        value_target: torch.Tensor,        # (B,)
        step: int,
        legal_mask: torch.Tensor | None = None,
    ) -> dict:
        """Compute and log batch-level diagnostics. Returns the metrics dict."""
        with torch.no_grad():
            # Mask illegal to -inf before softmax if given.
            logits = policy_logits.detach().float()
            if legal_mask is not None:
                neg_inf = torch.finfo(logits.dtype).min
                logits = logits.masked_fill(~legal_mask, neg_inf)

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            probs = log_probs.exp()

            # Policy entropy (over legal). 0*log(0)=0.
            ent = -(probs * log_probs).where(probs > 0, torch.zeros_like(probs)).sum(dim=-1)
            policy_entropy = float(ent.mean())

            # Top-1 agreement with the target's argmax.
            pred_argmax = logits.argmax(dim=-1)
            tgt_argmax = policy_target.argmax(dim=-1)
            top1_acc = float((pred_argmax == tgt_argmax).float().mean())

            # Value stats.
            v = value_pred.detach().float()
            z = value_target.detach().float()
            v_mean = float(v.mean())
            z_mean = float(z.mean())
            # Pearson correlation.
            vm = v - v.mean(); zm = z - z.mean()
            denom = (vm.norm() * zm.norm()).clamp_min(1e-8)
            corr = float((vm * zm).sum() / denom)

        metrics = {
            "policy/entropy": policy_entropy,
            "policy/top1_acc": top1_acc,
            "value/mean_pred": v_mean,
            "value/mean_target": z_mean,
            "value/corr": corr,
        }
        self.log(metrics, step=step)
        return metrics

    # ---- Gradient stats ----------------------------------------------------

    def log_grad_norm(self, model, step: int, clip_value: float | None = None) -> float:
        if not _is_main():
            return 0.0
        total_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_sq += float(p.grad.detach().float().norm() ** 2)
        gn = total_sq ** 0.5
        m = {"train/grad_norm": gn}
        if clip_value is not None:
            m["train/grad_clipped"] = float(gn > clip_value)
        self.log(m, step=step)
        return gn

    # ---- GPU stats ---------------------------------------------------------

    def log_gpu(self, step: int) -> None:
        if not _is_main() or not torch.cuda.is_available():
            return
        metrics = {}
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            used_gb = (total - free) / 1e9
            metrics[f"gpu/mem_used_gb/{i}"] = used_gb
            metrics[f"gpu/mem_frac/{i}"] = (total - free) / total
        # Utilization / power / temp via nvidia-ml-py if available.
        try:
            import pynvml
            pynvml.nvmlInit()
            for i in range(torch.cuda.device_count()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                metrics[f"gpu/util/{i}"] = util.gpu
                metrics[f"gpu/power_w/{i}"] = power
                metrics[f"gpu/temp_c/{i}"] = temp
        except Exception:
            pass
        self.log(metrics, step=step)

    # ---- Self-play stats ---------------------------------------------------

    def log_game(self, result: str, ply_count: int, resigned: bool,
                 root_value_mean: float | None, root_entropy: float | None,
                 step: int, buffer_size: int) -> None:
        """Per-game log from self-play worker; aggregated via W&B histograms."""
        if not _is_main():
            return
        m = {
            "rl/result_white_win": 1.0 if result == "1-0" else 0.0,
            "rl/result_draw": 1.0 if result == "1/2-1/2" else 0.0,
            "rl/result_black_win": 1.0 if result == "0-1" else 0.0,
            "rl/plies_per_game": ply_count,
            "rl/resigned": 1.0 if resigned else 0.0,
            "rl/buffer_size": buffer_size,
        }
        if root_value_mean is not None:
            m["rl/root_value_mean"] = root_value_mean
        if root_entropy is not None:
            m["rl/root_entropy"] = root_entropy
        self.log(m, step=step)

    # ---- Eval stats --------------------------------------------------------

    def log_eval(self, wins: int, draws: int, losses: int, elo_diff: float,
                 stderr_elo: float, opp_label: str, step: int) -> None:
        if not _is_main():
            return
        total = wins + draws + losses
        score = (wins + 0.5 * draws) / max(total, 1)
        self.log({
            f"eval/{opp_label}/wins": wins,
            f"eval/{opp_label}/draws": draws,
            f"eval/{opp_label}/losses": losses,
            f"eval/{opp_label}/score": score,
            f"eval/{opp_label}/elo_diff": elo_diff,
            f"eval/{opp_label}/elo_stderr": stderr_elo,
        }, step=step)

    def finish(self) -> None:
        if self.wandb is not None:
            self.wandb.finish()
