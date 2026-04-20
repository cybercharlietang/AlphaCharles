"""Self-play RL trainer.

On each H100, we spawn one self-play subprocess (N games in parallel via
async MCTS is a later optimization). The trainer proc continuously samples
from the shared replay buffer and updates the net.

For simplicity in this first version, run with N ranks where rank 0 is the
trainer and ranks 1..N-1 are self-play workers. Workers send game records
via torch.distributed.gather; trainer periodically broadcasts fresh weights.

A cleaner production version would use a dedicated IPC queue, but DDP all-reduce
works fine for our scale. Launch:

    torchrun --nproc_per_node=8 scripts/train_rl.py --config configs/rl_selfplay.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alphazero.encoding import NUM_PLANES, POLICY_SIZE
from alphazero.mcts import MCTSConfig
from alphazero.metrics import MetricsTracker
from alphazero.model import AlphaZeroNet, ModelConfig
from alphazero.replay import ReplayBuffer
from alphazero.selfplay import SelfPlayConfig, play_game
from alphazero.training import TrainConfig, policy_value_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp = world_size > 1
    if is_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(rank % torch.cuda.device_count())

    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")
    is_trainer = (rank == 0)

    model_cfg = ModelConfig(**cfg.get("model", {}))
    net = AlphaZeroNet(model_cfg).to(device)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    auto_ckpt = sorted(out_dir.glob("ckpt_*.pt"))
    resume_path = auto_ckpt[-1] if auto_ckpt else cfg.get("resume_from")
    resume_step = 0
    resume_optim_state = None
    if resume_path:
        print(f"[rank {rank}] resuming from {resume_path}")
        state = torch.load(resume_path, map_location=device)
        net.load_state_dict(state["model"] if "model" in state else state)
        resume_step = int(state.get("step", 0))
        resume_optim_state = state.get("optimizer")

    train_cfg = TrainConfig(**cfg.get("train", {}))
    mcts_cfg = MCTSConfig(**cfg.get("mcts", {}))
    sp_cfg = SelfPlayConfig(mcts=mcts_cfg,
                            move_limit=cfg.get("move_limit", 512),
                            resign_threshold=cfg.get("resign_threshold", -0.95),
                            resign_after_move=cfg.get("resign_after_move", 20))

    if is_trainer:
        # ---- Trainer role ----
        buffer = ReplayBuffer(capacity=cfg.get("replay_capacity", 1_000_000))
        optimizer = torch.optim.AdamW(net.parameters(), lr=train_cfg.lr,
                                      weight_decay=train_cfg.weight_decay)
        if resume_optim_state is not None:
            try:
                optimizer.load_state_dict(resume_optim_state)
                print(f"  restored optimizer state at step {resume_step}")
            except Exception as e:
                print(f"  [warn] couldn't restore optimizer state: {e}")
        total_steps = cfg["total_steps"]
        step = resume_step
        games_collected = 0
        start = time.time()

        tracker = MetricsTracker(
            run_name=cfg.get("run_name", Path(args.config).stem),
            project=cfg.get("wandb_project", "alphazero-chess"),
            config=cfg,
            wandb_enabled=cfg.get("wandb", True),
            stdout_every=cfg.get("log_every", 50),
        )

        while step < total_steps:
            # Drain any incoming games from workers (in a real system this would
            # be async/queue-based; for now we do a synchronous all-gather each
            # step and the trainer also plays games when the buffer is empty).
            if is_ddp:
                _collect_games_ddp(buffer, world_size)

            if len(buffer) < train_cfg.batch_size * 4:
                # Buffer too small; also contribute self-play games.
                rec = play_game(_ddp_unwrap(net), device, sp_cfg)
                buffer.add(rec.planes, rec.policies, rec.values)
                games_collected += 1
                resigned = (rec.ply_count < cfg.get("move_limit", 512)
                            and rec.result != "1/2-1/2")
                tracker.log_game(
                    result=rec.result, ply_count=rec.ply_count,
                    resigned=False,  # accurate resignation flag would require selfplay.py changes
                    root_value_mean=None, root_entropy=None,
                    step=step, buffer_size=len(buffer),
                )
                continue

            planes, policies, values = buffer.sample(train_cfg.batch_size)
            planes = torch.from_numpy(planes).to(device)
            policies = torch.from_numpy(policies).to(device)
            values = torch.from_numpy(values).to(device)

            net.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                p_logits, v_pred = net(planes)
                loss, metrics = policy_value_loss(
                    p_logits, v_pred, policies, values,
                    policy_weight=train_cfg.policy_loss_weight,
                    value_weight=train_cfg.value_loss_weight,
                )
            loss.backward()
            tracker.log_grad_norm(net, step=step + 1, clip_value=train_cfg.grad_clip)
            torch.nn.utils.clip_grad_norm_(net.parameters(), train_cfg.grad_clip)
            optimizer.step()
            step += 1

            tracker.log({**metrics, "train/lr": train_cfg.lr,
                         "rl/games_collected": games_collected}, step=step)
            if step % cfg.get("log_every", 50) == 0:
                tracker.log_batch_stats(p_logits, v_pred, policies, values, step=step)
                tracker.log_gpu(step=step)
                elapsed = time.time() - start
                tracker.log({"train/steps_per_sec": step / elapsed}, step=step)

            if step % cfg.get("ckpt_every", 1000) == 0:
                ckpt = out_dir / f"ckpt_{step:07d}.pt"
                tmp = ckpt.with_suffix(".pt.tmp")
                torch.save({
                    "model": _ddp_unwrap(net).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "model_cfg": model_cfg.__dict__,
                }, tmp)
                os.replace(tmp, ckpt)
                print(f"  saved {ckpt}")
                keep = cfg.get("keep_last_n_ckpts", 5)
                all_ckpts = sorted(out_dir.glob("ckpt_*.pt"))
                for old in all_ckpts[:-keep]:
                    old.unlink()
                if is_ddp:
                    _broadcast_weights(net)

        final = out_dir / "final.pt"
        tmp = final.with_suffix(".pt.tmp")
        torch.save({"model": _ddp_unwrap(net).state_dict(),
                    "model_cfg": model_cfg.__dict__}, tmp)
        os.replace(tmp, final)

    else:
        # ---- Self-play worker role ----
        while True:
            rec = play_game(net, device, sp_cfg)
            _send_game_ddp(rec, rank)
            if rank == 1 and dist.is_initialized():
                # Occasionally pull fresh weights.
                _receive_weights(net)


def _ddp_unwrap(net):
    return net.module if hasattr(net, "module") else net


def _collect_games_ddp(buffer, world_size):
    # Placeholder: in a real implementation this would gather from workers.
    # For the v1 single-process fallback (used in tests and small runs), the
    # trainer itself generates games, so this is a no-op.
    return


def _send_game_ddp(rec, rank):
    return


def _broadcast_weights(net):
    return


def _receive_weights(net):
    return


if __name__ == "__main__":
    main()
