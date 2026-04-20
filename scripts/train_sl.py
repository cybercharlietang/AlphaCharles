"""Supervised warm-start / puzzle fine-tune trainer.

Same trainer serves both stages:
  --data-dir points at a directory of shard .npz files produced by
  alphazero.data_pgn or alphazero.data_puzzles.

Reads config from YAML; supports multi-GPU via torchrun.

Example:
    torchrun --nproc_per_node=8 scripts/train_sl.py \
        --config configs/sl_warmstart.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alphazero.dataset import build_dataset_from_dir
from alphazero.metrics import MetricsTracker
from alphazero.model import AlphaZeroNet, ModelConfig
from alphazero.training import TrainConfig, train_step


def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return rank, world_size, True
    return 0, 1, False


def cleanup_ddp(is_ddp: bool):
    if is_ddp:
        dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    rank, world_size, is_ddp = setup_ddp()
    is_main = rank == 0
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")

    model_cfg = ModelConfig(**cfg.get("model", {}))
    net = AlphaZeroNet(model_cfg).to(device)

    # Auto-resume: prefer the latest ckpt_*.pt in output_dir over the `resume_from`
    # field, so that if training crashes halfway, the next run continues.
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    auto_ckpt = sorted(out_dir.glob("ckpt_*.pt"))
    resume_path = None
    resume_step = 0
    resume_optim_state = None
    if auto_ckpt:
        resume_path = auto_ckpt[-1]
        if is_main:
            print(f"auto-resuming from latest ckpt: {resume_path}")
    elif cfg.get("resume_from"):
        resume_path = cfg["resume_from"]
        if is_main:
            print(f"resuming from seed ckpt: {resume_path}")

    if resume_path:
        state = torch.load(resume_path, map_location=device)
        net.load_state_dict(state["model"] if "model" in state else state)
        resume_step = int(state.get("step", 0))
        resume_optim_state = state.get("optimizer")

    if is_ddp:
        net = DDP(net, device_ids=[device.index])

    train_cfg = TrainConfig(**cfg.get("train", {}))

    dataset = build_dataset_from_dir(cfg["data_dir"])
    if is_main:
        print(f"dataset size: {len(dataset):,}")

    sampler = DistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=True) if is_ddp else None
    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    if resume_optim_state is not None:
        try:
            optimizer.load_state_dict(resume_optim_state)
            if is_main:
                print(f"  restored optimizer state at step {resume_step}")
        except Exception as e:
            if is_main:
                print(f"  [warn] couldn't restore optimizer state: {e}")

    total_steps = cfg["total_steps"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=train_cfg.lr, total_steps=total_steps,
        pct_start=0.05, anneal_strategy="cos",
        last_epoch=resume_step - 1 if resume_step > 0 else -1,
    )

    scaler = None  # bfloat16 doesn't need a scaler
    use_amp = cfg.get("bf16", True) and torch.cuda.is_available()

    tracker = MetricsTracker(
        run_name=cfg.get("run_name", Path(args.config).stem),
        project=cfg.get("wandb_project", "alphazero-chess"),
        config=cfg,
        wandb_enabled=cfg.get("wandb", True),
        stdout_every=cfg.get("log_every", 50),
    )

    step = resume_step
    epoch = 0
    start = time.time()

    while step < total_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            if step >= total_steps:
                break
            # Inline amp-aware training step (training.py's train_step does bf16 autocast).
            net.train()
            planes = batch["planes"].to(device, non_blocking=True)
            policy_t = batch["policy"].to(device, non_blocking=True)
            value_t = batch["value"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if use_amp else torch.autocast(device_type="cpu", enabled=False)
            with ctx:
                policy_logits, value_pred = net(planes)
                from alphazero.training import policy_value_loss
                loss, metrics = policy_value_loss(
                    policy_logits, value_pred, policy_t, value_t,
                    policy_weight=train_cfg.policy_loss_weight,
                    value_weight=train_cfg.value_loss_weight,
                )
            loss.backward()
            # Log grad norm BEFORE clip so we can see spikes.
            grad_norm = tracker.log_grad_norm(net, step=step + 1, clip_value=train_cfg.grad_clip)
            torch.nn.utils.clip_grad_norm_(net.parameters(), train_cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            step += 1
            # Log loss + lr every step (cheap); batch stats + gpu every log_every.
            tracker.log({**metrics, "train/lr": scheduler.get_last_lr()[0]}, step=step)
            if is_main and step % cfg.get("log_every", 50) == 0:
                tracker.log_batch_stats(policy_logits, value_pred, policy_t, value_t, step=step)
                tracker.log_gpu(step=step)
                elapsed = time.time() - start
                samples = step * train_cfg.batch_size * world_size
                tracker.log({"train/samples_per_sec": samples / elapsed}, step=step)
            if is_main and step % cfg.get("ckpt_every", 5000) == 0:
                ckpt = out_dir / f"ckpt_{step:07d}.pt"
                tmp = ckpt.with_suffix(".pt.tmp")
                torch.save({
                    "model": (net.module if is_ddp else net).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "model_cfg": model_cfg.__dict__,
                }, tmp)
                os.replace(tmp, ckpt)  # atomic; never leave a partial ckpt
                print(f"  saved {ckpt}")
                # Prune old checkpoints to bound disk usage.
                keep = cfg.get("keep_last_n_ckpts", 5)
                all_ckpts = sorted(out_dir.glob("ckpt_*.pt"))
                for old in all_ckpts[:-keep]:
                    old.unlink()
        epoch += 1

    if is_main:
        ckpt = out_dir / "final.pt"
        tmp = ckpt.with_suffix(".pt.tmp")
        torch.save({
            "model": (net.module if is_ddp else net).state_dict(),
            "model_cfg": model_cfg.__dict__,
        }, tmp)
        os.replace(tmp, ckpt)
        print(f"saved final checkpoint: {ckpt}")
    tracker.finish()
    cleanup_ddp(is_ddp)


if __name__ == "__main__":
    main()
