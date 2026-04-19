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
    if cfg.get("resume_from"):
        state = torch.load(cfg["resume_from"], map_location=device)
        net.load_state_dict(state["model"] if "model" in state else state)
        if is_main:
            print(f"resumed from {cfg['resume_from']}")

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

    total_steps = cfg["total_steps"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=train_cfg.lr, total_steps=total_steps,
        pct_start=0.05, anneal_strategy="cos",
    )

    scaler = None  # bfloat16 doesn't need a scaler
    use_amp = cfg.get("bf16", True) and torch.cuda.is_available()

    step = 0
    epoch = 0
    start = time.time()
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

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
            torch.nn.utils.clip_grad_norm_(net.parameters(), train_cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            step += 1
            if is_main and step % 50 == 0:
                elapsed = time.time() - start
                samples = step * train_cfg.batch_size * world_size
                print(f"step {step}/{total_steps} | "
                      f"loss={metrics['loss/total']:.4f} "
                      f"p={metrics['loss/policy']:.4f} "
                      f"v={metrics['loss/value']:.4f} | "
                      f"lr={scheduler.get_last_lr()[0]:.2e} | "
                      f"{samples/elapsed:.0f} samples/s")
            if is_main and step % cfg.get("ckpt_every", 5000) == 0:
                ckpt = out_dir / f"ckpt_{step:07d}.pt"
                torch.save({
                    "model": (net.module if is_ddp else net).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "model_cfg": model_cfg.__dict__,
                }, ckpt)
                print(f"  saved {ckpt}")
        epoch += 1

    if is_main:
        ckpt = out_dir / "final.pt"
        torch.save({
            "model": (net.module if is_ddp else net).state_dict(),
            "model_cfg": model_cfg.__dict__,
        }, ckpt)
        print(f"saved final checkpoint: {ckpt}")
    cleanup_ddp(is_ddp)


if __name__ == "__main__":
    main()
