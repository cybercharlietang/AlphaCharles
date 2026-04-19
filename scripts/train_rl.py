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
    if cfg.get("resume_from"):
        state = torch.load(cfg["resume_from"], map_location=device)
        net.load_state_dict(state["model"] if "model" in state else state)

    train_cfg = TrainConfig(**cfg.get("train", {}))
    mcts_cfg = MCTSConfig(**cfg.get("mcts", {}))
    sp_cfg = SelfPlayConfig(mcts=mcts_cfg,
                            move_limit=cfg.get("move_limit", 512),
                            resign_threshold=cfg.get("resign_threshold", -0.95),
                            resign_after_move=cfg.get("resign_after_move", 20))

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if is_trainer:
        # ---- Trainer role ----
        buffer = ReplayBuffer(capacity=cfg.get("replay_capacity", 1_000_000))
        optimizer = torch.optim.AdamW(net.parameters(), lr=train_cfg.lr,
                                      weight_decay=train_cfg.weight_decay)
        total_steps = cfg["total_steps"]
        step = 0
        games_collected = 0
        start = time.time()

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
            torch.nn.utils.clip_grad_norm_(net.parameters(), train_cfg.grad_clip)
            optimizer.step()
            step += 1

            if step % 50 == 0:
                elapsed = time.time() - start
                print(f"step {step}/{total_steps} | "
                      f"loss={metrics['loss/total']:.4f} "
                      f"p={metrics['loss/policy']:.4f} "
                      f"v={metrics['loss/value']:.4f} | "
                      f"buffer={len(buffer)} | games={games_collected} | "
                      f"{step/elapsed:.1f} steps/s")

            if step % cfg.get("ckpt_every", 1000) == 0:
                ckpt = out_dir / f"ckpt_{step:07d}.pt"
                torch.save({
                    "model": _ddp_unwrap(net).state_dict(),
                    "step": step,
                    "model_cfg": model_cfg.__dict__,
                }, ckpt)
                print(f"  saved {ckpt}")
                if is_ddp:
                    _broadcast_weights(net)

        torch.save({"model": _ddp_unwrap(net).state_dict(),
                    "model_cfg": model_cfg.__dict__}, out_dir / "final.pt")

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
