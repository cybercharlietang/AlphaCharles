"""Parallel RL via file-based IPC.

Architecture
------------
One trainer process (rank 0) + N worker processes. Each process holds its own
net on its own GPU. Workers run self-play continuously; trainer consumes games
from disk, trains, periodically publishes fresh weights.

Layout under OUT_DIR:
    weights/current.pt             -- latest weights, read by workers
    games/pending/*.npz + *.pgn    -- worker-produced games awaiting ingestion
    games/consumed/*.pgn           -- PGNs kept for browsing
    ckpts/ckpt_<step>.pt           -- periodic trainer snapshots
    final.pt                       -- trainer's final weights
    pipeline.log                   -- aggregated stdout

Usage:
    trainer:  python scripts/rl_parallel.py --role trainer --gpu 0 --config <yaml>
    worker:   python scripts/rl_parallel.py --role worker  --gpu 1 --worker-id 1 --config <yaml>

Typically launched via a supervising bash script that spawns 1 trainer + N workers.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alphazero.mcts import MCTSConfig
from alphazero.metrics import MetricsTracker
from alphazero.model import AlphaZeroNet, ModelConfig
from alphazero.replay import ReplayBuffer
from alphazero.selfplay import SelfPlayConfig, play_game
from alphazero.training import TrainConfig, policy_value_loss


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def build_net(cfg: dict, device: torch.device, weights_path: str | None) -> AlphaZeroNet:
    mc = ModelConfig(**cfg.get("model", {}))
    net = AlphaZeroNet(mc).to(device)
    if weights_path and Path(weights_path).exists():
        state = torch.load(weights_path, map_location=device, weights_only=False)
        net.load_state_dict(state["model"] if "model" in state else state)
    return net


def atomic_save(state: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


# ============================================================================
# Worker
# ============================================================================

def run_worker(cfg: dict, gpu: int, worker_id: int) -> None:
    device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    out_dir = Path(cfg["output_dir"])
    weights_dir = out_dir / "weights"
    pending_dir = out_dir / "games" / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)

    weights_current = weights_dir / "current.pt"
    # Wait for initial weights to exist (trainer publishes them on startup).
    while not weights_current.exists():
        print(f"[worker {worker_id}] waiting for initial weights...", flush=True)
        time.sleep(5)

    net = build_net(cfg, device, str(weights_current))
    net.eval()
    last_weights_mtime = weights_current.stat().st_mtime

    mcts_cfg = MCTSConfig(**cfg.get("mcts", {}))
    sp_cfg = SelfPlayConfig(
        mcts=mcts_cfg,
        move_limit=cfg.get("move_limit", 400),
        resign_threshold=cfg.get("resign_threshold", -0.95),
        resign_after_move=cfg.get("resign_after_move", 20),
    )

    game_idx = 0
    print(f"[worker {worker_id}] ready on {device}", flush=True)
    while True:
        # Reload weights if they've changed.
        try:
            m = weights_current.stat().st_mtime
            if m > last_weights_mtime:
                state = torch.load(weights_current, map_location=device, weights_only=False)
                net.load_state_dict(state["model"] if "model" in state else state)
                last_weights_mtime = m
                print(f"[worker {worker_id}] refreshed weights (mtime {int(m)})", flush=True)
        except FileNotFoundError:
            pass

        t0 = time.time()
        rec = play_game(net, device, sp_cfg)
        dt = time.time() - t0
        game_idx += 1

        # Write tensor training data + PGN atomically.
        # np.savez auto-appends .npz; so we pass the stem and rename the resulting file.
        stem = f"w{worker_id}_g{game_idx:05d}_{int(time.time())}"
        tmp_stem = pending_dir / f"{stem}.tmp"
        npz_final = pending_dir / f"{stem}.npz"
        np.savez(tmp_stem, planes=rec.planes, policies=rec.policies, values=rec.values)
        os.replace(str(tmp_stem) + ".npz", npz_final)
        pgn_path = pending_dir / f"{stem}.pgn"
        with open(pgn_path, "w") as fh:
            fh.write(rec.pgn)
        print(f"[worker {worker_id}] game {game_idx} {rec.result} "
              f"plies={rec.ply_count} time={dt:.1f}s "
              f"H_prior={rec.avg_prior_entropy:.3f} H_mcts={rec.avg_mcts_entropy:.3f} "
              f"drop={rec.avg_entropy_drop:+.3f}", flush=True)


# ============================================================================
# Trainer
# ============================================================================

def run_trainer(cfg: dict, gpu: int) -> None:
    device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    out_dir = Path(cfg["output_dir"])
    weights_dir = out_dir / "weights"
    ckpt_dir = out_dir / "ckpts"
    pending_dir = out_dir / "games" / "pending"
    consumed_dir = out_dir / "games" / "consumed"
    for d in (weights_dir, ckpt_dir, pending_dir, consumed_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Seed weights: load from cfg.resume_from or cfg.seed_from.
    seed_path = cfg.get("resume_from") or cfg.get("seed_from")
    if not seed_path:
        raise ValueError("trainer needs 'resume_from' or 'seed_from' in config")

    net = build_net(cfg, device, seed_path)
    net.train()

    # Publish initial weights so workers can start.
    weights_current = weights_dir / "current.pt"
    atomic_save({"model": net.state_dict(),
                 "model_cfg": net.cfg.__dict__,
                 "step": 0}, weights_current)
    print(f"[trainer] published initial weights from {seed_path}", flush=True)

    train_cfg = TrainConfig(**cfg.get("train", {}))
    optimizer = torch.optim.AdamW(net.parameters(),
                                  lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    total_steps = cfg["total_steps"]
    replay_capacity = cfg.get("replay_capacity", 500_000)
    buffer = ReplayBuffer(capacity=replay_capacity)

    tracker = MetricsTracker(
        run_name=cfg.get("run_name", "rl_parallel"),
        project=cfg.get("wandb_project", "alphazero-chess"),
        config=cfg,
        wandb_enabled=cfg.get("wandb", True),
        stdout_every=cfg.get("log_every", 50),
    )

    step = 0
    games_ingested = 0
    last_publish = 0
    publish_every = cfg.get("publish_weights_every", 500)
    ckpt_every = cfg.get("ckpt_every", 2000)
    warmup_games = cfg.get("warmup_games", 32)
    train_step_per_games = float(cfg.get("train_step_per_games", 1.0))
    wall_clock_limit_s = float(cfg.get("wall_clock_limit_s", 999999))
    lr_warmup_steps = int(cfg.get("lr_warmup_steps", 0))
    lr_peak = train_cfg.lr

    start = time.time()
    print(f"[trainer] entering main loop: total_steps={total_steps}, "
          f"warmup_games={warmup_games}, games_per_step={train_step_per_games}, "
          f"wall_clock_limit_s={wall_clock_limit_s}", flush=True)

    while step < total_steps:
        # Ingest any pending games.
        pending = sorted(pending_dir.glob("*.npz"))
        for p in pending:
            try:
                data = np.load(p)
                buffer.add(data["planes"], data["policies"], data["values"])
                games_ingested += 1
                # Move PGN + npz to consumed so we don't re-read, but keep pgn for viewing.
                pgn = p.with_suffix(".pgn")
                if pgn.exists():
                    shutil.move(str(pgn), consumed_dir / pgn.name)
                p.unlink()
            except Exception as e:
                print(f"[trainer] failed to ingest {p}: {e}", flush=True)

        if games_ingested < warmup_games:
            if step == 0:
                print(f"[trainer] warmup: {games_ingested}/{warmup_games} games, "
                      f"buffer size {len(buffer)}", flush=True)
            time.sleep(3)
            continue

        # Throttle: only train when games_ingested ≥ warmup + step * games_per_step.
        required_games = warmup_games + step * train_step_per_games
        if games_ingested < required_games:
            time.sleep(2)
            continue

        # Wall-clock exit
        if time.time() - start > wall_clock_limit_s:
            print(f"[trainer] wall-clock limit hit ({wall_clock_limit_s}s), stopping", flush=True)
            break

        # LR warmup schedule
        if lr_warmup_steps > 0 and step < lr_warmup_steps:
            cur_lr = lr_peak * (0.3 + 0.7 * step / lr_warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

        # Train one step.
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
                     "rl/games_ingested": games_ingested,
                     "rl/buffer_size": len(buffer)}, step=step)
        if step % cfg.get("log_every", 50) == 0:
            tracker.log_batch_stats(p_logits, v_pred, policies, values, step=step)
            tracker.log_gpu(step=step)
            elapsed = time.time() - start
            print(f"[trainer] step {step}/{total_steps} "
                  f"loss={metrics['loss/total']:.3f} "
                  f"p={metrics['loss/policy']:.3f} v={metrics['loss/value']:.3f} "
                  f"buf={len(buffer)} games={games_ingested} "
                  f"steps/s={step/elapsed:.2f}", flush=True)

        # Periodically publish weights so workers pick them up.
        if step - last_publish >= publish_every:
            atomic_save({"model": net.state_dict(),
                         "model_cfg": net.cfg.__dict__,
                         "step": step}, weights_current)
            last_publish = step

        # Periodic checkpoint.
        if step % ckpt_every == 0:
            atomic_save({"model": net.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "model_cfg": net.cfg.__dict__,
                         "step": step}, ckpt_dir / f"ckpt_{step:07d}.pt")

    # Final publish + save.
    atomic_save({"model": net.state_dict(),
                 "model_cfg": net.cfg.__dict__,
                 "step": step}, weights_current)
    atomic_save({"model": net.state_dict(),
                 "model_cfg": net.cfg.__dict__,
                 "step": step}, out_dir / "final.pt")
    print(f"[trainer] done. final saved. games_ingested={games_ingested}", flush=True)
    tracker.finish()


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["trainer", "worker"], required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.role == "trainer":
        run_trainer(cfg, args.gpu)
    else:
        run_worker(cfg, args.gpu, args.worker_id)


if __name__ == "__main__":
    main()
