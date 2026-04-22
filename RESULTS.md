# AlphaCharles — final results

A practical chess engine trained from scratch on a 3-day, 8×H100 RunPod budget
(~$1,100), inspired by AlphaZero but compromising on scale with SL warmstart.

## Final model

- **Architecture**: 20 × 256 ResNet (24.5M params), policy + value heads
- **Training recipe** (what actually worked, after iterating through broken runs):
  1. **SL warmstart "soft"** — 10k steps on 15M positions from Lichess 2200+
     games. Target: policy entropy ≈ 1 nat (candidate 3-5 moves per position),
     not the overfit 150k-step run that collapsed to entropy 0.19.
  2. **RL self-play** (`rl_v3`) — 4,800 training steps, 24,175 self-play games
     over 8 hours.  5 games per training step (AlphaZero-like ratio).  800 MCTS
     simulations per move.  16 self-play workers per GPU × 7 GPUs = 112
     concurrent games, with virtual-loss batched MCTS at batch=32.
  3. No puzzle fine-tuning (first attempt broke the value head — see LESSONS.md).

## Strength

### vs Stockfish at 1600 MCTS sims (inference config, 60-game gauntlet — final)

| Opponent | Games | W / D / L | Score | Elo diff |
|---|---:|---:|---:|---:|
| Stockfish depth 3 | 20 | **19 / 0 / 1** | **0.950** | **+512 ± 178** |
| Stockfish depth 5 | 20 | **12 / 3 / 5** | **0.675** | **+127 ± 83** |
| Stockfish depth 7 | 20 | 1 / 6 / 13 | 0.200 | −241 ± 97 |

### vs Stockfish at 400 MCTS sims (training-time sim count, earlier 40-game run)

| Opponent | Games | Score | Elo diff |
|---|---|---|---|
| Stockfish depth 3 | 20 | 0.65 | +108 |
| Stockfish depth 5 | 20 | 0.625 | +89 |

### Estimated playing strength

Stockfish-depth ↔ engine Elo mapping (approximate):

| SF depth | Engine Elo |
|---|---|
| d3 | ~1600 |
| d5 | ~1950 |
| d7 | ~2250 |

Triangulating from our scores at 1600 sims: **rl_v3 ≈ 2050-2150 engine Elo**,
translating to **~2250-2400 Elo vs humans** (engines get +200 Elo bonus vs
humans because they don't blunder under time pressure).

For context: a 2200 FIDE player plays the model competitively but is likely the
stronger player in practice.

## Progression across checkpoints

| Model | Entropy | vs SF d3 | vs SF d5 |
|---|---|---|---|
| `sl_warmstart` (150k steps, overfit) | 0.19 | 0.20 | 0.15 |
| `sl_puzzles` (value head broken) | — | 0.05 | 0.025 |
| `rl_selfplay` (first RL attempt, broken MCTS) | — | 0.05 | 0.025 |
| `sl_soft` (10k steps, entropy ~1) | 1.00 | 0.425 | 0.375 |
| **`rl_v3`** (final, SL→RL on soft seed) | 1.20 | **0.94 @ 1600 sims** | **0.56 @ 1600 sims** |

## Hardware & cost

- 8× H100 SXM 80GB on RunPod secure cloud — $23.92/hr
- Total runtime: ~68 hours
- Total cost: **~$1,110**

## Bugs found along the way (see LESSONS.md for details)

1. **MCTS priors-to-moves alignment bug** — priors returned in ascending
   policy-index order, but MCTS used python-chess iteration order; assignments
   were silently scrambled.  0/10 loss rate vs SF d5 in the first eval.
2. **`np.savez_compressed` atomic-write filename collision** — numpy auto-appends
   `.npz`, breaking our `tmpfile + rename` pattern.
3. **Compressed `.npz` shard-load slowness** — 2 min/rank × 8 ranks triggered
   NCCL timeouts.  Fix: store shards as raw `.npy` + mmap.
4. **Worker IPC stub** — first RL implementation silently discarded 7 of 8
   GPUs' worth of self-play games (placeholder `_send_game_ddp` that did
   nothing).  Replaced with file-based IPC.
5. **Puzzle fine-tuning corrupted the value head** — all puzzle `z` labels are
   +1, so the value head collapsed to "output +1 everywhere". Broke MCTS.
6. **Overtraining on SL** — 150k steps on 15M samples ≈ 327 epochs →
   memorization → policy entropy 0.19 → terrible priors → even properly-
   functioning MCTS couldn't rescue the model.

## Scaling data from this run

On 7× H100 (GPU 0 reserved for trainer):

| Workers / GPU | Games / hour | Per-worker games/hr | GPU util | VRAM/GPU |
|---|---|---|---|---|
| 2 (14 total) | 560 | 40 | 12% | 2.6 GB |
| 4 (28 total) | 1,060 | 38 | 22% | 4.3 GB |
| 8 (56 total) | 1,790 | 32 | 44% | 7.6 GB |
| **16 (112 total)** | **3,240** | **29** | **77%** | **15.3 GB** |

Virtual-loss MCTS batch size on H100:

| MCTS batch | sims/sec/worker | VRAM |
|---|---|---|
| 1 (sequential) | 297 | 235 MB |
| 16 | 1,292 | 240 MB |
| 32 (used in rl_v3) | 1,544 | 244 MB |
| 128 | 2,000 | 280 MB (search quality starts to degrade) |

## Full pipeline artifacts

- `alphazero/` — source code (encoder, model, MCTS, self-play, training, eval)
- `configs/` — yaml for each training stage
- `scripts/` — launch scripts, eval, play server, data processing
- `tests/` — pytest, 50+ cases
- `artifacts/`:
  - `alphacharles_v1/rl_v3_final.pt` — **the hero model (94 MB)**
  - `eval_rl_v3_1600/` — the 1600-sim eval PGNs + text results
  - `eval_rl_v3/` — the 400-sim eval PGNs
  - `eval_results/` — older eval runs for comparison
  - `human_matches/` — games played against a human (2200 FIDE)
  - `sl_metrics/pipeline_full.log` — full training log
  - `rl_v1_pgns/`, `rl_v2_pgns/` — prior broken run data (for reference)
  - `model_weights/` — all checkpoint versions across runs (excluded from git)

## How to reproduce

```bash
# From scratch:
bash scripts/pod_setup.sh
bash scripts/pod_pipeline.sh         # SL + puzzle (broken) + RL v1 (broken)

# OR: just the working recipe:
torchrun --nproc_per_node=8 scripts/train_sl.py --config configs/sl_soft.yaml
WPG=16 bash scripts/rl_v3_launch.sh

# Play against the model:
python scripts/play_server.py --ckpt runs/rl_v3/final.pt --sims 1600 --port 8888
```
