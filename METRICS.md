# Metrics guide

This doc is the cheat sheet for reading W&B dashboards during training.

## Loss components

| Metric | Healthy SL range | Healthy RL range | Trouble signs |
|---|---|---|---|
| `loss/total` | 8 → 3 over warmup | ~2-3 stable | Increasing = LR too high or bad data |
| `loss/policy` | 8.4 → ~2 | ~1-2 | Drops but val stalls = overfit |
| `loss/value` | ~1 → ~0.7 | ~0.7 | Stalls while policy drops = value head bottleneck |

`ln(4672) ≈ 8.45` is the uniform-random baseline for policy loss.

## Policy entropy (single best signal)

`policy/entropy` measures how peaked the net's distribution over legal moves is.

| Phase | Typical range |
|---|---|
| Early SL | 4-5 nats |
| Mature SL | 1.5-2.5 nats |
| Puzzle FT | 1.0-2.0 nats |
| RL good | 1.0-2.0 nats |
| **RL mode collapse** | **<0.3 nats** — needs intervention |

If entropy drops below 0.3 in RL, bump Dirichlet epsilon or restart from a prior checkpoint.

## Value head health

- `value/mean_pred` — should hover near 0. Nonzero = color bias.
- `value/corr` — Pearson correlation between `v` and `z`. 0.3 → 0.7 over training. This is the single best honest number.

## Training dynamics

- `train/lr` — OneCycleLR schedule. Warmup 5% of steps, then cosine decay.
- `train/grad_norm` — pre-clip. Spikes >20 = instability. If always pinned at `grad_clip=5.0`, LR too high.
- `train/samples_per_sec` — SL target: ~1.5M samples/s on 8×H100. Lower = data loader starved.

## GPU stats (per device)

- `gpu/util/N` — target >90%. Lower = CPU bound or data starved.
- `gpu/mem_used_gb/N` — stable 35-45 GB at batch_size=2048. Growing = leak.
- `gpu/power_w/N` — ~650-700W at full tilt.
- `gpu/temp_c/N` — <85°C. Thermal throttle above that.

## RL-specific

- `rl/plies_per_game` — ~90 is normal AZ-chess. Shorter = decisive / resignation firing.
- `rl/result_draw` — 40-60% is healthy. >60% = too defensive. <20% = too tactical, likely blundering.
- `rl/buffer_size` — fills to `replay_capacity` then stays there.

## Eval

- `eval/sf_depth_N/score` — score vs Stockfish. 0.5 = equal, 0.75 ≈ +191 Elo.
- `eval/sf_depth_N/elo_diff` — derived Elo diff. **The headline number.**
- `eval/sf_depth_N/elo_stderr` — halves as n_games quadruples.

## Interpretation cheats

- **loss falling, entropy falling, corr rising** → training is on track.
- **loss plateaus, entropy keeps falling** → overfit to deterministic moves, risk of RL collapse.
- **loss rising, grad_norm rising** → instability. Drop LR or inspect recent batches.
- **samples/s drops** → data loader issue, not model issue.
