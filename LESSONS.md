# Lessons from the AlphaCharles training run

Written after the first full SL → puzzle FT → RL pipeline completed and we
began post-hoc analysis. Documenting bugs, surprises, and decisions that were
non-obvious in hindsight, so future iterations can skip them.

## Bug #1 — `np.savez_compressed` atomic-write filename collision

Background: all shard and checkpoint writers do `tmpfile → os.replace(final)`
to avoid half-written files on crash.

```python
# BROKEN
tmp = out + ".tmp"           # "shard_00000.npz.tmp"
np.savez_compressed(tmp, ...)  # numpy silently appends .npz → writes "shard_00000.npz.tmp.npz"
os.replace(tmp, out)           # FileNotFoundError: "shard_00000.npz.tmp" doesn't exist
```

`numpy.savez*` functions auto-append `.npz` to filenames that lack it. Our `.tmp`
suffix triggered the auto-append. `os.replace` then looked for the original
`.tmp` name and failed.

**Fix**: write to a stem path without `.npz` so numpy's auto-append becomes the
intended final suffix, then rename:

```python
tmp_stem = os.path.join(out_dir, f"shard_{i:05d}.tmp")   # no .npz
np.savez_compressed(tmp_stem, ...)                         # writes "shard_00000.tmp.npz"
os.replace(tmp_stem + ".npz", final_path)
```

Affected files: `data_pgn.py`, `data_puzzles.py`, `rl_parallel.py`.

**Cost of the bug**: lost the first data-prep attempt (~40 min wall, ~$16).
Caught because the pipeline stage failed fast on the first flush.

## Bug #2 — Shard-loading 100× slowdown from compressed .npz

`np.load("shard.npz")` is lazy at the outer level but `data["planes"]`
decompresses the whole array on first access. With 15 shards × ~128 MB of
compressed planes, dataset construction took **122 seconds per rank**. With
8 ranks doing it concurrently, the wall-clock stretched long enough for
DDP's NCCL collective to hit the 600s watchdog timeout.

**Symptoms before diagnosis**: training appeared stalled; 8 GPUs at 97%
CPU (not GPU); NCCL watchdog killed rank 3 with `BROADCAST timeout`.

**Fix**: convert shards to uncompressed `.npy` files and load with
`np.load(path, mmap_mode='r')`. Memory-mapping the (planes, policy_idx, values)
arrays separately makes dataset construction ~instant. Disk cost is 4× larger
for the raw arrays, still well within our 400 GB volume.

## Bug #3 — The MCTS priors/moves misalignment (the big one)

The most expensive bug by far. Manifested as: **all three trained models lost
0 of 10, 1 of 10, 0 of 10 vs Stockfish depth 5**, despite SL training metrics
showing excellent convergence (`policy_loss=0.19, value_corr=0.91`).

Root cause: two functions both produced "a list of legal moves" in two
**different orders**.

```python
# mcts._evaluate()
mask = legal_move_mask(board)        # True at indices 129, 136, 497, 501, 584, 592, ...
legal_logits = logits[mask]          # logits in ASCENDING INDEX ORDER

# mcts._expand()
node.legal_moves = list(board.legal_moves)   # python-chess iteration order
node.P = priors                              # WRONG — misaligned with legal_moves
```

Python-chess iterates moves by piece type × from-square (pawn pushes before
knight moves, etc.). NumPy indexing with a boolean mask returns elements in
ascending index order. The two orderings are totally different, and the
mis-assignment was silent — neither numpy nor python complained.

**Effect**: every MCTS call assigned the net's priors to the WRONG moves. A
high prior that should have pushed the search toward `c2c4` got attached to
`f2f3`. The net's actual learning was never used at inference time.

**What masked the bug**:
- SL training never invokes MCTS. It regresses the net directly on one-hot
  `move_to_index(human_move)` targets. The bug has zero effect on gradients.
- Unit tests only verified `move_to_index ↔ index_to_move` roundtrip on
  individual moves, not that MCTS's `priors` array aligned with its
  `legal_moves` array.
- The training metrics looked great because they WERE great — the net learned
  correct policies. The bug was purely in how we consumed them.

**Fix**:
```python
legal_moves = list(board.legal_moves)
idx = np.fromiter((move_to_index(m, board) for m in legal_moves), dtype=np.int64)
legal_logits = logits[idx]     # explicit indexing → aligned by construction
```

**Regression test added**: a "spiked net" with a single hot logit on `c2c4`'s
index must cause MCTS to visit `c2c4` most. Takes ~1s.

**Cost of the bug**:
- RL stage was not just useless but actively damaging — it trained the net on
  MCTS visit distributions that were scrambled random noise. The resulting
  `rl_selfplay` model scores 0.05 vs SF d3 (vs 0.20 for the un-trained
  `sl_warmstart`).
- ~$50 of compute on the broken RL run.
- ~1 day of "everything looks fine in the metrics, but something is off"
  before the eval surfaced it.

**Generalizable lesson**: when two independent code paths both compute "a
list of things", verify the ORDER is the same, not just the CONTENTS. For
anything that gets indexed by position (priors, visit counts, legal move
lists), write a test that spikes one slot and verifies it shows up at the
correct downstream slot. The roundtrip check alone is insufficient.

## Bug #4 — Worker game records going to /dev/null

My first `train_rl.py` was a "single-process DDP with role-by-rank" design.
Rank 0 trained; ranks 1-7 played self-play. But the function meant to move
worker game records to rank 0's buffer was a placeholder:

```python
def _send_game_ddp(rec, rank):
    return  # TODO
```

Workers dutifully played games and piped them into a no-op. Only rank 0's
self-play games contributed to the replay buffer.

**Effect**: effectively a 1-GPU RL run (rank 0 doing both training and
self-play) while paying for 8 GPUs. Buffer filled with ~50 games from rank 0,
then trainer cycled through those 50 games for 100k steps → memorization.

**Fix**: replaced DDP approach with file-based IPC. Each worker writes
`game_<worker>_<ts>.npz` + `.pgn` to `runs/rl_parallel/games/pending/`. Trainer
polls, ingests into buffer, moves files to `consumed/`. Weights are republished
every N steps via an atomic file write that workers mtime-poll.

This is also strictly easier to debug and doesn't depend on NCCL being healthy.

## Surprise #1 — Puzzle fine-tuning hurt the model

Going in, I expected puzzle FT to boost tactical sharpness. It regressed
performance significantly:

| Model | Score vs SF d3 | vs SF d5 | vs SF d7 |
|---|---|---|---|
| sl_warmstart | 0.20 | 0.15 | 0.05 |
| sl_puzzles | **0.05** | **0.025** | **0.025** |

### Why it hurt

The puzzle dataset has a pathological label distribution for value learning:
**every z target is +1** (by construction — the solver always wins the puzzle).
With `value_loss_weight=0.3`, the value head was still trained on these
constant targets and learned to output `v ≈ 1.0` for every position.

Post-hoc diagnostic (from the broken-MCTS era) showed `value=1.000` on every
position the model saw, even startpos.

**Cascading failure**:
1. Broken value head → MCTS Q values don't discriminate good vs bad positions
2. With uninformative Q, PUCT selection collapses to prior-following-only
3. MCTS becomes a near-greedy policy sampler, losing the benefit of search
4. Net effective strength drops because search was meaningful upside

The POLICY head also drifted: puzzle positions are tactically biased (sharp
attacking positions), so training on them pulled the policy toward "find the
aggressive move" heuristics even in quiet positions where passive moves are
correct. Combined with broken value, the model plays suicidal king walks in
middle games.

### What a correct puzzle FT would look like

- **Option A (cleanest)**: `value_loss_weight: 0.0`. Train puzzles for policy
  only, never touch the value head. Freeze its input gradient too if possible.
- **Option B**: mix puzzle batches with regular game batches at a low ratio
  (e.g. 20% puzzles / 80% games). The puzzle-z=+1 bias gets diluted by the
  game-z distribution.
- **Option C**: use MCTS on each puzzle position at generation time to produce
  a proper value target. Slow (each puzzle needs ~400 sims), but gives us an
  estimate of `v*(s)` for the puzzle position that isn't a constant.
- **Option D**: skip puzzles entirely. Rely on RL self-play to sharpen tactics.

## History — what AlphaGo and AlphaZero actually fine-tuned on (for chess)

To answer your original question precisely:

### AlphaGo (2016) — Go only, not chess
- **Phase 1 (SL)**: policy network trained on 30M moves from KGS Go games
  (amateur and pro 6-9 dan). Cross-entropy on the move actually played.
- **Phase 2 (RL policy)**: the SL network fine-tuned with policy gradient
  against randomly-chosen prior versions of itself (~one million self-play
  games). Goal: optimize for winning, not for imitation.
- **Phase 3 (value network)**: separate value network trained on 30M positions
  sampled from **self-play games** (not human games). Target: final outcome of
  the self-play game that the position was sampled from. Crucial design
  choice — positions from human games would have introduced biased outcomes.
- No "puzzle" phase. No problem-set fine-tuning. The RL self-play phase was
  the only fine-tuning.

### AlphaGo Zero (2017)
- Title literally says "without human knowledge."
- Start from random-init network weights.
- Pure self-play from scratch. No SL. No fine-tuning.
- Single combined network (policy + value heads) rather than separate networks.

### AlphaZero (2017) — the chess/shogi/Go paper
- Inherited the AlphaGo Zero architecture and training recipe unchanged,
  generalized to chess and shogi.
- **Pure self-play from random initialization**. No human games. No puzzles.
  No fine-tuning of any kind.
- Ran ~700k training steps on 5000 first-gen TPUs + 64 second-gen TPUs for
  9 hours. That's ~44,000 TPU-hours of compute.
- Reached ~3500 Elo, beating Stockfish 8 decisively.

### So for chess specifically
Neither paper fine-tuned on anything beyond self-play. Our pipeline
(SL warmstart + puzzle FT) is a **practical compromise** for smaller compute
budgets, inspired by AlphaGo's SL phase but not by AlphaZero. The puzzle FT
stage was my addition based on Leela Chess Zero community lore; it turned out
to be harmful with our specific data shape (z=+1 targets).

### A cleaner alternative for next time
If we were doing this again with the same budget, the strongest recipe would
be: **SL warmstart on quality human games + value-preserving tune + long RL
self-play**. Specifically:

1. SL on Lichess Elite (~50M positions, 1-2 days on 8× H100)
2. MCTS-distillation from SL checkpoint: generate ~1M positions with MCTS
   visits as policy targets and MCTS root-Q as value targets. Train on those
   for a few thousand steps. This sharpens the network against a proper
   target without degenerate labels.
3. RL self-play with proper parallel workers (as we have now)
4. Optional: distillation from Stockfish eval at each position, for very
   tight tactical grounding. Not "AlphaZero" anymore but pragmatic.

## Lessons for the pipeline design itself

### What worked
- **Atomic file writes + resume sentinels**: pipeline survived several
  restarts cleanly
- **File-based IPC for parallel RL**: simpler and more robust than DDP for
  this workload; workers don't care whether the trainer is alive
- **W&B offline mode**: metrics survived even when we never logged in
- **Live dashboard via SSH tunnel**: non-intrusive, zero-cost monitoring
- **2h cron monitoring**: caught the "RL is running but idle on 7 GPUs"
  issue within 2h of it starting

### What I'd do differently
- **Unit-test end-to-end MCTS, not just encoding**: the priors-to-moves
  regression test takes 1 second and would have caught Bug #3 instantly.
  Test the invariant "net predictions → MCTS action" directly.
- **Held-out validation set during SL**: would have caught the overfitting
  story; the 1M-epoch memorization pattern should have raised flags earlier
- **Smoke test the full inference pipeline on real PGNs, not synthetic nets**:
  Run `python scripts/eval_vs_stockfish.py --n-games 2 --opp-depth 1` as
  part of CI / before any long run
- **Don't rely on DDP + complex launcher for the first RL implementation**:
  start with the file-based IPC design
- **Separate value-head training signal when using degenerate-label data**:
  don't just down-weight, zero it
