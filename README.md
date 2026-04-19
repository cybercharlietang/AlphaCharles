# alphazero-chess

AlphaZero-style chess engine. Target: beat a 2200 FIDE player.

## Plan
1. **SL warm-start** on Lichess Elite (2200+) + CCRL PGNs.
2. **Puzzle fine-tune** on Lichess puzzle DB (rated 1500-3000) with reduced value-loss weight.
3. **Self-play RL** starting from the SL-pretrained net.
4. **Evaluation** vs Stockfish at fixed depth, then a 10-game gauntlet vs the user.

## Architecture
- ResNet: 20 blocks × 256 channels (~24.5M params).
- Input: (119, 8, 8) planes per AlphaZero paper §2.1 (8-ply history + 7 constants).
- Output: (4672,) policy logits over (from_sq × 73 move-types) + scalar value.
- MCTS: sequential PUCT, 400 sims/move in self-play, 800+ in eval.

## Layout
```
alphazero/
  encoding.py    # board <-> tensor, move <-> index
  model.py       # ResNet tower + policy/value heads
  mcts.py        # PUCT MCTS
  selfplay.py    # game generator
  replay.py      # ring buffer (memory + mmap)
  training.py    # loss + train step
  data_pgn.py    # PGN -> shard .npz
  data_puzzles.py
  dataset.py     # torch Dataset over shards
  eval.py        # match vs UCI engine + Elo estimate
scripts/
  train_sl.py, train_rl.py, eval_vs_stockfish.py
configs/
  sl_warmstart.yaml, sl_puzzles.yaml, rl_selfplay.yaml
tests/           # pytest suite, 40+ cases
```

## Data sources
- **Lichess Elite DB**: <https://database.nikonoel.fr/> (~120k games, all players >=2400, titled).
- **CCRL 40/4**: <https://ccrl.chessdom.com/ccrl/404/> (engine games).
- **Lichess puzzles**: <https://database.lichess.org/#puzzles> (4M+ positions CSV).

## Workflow
```bash
# 1. Prepare SL data (PGN -> shard npz).
python -m alphazero.data_pgn --pgn path/to/*.pgn --out data/sl_shards --min-rating 2200

# 2. Prepare puzzle data.
python -m alphazero.data_puzzles --csv lichess_db_puzzle.csv --out data/puzzle_shards

# 3. SL warm-start (8 GPUs).
torchrun --nproc_per_node=8 scripts/train_sl.py --config configs/sl_warmstart.yaml

# 4. Puzzle fine-tune.
torchrun --nproc_per_node=8 scripts/train_sl.py --config configs/sl_puzzles.yaml

# 5. RL self-play.
torchrun --nproc_per_node=8 scripts/train_rl.py --config configs/rl_selfplay.yaml

# 6. Evaluate.
python scripts/eval_vs_stockfish.py --ckpt runs/rl_selfplay/final.pt --stockfish /usr/games/stockfish
```

## Tests
```bash
PYTHONPATH=. .venv/bin/pytest tests/
```

## Status
- Encoder, ResNet, MCTS, self-play, training loss, eval harness: done, tested.
- Next: RunPod launch script + data download.
