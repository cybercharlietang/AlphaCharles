"""Convert Lichess puzzle CSV to training samples.

Lichess puzzles CSV format:
    PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags

FEN is the position BEFORE the setup move; the first move in `Moves` is the
opponent's move that sets up the puzzle; subsequent moves are the solution
alternating between solver and opponent.

We produce one sample per solver move:
    planes = encode_board(position after applying prior moves)
    policy = one-hot on the correct solver move
    value  = +1 (solver wins by following puzzle)  [or 0 for draw puzzles; rare]

We skip puzzles below a rating threshold (default 1500) to keep quality high.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass

import chess
import numpy as np

from .encoding import NUM_PLANES, POLICY_SIZE, encode_board_uint8, move_to_index


@dataclass
class PuzzleFilterConfig:
    min_rating: int = 1500
    max_rating: int = 3000


def extract_puzzle_samples(fen: str, moves_str: str):
    """Apply the setup move; then yield (planes, policy_idx, value) for each
    solver move. The opponent's replies are applied but not yielded."""
    board = chess.Board(fen)
    uci_moves = moves_str.split()
    if len(uci_moves) < 2:
        return
    # Apply setup (opponent) move.
    try:
        setup = chess.Move.from_uci(uci_moves[0])
        if not board.is_legal(setup):
            return
        board.push(setup)
    except ValueError:
        return

    # Remaining moves alternate solver, opponent, solver, opponent, ...
    for i, uci in enumerate(uci_moves[1:]):
        try:
            mv = chess.Move.from_uci(uci)
        except ValueError:
            return
        if not board.is_legal(mv):
            return
        if i % 2 == 0:  # solver's move
            planes = encode_board_uint8(board)
            policy_idx = move_to_index(mv, board)
            yield planes, policy_idx, 1.0  # solver wins these
        board.push(mv)


def build_shards(csv_paths: list[str], out_dir: str, cfg: PuzzleFilterConfig,
                 shard_size: int = 500_000, max_samples: int | None = None) -> int:
    os.makedirs(out_dir, exist_ok=True)
    existing = sorted([f for f in os.listdir(out_dir) if f.startswith("puzzles_") and f.endswith(".npz")])
    if existing:
        total = 0
        for f in existing:
            with np.load(os.path.join(out_dir, f)) as d:
                total += len(d["values"])
        shard_idx = len(existing)
        print(f"  found {shard_idx} existing puzzle shards with {total:,} samples, resuming")
        if max_samples is not None and total >= max_samples:
            return total
    else:
        shard_idx = 0
        total = 0
    cur_planes = np.zeros((shard_size, NUM_PLANES, 8, 8), dtype=np.uint8)
    cur_policy_idx = np.zeros(shard_size, dtype=np.int32)
    cur_values = np.zeros(shard_size, dtype=np.float32)
    cur_n = 0

    def flush():
        nonlocal shard_idx, cur_n
        if cur_n == 0:
            return
        out = os.path.join(out_dir, f"puzzles_{shard_idx:05d}.npz")
        tmp = out + ".tmp"
        np.savez_compressed(tmp,
                            planes=cur_planes[:cur_n],
                            policy_idx=cur_policy_idx[:cur_n],
                            values=cur_values[:cur_n])
        os.replace(tmp, out)
        print(f"  wrote {out} with {cur_n} samples")
        shard_idx += 1
        cur_n = 0

    for path in csv_paths:
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if max_samples is not None and total >= max_samples:
                    flush()
                    return total
                try:
                    rating = int(row["Rating"])
                except (ValueError, KeyError):
                    continue
                if not (cfg.min_rating <= rating <= cfg.max_rating):
                    continue
                for planes, pol_idx, value in extract_puzzle_samples(row["FEN"], row["Moves"]):
                    cur_planes[cur_n] = planes
                    cur_policy_idx[cur_n] = pol_idx
                    cur_values[cur_n] = value
                    cur_n += 1
                    total += 1
                    if cur_n == shard_size:
                        flush()
                    if max_samples is not None and total >= max_samples:
                        flush()
                        return total
    flush()
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-rating", type=int, default=1500)
    ap.add_argument("--max-rating", type=int, default=3000)
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()
    cfg = PuzzleFilterConfig(min_rating=args.min_rating, max_rating=args.max_rating)
    total = build_shards(args.csv, args.out, cfg, max_samples=args.max_samples)
    print(f"total puzzle samples: {total}")


if __name__ == "__main__":
    main()
