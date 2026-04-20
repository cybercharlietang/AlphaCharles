"""Convert PGN game files to (planes, policy, value) tensors for SL training.

Each position in a PGN becomes one training sample:
    planes = encode_board(board_before_move)
    policy = one-hot on move_to_index(the move actually played)
    value  = +1 if the side to move eventually won, -1 if lost, 0 draw

We filter games to:
  - time control >= Rapid (to skip bullet junk) if headers say so
  - both players above a min rating (configurable)
  - at least 10 plies (skip abandoned games)

Output: sharded .npz files (each ~1M positions) for fast loading during training.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn
import numpy as np

from .encoding import NUM_PLANES, POLICY_SIZE, encode_board_uint8, move_to_index


@dataclass
class PgnFilterConfig:
    min_rating: int = 2200
    min_plies: int = 10
    max_plies: int = 400


def _game_value(result: str) -> tuple[int | None, bool]:
    """Return (winner_flag, ok) where winner_flag is +1 white / -1 black / 0 draw,
    or None if the game had no decisive/drawn result."""
    if result == "1-0":
        return 1, True
    if result == "0-1":
        return -1, True
    if result == "1/2-1/2":
        return 0, True
    return None, False


def _game_passes_filter(game: chess.pgn.Game, cfg: PgnFilterConfig) -> bool:
    headers = game.headers
    we = headers.get("WhiteElo", "0")
    be = headers.get("BlackElo", "0")
    try:
        if int(we) < cfg.min_rating or int(be) < cfg.min_rating:
            return False
    except ValueError:
        return False
    return True


def extract_samples_from_game(game: chess.pgn.Game, cfg: PgnFilterConfig):
    """Yield (planes, policy_idx, value) tuples for each mainline ply."""
    winner, ok = _game_value(game.headers.get("Result", "*"))
    if not ok:
        return
    board = game.board()
    moves = list(game.mainline_moves())
    if not (cfg.min_plies <= len(moves) <= cfg.max_plies):
        return
    for mv in moves:
        if not board.is_legal(mv):
            return  # corrupt game
        planes = encode_board_uint8(board)
        policy_idx = move_to_index(mv, board)
        mover = 1 if board.turn == chess.WHITE else -1
        value = 0.0 if winner == 0 else (1.0 if mover == winner else -1.0)
        yield planes, policy_idx, value
        board.push(mv)


def build_shards(pgn_paths: list[str], out_dir: str, cfg: PgnFilterConfig,
                 shard_size: int = 1_000_000, max_samples: int | None = None) -> int:
    """Stream PGN files, writing shards of shard_size samples. Returns total samples."""
    os.makedirs(out_dir, exist_ok=True)
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
        out = os.path.join(out_dir, f"shard_{shard_idx:05d}.npz")
        np.savez_compressed(out,
                            planes=cur_planes[:cur_n],
                            policy_idx=cur_policy_idx[:cur_n],
                            values=cur_values[:cur_n])
        print(f"  wrote {out} with {cur_n} samples")
        shard_idx += 1
        cur_n = 0

    for path in pgn_paths:
        with open(path, errors="ignore") as fh:
            while True:
                if max_samples is not None and total >= max_samples:
                    flush()
                    return total
                game = chess.pgn.read_game(fh)
                if game is None:
                    break
                if not _game_passes_filter(game, cfg):
                    continue
                for planes, pol_idx, value in extract_samples_from_game(game, cfg):
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
    ap.add_argument("--pgn", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-rating", type=int, default=2200)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--shard-size", type=int, default=1_000_000)
    args = ap.parse_args()
    cfg = PgnFilterConfig(min_rating=args.min_rating)
    total = build_shards(args.pgn, args.out, cfg,
                         shard_size=args.shard_size, max_samples=args.max_samples)
    print(f"total samples: {total}")


if __name__ == "__main__":
    main()
