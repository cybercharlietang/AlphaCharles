"""CLI: evaluate a checkpoint vs Stockfish at fixed depth."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alphazero.eval import MatchConfig, play_match
from alphazero.model import AlphaZeroNet, ModelConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stockfish", default="stockfish")
    ap.add_argument("--n-games", type=int, default=10)
    ap.add_argument("--our-sims", type=int, default=400)
    ap.add_argument("--opp-depth", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pgn-dir", default=None, help="save each game's PGN here")
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location=args.device)
    model_cfg_dict = state.get("model_cfg", {})
    net = AlphaZeroNet(ModelConfig(**model_cfg_dict)).to(args.device)
    net.load_state_dict(state["model"] if "model" in state else state)
    net.eval()

    cfg = MatchConfig(n_games=args.n_games, our_sims=args.our_sims, opp_depth=args.opp_depth,
                      pgn_dir=args.pgn_dir)
    result = play_match(net, torch.device(args.device), args.stockfish, cfg, print_games=True)
    print()
    print(f"W/D/L: {result.wins}/{result.draws}/{result.losses} of {result.total}")
    print(f"score: {result.score:.3f}")
    print(f"elo diff vs stockfish depth={args.opp_depth}: {result.elo_diff:+.0f} ± {result.stderr_elo:.0f}")


if __name__ == "__main__":
    main()
