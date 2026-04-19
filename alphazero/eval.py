"""Evaluation: play our net vs a UCI engine (e.g. Stockfish) at fixed depth/nodes.

Also includes an Elo estimator from match results (log5 / BayesElo-style point
estimate using the logistic curve inverse).

Usage in-process: `play_match(net, stockfish_path, n_games=20, depth=5)`.

Elo mapping for Stockfish at fixed depth is approximate but useful as a proxy:
    depth  1  -> ~1300
    depth  3  -> ~1600
    depth  5  -> ~1900
    depth  8  -> ~2200
    depth 12  -> ~2600
    depth 18  -> ~3100
These are rough — real Elo depends on hardware and SF version. We treat Stockfish
at chosen depth as a fixed opponent; the INTERESTING number is our net's Elo diff
against it, which we convert to a reference Elo.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import chess
import chess.engine
import torch

from .mcts import MCTS, MCTSConfig


@dataclass
class MatchConfig:
    n_games: int = 20
    our_sims: int = 400
    opp_depth: int = 5          # stockfish depth
    opp_nodes: int | None = None
    opp_movetime_ms: int | None = None
    our_temperature: float = 0.0
    starting_fens: list[str] | None = None  # sample from these to diversify openings


@dataclass
class MatchResult:
    wins: int
    losses: int
    draws: int
    total: int
    score: float   # = (wins + 0.5 * draws) / total
    elo_diff: float  # our Elo - opp Elo, estimated
    stderr_elo: float


def _elo_from_score(score: float, n: int) -> tuple[float, float]:
    """Invert the logistic: elo_diff = -400 * log10(1/score - 1)."""
    score = max(min(score, 1 - 1e-6), 1e-6)
    elo = -400.0 * math.log10(1.0 / score - 1.0)
    # Standard error on score, propagated: SE(elo) approx dElo/dScore * SE(score).
    # dElo/dScore = 400 / (ln(10) * score * (1 - score))
    var = score * (1 - score) / max(n, 1)
    se_score = math.sqrt(var)
    d = 400.0 / (math.log(10) * score * (1 - score))
    return elo, d * se_score


def play_match(
    net,
    device: torch.device,
    engine_path: str,
    cfg: MatchConfig | None = None,
    our_color_alternates: bool = True,
    print_games: bool = False,
) -> MatchResult:
    cfg = cfg or MatchConfig()
    mcts_cfg = MCTSConfig(num_simulations=cfg.our_sims, add_root_noise=False)
    mcts = MCTS(net, device, mcts_cfg)

    engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    if cfg.opp_movetime_ms is not None:
        limit = chess.engine.Limit(time=cfg.opp_movetime_ms / 1000.0)
    elif cfg.opp_nodes is not None:
        limit = chess.engine.Limit(nodes=cfg.opp_nodes)
    else:
        limit = chess.engine.Limit(depth=cfg.opp_depth)

    wins = losses = draws = 0
    try:
        for g in range(cfg.n_games):
            board = chess.Board(cfg.starting_fens[g % len(cfg.starting_fens)]
                                if cfg.starting_fens else chess.STARTING_FEN)
            our_color = chess.WHITE if (not our_color_alternates or g % 2 == 0) else chess.BLACK

            while not board.is_game_over(claim_draw=True) and board.fullmove_number < 400:
                if board.turn == our_color:
                    root = mcts.run(board)
                    move, _ = mcts.choose_move(root, temperature=cfg.our_temperature)
                else:
                    result = engine.play(board, limit)
                    move = result.move
                board.push(move)

            outcome = board.outcome(claim_draw=True)
            if outcome is None or outcome.winner is None:
                draws += 1
                res_str = "1/2-1/2"
            elif outcome.winner == our_color:
                wins += 1
                res_str = "1-0" if our_color == chess.WHITE else "0-1"
            else:
                losses += 1
                res_str = "0-1" if our_color == chess.WHITE else "1-0"

            if print_games:
                print(f"game {g+1}: we={('W' if our_color==chess.WHITE else 'B')} "
                      f"result={res_str}  score so far: {wins}/{draws}/{losses}")
    finally:
        engine.quit()

    total = wins + losses + draws
    score = (wins + 0.5 * draws) / max(total, 1)
    elo, se = _elo_from_score(score, total)
    return MatchResult(wins, losses, draws, total, score, elo, se)
