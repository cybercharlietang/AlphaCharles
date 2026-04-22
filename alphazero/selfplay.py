"""Self-play game generator.

Produces training tuples (state_planes, policy_target, value_target) from
games played between two copies of the same network. Value target is the
GAME OUTCOME from the mover's perspective at that ply (+1 won, -1 lost, 0 draw).

Each game:
  - Run MCTS from the current board (with Dirichlet noise at root).
  - Sample a move from N^(1/tau); tau=1 for first `temperature_moves` plies, 0 after.
  - Append (encoded_board, visit_policy, side_to_move) to the trajectory.
  - Play the move, repeat until terminal or move_limit reached.
  - Assign outcome z to each ply: +1 if side_to_move at that ply won, -1 if lost, 0 draw.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess
import chess.pgn
import numpy as np
import torch

from .encoding import POLICY_SIZE, encode_board
from .mcts import MCTS, MCTSConfig


@dataclass
class SelfPlayConfig:
    mcts: MCTSConfig = None
    move_limit: int = 512        # hard cap on plies; game declared draw if hit
    resign_threshold: float | None = -0.95  # None = no resignation
    resign_after_move: int = 20  # don't resign in opening


@dataclass
class GameRecord:
    planes: np.ndarray           # (T, 119, 8, 8) float32
    policies: np.ndarray         # (T, 4672) float32
    values: np.ndarray           # (T,) float32 in {-1, 0, 1}
    result: str                  # '1-0', '0-1', '1/2-1/2'
    ply_count: int
    pgn: str = ""                # full game PGN text, empty if not tracked
    # Per-game averages of entropy (in nats), useful for diagnosing exploration
    avg_prior_entropy: float = 0.0   # mean entropy of net's raw prior over legal moves
    avg_mcts_entropy: float = 0.0    # mean entropy of MCTS visit distribution
    avg_entropy_drop: float = 0.0    # mean (prior - mcts) — MCTS's "decisiveness gain"


def play_game(net, device: torch.device, cfg: SelfPlayConfig | None = None,
              starting_fen: str | None = None) -> GameRecord:
    cfg = cfg or SelfPlayConfig(mcts=MCTSConfig())
    if cfg.mcts is None:
        cfg.mcts = MCTSConfig()
    mcts = MCTS(net, device, cfg.mcts)

    board = chess.Board(starting_fen) if starting_fen else chess.Board()

    traj_planes: list[np.ndarray] = []
    traj_policies: list[np.ndarray] = []
    traj_movers: list[bool] = []  # True = white to move at this ply
    prior_entropies: list[float] = []   # H(root.P) per ply
    mcts_entropies: list[float] = []    # H(root.N / sum(root.N)) per ply

    # Build PGN incrementally.
    pgn_game = chess.pgn.Game()
    pgn_game.headers["Event"] = "AlphaCharles self-play"
    pgn_game.headers["White"] = "AlphaCharles"
    pgn_game.headers["Black"] = "AlphaCharles"
    pgn_node = pgn_game

    result_winner: int | None = None  # 1 white, -1 black, 0 draw
    reuse_root = None
    ply = 0

    while ply < cfg.move_limit:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            result_winner = (
                0 if outcome.winner is None
                else (1 if outcome.winner == chess.WHITE else -1)
            )
            break

        temperature = 1.0 if ply < cfg.mcts.temperature_moves else 0.0

        root = mcts.run(board, reuse_root=reuse_root)

        # Resignation check: if root value (approx by max Q of visited edges, from
        # mover's POV) is very negative and we've played enough moves, resign.
        if (cfg.resign_threshold is not None and ply >= cfg.resign_after_move
                and root.N.sum() > 0):
            visited = root.N > 0
            if visited.any():
                best_q = float((root.W[visited] / root.N[visited]).max())
                if best_q < cfg.resign_threshold:
                    result_winner = -1 if board.turn == chess.WHITE else 1
                    break

        pi = mcts.policy_from_root(root, temperature)
        traj_planes.append(encode_board(board))
        traj_policies.append(pi)
        traj_movers.append(board.turn == chess.WHITE)

        # Track per-ply entropy diagnostics.
        if len(root.P) > 0:
            p_legal = root.P[root.P > 0]
            prior_entropies.append(float(-(p_legal * np.log(p_legal)).sum()))
            total_visits = float(root.N.sum())
            if total_visits > 0:
                vp = root.N.astype(np.float64) / total_visits
                vp = vp[vp > 0]
                mcts_entropies.append(float(-(vp * np.log(vp)).sum()))

        move, edge = mcts.choose_move(root, temperature)
        # Re-root: the chosen edge's subtree becomes the new root.
        next_root = root.children[edge]
        board.push(move)
        pgn_node = pgn_node.add_variation(move)
        reuse_root = next_root if next_root is not None else None
        ply += 1

    if result_winner is None:
        result_winner = 0  # move_limit hit -> treat as draw

    # Assign z to each ply from the mover's perspective at that ply.
    values = np.empty(len(traj_movers), dtype=np.float32)
    for i, mover_white in enumerate(traj_movers):
        if result_winner == 0:
            values[i] = 0.0
        else:
            mover_side = 1 if mover_white else -1
            values[i] = 1.0 if mover_side == result_winner else -1.0

    result_str = {1: "1-0", -1: "0-1", 0: "1/2-1/2"}[result_winner]
    pgn_game.headers["Result"] = result_str
    pgn_game.headers["PlyCount"] = str(ply)

    avg_prior = float(np.mean(prior_entropies)) if prior_entropies else 0.0
    avg_mcts = float(np.mean(mcts_entropies)) if mcts_entropies else 0.0
    pgn_game.headers["PriorEntropy"] = f"{avg_prior:.3f}"
    pgn_game.headers["MctsEntropy"] = f"{avg_mcts:.3f}"

    return GameRecord(
        planes=np.stack(traj_planes) if traj_planes else np.empty((0, 119, 8, 8), dtype=np.float32),
        policies=np.stack(traj_policies) if traj_policies else np.empty((0, POLICY_SIZE), dtype=np.float32),
        values=values,
        result=result_str,
        ply_count=ply,
        pgn=str(pgn_game),
        avg_prior_entropy=avg_prior,
        avg_mcts_entropy=avg_mcts,
        avg_entropy_drop=avg_prior - avg_mcts,
    )
