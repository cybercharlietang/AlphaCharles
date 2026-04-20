"""Interactive play: user vs trained net.

Runs on the pod after training completes. The pod stays alive so you can
connect via SSH and play 10 games, or however many you want. Each game is
logged (PGN saved to disk) and optional W&B logging at end.

Usage on the pod:
    python scripts/play_human.py --ckpt runs/rl_selfplay/final.pt \
        --games 10 --our-sims 1600

Controls inside the match:
    - Enter a move in UCI ("e2e4") or SAN ("e4", "Nf3", "O-O")
    - "resign" to resign the game
    - "draw" to offer / accept a draw
    - "board" to reprint the board
    - "hint" for the net's current top choice + its MCTS visit distribution
    - "quit" to end the session (won't kill the server if run via pod_pipeline)
"""

from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path
import sys

import chess
import chess.pgn
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alphazero.mcts import MCTS, MCTSConfig
from alphazero.model import AlphaZeroNet, ModelConfig


def print_board(board: chess.Board):
    print()
    print(board.unicode(borders=True, invert_color=True))
    print(f"FEN: {board.fen()}")
    print(f"Turn: {'White' if board.turn else 'Black'}  "
          f"Fullmove: {board.fullmove_number}")
    print()


def parse_user_move(s: str, board: chess.Board) -> chess.Move | None:
    s = s.strip()
    # Try UCI first (a1b2, e7e8q for promotion).
    try:
        mv = chess.Move.from_uci(s)
        if mv in board.legal_moves:
            return mv
    except ValueError:
        pass
    # Try SAN (Nf3, O-O, e4, Rxe5).
    try:
        return board.parse_san(s)
    except (ValueError, chess.IllegalMoveError, chess.AmbiguousMoveError):
        return None


def show_hint(mcts: MCTS, board: chess.Board, top_k: int = 5):
    root = mcts.run(board, add_root_noise=False)
    # Rank edges by visit count.
    order = sorted(range(len(root.N)), key=lambda i: -int(root.N[i]))
    print(f"\nNet's top {top_k} candidates (visits | Q | move):")
    for i in order[:top_k]:
        mv = root.legal_moves[i]
        n = int(root.N[i])
        q = float(root.W[i] / max(n, 1))
        san = board.san(mv)
        print(f"  {n:>4} | {q:+.3f} | {san}")
    print()


def play_game(net, device, mcts: MCTS, user_color: chess.Color,
              game_num: int, pgn_dir: Path, our_sims: int):
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Human vs AlphaCharles"
    game.headers["White"] = "Human" if user_color == chess.WHITE else "AlphaCharles"
    game.headers["Black"] = "AlphaCharles" if user_color == chess.WHITE else "Human"
    game.headers["Date"] = datetime.date.today().isoformat()
    game.headers["Round"] = str(game_num)
    node = game

    print(f"\n=== Game {game_num}: You play {'White' if user_color == chess.WHITE else 'Black'} ===")
    print_board(board)

    while not board.is_game_over(claim_draw=True):
        if board.turn == user_color:
            while True:
                raw = input("Your move (or command: hint/resign/draw/board/quit): ").strip().lower()
                if raw == "quit":
                    print("Quitting session.")
                    return "aborted"
                if raw == "board":
                    print_board(board); continue
                if raw == "hint":
                    show_hint(mcts, board); continue
                if raw == "resign":
                    print("You resigned.")
                    result = "0-1" if user_color == chess.WHITE else "1-0"
                    game.headers["Result"] = result
                    _save_pgn(game, pgn_dir, game_num)
                    return result
                if raw == "draw":
                    # Accept draw by declaring it.
                    print("Draw agreed.")
                    game.headers["Result"] = "1/2-1/2"
                    _save_pgn(game, pgn_dir, game_num)
                    return "1/2-1/2"
                mv = parse_user_move(raw, board)
                if mv is None:
                    print("Invalid or illegal move. Try again (UCI like e2e4 or SAN like e4).")
                    continue
                board.push(mv)
                node = node.add_variation(mv)
                print_board(board)
                break
        else:
            print("AlphaCharles is thinking...")
            root = mcts.run(board, add_root_noise=False)
            best_edge = int(root.N.argmax())
            mv = root.legal_moves[best_edge]
            visits = int(root.N[best_edge])
            q = float(root.W[best_edge] / max(visits, 1))
            san = board.san(mv)
            board.push(mv)
            node = node.add_variation(mv)
            print(f"  {san}   (visits={visits}, value={q:+.3f})")
            print_board(board)

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        result = "1/2-1/2"
    elif outcome.winner is None:
        result = "1/2-1/2"
    elif outcome.winner == chess.WHITE:
        result = "1-0"
    else:
        result = "0-1"
    game.headers["Result"] = result
    print(f"\nGame over: {result}  ({outcome.termination.name if outcome else 'move limit'})")
    _save_pgn(game, pgn_dir, game_num)
    return result


def _save_pgn(game: chess.pgn.Game, pgn_dir: Path, game_num: int):
    pgn_dir.mkdir(parents=True, exist_ok=True)
    path = pgn_dir / f"game_{game_num:02d}_{datetime.datetime.now().strftime('%H%M%S')}.pgn"
    with open(path, "w") as fh:
        print(game, file=fh, end="\n\n")
    print(f"  saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--our-sims", type=int, default=1600,
                    help="MCTS sims per move for the model (higher = stronger)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pgn-dir", default="runs/human_matches")
    ap.add_argument("--c-puct", type=float, default=2.5)
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location=args.device)
    model_cfg_dict = state.get("model_cfg", {})
    net = AlphaZeroNet(ModelConfig(**model_cfg_dict)).to(args.device)
    net.load_state_dict(state["model"] if "model" in state else state)
    net.eval()
    print(f"Loaded {args.ckpt}  ({sum(p.numel() for p in net.parameters())/1e6:.1f}M params, "
          f"{args.device})")

    mcts_cfg = MCTSConfig(
        num_simulations=args.our_sims, c_puct=args.c_puct,
        add_root_noise=False, temperature_moves=0,
    )
    mcts = MCTS(net, torch.device(args.device), mcts_cfg)

    pgn_dir = Path(args.pgn_dir)
    tally = {"human_wins": 0, "net_wins": 0, "draws": 0}

    for g in range(1, args.games + 1):
        user_color = chess.WHITE if g % 2 == 1 else chess.BLACK
        result = play_game(net, args.device, mcts, user_color, g, pgn_dir, args.our_sims)
        if result == "aborted":
            break
        if result == "1/2-1/2":
            tally["draws"] += 1
        elif (result == "1-0" and user_color == chess.WHITE) or \
             (result == "0-1" and user_color == chess.BLACK):
            tally["human_wins"] += 1
        else:
            tally["net_wins"] += 1

        print(f"\nRunning score — You: {tally['human_wins']}  "
              f"Draws: {tally['draws']}  AlphaCharles: {tally['net_wins']}")

    print("\n====== Final ======")
    print(f"You: {tally['human_wins']}   Draws: {tally['draws']}   AlphaCharles: {tally['net_wins']}")


if __name__ == "__main__":
    main()
