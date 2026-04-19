"""Tests for board/move encoding.

Correctness checks we care about:
  1. encode_board produces the right shape and dtype.
  2. move_to_index is a bijection with index_to_move on every legal move in
     a variety of positions (startpos, midgame, tactical puzzle, endgame with
     underpromotion options).
  3. Mirror-invariance: encoding a position with black to move is the same as
     encoding the color-flipped position with white to move.
"""

from __future__ import annotations

import chess
import numpy as np
import pytest

from alphazero.encoding import (
    NUM_PLANES,
    POLICY_SIZE,
    encode_board,
    index_to_move,
    legal_move_mask,
    move_to_index,
)


POSITIONS = [
    # Startpos.
    chess.STARTING_FEN,
    # Early Najdorf, complex piece activity.
    "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
    # Endgame with pawn on 7th — all 4 promotions are legal.
    "8/P7/8/8/8/8/k7/7K w - - 0 1",
    # Black to move, checks mirror path.
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 2 2",
    # Position where underpromotions (N/B/R) are all individually reachable.
    "1n6/P7/8/8/8/8/k7/7K w - - 0 1",
]


@pytest.mark.parametrize("fen", POSITIONS)
def test_encode_shape(fen):
    board = chess.Board(fen)
    x = encode_board(board)
    assert x.shape == (NUM_PLANES, 8, 8)
    assert x.dtype == np.float32


@pytest.mark.parametrize("fen", POSITIONS)
def test_move_roundtrip(fen):
    board = chess.Board(fen)
    for mv in board.legal_moves:
        idx = move_to_index(mv, board)
        assert 0 <= idx < POLICY_SIZE
        decoded = index_to_move(idx, board)
        assert decoded == mv, f"roundtrip failed: {mv.uci()} -> {idx} -> {decoded.uci()}"


@pytest.mark.parametrize("fen", POSITIONS)
def test_legal_mask_counts(fen):
    board = chess.Board(fen)
    mask = legal_move_mask(board)
    assert mask.sum() == board.legal_moves.count()


def test_all_index_are_distinct_per_position():
    """Every legal move in a crowded middlegame maps to a unique index."""
    board = chess.Board(POSITIONS[1])
    indices = {move_to_index(m, board) for m in board.legal_moves}
    assert len(indices) == board.legal_moves.count()


def test_mirror_invariance():
    """A position viewed with black to move should produce the same planes as
    the color-mirrored position with white to move."""
    fen_black = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 2 2"
    b_black = chess.Board(fen_black)
    b_white = b_black.mirror()  # swaps colors AND flips vertically
    x1 = encode_board(b_black)
    x2 = encode_board(b_white)
    # Piece planes (first 112) should match. Constant planes differ in
    # side-to-move and move-count scaling, so compare only piece history.
    assert np.array_equal(x1[:112], x2[:112])


def test_underpromotion_encodes_separately():
    """Promoting to N, B, R on the same square should yield different indices,
    while queen promo uses the queen-move plane."""
    board = chess.Board("8/P7/8/8/8/8/k7/7K w - - 0 1")
    promos = {}
    for mv in board.legal_moves:
        if mv.from_square == chess.A7 and mv.to_square == chess.A8:
            promos[mv.promotion] = move_to_index(mv, board)
    assert len(promos) == 4
    assert len(set(promos.values())) == 4
