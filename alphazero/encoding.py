"""Board and move encoding.

Encoding follows AlphaZero (Silver et al., 2017, arXiv:1712.01815) §2.1:

Input:  (119, 8, 8) float32 tensor per position.
Output policy target: flat vector of length 4672 = 64 squares * 73 move-types.

Move-type layout per from-square (73 planes total):
    planes  0..55  : "queen" moves  = 7 distances (1..7) * 8 compass dirs (N,NE,E,SE,S,SW,W,NW)
                     Also used for queen promotions (promotion piece inferred from rank).
    planes 56..63  : knight moves, 8 L-shapes
    planes 64..72  : underpromotions, 3 directions (capture-left, straight, capture-right)
                     x 3 pieces (knight, bishop, rook)

The planes are always oriented from the side-to-move's perspective: if black is to
move we flip the board vertically and swap piece colors, so "forward" is always up.
"""

from __future__ import annotations

import numpy as np
import chess

# ---- Constants -------------------------------------------------------------

NUM_PLANES = 119
HISTORY_LEN = 8
PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
# Per history frame: 6 us + 6 them + 2 repetition flags = 14 planes
PLANES_PER_FRAME = 14
# Constant tail: side-to-move, total-move-count, us-castle-kingside, us-castle-queenside,
# them-castle-kingside, them-castle-queenside, no-progress-count = 7 planes
CONST_PLANES = 7
assert PLANES_PER_FRAME * HISTORY_LEN + CONST_PLANES == NUM_PLANES

POLICY_SIZE = 64 * 73

# Compass directions: (dr, df) pairs, ordered N, NE, E, SE, S, SW, W, NW.
# From white's perspective "N" means rank +1.
QUEEN_DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
KNIGHT_DELTAS = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
# Underpromotion: (file_delta, piece). file_delta: -1 capture-left, 0 push, +1 capture-right.
UNDERPROMO_FILES = [-1, 0, 1]
UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


# ---- Move <-> index --------------------------------------------------------

def _mirror_square(sq: int) -> int:
    """Flip square vertically (a1<->a8). Used when black is to move so that
    the network always sees itself at the bottom."""
    return chess.square(chess.square_file(sq), 7 - chess.square_rank(sq))


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """Encode a move as an integer in [0, 4672).

    The encoding is always from the perspective of the side to move:
    when board.turn == BLACK we vertically mirror the from/to squares before
    computing the index, matching the orientation of the input planes.
    """
    from_sq = move.from_square
    to_sq = move.to_square
    if not board.turn:  # black to move -> mirror so "forward" is +rank
        from_sq = _mirror_square(from_sq)
        to_sq = _mirror_square(to_sq)

    fr, ff = chess.square_rank(from_sq), chess.square_file(from_sq)
    tr, tf = chess.square_rank(to_sq), chess.square_file(to_sq)
    dr, df = tr - fr, tf - ff

    promo = move.promotion
    # Underpromotions (knight/bishop/rook only — queen promo uses queen-move plane).
    if promo is not None and promo != chess.QUEEN:
        # Only happens when moving from rank 6 to rank 7 (white perspective).
        file_delta = df  # -1, 0, or +1
        file_idx = UNDERPROMO_FILES.index(file_delta)
        piece_idx = UNDERPROMO_PIECES.index(promo)
        plane = 64 + file_idx * 3 + piece_idx
        return from_sq * 73 + plane

    # Knight jumps.
    if (dr, df) in KNIGHT_DELTAS:
        plane = 56 + KNIGHT_DELTAS.index((dr, df))
        return from_sq * 73 + plane

    # Queen-like (sliding or 1-step: pawn push, king step, castling, etc.).
    # Determine direction unit vector and distance.
    step_r = (dr > 0) - (dr < 0)
    step_f = (df > 0) - (df < 0)
    dist = max(abs(dr), abs(df))
    assert 1 <= dist <= 7, f"bad distance {dist} for move {move}"
    dir_idx = QUEEN_DIRS.index((step_r, step_f))
    plane = (dist - 1) * 8 + dir_idx
    return from_sq * 73 + plane


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Decode index back to a Move. Returns a pseudo-legal move; caller must
    verify legality (e.g. by checking board.legal_moves)."""
    from_sq = index // 73
    plane = index % 73
    mirror = not board.turn

    def unmirror(sq: int) -> int:
        return _mirror_square(sq) if mirror else sq

    fr, ff = chess.square_rank(from_sq), chess.square_file(from_sq)

    if plane < 56:
        # Queen move.
        dist = plane // 8 + 1
        dir_idx = plane % 8
        dr, df = QUEEN_DIRS[dir_idx]
        tr, tf = fr + dr * dist, ff + df * dist
        to_sq = chess.square(tf, tr)
        # Check if this is a pawn reaching last rank -> queen promotion.
        piece = board.piece_at(unmirror(from_sq))
        promo = None
        if piece and piece.piece_type == chess.PAWN and tr == 7:
            promo = chess.QUEEN
        return chess.Move(unmirror(from_sq), unmirror(to_sq), promotion=promo)

    if plane < 64:
        # Knight.
        dr, df = KNIGHT_DELTAS[plane - 56]
        tr, tf = fr + dr, ff + df
        to_sq = chess.square(tf, tr)
        return chess.Move(unmirror(from_sq), unmirror(to_sq))

    # Underpromotion.
    u = plane - 64
    file_idx, piece_idx = u // 3, u % 3
    df = UNDERPROMO_FILES[file_idx]
    tr, tf = fr + 1, ff + df  # always 1 rank forward (from side-to-move POV)
    to_sq = chess.square(tf, tr)
    return chess.Move(unmirror(from_sq), unmirror(to_sq), promotion=UNDERPROMO_PIECES[piece_idx])


def legal_move_mask(board: chess.Board) -> np.ndarray:
    """Return a (4672,) bool mask of legal move indices for the given board."""
    mask = np.zeros(POLICY_SIZE, dtype=bool)
    for mv in board.legal_moves:
        mask[move_to_index(mv, board)] = True
    return mask


# ---- Board -> planes -------------------------------------------------------

def _piece_planes(board: chess.Board, perspective_white: bool) -> np.ndarray:
    """14 planes for a single position frame: 6 us, 6 them, 2 repetition flags."""
    planes = np.zeros((PLANES_PER_FRAME, 8, 8), dtype=np.float32)
    for sq, piece in board.piece_map().items():
        # Orient planes from side-to-move perspective.
        osq = sq if perspective_white else _mirror_square(sq)
        r, f = chess.square_rank(osq), chess.square_file(osq)
        is_us = (piece.color == chess.WHITE) == perspective_white
        base = 0 if is_us else 6
        planes[base + PIECE_TYPES.index(piece.piece_type), r, f] = 1.0
    # Repetition flags: 1 plane "has this position occurred once before",
    # 1 plane "...twice before". python-chess's is_repetition(n) counts this
    # position plus prior occurrences, so is_repetition(2) means >=1 prior.
    if board.is_repetition(2):
        planes[12].fill(1.0)
    if board.is_repetition(3):
        planes[13].fill(1.0)
    return planes


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a board (with its move stack) as a (119, 8, 8) float32 tensor.

    History is derived by popping moves off a copy of the board. If fewer than
    HISTORY_LEN prior positions exist, older frames are zero-filled.
    """
    perspective_white = board.turn == chess.WHITE
    frames = []
    b = board.copy(stack=True)
    for _ in range(HISTORY_LEN):
        if b is None:
            frames.append(np.zeros((PLANES_PER_FRAME, 8, 8), dtype=np.float32))
            continue
        frames.append(_piece_planes(b, perspective_white))
        if b.move_stack:
            b.pop()
        else:
            b = None

    history = np.concatenate(frames, axis=0)  # (112, 8, 8)

    # 7 constant planes.
    us_castle_k = board.has_kingside_castling_rights(board.turn)
    us_castle_q = board.has_queenside_castling_rights(board.turn)
    them_castle_k = board.has_kingside_castling_rights(not board.turn)
    them_castle_q = board.has_queenside_castling_rights(not board.turn)
    const = np.zeros((CONST_PLANES, 8, 8), dtype=np.float32)
    const[0].fill(0.0 if perspective_white else 1.0)  # colour plane (AZ uses 1 for black to move)
    const[1].fill(board.fullmove_number / 100.0)       # scaled move count
    const[2].fill(1.0 if us_castle_k else 0.0)
    const[3].fill(1.0 if us_castle_q else 0.0)
    const[4].fill(1.0 if them_castle_k else 0.0)
    const[5].fill(1.0 if them_castle_q else 0.0)
    const[6].fill(board.halfmove_clock / 100.0)        # 50-move-rule counter

    return np.concatenate([history, const], axis=0)  # (119, 8, 8)
