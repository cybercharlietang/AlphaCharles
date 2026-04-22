"""Sequential MCTS with PUCT.

This is the unbatched teaching version. Each simulation runs one forward pass
through the net; later we'll swap in leaf-parallelization (virtual loss) to
batch NN calls.

Node state (per edge a out of state s):
    P(s, a)  prior from net, fixed at expansion time
    N(s, a)  visit count
    W(s, a)  total value (sum of backup values)
    Q(s, a)  mean value = W/N (stored as a derived property)

PUCT:
    U(s, a) = c_puct * P(s, a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))
    score(s, a) = Q(s, a) + U(s, a)

Values are always "from the POV of the side to move at s". At backup, we flip
sign going up the tree so each node sees value from its own mover's perspective.

Terminal values: +1 for checkmate delivered by the mover of the PARENT (so the
mover of this node just lost -> their Q is -1), 0 for draw. python-chess's
`outcome()` returns the winner or None for draws.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import chess
import numpy as np
import torch

from .encoding import POLICY_SIZE, encode_board, legal_move_mask, move_to_index


@dataclass
class MCTSConfig:
    c_puct: float = 2.5
    num_simulations: int = 800
    dirichlet_alpha: float = 0.3       # AZ uses 0.3 for chess
    dirichlet_epsilon: float = 0.25    # fraction of noise to mix into root priors
    add_root_noise: bool = True        # False during eval/match play
    temperature_moves: int = 30        # plies with tau=1, then tau->0
    fpu: float = 0.0                   # First Play Urgency: Q value for unvisited children
    batch_size: int = 1                # leaf-parallel batch size (1 = sequential/legacy)
    virtual_loss: float = 1.0          # negative value added to in-flight edges
                                       # to make PUCT pick different paths


class Node:
    """A node in the MCTS tree. Keeps per-edge arrays sized to the number of
    legal moves at this position (not 4672), to save memory/time."""

    __slots__ = ("board", "legal_moves", "move_indices", "P", "N", "W",
                 "children", "is_terminal", "terminal_value", "is_expanded")

    def __init__(self, board: chess.Board):
        self.board = board
        self.legal_moves: list[chess.Move] = []
        self.move_indices: np.ndarray = np.empty(0, dtype=np.int32)
        self.P: np.ndarray = np.empty(0, dtype=np.float32)
        self.N: np.ndarray = np.empty(0, dtype=np.int32)
        self.W: np.ndarray = np.empty(0, dtype=np.float32)
        self.children: list["Node | None"] = []
        self.is_terminal: bool = False
        self.terminal_value: float = 0.0
        self.is_expanded: bool = False

    @property
    def total_visits(self) -> int:
        return int(self.N.sum())

    def Q(self) -> np.ndarray:
        """Mean action value per edge. Unvisited edges default to fpu (set by MCTS)."""
        q = np.zeros_like(self.W)
        visited = self.N > 0
        q[visited] = self.W[visited] / self.N[visited]
        return q


def _terminal_value_for_mover(board: chess.Board) -> float | None:
    """Return value from POV of side-to-move if the game is over, else None.

    Checkmate at this position means the side to move has been mated -> -1.
    Stalemate / insufficient material / 50-move / repetition -> 0.
    """
    if board.is_checkmate():
        return -1.0
    # outcome() covers stalemate, insufficient material, 75-move, 5-fold, etc.
    if board.outcome(claim_draw=True) is not None:
        return 0.0
    return None


class MCTS:
    """Sequential PUCT MCTS."""

    def __init__(self, net, device: torch.device, cfg: MCTSConfig | None = None):
        self.net = net
        self.device = device
        self.cfg = cfg or MCTSConfig()

    # ---- Network eval ------------------------------------------------------

    @torch.no_grad()
    def _evaluate(self, board: chess.Board) -> tuple[np.ndarray, float, list[chess.Move]]:
        """Run the net on a single position. Returns (priors, value, legal_moves).

        CRITICAL: priors[i] corresponds to legal_moves[i] (python-chess iteration order),
        not to ascending policy-index order. This is the order used throughout MCTS.
        Getting the alignment wrong silently scrambles priors w.r.t. moves.
        """
        x = torch.from_numpy(encode_board(board)).unsqueeze(0).to(self.device)
        logits, value = self.net(x)
        logits = logits.squeeze(0).cpu().numpy()   # (4672,)
        value = float(value.squeeze(0).cpu().item())

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return np.empty(0, dtype=np.float32), value, legal_moves
        idx = np.fromiter(
            (move_to_index(m, board) for m in legal_moves),
            dtype=np.int64, count=len(legal_moves),
        )
        legal_logits = logits[idx]
        legal_logits -= legal_logits.max()
        exp = np.exp(legal_logits)
        priors = (exp / exp.sum()).astype(np.float32)
        return priors, value, legal_moves

    # ---- Node expansion ----------------------------------------------------

    def _expand(self, node: Node) -> float:
        """Expand a leaf node. Returns the value to back up (from node's mover POV)."""
        term = _terminal_value_for_mover(node.board)
        if term is not None:
            node.is_terminal = True
            node.terminal_value = term
            node.is_expanded = True
            return term

        priors, value, legal = self._evaluate(node.board)
        node.legal_moves = legal
        node.move_indices = np.fromiter(
            (move_to_index(m, node.board) for m in legal),
            dtype=np.int32, count=len(legal),
        )
        node.P = priors  # aligned with `legal` by construction
        node.N = np.zeros(len(legal), dtype=np.int32)
        node.W = np.zeros(len(legal), dtype=np.float32)
        node.children = [None] * len(legal)
        node.is_expanded = True
        return value

    def _add_dirichlet_noise(self, node: Node) -> None:
        if len(node.P) == 0:
            return
        noise = np.random.dirichlet([self.cfg.dirichlet_alpha] * len(node.P)).astype(np.float32)
        eps = self.cfg.dirichlet_epsilon
        node.P = (1 - eps) * node.P + eps * noise

    # ---- One simulation ----------------------------------------------------

    def _simulate(self, root: Node) -> None:
        """One MCTS simulation: select down to a leaf, expand, backup."""
        node = root
        path: list[tuple[Node, int]] = []  # (node, edge_idx chosen at that node)

        # SELECT: descend to a leaf.
        while node.is_expanded and not node.is_terminal:
            edge = self._select_edge(node)
            path.append((node, edge))
            child = node.children[edge]
            if child is None:
                # Create child board lazily.
                b = node.board.copy(stack=True)
                b.push(node.legal_moves[edge])
                child = Node(b)
                node.children[edge] = child
            node = child
            if not node.is_expanded:
                break

        # EXPAND + evaluate (or use terminal value).
        if node.is_terminal:
            value = node.terminal_value
        else:
            value = self._expand(node)

        # BACKUP: value is from POV of `node`'s mover. Going up one ply flips sign.
        for parent, edge in reversed(path):
            value = -value
            parent.N[edge] += 1
            parent.W[edge] += value

    def _select_edge(self, node: Node) -> int:
        """PUCT edge selection."""
        total_visits = node.total_visits
        sqrt_total = math.sqrt(total_visits + 1e-8)
        Q = node.Q()
        # FPU: unvisited children get Q = cfg.fpu. (AZ paper uses 0; some
        # implementations use parent-Q - 0.25 for better exploration.)
        Q[node.N == 0] = self.cfg.fpu
        U = self.cfg.c_puct * node.P * sqrt_total / (1.0 + node.N)
        score = Q + U
        return int(np.argmax(score))

    # ---- Batched simulation (leaf-parallel with virtual loss) --------------

    @torch.no_grad()
    def _batch_evaluate(self, boards: list[chess.Board]):
        """Batch-evaluate many boards in one forward pass.

        Returns list of (priors, value, legal_moves) tuples, one per board.
        """
        if not boards:
            return []
        planes = np.stack([encode_board(b) for b in boards])  # (B, 119, 8, 8)
        x = torch.from_numpy(planes).to(self.device)
        logits, values = self.net(x)
        logits_np = logits.cpu().numpy()     # (B, 4672)
        values_np = values.cpu().numpy()     # (B,)
        results = []
        for i, board in enumerate(boards):
            legal = list(board.legal_moves)
            if not legal:
                results.append((np.empty(0, dtype=np.float32), float(values_np[i]), legal))
                continue
            idx = np.fromiter((move_to_index(m, board) for m in legal),
                              dtype=np.int64, count=len(legal))
            ll = logits_np[i, idx]
            ll = ll - ll.max()
            exp = np.exp(ll)
            priors = (exp / exp.sum()).astype(np.float32)
            results.append((priors, float(values_np[i]), legal))
        return results

    def _simulate_batch(self, root: Node, batch_size: int) -> None:
        """Run `batch_size` sims in parallel via virtual loss.

        Phase 1: descend `batch_size` paths down the tree, marking each traversed
                 edge with a negative virtual loss so other in-flight paths pick
                 different routes.
        Phase 2: collect leaves (non-terminal, un-expanded) and evaluate them in
                 one batched NN forward pass.
        Phase 3: expand leaves using fresh priors/values.
        Phase 4: back up each path's value, reversing virtual losses along the way.

        Safety: equivalent to `batch_size` calls to `_simulate` when batch_size=1.
        """
        vloss = self.cfg.virtual_loss
        paths: list[list[tuple[Node, int]]] = []
        leaves: list[Node] = []

        # ---- Phase 1: collect batch_size paths + leaves with virtual loss ----
        for _ in range(batch_size):
            node = root
            path: list[tuple[Node, int]] = []
            while node.is_expanded and not node.is_terminal:
                edge = self._select_edge(node)
                path.append((node, edge))
                # Apply virtual loss: bump N, subtract vloss from W
                # (so Q[edge] = W/N looks worse to the NEXT in-flight sim)
                node.N[edge] += 1
                node.W[edge] -= vloss
                child = node.children[edge]
                if child is None:
                    b = node.board.copy(stack=True)
                    b.push(node.legal_moves[edge])
                    child = Node(b)
                    node.children[edge] = child
                node = child
                if not node.is_expanded:
                    break
            paths.append(path)
            leaves.append(node)

        # ---- Phase 2: determine which leaves need NN evaluation ---------------
        # Terminal leaves have a known value; unexpanded leaves need NN eval;
        # already-expanded leaves (another in-flight sim expanded them) use their
        # stored value again (since they'd give same priors — we re-evaluate for
        # simplicity and it's cheap once per game).
        eval_indices = []
        eval_boards = []
        leaf_values: list[float | None] = [None] * batch_size
        for i, leaf in enumerate(leaves):
            term = _terminal_value_for_mover(leaf.board)
            if term is not None:
                leaf.is_terminal = True
                leaf.terminal_value = term
                leaf.is_expanded = True
                leaf_values[i] = term
            elif leaf.is_expanded:
                # Another sim in this batch already expanded it; pick any prior
                # value. Simplest: treat as 0 (neutral). Rarely fires.
                leaf_values[i] = 0.0
            else:
                eval_indices.append(i)
                eval_boards.append(leaf.board)

        # ---- Phase 3: single batched NN call, then expand ---------------------
        if eval_boards:
            results = self._batch_evaluate(eval_boards)
            for k, i in enumerate(eval_indices):
                priors, value, legal = results[k]
                leaf = leaves[i]
                if not leaf.is_expanded:  # might have been expanded by earlier i in this batch (rare)
                    leaf.legal_moves = legal
                    leaf.move_indices = np.fromiter(
                        (move_to_index(m, leaf.board) for m in legal),
                        dtype=np.int32, count=len(legal),
                    )
                    leaf.P = priors
                    leaf.N = np.zeros(len(legal), dtype=np.int32)
                    leaf.W = np.zeros(len(legal), dtype=np.float32)
                    leaf.children = [None] * len(legal)
                    leaf.is_expanded = True
                leaf_values[i] = value

        # ---- Phase 4: back up each path, reversing virtual losses -----------
        for i, path in enumerate(paths):
            value = leaf_values[i] if leaf_values[i] is not None else 0.0
            for parent, edge in reversed(path):
                # Reverse virtual loss we added in Phase 1
                parent.N[edge] -= 1
                parent.W[edge] += vloss
                # Real backup (sign-flip across plies)
                value = -value
                parent.N[edge] += 1
                parent.W[edge] += value

    # ---- Public API --------------------------------------------------------

    def run(self, board: chess.Board, *, add_root_noise: bool | None = None,
            reuse_root: Node | None = None) -> Node:
        """Run num_simulations from `board`. Returns the root node (with stats).

        If `reuse_root` is provided and its board matches, we reuse its subtree.
        """
        if reuse_root is not None and reuse_root.board.fen() == board.fen():
            root = reuse_root
        else:
            root = Node(board.copy(stack=True))

        if not root.is_expanded:
            self._expand(root)

        if (add_root_noise if add_root_noise is not None else self.cfg.add_root_noise):
            self._add_dirichlet_noise(root)

        bs = max(1, self.cfg.batch_size)
        if bs == 1:
            for _ in range(self.cfg.num_simulations):
                self._simulate(root)
        else:
            sims_done = 0
            while sims_done < self.cfg.num_simulations:
                this_batch = min(bs, self.cfg.num_simulations - sims_done)
                self._simulate_batch(root, this_batch)
                sims_done += this_batch

        return root

    def policy_from_root(self, root: Node, temperature: float) -> np.ndarray:
        """Return a (POLICY_SIZE,) visit-count distribution for training targets.

        temperature=1 -> proportional to N. temperature->0 -> one-hot on argmax.
        """
        pi = np.zeros(POLICY_SIZE, dtype=np.float32)
        if len(root.N) == 0:
            return pi
        if temperature < 1e-3:
            best = int(np.argmax(root.N))
            pi[root.move_indices[best]] = 1.0
            return pi
        counts = root.N.astype(np.float64) ** (1.0 / temperature)
        counts /= counts.sum()
        pi[root.move_indices] = counts.astype(np.float32)
        return pi

    def choose_move(self, root: Node, temperature: float) -> tuple[chess.Move, int]:
        """Sample a move from the visit distribution. Returns (move, edge_idx)."""
        if temperature < 1e-3:
            edge = int(np.argmax(root.N))
        else:
            counts = root.N.astype(np.float64) ** (1.0 / temperature)
            probs = counts / counts.sum()
            edge = int(np.random.choice(len(root.N), p=probs))
        return root.legal_moves[edge], edge
