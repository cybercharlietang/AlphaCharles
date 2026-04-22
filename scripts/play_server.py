"""Web server for playing against AlphaCharles on an interactive board.

Uses chessboard.js (from CDN) for the visual board, chess.js for move validation,
and our MCTS for the engine. Time control: configurable (default 3+2).

Endpoints:
  GET  /           — serves HTML UI
  POST /new        — start fresh game; body: {user_color: "w"|"b", base_s, inc_s}
  POST /move       — user plays a move; body: {uci: "e2e4"}, returns AI's reply
  GET  /state      — current FEN + PGN + clocks
  POST /resign     — user resigns
  POST /offer_draw — user offers/accepts draw
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import threading
import time
from pathlib import Path

import chess
import chess.pgn
import torch
from flask import Flask, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alphazero.mcts import MCTS, MCTSConfig
from alphazero.model import AlphaZeroNet, ModelConfig

app = Flask(__name__)

# Global game state (single-user app — one game at a time)
STATE = {
    "board": chess.Board(),
    "user_color": chess.WHITE,
    "clock": {"w": 180.0, "b": 180.0},   # seconds remaining
    "base_s": 180.0, "inc_s": 2.0,
    "last_move_time": None,
    "turn_start_time": None,
    "game": None,          # chess.pgn.Game() running accumulator
    "pgn_node": None,
    "result": None,        # "1-0", "0-1", "1/2-1/2" or None
    "thinking": False,     # AI is computing a move
    "ai_last_move": None,  # for UI display: dict {from, to, san, visits, q, time}
    "save_dir": "/workspace/AlphaCharles/runs/human_matches",
}
STATE_LOCK = threading.Lock()

# Model + MCTS loaded lazily
NET = None
MCTS_OBJ = None
DEVICE = None


def load_model(ckpt: str, sims: int, device: str):
    global NET, MCTS_OBJ, DEVICE
    DEVICE = torch.device(device if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    cfg_dict = state.get("model_cfg", {})
    NET = AlphaZeroNet(ModelConfig(**cfg_dict)).to(DEVICE)
    NET.load_state_dict(state["model"] if "model" in state else state)
    NET.eval()
    MCTS_OBJ = MCTS(NET, DEVICE, MCTSConfig(num_simulations=sims, add_root_noise=False,
                                            temperature_moves=0))
    print(f"Loaded {ckpt} on {DEVICE} with {sims} sims/move")


def tick_clock():
    """Subtract elapsed time from current mover's clock. Call before any state change."""
    now = time.time()
    s = STATE
    if s["result"] is not None or s["turn_start_time"] is None:
        return
    elapsed = now - s["turn_start_time"]
    mover = "w" if s["board"].turn == chess.WHITE else "b"
    s["clock"][mover] = max(0.0, s["clock"][mover] - elapsed)
    s["turn_start_time"] = now
    if s["clock"][mover] == 0.0:
        # flag fall
        s["result"] = "0-1" if mover == "w" else "1-0"
        try:
            save_pgn()
        except Exception as e:
            print(f"save_pgn on flag failed: {e}")


def apply_move(mv: chess.Move):
    """Apply a move, update clock with increment, advance turn."""
    s = STATE
    mover = "w" if s["board"].turn == chess.WHITE else "b"
    s["board"].push(mv)
    s["pgn_node"] = s["pgn_node"].add_variation(mv)
    s["clock"][mover] += s["inc_s"]
    s["turn_start_time"] = time.time()

    if s["board"].is_game_over(claim_draw=True):
        outcome = s["board"].outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            s["result"] = "1/2-1/2"
        else:
            s["result"] = "1-0" if outcome.winner == chess.WHITE else "0-1"
        try:
            save_pgn()
        except Exception as e:
            print(f"save_pgn failed: {e}")


def save_pgn():
    s = STATE
    os.makedirs(s["save_dir"], exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(s["save_dir"], f"human_vs_ac_{ts}.pgn")
    s["game"].headers["Result"] = s["result"] or "*"
    with open(path, "w") as f:
        print(s["game"], file=f, end="\n\n")


def think_and_move():
    """Called when it's the AI's turn. Computes move via MCTS."""
    s = STATE
    s["thinking"] = True
    t0 = time.time()
    board = s["board"].copy(stack=True)
    root = MCTS_OBJ.run(board, add_root_noise=False)
    best = int(root.N.argmax())
    mv = root.legal_moves[best]
    visits = int(root.N[best])
    q = float(root.W[best] / max(visits, 1))
    dt = time.time() - t0
    san = s["board"].san(mv)
    # Apply with clock update
    with STATE_LOCK:
        tick_clock()
        if s["result"] is None:  # not flagged
            apply_move(mv)
            s["ai_last_move"] = {
                "uci": mv.uci(), "san": san, "visits": visits,
                "q": round(q, 3), "time_s": round(dt, 2),
            }
        s["thinking"] = False


@app.route("/")
def home():
    return HTML


@app.route("/play")
def play_page():
    return HTML


# Serve the existing static dashboards from /workspace so a single pod port works.
@app.route("/games.html")
def serve_games():
    try:
        return open("/workspace/games.html").read()
    except Exception as e:
        return f"games.html not found: {e}", 404


@app.route("/losses.html")
def serve_losses():
    try:
        return open("/workspace/losses.html").read()
    except Exception as e:
        return f"losses.html not found: {e}", 404


@app.route("/telemetry.html")
def serve_telemetry():
    try:
        return open("/workspace/telemetry.html").read()
    except Exception as e:
        return f"telemetry.html not found: {e}", 404


@app.route("/state")
def get_state():
    with STATE_LOCK:
        tick_clock()
        s = STATE
        return jsonify({
            "fen": s["board"].fen(),
            "turn": "w" if s["board"].turn == chess.WHITE else "b",
            "user_color": "w" if s["user_color"] == chess.WHITE else "b",
            "clock": s["clock"],
            "result": s["result"],
            "legal_moves": [m.uci() for m in s["board"].legal_moves],
            "history": [m.uci() for m in s["board"].move_stack],
            "ai_last_move": s["ai_last_move"],
            "thinking": s["thinking"],
            "game_over": s["result"] is not None,
        })


@app.route("/new", methods=["POST"])
def new_game():
    data = request.get_json() or {}
    user_color_str = data.get("user_color", "w")
    base_s = float(data.get("base_s", 180.0))
    inc_s = float(data.get("inc_s", 2.0))
    with STATE_LOCK:
        STATE["board"] = chess.Board()
        STATE["user_color"] = chess.WHITE if user_color_str == "w" else chess.BLACK
        STATE["clock"] = {"w": base_s, "b": base_s}
        STATE["base_s"] = base_s
        STATE["inc_s"] = inc_s
        STATE["result"] = None
        STATE["ai_last_move"] = None
        STATE["thinking"] = False
        game = chess.pgn.Game()
        game.headers["Event"] = "Human vs AlphaCharles"
        game.headers["White"] = "Human" if STATE["user_color"] == chess.WHITE else "AlphaCharles"
        game.headers["Black"] = "AlphaCharles" if STATE["user_color"] == chess.WHITE else "Human"
        game.headers["TimeControl"] = f"{int(base_s)}+{int(inc_s)}"
        game.headers["Date"] = datetime.date.today().isoformat()
        STATE["game"] = game
        STATE["pgn_node"] = game
        STATE["turn_start_time"] = time.time()
    # If AI plays white, trigger its move immediately
    if STATE["user_color"] == chess.BLACK:
        threading.Thread(target=think_and_move, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/move", methods=["POST"])
def user_move():
    data = request.get_json() or {}
    uci = data.get("uci", "")
    with STATE_LOCK:
        tick_clock()
        s = STATE
        if s["result"] is not None:
            return jsonify({"error": "game over"}), 400
        if s["board"].turn != s["user_color"]:
            return jsonify({"error": "not your turn"}), 400
        try:
            mv = chess.Move.from_uci(uci)
        except ValueError:
            return jsonify({"error": "bad uci"}), 400
        if mv not in s["board"].legal_moves:
            return jsonify({"error": "illegal"}), 400
        apply_move(mv)
    # Queue AI move
    if STATE["result"] is None:
        threading.Thread(target=think_and_move, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/resign", methods=["POST"])
def resign():
    with STATE_LOCK:
        s = STATE
        if s["result"] is None:
            s["result"] = "0-1" if s["user_color"] == chess.WHITE else "1-0"
            save_pgn()
    return jsonify({"ok": True})


# ---- HTML frontend ----
HTML = r"""
<!DOCTYPE html><html><head><meta charset="utf-8"><title>Play AlphaCharles</title>
<link rel="stylesheet" href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css">
<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<script src="https://unpkg.com/chess.js@0.10.3/chess.js"></script>
<script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
<style>
body{font-family:system-ui;background:#0d1117;color:#c9d1d9;margin:0;padding:20px;display:flex;gap:24px;flex-wrap:wrap}
.board-col{flex:0 0 480px}
.side-col{flex:1 1 360px;max-width:480px}
#board{width:460px}
h1{color:#58a6ff;margin:0 0 10px;font-size:1.3em}
.clock{background:#161b22;border:1px solid #30363d;padding:12px;border-radius:6px;
  margin:8px 0;font-family:ui-monospace;font-size:1.8em;display:flex;justify-content:space-between}
.clock.active{border-color:#58a6ff;background:#1a2a3a}
.clock.low{color:#ff7b72}
.ctrl{margin:12px 0;display:flex;gap:8px;flex-wrap:wrap}
.ctrl button{background:#238636;color:#fff;border:none;padding:8px 14px;border-radius:6px;cursor:pointer}
.ctrl button:hover{background:#2ea043}
.ctrl button.danger{background:#da3633}
.ctrl select,.ctrl input{background:#21262d;color:#c9d1d9;border:1px solid #30363d;padding:6px 10px;border-radius:4px}
.info{background:#161b22;border:1px solid #30363d;padding:10px 14px;border-radius:6px;margin:6px 0;font-size:.9em}
.ai-move{color:#7ee787;font-family:ui-monospace}
.result{font-weight:bold;color:#58a6ff;font-size:1.1em}
.history{background:#161b22;border:1px solid #30363d;padding:10px;border-radius:6px;
  font-family:ui-monospace;font-size:.85em;max-height:220px;overflow-y:auto}
</style></head><body>

<div class="board-col">
<h1>Play AlphaCharles</h1>
<div class="clock" id="clock-opp"><span>Opponent</span><span id="clock-opp-time">3:00</span></div>
<div id="board"></div>
<div class="clock" id="clock-me"><span>You</span><span id="clock-me-time">3:00</span></div>
</div>

<div class="side-col">
<div class="ctrl">
<label>Color:
<select id="sel-color"><option value="w">White</option><option value="b">Black</option></select>
</label>
<label>Time (s):
<input type="number" id="sel-base" value="180" min="30" max="1800" style="width:70px">
</label>
<label>Inc (s):
<input type="number" id="sel-inc" value="2" min="0" max="30" style="width:50px">
</label>
<button onclick="newGame()">New game</button>
<button class="danger" onclick="resign()">Resign</button>
</div>

<div class="info">
<span id="status">Click "New game" to start.</span>
<span id="premove-badge" style="display:none;background:#3b82f6;color:#fff;padding:2px 8px;border-radius:4px;margin-left:8px;font-size:.85em">
premove: <span id="premove-text">-</span> (right-click or Esc to cancel)
</span>
</div>
<div class="info" id="ai-info" style="display:none">
AI move: <span class="ai-move" id="ai-san"></span>
<small> &middot; visits=<span id="ai-visits">-</span>, Q=<span id="ai-q">-</span>, time=<span id="ai-time">-</span>s</small>
</div>
<div class="info">Moves:
<div class="history" id="history"></div>
</div>
<div class="info" id="result-info" style="display:none">
<span class="result" id="result-text"></span>
</div>
</div>

<script>
let board = null;
let game = new Chess();
let myColor = 'w';
let gameOver = false;
let premove = null;   // {from, to} pending premove
let lastServerFen = null;

function cfg() { return {
  draggable: true,
  position: 'start',
  orientation: myColor === 'w' ? 'white' : 'black',
  onDragStart: onDragStart,
  onDrop: onDrop,
  onSnapEnd: onSnapEnd,
  pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png',
}; }

function onDragStart(source, piece) {
  if (gameOver) return false;
  // Only allow dragging OUR pieces, regardless of turn (premove allowed)
  if ((myColor==='w' && piece.startsWith('b')) || (myColor==='b' && piece.startsWith('w'))) return false;
}

function onDrop(source, target) {
  if (gameOver) return 'snapback';
  // If it's my turn, try normal move
  if (game.turn() === myColor) {
    const move = game.move({from:source, to:target, promotion:'q'});
    if (!move) return 'snapback';
    premove = null;
    document.getElementById('premove-badge').style.display = 'none';
    fetch('/move', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({uci: source+target+(move.promotion||'')})
    }).then(r=>r.json()).then(d=>{
      if (d.error) {
        alert('server rejected: ' + d.error);
        game.undo();
        board.position(game.fen());
      } else {
        refresh();
      }
    });
  } else {
    // Not my turn: store as premove
    premove = {from: source, to: target};
    document.getElementById('premove-badge').style.display = 'inline-block';
    document.getElementById('premove-text').textContent = source + '→' + target;
    return 'snapback';  // visually snap back; we'll execute once it's our turn
  }
}

function onSnapEnd() { board.position(game.fen()); }

function tryExecutePremove() {
  if (!premove) return;
  if (game.turn() !== myColor) return;
  if (gameOver) { premove = null; return; }
  // Validate the premove is legal NOW
  const move = game.move({from:premove.from, to:premove.to, promotion:'q'});
  if (!move) {
    // Illegal once opponent moved — cancel
    premove = null;
    document.getElementById('premove-badge').style.display = 'none';
    return;
  }
  // Legal — send immediately
  const uci = premove.from + premove.to + (move.promotion||'');
  premove = null;
  document.getElementById('premove-badge').style.display = 'none';
  board.position(game.fen());
  fetch('/move', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({uci: uci})
  }).then(r=>r.json()).then(d=>{
    if (d.error) {
      game.undo();
      board.position(game.fen());
    } else {
      refresh();
    }
  });
}

function cancelPremove() {
  premove = null;
  document.getElementById('premove-badge').style.display = 'none';
}

function fmtTime(s) {
  if (s <= 0) return "0:00";
  const m = Math.floor(s/60);
  const ss = Math.floor(s%60);
  return m + ':' + (ss<10?'0':'') + ss;
}

function refresh() {
  fetch('/state').then(r=>r.json()).then(d=>{
    // Sync board — detect when server's state (which includes AI's move) differs
    if (d.fen !== lastServerFen) {
      lastServerFen = d.fen;
      game.load(d.fen);
      board.position(game.fen());
      // Attempt premove now (if it's our turn and we have one queued)
      setTimeout(tryExecutePremove, 30);
    }
    // Clocks
    const opp = myColor==='w' ? d.clock.b : d.clock.w;
    const me = myColor==='w' ? d.clock.w : d.clock.b;
    document.getElementById('clock-opp-time').textContent = fmtTime(opp);
    document.getElementById('clock-me-time').textContent = fmtTime(me);
    document.getElementById('clock-opp').className = 'clock' + (d.turn !== myColor ? ' active' : '') + (opp < 30 ? ' low' : '');
    document.getElementById('clock-me').className = 'clock' + (d.turn === myColor ? ' active' : '') + (me < 30 ? ' low' : '');
    // Status
    let status = '';
    if (d.thinking) status = 'AI thinking...';
    else if (d.result) status = 'Game over: ' + d.result;
    else if (d.turn === myColor) status = 'Your turn';
    else status = "Opponent's turn";
    document.getElementById('status').textContent = status;
    // AI move
    if (d.ai_last_move) {
      document.getElementById('ai-info').style.display = 'block';
      document.getElementById('ai-san').textContent = d.ai_last_move.san;
      document.getElementById('ai-visits').textContent = d.ai_last_move.visits;
      document.getElementById('ai-q').textContent = d.ai_last_move.q;
      document.getElementById('ai-time').textContent = d.ai_last_move.time_s;
    }
    // History
    const hist = d.history;
    let html = '';
    for (let i = 0; i < hist.length; i += 2) {
      const num = Math.floor(i/2) + 1;
      const w = hist[i] || '';
      const b = hist[i+1] || '';
      html += num + '. ' + w + ' ' + b + '  ';
    }
    document.getElementById('history').textContent = html;
    // Game over
    gameOver = d.game_over;
    if (d.result) {
      document.getElementById('result-info').style.display = 'block';
      document.getElementById('result-text').textContent = 'Result: ' + d.result;
    }
  });
}

function newGame() {
  myColor = document.getElementById('sel-color').value;
  const base = parseFloat(document.getElementById('sel-base').value);
  const inc = parseFloat(document.getElementById('sel-inc').value);
  fetch('/new', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({user_color: myColor, base_s: base, inc_s: inc})
  }).then(()=>{
    game.reset();
    board = Chessboard('board', cfg());
    gameOver = false;
    document.getElementById('result-info').style.display = 'none';
    document.getElementById('ai-info').style.display = 'none';
    refresh();
  });
}

board = Chessboard('board', cfg());
setInterval(refresh, 200);
refresh();

// Right-click or Escape cancels premove
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') cancelPremove(); });
document.getElementById('board').addEventListener('contextmenu', (e) => {
  e.preventDefault(); cancelPremove();
});
</script>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/AlphaCharles/runs/rl_v3/final.pt")
    ap.add_argument("--sims", type=int, default=1600)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--port", type=int, default=8889)
    ap.add_argument("--save-dir", default="/workspace/AlphaCharles/runs/human_matches")
    args = ap.parse_args()
    STATE["save_dir"] = args.save_dir
    load_model(args.ckpt, args.sims, args.device)
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
