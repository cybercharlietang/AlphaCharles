"""Serve a game browser that lists recent PGNs from self-play and eval games.

Output: /workspace/games.html (auto-refresh). Uses lichess.org's embedded
analysis board for interactive replay — click a game row, paste PGN into the
inline analysis widget.
"""

from __future__ import annotations

import glob
import os
import time
from pathlib import Path

OUT = "/workspace/games.html"

SELFPLAY_DIRS = [
    "/workspace/AlphaCharles/runs/rl_parallel/games/consumed",
    "/workspace/AlphaCharles/runs/rl_parallel/games/pending",
]
EVAL_DIR = "/workspace/AlphaCharles/runs/eval2/pgns"
HUMAN_DIR = "/workspace/AlphaCharles/runs/human_matches"


def list_games():
    items = []
    for d in SELFPLAY_DIRS:
        for p in sorted(glob.glob(f"{d}/*.pgn"), key=os.path.getmtime, reverse=True)[:100]:
            items.append(("self-play", p))
    for p in sorted(glob.glob(f"{EVAL_DIR}/*.pgn"), key=os.path.getmtime, reverse=True)[:60]:
        items.append(("eval", p))
    for p in sorted(glob.glob(f"{HUMAN_DIR}/*.pgn"), key=os.path.getmtime, reverse=True)[:30]:
        items.append(("human", p))
    return items


def read_pgn(path):
    try:
        with open(path) as fh:
            return fh.read()
    except Exception:
        return ""


def parse_headers(pgn):
    h = {}
    for line in pgn.splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            inside = line[1:-1]
            if " " in inside:
                k, v = inside.split(" ", 1)
                h[k] = v.strip('"')
    return h


def write_html():
    items = list_games()
    # Build list of (kind, path, headers, body_moves, mtime)
    entries = []
    for kind, path in items:
        pgn = read_pgn(path)
        hdrs = parse_headers(pgn)
        mt = os.path.getmtime(path)
        # Extract moves portion (after last blank line within header block)
        lines = pgn.splitlines()
        split = 0
        for i, l in enumerate(lines):
            if not l.strip() and i > 2:
                split = i + 1
                break
        moves = "\n".join(lines[split:]).strip()
        entries.append({
            "kind": kind, "path": path, "headers": hdrs,
            "moves": moves, "pgn": pgn, "mtime": mt,
        })

    html_parts = []
    html_parts.append("""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>AlphaCharles games</title>
<meta http-equiv="refresh" content="60">
<style>
body{font-family:system-ui;background:#0d1117;color:#c9d1d9;padding:20px;margin:0}
h1{color:#58a6ff}
.filter{margin:10px 0}
.filter button{background:#21262d;color:#c9d1d9;border:1px solid #30363d;
  padding:6px 12px;border-radius:6px;cursor:pointer;margin-right:6px}
.filter button.active{background:#1f6feb}
.game{background:#161b22;border:1px solid #30363d;border-radius:6px;
  padding:12px 16px;margin:10px 0}
.game h3{margin:0 0 6px;color:#58a6ff;font-size:1em}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;
  font-size:.75em;margin-right:8px}
.badge.self-play{background:#34b77c}
.badge.eval{background:#f0883e}
.badge.human{background:#bc6cff}
.result{font-weight:bold}
.result.win{color:#7ee787}
.result.draw{color:#d29922}
.result.loss{color:#ff7b72}
.meta{color:#8b949e;font-size:.85em}
.pgn{font-family:ui-monospace;font-size:.85em;background:#0d1117;
  border:1px solid #30363d;padding:8px;margin-top:8px;
  border-radius:4px;max-height:200px;overflow-y:auto;white-space:pre-wrap;display:none}
.show-pgn{background:#21262d;color:#58a6ff;border:none;
  padding:4px 10px;border-radius:4px;cursor:pointer;font-size:.85em;margin-right:6px}
.lichess{color:#58a6ff;text-decoration:none}
</style></head><body>
<h1>AlphaCharles games</h1>
""")
    html_parts.append(f'<p class="meta">Updated {time.strftime("%Y-%m-%d %H:%M:%S UTC")} &bull; {len(entries)} games listed &bull; auto-refresh 60s</p>')
    html_parts.append("""<div class="filter">
  <button class="active" onclick="filt(\'all\')">All</button>
  <button onclick="filt(\'self-play\')">Self-play</button>
  <button onclick="filt(\'eval\')">Eval</button>
  <button onclick="filt(\'human\')">Human</button>
</div>
<div id="games">""")

    for e in entries:
        h = e["headers"]
        white = h.get("White", "?")
        black = h.get("Black", "?")
        result = h.get("Result", "*")
        plies = h.get("PlyCount", "?")
        event = h.get("Event", "")
        # Colour the result relative to AlphaCharles.
        win_cls = "draw"
        if result == "1/2-1/2":
            win_cls = "draw"
        elif result == "1-0":
            win_cls = "win" if "AlphaCharles" in white else "loss"
        elif result == "0-1":
            win_cls = "win" if "AlphaCharles" in black else "loss"
        basename = os.path.basename(e["path"])
        # Escape PGN for inclusion as text
        pgn_escaped = (e["pgn"].replace("&", "&amp;").replace("<", "&lt;")
                       .replace(">", "&gt;").replace("\"", "&quot;"))
        html_parts.append(f'''<div class="game" data-kind="{e["kind"]}">
<h3><span class="badge {e["kind"]}">{e["kind"]}</span>
{white} vs {black}
<span class="result {win_cls}">{result}</span>
<span class="meta">&bull; {plies} plies &bull; {event}</span></h3>
<div class="meta">{basename}</div>
<button class="show-pgn" onclick="togl(this)">Show PGN</button>
<a class="lichess" href="https://lichess.org/paste" target="_blank">→ Paste into lichess.org/paste for interactive replay</a>
<div class="pgn">{pgn_escaped}</div>
</div>''')

    html_parts.append("""</div>
<script>
function filt(k) {
  document.querySelectorAll('.filter button').forEach(b=>b.classList.remove('active'));
  event.target.classList.add('active');
  document.querySelectorAll('.game').forEach(g=>{
    g.style.display = (k==='all'||g.dataset.kind===k)?'':'none';
  });
}
function togl(btn){
  const pre = btn.parentElement.querySelector('.pgn');
  if (pre.style.display==='block'){pre.style.display='none';btn.textContent='Show PGN';}
  else {pre.style.display='block';btn.textContent='Hide PGN';}
}
</script></body></html>""")

    with open(OUT, "w") as f:
        f.write("".join(html_parts))


if __name__ == "__main__":
    while True:
        write_html()
        time.sleep(30)
