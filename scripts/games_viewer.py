"""Unified dashboard: each tab has metric charts (if applicable) + games list.

Tabs:
    SL warmstart (training)   → metrics only (no games; trained on PGN files)
    SL puzzles  (training)    → metrics only
    RL v1 self-play           → metrics + games (broken run, for reference)
    RL v2 self-play           → metrics + games (current training)
    Eval: SL vs SF d3         → games only
    Eval: post-RL vs SF       → games only
    Human matches             → games only (empty until played)

Log scale uses decade-only ticks so the grid reads cleanly (10, 100, 1000).
"""
from __future__ import annotations

import glob
import json
import os
import re
import time
from collections import Counter

OUT = "/workspace/games.html"

# Training log sources
LOG_PIPELINE = "/workspace/AlphaCharles/runs/pipeline.log"
LOG_SL_SOFT = "/workspace/AlphaCharles/runs/sl_soft/pipeline.log"
LOG_RL_V2 = "/workspace/AlphaCharles/runs/rl_v2_test/trainer.log"
LOG_RL_V3 = "/workspace/AlphaCharles/runs/rl_v3/trainer.log"

# Categories: label, metric_stage (or None), list of game dirs
CATEGORIES = [
    ("SL warmstart (original, 150k steps, overfit)", "sl_warmstart", []),
    ("SL soft (fresh 10k, target entropy ~1)", "sl_soft",         []),
    ("SL puzzles (training, broken value head)", "sl_puzzles",    []),
    ("RL v1 self-play (broken MCTS)",  "rl_selfplay",      [
        "/workspace/AlphaCharles/runs/rl_parallel/games/consumed",
    ]),
    ("RL v2 self-play (overfit seed, aborted)",     "rl_v2_test",       [
        "/workspace/AlphaCharles/runs/rl_v2_test/games/consumed",
        "/workspace/AlphaCharles/runs/rl_v2_test/games/pending",
    ]),
    ("RL v3 current (5 games/step, 16 workers/GPU, bs=32)", "rl_v3", [
        "/workspace/AlphaCharles/runs/rl_v3/games/consumed",
        "/workspace/AlphaCharles/runs/rl_v3/games/pending",
    ]),
    ("Eval: sl_soft vs Stockfish (d3+d5)", None,           [
        "/workspace/AlphaCharles/runs/eval_sl_soft/pgns",
    ]),
    ("Eval: sl_warmstart vs Stockfish d3",      None,               [
        "/workspace/AlphaCharles/runs/eval_sf3_pgns/pgns",
        "/workspace/AlphaCharles/runs/eval_v2_fixed_mcts/pgns",
    ]),
    ("Eval: RL v3 @ 1600 sims vs SF (d3+d5+d7)", None,       [
        "/workspace/AlphaCharles/runs/eval_rl_v3_1600/pgns",
    ]),
    ("Eval: RL v3 @ 400 sims vs SF (d3+d5)", None,           [
        "/workspace/AlphaCharles/runs/eval_rl_v3/pgns",
    ]),
    ("Eval: post-RL (rl_parallel) vs SF", None,           [
        "/workspace/AlphaCharles/runs/eval_post_rl/pgns",
    ]),
    ("Human matches",                 None,               [
        "/workspace/AlphaCharles/runs/human_matches",
    ]),
]

HEADER_RE = re.compile(r'\[(\w+)\s+"(.*?)"\]')
LOSS_PAT = re.compile(r"step=(\d+).*?total=([\d.]+).*?policy=([\d.]+).*?value=([\d.]+).*?lr=([\d.e+-]+)")
STAT_PAT = re.compile(r"step=(\d+).*?entropy=([\d.]+).*?corr=([-\d.]+)")
GRAD_PAT = re.compile(r"step=(\d+).*?grad_norm=([\d.]+).*?steps/s=([\d.]+)")
STAGE_MARK = re.compile(r"=== \[run\] (sl_warmstart|sl_puzzles|rl_selfplay) ===")


def parse_file(path, default_stage=None):
    stages = {}
    if not os.path.exists(path):
        return stages
    current = default_stage
    try:
        with open(path) as f:
            for line in f:
                m = STAGE_MARK.search(line)
                if m:
                    current = m.group(1); continue
                if current is None:
                    continue
                buf = stages.setdefault(current, {})
                for pat, keys in [(LOSS_PAT, ("total", "policy", "value", "lr")),
                                  (STAT_PAT, ("entropy", "corr")),
                                  (GRAD_PAT, ("grad_norm", "steps_per_s"))]:
                    m = pat.search(line)
                    if m:
                        step = int(m.group(1))
                        rec = buf.setdefault(step, {"step": step})
                        for i, k in enumerate(keys):
                            rec[k] = float(m.group(i + 2))
                        break
    except Exception:
        pass
    return {k: sorted(v.values(), key=lambda p: p["step"]) for k, v in stages.items()}


def all_stage_metrics():
    stages = {}
    stages.update(parse_file(LOG_PIPELINE))
    stages.update(parse_file(LOG_SL_SOFT, default_stage="sl_soft"))
    stages.update(parse_file(LOG_RL_V2, default_stage="rl_v2_test"))
    stages.update(parse_file(LOG_RL_V3, default_stage="rl_v3"))
    return stages


def parse_pgn(path):
    try:
        txt = open(path, errors="ignore").read()
    except Exception:
        return None
    headers = {m.group(1): m.group(2) for m in HEADER_RE.finditer(txt)}
    lines = txt.splitlines()
    move_start = 0
    for i, l in enumerate(lines):
        if l.strip() and not l.startswith("["):
            move_start = i; break
    move_text = " ".join(lines[move_start:]).strip()
    tokens = move_text.split()
    plies = sum(1 for t in tokens if not t.endswith(".") and t not in ("1-0", "0-1", "1/2-1/2", "*"))
    open_toks, pc = [], 0
    for t in tokens:
        if t.endswith("."): continue
        if t in ("1-0", "0-1", "1/2-1/2", "*"): break
        open_toks.append(t); pc += 1
        if pc >= 4: break
    mtime = os.path.getmtime(path) if os.path.exists(path) else 0
    return {"path": path, "headers": headers, "pgn": txt,
            "plies": plies, "opening": " ".join(open_toks), "mtime": mtime}


def collect_category(dirs):
    entries = []
    seen = set()
    for d in dirs:
        for p in glob.glob(f"{d}/*.pgn"):
            if p in seen: continue
            seen.add(p)
            e = parse_pgn(p)
            if e: entries.append(e)
    entries.sort(key=lambda e: -e["mtime"])
    return entries


def category_stats(entries):
    n = len(entries)
    if n == 0: return {"n": 0}
    results = Counter(e["headers"].get("Result", "*") for e in entries)
    plies = [e["plies"] for e in entries if e["plies"] > 0]
    ac_w = ac_d = ac_l = 0
    for e in entries:
        r = e["headers"].get("Result", "*")
        w = e["headers"].get("White", ""); b = e["headers"].get("Black", "")
        ac_white = "AlphaCharles" in w; ac_black = "AlphaCharles" in b
        if r == "1/2-1/2":
            ac_d += 1
        elif ac_white and ac_black:
            pass  # self-play; decisive counted differently below
        elif r == "1-0":
            if ac_white: ac_w += 1
            elif ac_black: ac_l += 1
        elif r == "0-1":
            if ac_black: ac_w += 1
            elif ac_white: ac_l += 1
    openings = Counter(e["opening"] for e in entries if e["opening"]).most_common(5)
    return {
        "n": n,
        "white_wins": results.get("1-0", 0),
        "black_wins": results.get("0-1", 0),
        "draws": results.get("1/2-1/2", 0),
        "ac_w": ac_w, "ac_d": ac_d, "ac_l": ac_l,
        "avg_plies": round(sum(plies) / len(plies), 1) if plies else 0,
        "min_plies": min(plies) if plies else 0,
        "max_plies": max(plies) if plies else 0,
        "top_openings": openings,
    }


def write_html():
    stage_metrics = all_stage_metrics()

    cats_data = []
    for label, stage, dirs in CATEGORIES:
        entries = collect_category(dirs)
        stats = category_stats(entries)
        metric_data = stage_metrics.get(stage, []) if stage else []
        cats_data.append({
            "label": label, "stage": stage, "dirs": dirs,
            "entries": entries, "stats": stats, "metrics": metric_data,
        })

    metrics_json = json.dumps({c["stage"]: c["metrics"] for c in cats_data if c["stage"]})

    html_parts = [f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>AlphaCharles dashboard</title>
<meta http-equiv="refresh" content="60">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body{{font-family:system-ui;background:#0d1117;color:#c9d1d9;padding:16px;margin:0;max-width:1500px}}
h1{{color:#58a6ff;margin-bottom:4px}}
h2{{color:#7ee787;margin-top:18px;border-bottom:1px solid #30363d;padding-bottom:6px}}
h3{{color:#ffa657;margin-top:16px;font-size:.95em}}
.nav{{position:sticky;top:0;background:#0d1117;padding:8px 0;z-index:10;
  border-bottom:1px solid #30363d;margin-bottom:16px;display:flex;flex-wrap:wrap;gap:4px}}
.tab{{padding:6px 12px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;
  border-radius:6px;cursor:pointer;font-family:system-ui;font-size:.9em}}
.tab.active{{background:#1f6feb;border-color:#1f6feb;color:#fff}}
.stats{{background:#161b22;border:1px solid #30363d;border-radius:6px;
  padding:12px 16px;margin:8px 0;line-height:1.6;font-size:.9em}}
.stats .k{{color:#8b949e;font-size:.85em}}
.stats .v{{color:#7ee787;font-weight:600}}
.grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}}
.grid>div{{background:#0d1117;padding:6px 12px;border-radius:4px}}
.chart-grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin:8px 0}}
.chart-card{{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:10px}}
.chart-card canvas{{max-width:100%;height:220px!important}}
.game{{background:#161b22;border:1px solid #30363d;border-radius:6px;
  padding:8px 14px;margin:6px 0;font-size:.9em}}
.game h4{{margin:0 0 4px;color:#58a6ff;font-size:.9em;font-weight:600}}
.result{{font-weight:bold;margin-left:6px}}
.result.draw{{color:#d29922}} .result.win{{color:#7ee787}} .result.loss{{color:#ff7b72}}
.meta{{color:#8b949e;font-size:.82em}}
.pgn{{font-family:ui-monospace;font-size:.8em;background:#0d1117;border:1px solid #30363d;
  padding:6px;margin-top:6px;border-radius:4px;max-height:200px;overflow-y:auto;
  white-space:pre-wrap;display:none}}
.show-pgn{{background:#21262d;color:#58a6ff;border:none;
  padding:3px 8px;border-radius:4px;cursor:pointer;font-size:.8em;margin-right:6px}}
.lichess{{color:#58a6ff;text-decoration:none;font-size:.82em}}
.note{{color:#8b949e;font-size:.82em}}
.no-metric{{color:#8b949e;font-style:italic;padding:8px}}
</style></head><body>
<h1>AlphaCharles dashboard</h1>
<p class="note">updated {time.strftime("%Y-%m-%d %H:%M:%S UTC")} &bull; auto-refresh 60s</p>
<div class="nav">"""]

    # Default-open the RL v3 tab (index depends on ordering; find it)
    default_idx = next((i for i, c in enumerate(cats_data) if "v3" in c["label"].lower()), 0)
    for i, c in enumerate(cats_data):
        cls = "tab active" if i == default_idx else "tab"
        html_parts.append(f'<button class="{cls}" onclick="showCat({i})">{c["label"]} ({c["stats"].get("n", 0)})</button>')
    html_parts.append('</div>')

    for i, c in enumerate(cats_data):
        vis = "block" if i == default_idx else "none"
        html_parts.append(f'<div class="cat" id="cat{i}" style="display:{vis}">')
        html_parts.append(f'<h2>{c["label"]}</h2>')

        # Metric charts at top (if this category has a metric stage)
        if c["stage"]:
            if c["metrics"]:
                html_parts.append('<h3>Training metrics</h3>')
                html_parts.append(f'''<div class="chart-grid">
  <div class="chart-card"><canvas id="chL_{i}"></canvas></div>
  <div class="chart-card"><canvas id="chP_{i}"></canvas></div>
  <div class="chart-card"><canvas id="chV_{i}"></canvas></div>
  <div class="chart-card"><canvas id="chE_{i}"></canvas></div>
  <div class="chart-card"><canvas id="chG_{i}"></canvas></div>
  <div class="chart-card"><canvas id="chC_{i}"></canvas></div>
</div>''')
            else:
                html_parts.append(f'<div class="no-metric">no training metrics yet for stage <code>{c["stage"]}</code></div>')

        # Game stats + list
        stats = c["stats"]
        if stats["n"] > 0:
            html_parts.append('<h3>Game statistics</h3>')
            topops = "".join(f'<div>{o or "—"}: <span class="v">{ct}</span></div>'
                             for o, ct in stats["top_openings"])
            html_parts.append(f'''<div class="stats">
<div class="grid">
  <div><span class="k">Total:</span> <span class="v">{stats["n"]}</span></div>
  <div><span class="k">Result split (W/D/L):</span> <span class="v">{stats["white_wins"]}/{stats["draws"]}/{stats["black_wins"]}</span></div>
  <div><span class="k">AC vs opponent (W/D/L):</span> <span class="v">{stats["ac_w"]}/{stats["ac_d"]}/{stats["ac_l"]}</span></div>
  <div><span class="k">Avg plies:</span> <span class="v">{stats["avg_plies"]}</span></div>
  <div><span class="k">Min plies:</span> <span class="v">{stats["min_plies"]}</span></div>
  <div><span class="k">Max plies:</span> <span class="v">{stats["max_plies"]}</span></div>
</div>
<div style="margin-top:8px"><span class="k">Top 5 opening 4-ply sequences:</span>
<div class="grid" style="grid-template-columns:1fr 1fr">{topops}</div></div>
</div>''')

            display = c["entries"][:40]
            if len(c["entries"]) > 40:
                html_parts.append(f'<p class="note">showing 40 most recent of {len(c["entries"])}</p>')
            for j, e in enumerate(display):
                h = e["headers"]
                w = h.get("White", "?"); b = h.get("Black", "?")
                result = h.get("Result", "*")
                cls = "draw"
                if result == "1-0":
                    cls = "win" if ("AlphaCharles" in w and "AlphaCharles" not in b) else ("loss" if "AlphaCharles" in b and "AlphaCharles" not in w else "draw")
                elif result == "0-1":
                    cls = "win" if ("AlphaCharles" in b and "AlphaCharles" not in w) else ("loss" if "AlphaCharles" in w and "AlphaCharles" not in b else "draw")
                basename = os.path.basename(e["path"])
                pgn_esc = (e["pgn"].replace("&", "&amp;").replace("<", "&lt;")
                           .replace(">", "&gt;").replace('"', "&quot;"))
                gid = f"g{i}_{j}"
                html_parts.append(f'''<div class="game">
<h4>{w} vs {b} <span class="result {cls}">{result}</span>
<span class="meta">&bull; {e["plies"]} plies &bull; {e["opening"] or "-"}</span></h4>
<div class="meta">{basename}</div>
<button class="show-pgn" onclick="togl(\'{gid}\', this)">Show PGN</button>
<a class="lichess" href="https://lichess.org/paste" target="_blank">→ lichess.org/paste for replay</a>
<div class="pgn" id="{gid}">{pgn_esc}</div>
</div>''')
        elif c["stage"] is None:
            html_parts.append('<div class="no-metric">no games in this category yet</div>')

        html_parts.append('</div>')

    # JS: tab switcher + chart renderer with decade-only log-ticks.
    html_parts.append(f'''<script>
const M = {metrics_json};
const COLORS = {{
  "sl_warmstart":"#58a6ff", "sl_soft":"#9cdcfe",
  "sl_puzzles":"#f0883e",
  "rl_selfplay":"#bc6cff",  "rl_v2_test":"#7ee787", "rl_v3":"#2ea043"
}};

// Only show tick labels at exact powers of 10 for log-scale y axes.
function decadeTicks(val) {{
  const log = Math.log10(val);
  return Math.abs(log - Math.round(log)) < 0.01 ? val.toString() : '';
}}

function mkChart(canvasId, stage, title, metricKey, logy) {{
  const rows = M[stage] || [];
  const pts = rows.map(r => ({{x: r.step, y: r[metricKey]}}))
                  .filter(p => p.y !== undefined && p.y !== null && (!logy || p.y > 0));
  if (!pts.length) return;
  new Chart(document.getElementById(canvasId), {{
    type: 'line',
    data: {{
      datasets: [{{
        label: metricKey, data: pts,
        borderColor: COLORS[stage] || "#c9d1d9",
        backgroundColor: (COLORS[stage] || "#c9d1d9") + "22",
        tension: 0.1, pointRadius: 0, borderWidth: 2,
      }}]
    }},
    options: {{
      plugins: {{
        title: {{display:true, text: title, color:'#c9d1d9', font:{{size:12}} }},
        legend: {{display:false}}
      }},
      scales: {{
        x: {{type:'linear', ticks:{{color:'#8b949e',maxTicksLimit:6}}, grid:{{color:'#30363d'}},
             title:{{display:true, text:'training step', color:'#8b949e', font:{{size:11}} }} }},
        y: {{
          type: logy ? 'logarithmic' : 'linear',
          ticks: {{
            color:'#8b949e',
            callback: logy ? decadeTicks : undefined,
          }},
          grid: {{
            color: (ctx) => {{
              if (!logy) return '#30363d';
              const v = ctx.tick.value;
              const log = Math.log10(v);
              return Math.abs(log - Math.round(log)) < 0.01 ? '#30363d' : '#1a1f26';
            }}
          }},
          title: {{display:true, text: (function(){{
            const titles = {{
              "Total loss (log)": "total loss (nats)",
              "Policy loss (log)": "cross-entropy (nats)",
              "Value loss (log)": "MSE",
              "Policy entropy (log, nats)": "entropy (nats)",
              "Grad norm (log)": "gradient L2 norm",
              "Value corr": "Pearson correlation"
            }};
            return titles[title] || "";
          }})(), color:'#8b949e', font:{{size:11}} }}
        }}
      }}
    }}
  }});
}}

// Render charts for each stage that has data.
const STAGES = {{'''
                       + ", ".join(f'{i}: "{c["stage"]}"' for i, c in enumerate(cats_data) if c["stage"] and c["metrics"])
                       + '''};
Object.entries(STAGES).forEach(([i, stage]) => {
  mkChart(`chL_${i}`, stage, "Total loss (log)", "total", true);
  mkChart(`chP_${i}`, stage, "Policy loss (log)", "policy", true);
  mkChart(`chV_${i}`, stage, "Value loss (log)", "value", true);
  mkChart(`chE_${i}`, stage, "Policy entropy (log, nats)", "entropy", true);
  mkChart(`chG_${i}`, stage, "Grad norm (log)", "grad_norm", true);
  mkChart(`chC_${i}`, stage, "Value corr", "corr", false);
});

function showCat(i) {
  document.querySelectorAll('.cat').forEach((c, idx) => c.style.display = idx === i ? 'block' : 'none');
  document.querySelectorAll('.tab').forEach((t, idx) => t.classList.toggle('active', idx === i));
}
function togl(id, btn) {
  const p = document.getElementById(id);
  if (p.style.display === 'block') { p.style.display='none'; btn.textContent='Show PGN'; }
  else { p.style.display='block'; btn.textContent='Hide PGN'; }
}
</script></body></html>''')

    with open(OUT, "w") as f:
        f.write("".join(html_parts))


if __name__ == "__main__":
    while True:
        try:
            write_html()
        except Exception as e:
            print(f"[viewer err] {e}", flush=True)
        time.sleep(30)
