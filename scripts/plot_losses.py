"""Live loss-curve plotter. Parses ALL training log files and segments them by
the pipeline stage markers ('=== [run] <stage> ===').

Stages parsed:
    sl_warmstart       — from pipeline.log
    sl_puzzles         — from pipeline.log
    rl_selfplay (v1)   — from pipeline.log
    rl_v2_test         — from runs/rl_v2_test/trainer.log
"""
from __future__ import annotations

import json
import os
import re
import time

LOG_PIPELINE = "/workspace/AlphaCharles/runs/pipeline.log"
LOG_RL_V2 = "/workspace/AlphaCharles/runs/rl_v2_test/trainer.log"
OUT = "/workspace/losses.html"

LOSS_PAT = re.compile(r"step=(\d+).*?total=([\d.]+).*?policy=([\d.]+).*?value=([\d.]+).*?lr=([\d.e+-]+)")
STAT_PAT = re.compile(r"step=(\d+).*?entropy=([\d.]+).*?corr=([-\d.]+)")
GRAD_PAT = re.compile(r"step=(\d+).*?grad_norm=([\d.]+).*?steps/s=([\d.]+)")
STAGE_MARK = re.compile(r"=== \[run\] (sl_warmstart|sl_puzzles|rl_selfplay) ===")


def parse_file(path, default_stage=None):
    """Return dict: stage_label -> list of point dicts."""
    stages = {}
    if not os.path.exists(path):
        return stages
    current = default_stage
    try:
        with open(path) as f:
            for line in f:
                m = STAGE_MARK.search(line)
                if m:
                    current = m.group(1)
                    continue
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


def parse_all():
    all_stages = {}
    pipeline = parse_file(LOG_PIPELINE)
    for k, v in pipeline.items():
        all_stages[k] = v
    # RL v2 trainer log doesn't have stage markers, so label it directly.
    v2 = parse_file(LOG_RL_V2, default_stage="rl_v2_test")
    for k, v in v2.items():
        all_stages[k] = v
    return all_stages


def write_html():
    stages = parse_all()
    data = {k: v for k, v in stages.items() if v}
    total_pts = sum(len(v) for v in data.values())
    data_json = json.dumps(data)

    # Short summary line per stage
    summary_lines = []
    for s, rows in data.items():
        if rows:
            last = rows[-1]
            summary_lines.append(
                f"<b style='color:#58a6ff'>{s}</b>: {len(rows)} pts, "
                f"final step {last['step']}, "
                f"total_loss={last.get('total', 0):.3f}, "
                f"entropy={last.get('entropy', 0):.3f}, "
                f"corr={last.get('corr', 0):.3f}"
            )

    summary_html = " &bull; ".join(summary_lines) or "no data yet"

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>AlphaCharles loss curves</title>
<meta http-equiv="refresh" content="30">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body{{font-family:system-ui;background:#0d1117;color:#c9d1d9;padding:20px;margin:0;max-width:1500px}}
h1{{color:#58a6ff;margin-bottom:4px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:16px}}
.card{{background:#161b22;border:1px solid #30363d;padding:16px;border-radius:6px}}
.card canvas{{max-width:100%;height:340px!important}}
.note{{color:#8b949e;font-size:.82em;margin-top:4px;line-height:1.4}}
.summary{{background:#161b22;border:1px solid #30363d;padding:10px 16px;
  border-radius:6px;margin-top:10px;font-size:.9em}}
</style></head><body>
<h1>AlphaCharles — loss curves (all runs)</h1>
<p style="color:#8b949e;margin:0">auto-refresh 30s &bull; {total_pts} total points &bull; updated {time.strftime("%Y-%m-%d %H:%M:%S UTC")}</p>
<p><a href="/games.html" style="color:#58a6ff">games →</a> &bull;
<a href="/telemetry.html" style="color:#58a6ff">telemetry →</a></p>
<div class="summary">{summary_html}</div>
<div class="grid">
  <div class="card"><canvas id="ch_loss"></canvas>
    <p class="note"><b>Total loss</b> = policy_CE + value_MSE + weight_decay. Log y.</p></div>
  <div class="card"><canvas id="ch_policy"></canvas>
    <p class="note"><b>Policy loss</b> (cross-entropy on move target). Log y.</p></div>
  <div class="card"><canvas id="ch_value"></canvas>
    <p class="note"><b>Value loss</b> (MSE on z). Log y. Puzzle stage dropped to 0 because z=+1 constant — the smoking gun.</p></div>
  <div class="card"><canvas id="ch_entropy"></canvas>
    <p class="note"><b>Policy entropy</b> (nats). Starts ~8 (uniform over 4672), drops as net concentrates. &lt;0.3 = overfit risk. Log y.</p></div>
  <div class="card"><canvas id="ch_corr"></canvas>
    <p class="note"><b>Value correlation</b> (Pearson with z). Rises as value head learns. Linear y.</p></div>
  <div class="card"><canvas id="ch_grad"></canvas>
    <p class="note"><b>Gradient norm</b> (pre-clip, clip=5). Log y.</p></div>
</div>

<script>
const DATA = {data_json};
const COLORS = {{
  "sl_warmstart":"#58a6ff",
  "sl_puzzles":"#f0883e",
  "rl_selfplay":"#bc6cff",
  "rl_v2_test":"#7ee787"
}};

function makeChart(id, title, key, logy) {{
  const datasets = [];
  for (const stage of Object.keys(DATA)) {{
    const pts = DATA[stage].map(r => ({{x: r.step, y: r[key]}})).filter(p => p.y !== undefined && p.y !== null);
    if (!pts.length) continue;
    datasets.push({{
      label: stage,
      data: pts,
      borderColor: COLORS[stage] || "#c9d1d9",
      backgroundColor: (COLORS[stage] || "#c9d1d9") + "22",
      tension: 0.1, pointRadius: 0, borderWidth: 2,
    }});
  }}
  new Chart(document.getElementById(id), {{
    type: 'line', data: {{datasets}},
    options: {{
      plugins: {{
        title: {{display:true, text:title, color:'#c9d1d9', font:{{size:14}} }},
        legend: {{labels: {{color:'#c9d1d9'}} }}
      }},
      scales: {{
        x: {{type:'linear', title:{{display:true, text:'step (within stage)', color:'#8b949e'}},
             ticks:{{color:'#8b949e'}}, grid:{{color:'#30363d'}} }},
        y: {{type: logy ? 'logarithmic' : 'linear',
             ticks:{{color:'#8b949e'}}, grid:{{color:'#30363d'}} }}
      }}
    }}
  }});
}}

makeChart("ch_loss","Total loss","total",true);
makeChart("ch_policy","Policy loss","policy",true);
makeChart("ch_value","Value loss","value",true);
makeChart("ch_entropy","Policy entropy (nats)","entropy",true);
makeChart("ch_corr","Value correlation","corr",false);
makeChart("ch_grad","Gradient norm","grad_norm",true);
</script></body></html>"""
    with open(OUT, "w") as f:
        f.write(html)


if __name__ == "__main__":
    while True:
        try:
            write_html()
        except Exception as e:
            print(f"[plotter error] {e}", flush=True)
        time.sleep(20)
