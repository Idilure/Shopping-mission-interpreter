"""generate_report.py

Reads results/scores.json and produces results/report.html —
a clean, styled HTML report with all the tables you need for the writeup.

Run from src/:
    python generate_report.py
Then open results/report.html in any browser.
To get a PDF: File -> Print -> Save as PDF (no margins, landscape).
"""
import json, os
from collections import defaultdict

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")
RES_DIR  = os.path.join(ROOT_DIR, "results")

with open(os.path.join(RES_DIR, "scores.json"), encoding="utf-8") as f:
    records = json.load(f)

SYSTEMS = list(dict.fromkeys(r["system"] for r in records))
TIERS   = ["all", "easy", "medium", "hard"]
TIER_N  = {t: len(set(r["id"] for r in records if r["tier"] == t or t == "all"))
           for t in TIERS}

SINGLE_FIELDS = ["meal_occasion", "mission_type", "budget_sensitivity"]
LIST_FIELDS   = ["dietary_constraints", "preference_constraints",
                 "product_categories", "risk_flags", "excluded_items"]

def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2*p*r / (p+r) if (p+r) > 0 else 0.0
    return p, r, f

# ── aggregate ─────────────────────────────────────────────────────────────────
def fresh():
    return {
        t: {
            **{f: {"correct": 0, "total": 0} for f in SINGLE_FIELDS},
            **{f: {"tp": 0, "fp": 0, "fn": 0, "exact": 0, "total": 0}
               for f in LIST_FIELDS},
        }
        for t in TIERS
    }

agg = {s: fresh() for s in SYSTEMS}

for rec in records:
    s  = rec["system"]
    t  = rec["tier"]
    sc = rec["scores"]
    for bucket in [t, "all"]:
        for f in SINGLE_FIELDS:
            agg[s][bucket][f]["correct"] += sc[f]
            agg[s][bucket][f]["total"]   += 1
        for f in LIST_FIELDS:
            agg[s][bucket][f]["tp"]    += sc[f]["tp"]
            agg[s][bucket][f]["fp"]    += sc[f]["fp"]
            agg[s][bucket][f]["fn"]    += sc[f]["fn"]
            agg[s][bucket][f]["exact"] += sc[f]["exact"]
            agg[s][bucket][f]["total"] += 1

def acc(s, t, f):
    d = agg[s][t][f]
    return d["correct"] / d["total"] if d["total"] else 0.0

def f1(s, t, f):
    d = agg[s][t][f]
    _, _, v = prf(d["tp"], d["fp"], d["fn"])
    return v

def color(val, lo=0.5, hi=0.85):
    if val >= hi:   return "#d4edda"   # green
    if val >= lo:   return "#fff3cd"   # amber
    return "#f8d7da"                   # red

def delta_color(d):
    if d >  0.05: return "#28a745"
    if d < -0.05: return "#dc3545"
    return "#6c757d"

def fmt(v): return f"{v:.3f}"

# ── failures ─────────────────────────────────────────────────────────────────
failures = {}
for rec in records:
    sc = rec["scores"]
    bad = []
    for f in SINGLE_FIELDS:
        if sc[f] == 0: bad.append(f)
    for f in LIST_FIELDS:
        if sc[f]["exact"] == 0: bad.append(f)
    if bad:
        key = (rec["system"], rec["id"])
        failures[key] = {"request": rec["request"], "tier": rec["tier"],
                         "failed": bad, "pred": rec["pred"], "gold": rec["gold"]}

# ── HTML ─────────────────────────────────────────────────────────────────────
def th(txt, extra=""):
    return f'<th {extra}>{txt}</th>'

def td(val, bg="", extra=""):
    style = f'background:{bg};' if bg else ""
    return f'<td style="{style}{extra}">{val}</td>'

rows_html = []

# ── Table 1: Per-system per-tier ──────────────────────────────────────────────
for sys in SYSTEMS:
    label = "Keyword Baseline" if sys == "keyword" else "LLM (gpt-4o-mini)"
    rows_html.append(f"""
    <h2 style="margin-top:2rem">System: {label}</h2>
    <table>
      <thead><tr>
        <th>Field</th><th>Metric</th>
        <th>ALL ({TIER_N['all']})</th>
        <th>EASY ({TIER_N['easy']})</th>
        <th>MEDIUM ({TIER_N['medium']})</th>
        <th>HARD ({TIER_N['hard']})</th>
      </tr></thead><tbody>""")

    for f in SINGLE_FIELDS:
        vals = {t: acc(sys, t, f) for t in TIERS}
        rows_html.append(f"<tr><td><b>{f}</b></td><td>Accuracy</td>"
            + "".join(td(fmt(vals[t]), color(vals[t])) for t in TIERS)
            + "</tr>")

    rows_html.append('<tr><td colspan="6" style="background:#f0f0f0;font-size:0.8rem;padding:4px 8px">List fields — Micro F1</td></tr>')

    for f in LIST_FIELDS:
        vals = {t: f1(sys, t, f) for t in TIERS}
        rows_html.append(f"<tr><td><b>{f}</b></td><td>F1</td>"
            + "".join(td(fmt(vals[t]), color(vals[t])) for t in TIERS)
            + "</tr>")

    rows_html.append("</tbody></table>")

# ── Table 2: Side-by-side comparison ─────────────────────────────────────────
if len(SYSTEMS) == 2:
    kw, lm = SYSTEMS[0], SYSTEMS[1]
    rows_html.append("""
    <h2 style="margin-top:2rem">Keyword vs LLM — Side by Side (All Tiers)</h2>
    <table>
      <thead><tr>
        <th>Field</th><th>Metric</th>
        <th>Keyword</th><th>LLM</th><th>Delta (LLM − KW)</th>
      </tr></thead><tbody>""")

    for f in SINGLE_FIELDS:
        kv = acc(kw, "all", f)
        lv = acc(lm, "all", f)
        d  = lv - kv
        sign = "+" if d >= 0 else ""
        rows_html.append(
            f"<tr><td><b>{f}</b></td><td>Accuracy</td>"
            + td(fmt(kv), color(kv))
            + td(fmt(lv), color(lv))
            + f'<td style="color:{delta_color(d)};font-weight:bold">{sign}{fmt(d)}</td>'
            + "</tr>")

    rows_html.append('<tr><td colspan="5" style="background:#f0f0f0;font-size:0.8rem;padding:4px 8px">List fields — Micro F1</td></tr>')

    for f in LIST_FIELDS:
        kv = f1(kw, "all", f)
        lv = f1(lm, "all", f)
        d  = lv - kv
        sign = "+" if d >= 0 else ""
        rows_html.append(
            f"<tr><td><b>{f}</b></td><td>F1</td>"
            + td(fmt(kv), color(kv))
            + td(fmt(lv), color(lv))
            + f'<td style="color:{delta_color(d)};font-weight:bold">{sign}{fmt(d)}</td>'
            + "</tr>")

    rows_html.append("</tbody></table>")

# ── Table 3: Failure analysis ─────────────────────────────────────────────────
for sys in SYSTEMS:
    label = "Keyword Baseline" if sys == "keyword" else "LLM (gpt-4o-mini)"
    sys_fails = {k: v for k, v in failures.items() if k[0] == sys}
    tier_counts = defaultdict(int)
    for v in sys_fails.values(): tier_counts[v["tier"]] += 1

    rows_html.append(f"""
    <h2 style="margin-top:2rem">Failure Analysis — {label}</h2>
    <p style="color:#555">
      {len(sys_fails)} / 50 examples had at least one incorrect field &nbsp;|&nbsp;
      Easy: {tier_counts['easy']} &nbsp; Medium: {tier_counts['medium']} &nbsp; Hard: {tier_counts['hard']}
    </p>
    <table>
      <thead><tr>
        <th style="width:5%">ID</th>
        <th style="width:8%">Tier</th>
        <th style="width:37%">Request</th>
        <th style="width:25%">Failed Fields</th>
        <th style="width:25%">Key difference</th>
      </tr></thead><tbody>""")

    tier_order = {"easy": 0, "medium": 1, "hard": 2}
    for (s, eid), info in sorted(sys_fails.items(),
                                  key=lambda x: (tier_order[x[1]["tier"]], x[0][1])):
        diff_parts = []
        for f in info["failed"]:
            g = info["gold"].get(f, "")
            p = info["pred"].get(f, "")
            if isinstance(g, list):
                g_s = ", ".join(g) if g else "∅"
                p_s = ", ".join(p) if p else "∅"
            else:
                g_s, p_s = str(g), str(p)
            diff_parts.append(f"<b>{f}:</b> gold=<i>{g_s}</i> pred=<i>{p_s}</i>")

        tier_bg = {"easy": "#d4edda", "medium": "#fff3cd", "hard": "#f8d7da"}
        rows_html.append(
            f'<tr>'
            f'<td>{eid}</td>'
            f'<td style="background:{tier_bg[info["tier"]]}">{info["tier"]}</td>'
            f'<td>{info["request"]}</td>'
            f'<td style="font-size:0.82rem">{", ".join(info["failed"])}</td>'
            f'<td style="font-size:0.78rem">{"<br>".join(diff_parts)}</td>'
            f'</tr>')

    rows_html.append("</tbody></table>")

# ── assemble ──────────────────────────────────────────────────────────────────
HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Shopping Mission Interpreter — Evaluation Report</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 14px; color: #212529;
    max-width: 1100px; margin: 2rem auto; padding: 0 1.5rem;
  }}
  h1 {{ font-size: 1.5rem; border-bottom: 2px solid #dee2e6; padding-bottom: .5rem; }}
  h2 {{ font-size: 1.1rem; color: #495057; margin-bottom: .5rem; }}
  table {{
    width: 100%; border-collapse: collapse;
    margin-bottom: 1.5rem; font-size: 13px;
  }}
  th {{
    background: #343a40; color: #fff;
    padding: 8px 12px; text-align: left;
    font-weight: 600; font-size: 12px;
  }}
  td {{ padding: 6px 12px; border-bottom: 1px solid #dee2e6; }}
  tr:hover td {{ background: #f8f9fa !important; }}
  .legend {{ display:flex; gap:1.5rem; font-size:12px; margin-bottom:1rem; }}
  .dot {{ width:14px; height:14px; border-radius:3px; display:inline-block;
          margin-right:4px; vertical-align:middle; }}
  @media print {{
    body {{ max-width: 100%; margin: 0; padding: 0; }}
    h2 {{ page-break-before: auto; }}
  }}
</style>
</head>
<body>
<h1>Shopping Mission Interpreter — Evaluation Report</h1>
<p style="color:#6c757d; font-size:13px">
  50 test examples &nbsp;·&nbsp; 3 tiers (Easy 16 / Medium 18 / Hard 16)
  &nbsp;·&nbsp; Systems: {', '.join(SYSTEMS)}
</p>
<div class="legend">
  <span><span class="dot" style="background:#d4edda"></span>≥ 0.85 (good)</span>
  <span><span class="dot" style="background:#fff3cd"></span>0.50 – 0.85 (moderate)</span>
  <span><span class="dot" style="background:#f8d7da"></span>&lt; 0.50 (weak)</span>
  <span style="color:#28a745;font-weight:bold">+0.05+ delta = LLM wins</span>
  <span style="color:#6c757d;font-weight:bold">~0 delta = tie</span>
</div>
{"".join(rows_html)}
</body>
</html>"""

out_path = os.path.join(RES_DIR, "report.html")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(HTML)

print(f"Report written to: {out_path}")
print("Open it in any browser. File → Print → Save as PDF for a clean PDF.")
