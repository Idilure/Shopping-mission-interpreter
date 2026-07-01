"""evaluate.py

Runs KeywordInterpreter and LLMInterpreter over every example in
testset.json, scores each field, and produces:
  - results/scores.json          full per-example predictions + scores
  - results/report.txt           human-readable summary table
  - results/failures.json        examples where either system failed
                                 on at least one field (for failure analysis)

Metrics:
  Single-label fields -> Accuracy
  List fields         -> Micro Precision / Recall / F1
                         (treating each label as a separate binary decision)

Usage:
  # keyword only (free, no API needed):
  python evaluate.py --keyword-only

  # both interpreters (needs OPENAI_API_KEY in ../.env):
  python evaluate.py

  # quick smoke test on first 5 examples:
  python evaluate.py --n 5
"""
import json, os, sys, time, argparse
from collections import defaultdict

# ── path setup ────────────────────────────────────────────────────────────────
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")
DATA_DIR = os.path.join(ROOT_DIR, "data")
RES_DIR  = os.path.join(ROOT_DIR, "results")
os.makedirs(RES_DIR, exist_ok=True)

sys.path.insert(0, SRC_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT_DIR, ".env"))

from schema_utils import SINGLE_FIELDS, CLOSED_LIST_FIELDS, OPEN_LIST_FIELDS
from keyword_interpreter import interpret_keyword

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--keyword-only", action="store_true",
                    help="Skip the LLM; score keyword baseline only.")
parser.add_argument("--n", type=int, default=None,
                    help="Only evaluate the first N examples (smoke test).")
args = parser.parse_args()

# ── load test set ─────────────────────────────────────────────────────────────
with open(os.path.join(DATA_DIR, "testset.json"), encoding="utf-8") as f:
    TESTSET = json.load(f)

if args.n:
    TESTSET = TESTSET[:args.n]
    print(f"[smoke test] evaluating first {args.n} examples only")

# ── LLM client (lazy — only created if needed) ────────────────────────────────
_llm_client = None
def get_llm_client():
    global _llm_client
    if _llm_client is None:
        from openai import OpenAI
        _llm_client = OpenAI()
    return _llm_client

# ── scoring helpers ───────────────────────────────────────────────────────────
def score_single(pred_val, gold_val):
    """1 if exact match, 0 otherwise."""
    return int(pred_val == gold_val)


def score_list(pred_list, gold_list):
    """Micro P / R / F1 for a single example's list field."""
    pred = set(pred_list or [])
    gold = set(gold_list or [])
    if not gold and not pred:
        return {"tp": 0, "fp": 0, "fn": 0, "exact": 1}
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    exact = int(pred == gold)
    return {"tp": tp, "fp": fp, "fn": fn, "exact": exact}


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(p, 3), round(r, 3), round(f, 3)


def score_example(pred, gold):
    """Score a single prediction dict against a gold dict.
    Returns a flat dict of field-level scores."""
    scores = {}

    # single-label fields: accuracy (0 or 1)
    for f in SINGLE_FIELDS:
        scores[f] = score_single(pred.get(f), gold.get(f))

    # closed list fields: tp/fp/fn for micro F1
    for f in CLOSED_LIST_FIELDS:
        scores[f] = score_list(pred.get(f, []), gold.get(f, []))

    # excluded_items: open list, scored as set equality only
    # (free strings; we report exact-match rate, not F1)
    ei_pred = set(x.lower() for x in pred.get("excluded_items", []))
    ei_gold = set(x.lower() for x in gold.get("excluded_items", []))
    scores["excluded_items"] = {
        "tp": len(ei_pred & ei_gold),
        "fp": len(ei_pred - ei_gold),
        "fn": len(ei_gold - ei_pred),
        "exact": int(ei_pred == ei_gold),
    }

    return scores


# ── main evaluation loop ──────────────────────────────────────────────────────
SYSTEMS = ["keyword"]
if not args.keyword_only:
    SYSTEMS.append("llm")
    from llm_interpreter import interpret_llm

print(f"\nEvaluating {len(TESTSET)} examples on: {SYSTEMS}")
print("=" * 60)

records = []   # one record per example, per system

for i, item in enumerate(TESTSET):
    req   = item["request"]
    gold  = item["gold"]
    tier  = item["tier"]
    ex_id = item["id"]

    preds = {}
    preds["keyword"] = interpret_keyword(req)

    if "llm" in SYSTEMS:
        try:
            preds["llm"] = interpret_llm(req, client=get_llm_client())
        except Exception as e:
            print(f"  [WARN] LLM failed on {ex_id}: {e}")
            from schema_utils import empty_output
            preds["llm"] = empty_output()
        # small sleep to avoid rate-limit on burst of 50 calls
        time.sleep(0.3)

    for sys_name, pred in preds.items():
        scores = score_example(pred, gold)
        records.append({
            "id":      ex_id,
            "tier":    tier,
            "system":  sys_name,
            "request": req,
            "pred":    pred,
            "gold":    gold,
            "scores":  scores,
        })

    done = i + 1
    if done % 10 == 0 or done == len(TESTSET):
        print(f"  {done}/{len(TESTSET)} done")

# ── aggregate metrics ─────────────────────────────────────────────────────────
# Structure: agg[system][tier_or_'all'][field] -> accumulated counts
TIERS = ["easy", "medium", "hard", "all"]

def fresh_agg():
    return {
        t: {
            **{f: {"correct": 0, "total": 0} for f in SINGLE_FIELDS},
            **{f: {"tp": 0, "fp": 0, "fn": 0, "exact": 0, "total": 0}
               for f in CLOSED_LIST_FIELDS + ["excluded_items"]},
        }
        for t in TIERS
    }

agg = {s: fresh_agg() for s in SYSTEMS}

for rec in records:
    s = rec["system"]
    t = rec["tier"]
    sc = rec["scores"]

    for bucket in [t, "all"]:
        for f in SINGLE_FIELDS:
            agg[s][bucket][f]["correct"] += sc[f]
            agg[s][bucket][f]["total"]   += 1

        for f in CLOSED_LIST_FIELDS + ["excluded_items"]:
            agg[s][bucket][f]["tp"]    += sc[f]["tp"]
            agg[s][bucket][f]["fp"]    += sc[f]["fp"]
            agg[s][bucket][f]["fn"]    += sc[f]["fn"]
            agg[s][bucket][f]["exact"] += sc[f]["exact"]
            agg[s][bucket][f]["total"] += 1


# ── build report ──────────────────────────────────────────────────────────────
lines = []

def ln(s=""): lines.append(s)

ln("=" * 72)
ln("SHOPPING MISSION INTERPRETER — EVALUATION REPORT")
ln(f"Examples: {len(TESTSET)}  |  Systems: {SYSTEMS}")
ln("=" * 72)

# ── per-system, per-tier summary table ───────────────────────────────────────
for sys_name in SYSTEMS:
    ln(f"\n{'─'*72}")
    ln(f"  SYSTEM: {sys_name.upper()}")
    ln(f"{'─'*72}")

    # header
    col_w = 24
    tier_labels = ["ALL", "EASY", "MEDIUM", "HARD"]
    ln(f"  {'FIELD':<{col_w}}" + "".join(f"{t:>14}" for t in tier_labels))
    ln(f"  {'─'*col_w}" + "─" * (14 * len(tier_labels)))

    for f in SINGLE_FIELDS:
        row = f"  {f:<{col_w}}"
        for t in ["all", "easy", "medium", "hard"]:
            d = agg[sys_name][t][f]
            acc = d["correct"] / d["total"] if d["total"] else 0
            row += f"  acc={acc:.2f}({d['total']:2d})"
        ln(row)

    ln()
    for f in CLOSED_LIST_FIELDS + ["excluded_items"]:
        row = f"  {f:<{col_w}}"
        for t in ["all", "easy", "medium", "hard"]:
            d = agg[sys_name][t][f]
            _, _, f1 = prf(d["tp"], d["fp"], d["fn"])
            row += f"    f1={f1:.2f}({d['total']:2d})"
        ln(row)
    ln()

# ── side-by-side comparison (all tiers combined) ─────────────────────────────
if len(SYSTEMS) == 2:
    ln(f"\n{'═'*72}")
    ln("  KEYWORD vs LLM — SIDE BY SIDE (ALL TIERS)")
    ln(f"{'═'*72}")
    col_w = 26
    ln(f"  {'FIELD':<{col_w}}{'KEYWORD':>12}{'LLM':>12}{'DELTA':>10}")
    ln(f"  {'─'*col_w}{'─'*12}{'─'*12}{'─'*10}")

    for f in SINGLE_FIELDS:
        kw  = agg["keyword"]["all"][f]
        lm  = agg["llm"]["all"][f]
        k_acc = kw["correct"] / kw["total"] if kw["total"] else 0
        l_acc = lm["correct"] / lm["total"] if lm["total"] else 0
        delta = l_acc - k_acc
        sign  = "+" if delta >= 0 else ""
        ln(f"  {f:<{col_w}}{k_acc:>11.3f}{l_acc:>12.3f}  {sign}{delta:.3f}")

    ln()
    for f in CLOSED_LIST_FIELDS + ["excluded_items"]:
        kw  = agg["keyword"]["all"][f]
        lm  = agg["llm"]["all"][f]
        _, _, k_f1 = prf(kw["tp"], kw["fp"], kw["fn"])
        _, _, l_f1 = prf(lm["tp"], lm["fp"], lm["fn"])
        delta = l_f1 - k_f1
        sign  = "+" if delta >= 0 else ""
        ln(f"  {f:<{col_w}}{k_f1:>11.3f}{l_f1:>12.3f}  {sign}{delta:.3f}")
    ln()

# ── failure log ──────────────────────────────────────────────────────────────
failures = []
for rec in records:
    sc = rec["scores"]
    failed_fields = []
    for f in SINGLE_FIELDS:
        if sc[f] == 0:
            failed_fields.append(f)
    for f in CLOSED_LIST_FIELDS + ["excluded_items"]:
        if sc[f]["exact"] == 0:
            failed_fields.append(f)
    if failed_fields:
        failures.append({**rec, "failed_fields": failed_fields})

ln(f"\n{'─'*72}")
ln(f"  FAILURE SUMMARY")
ln(f"  Total failures (any field wrong): "
   f"{sum(1 for f in failures if f['system']=='keyword')} keyword  |  "
   f"{sum(1 for f in failures if f['system']=='llm')} LLM  "
   f"(out of {len(TESTSET)} examples each)")
ln()

# Show the most instructive failures: hard-tier, LLM still wrong
hard_llm_fails = [f for f in failures
                  if f["system"] == "llm" and f["tier"] == "hard"]
ln(f"  Hard-tier LLM failures ({len(hard_llm_fails)}):")
for rec in hard_llm_fails[:10]:   # cap at 10 in the report
    ln(f"    [{rec['id']}] {rec['request'][:60]}")
    ln(f"           failed: {rec['failed_fields']}")
ln()
ln("=" * 72)

report_text = "\n".join(lines)
print(report_text)

# ── write files ───────────────────────────────────────────────────────────────
with open(os.path.join(RES_DIR, "report.txt"), "w", encoding="utf-8") as f:
    f.write(report_text)

with open(os.path.join(RES_DIR, "scores.json"), "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)

with open(os.path.join(RES_DIR, "failures.json"), "w", encoding="utf-8") as f:
    json.dump(failures, f, indent=2, ensure_ascii=False)

print(f"\nFiles written to results/:")
print(f"  scores.json   — full per-example predictions + scores")
print(f"  report.txt    — the table above")
print(f"  failures.json — examples with at least one wrong field")
