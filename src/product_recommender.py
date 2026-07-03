"""product_recommender.py

Second-stage LLM call: given the interpreter's structured output and a pool
of candidate products from the matched aisles, ask gpt-4o-mini to pick and
rank the best-fitting products with one-line reasons.

This is the semantic reasoning step: the LLM reads real product names and
understands that "Organic Steel Cut Oats" fits a diabetic low-sugar breakfast
mission better than "Honey Nut Cheerios" — without the word "oats" appearing
in the original request.

Usage:
    from product_recommender import recommend_products
    recs = recommend_products(interpreter_output, product_index)
    # returns [{"product": "...", "reason": "...", "aisle": "..."}, ...]
"""
import json, os, random
from schema_utils import coerce_to_schema

MODEL        = "gpt-4o-mini"
MAX_TOKENS   = 500
TEMPERATURE  = 0.2          # slight creativity ok for recommendations
MAX_CANDIDATES = 80         # products sent to LLM per request
MAX_RECS       = 8          # products returned to user
RANDOM_SEED    = None       # set to int for reproducible sampling

SYSTEM_PROMPT = """You are a smart grocery product recommender.

You receive:
1. A structured shopping mission (JSON)
2. A list of candidate products from the relevant aisles

Your job: select the {n} products that BEST match the shopper's mission,
constraints, and preferences. Think semantically:
  - "Organic" often signals health-conscious shoppers
  - "Family Size" / "Party Pack" signals party_hosting or household_stock_up
  - "Low Fat" / "Light" signals health_goal or low_calorie preference
  - "Quick Cook" / "Instant" signals convenience_meal or quick preference
  - Product names implying shared portion signal party/group missions
  - EXCLUDE any product that conflicts with dietary_constraints or excluded_items

Return ONLY a JSON object with this shape:
{{
  "recommendations": [
    {{
      "product"     : the exact product name from the list,
      "reason"      : one short sentence explaining why it fits this mission,
      "aisle"       : the aisle name it came from,
      "match_score" : integer 1-10 scoring how well this product fits
    }}
  ]
}}

Return exactly {n} recommendation objects. Score based on:
  - dietary_constraints satisfied (hard requirement, fails if not)
  - preference_constraints matched (e.g. low_sugar, high_protein)
  - mission_type alignment (e.g. health_goal, convenience_meal)
Be honest: score 9-10 only for near-perfect fits, 7-8 for good fits,
5-6 for partial, below 5 rarely.

No preamble, no markdown, no explanation outside the JSON object.
""".strip()

def _build_prompt(interp: dict, candidates: list[dict]) -> list[dict]:
    mission_str = json.dumps({
        "meal_occasion":          interp.get("meal_occasion"),
        "mission_type":           interp.get("mission_type"),
        "mission_summary":        interp.get("mission_summary"),
        "dietary_constraints":    interp.get("dietary_constraints"),
        "preference_constraints": interp.get("preference_constraints"),
        "budget_sensitivity":     interp.get("budget_sensitivity"),
        "excluded_items":         interp.get("excluded_items"),
    }, ensure_ascii=False)

    product_list = "\n".join(
        f'- "{c["product"]}" [{c["aisle"]}]' for c in candidates
    )

    user_content = (
        f"SHOPPING MISSION:\n{mission_str}\n\n"
        f"CANDIDATE PRODUCTS ({len(candidates)}):\n{product_list}"
    )

    return [
        {"role": "system",
         "content": SYSTEM_PROMPT.format(n=MAX_RECS)},
        {"role": "user", "content": user_content},
    ]


def recommend_products(
    interpreter_output: dict,
    product_index: dict,
    client=None,
) -> list[dict]:
    """
    interpreter_output : result from interpret_llm() or interpret_keyword()
    product_index      : the dict loaded from data/product_index.json
    client             : OpenAI client (created lazily if None)
    Returns a list of recommendation dicts, or [] on failure.
    """
    if client is None:
        from openai import OpenAI
        client = OpenAI()

    # ── gather candidates from matched aisles ─────────────────────────────────
    cats = interpreter_output.get("product_categories", [])
    excluded = [x.lower() for x in interpreter_output.get("excluded_items", [])]
    dietary  = interpreter_output.get("dietary_constraints", [])

    # pre-filter: remove obviously incompatible products
    def is_excluded(name: str) -> bool:
        n = name.lower()
        # explicit exclusions
        if any(ex in n for ex in excluded):
            return True
        # dietary safety filters (fast keyword check as a safety net —
        # the LLM does the deeper reasoning, this just removes the obvious)
        if "nut_allergy" in dietary and any(
                w in n for w in ["nut", "almond", "peanut", "cashew", "walnut"]):
            return True
        if "gluten_free" in dietary and any(
                w in n for w in ["wheat", "rye", "barley", "spelt"]):
            return True
        if "vegan" in dietary and any(
                w in n for w in ["beef", "chicken", "pork", "fish", "salmon",
                                  "turkey", "lamb", "tuna", "shrimp"]):
            return True
        if "lactose_free" in dietary and any(
                w in n for w in ["milk", "cheese", "butter", "cream", "yogurt",
                                  "whey"]):
            return True
        return False

    candidates = []
    for aisle in cats:
        products = product_index.get(aisle, [])
        for p in products:
            if not is_excluded(p):
                candidates.append({"product": p, "aisle": aisle})

    if not candidates:
        return []

    # sample down to MAX_CANDIDATES for the prompt
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
    if len(candidates) > MAX_CANDIDATES:
        # stratified: keep variety across aisles
        per_aisle = max(5, MAX_CANDIDATES // len(cats))
        sampled = []
        for aisle in cats:
            pool = [c for c in candidates if c["aisle"] == aisle]
            random.shuffle(pool)
            sampled.extend(pool[:per_aisle])
        candidates = sampled[:MAX_CANDIDATES]

    # ── call the LLM ──────────────────────────────────────────────────────────
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=_build_prompt(interpreter_output, candidates),
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
            timeout=30,
        )
        raw = json.loads(resp.choices[0].message.content)

        if not isinstance(raw, dict):
            return []
        recs = raw.get("recommendations", [])
        if not isinstance(recs, list):
            return []

        # validate shape
        out = []
        for r in recs:
            if isinstance(r, dict) and "product" in r and "reason" in r:
                try:
                    score = max(1, min(10, int(r.get("match_score", 7))))
                except (TypeError, ValueError):
                    score = 7
                out.append({
                    "product":     str(r["product"]),
                    "reason":      str(r.get("reason", "")),
                    "aisle":       str(r.get("aisle", "")),
                    "match_score": score,
                })
        return out[:MAX_RECS]

    except Exception as e:
        print(f"[product_recommender] LLM call failed: {e}")
        return []


# ── smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.join(SRC_DIR, "..")
    DATA_DIR = os.path.join(ROOT_DIR, "data")

    index_path = os.path.join(DATA_DIR, "product_index.json")
    if not os.path.exists(index_path):
        print("ERROR: run build_product_index.py first to create product_index.json")
        sys.exit(1)

    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT_DIR, ".env"))

    with open(index_path, encoding="utf-8") as f:
        idx = json.load(f)

    # use a hard test case: diabetic low-sugar breakfast
    test_interp = {
        "meal_occasion":          "breakfast",
        "mission_type":           "health_goal",
        "mission_summary":        "filling breakfast that won't spike blood sugar",
        "dietary_constraints":    [],
        "preference_constraints": ["low_sugar", "filling", "healthy"],
        "budget_sensitivity":     "not_mentioned",
        "excluded_items":         [],
        "product_categories":     ["breakfast", "dairy_eggs", "produce"],
        "risk_flags":             ["allergy_or_safety"],
        "confidence":             0.9,
        "explanation":            "diabetic -> low_sugar; breakfast -> meal_occasion",
    }

    print("Mission:", test_interp["mission_summary"])
    print("Calling recommender …\n")
    recs = recommend_products(test_interp, idx)
    for i, r in enumerate(recs, 1):
        score = r.get("match_score", "?")
        bar   = "█" * score + "░" * (10 - score) if isinstance(score, int) else ""
        print(f"{i}. {r['product']}  [{r['aisle']}]")
        print(f"   → {r['reason']}")
        print(f"   match: {bar} {score}/10")
    print(f"\n{len(recs)} recommendations returned.")
