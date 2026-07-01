import json, os, time
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
"""LLMInterpreter: gpt-4o-mini behind the same interface as the baseline.

Cost-control measures (important for the 2.50 EUR budget):
  - JSON mode forces well-formed JSON, no rambling explanations
  - max_tokens cap prevents runaway outputs
  - temperature=0 for reproducibility (your eval numbers shouldn't drift
    between runs; that would make the keyword-vs-LLM comparison noisy)
  - the schema is injected from schema.json so the prompt always matches
    what the harness scores against
"""
import json, os, time
from schema_utils import SCHEMA, coerce_to_schema, empty_output, allowed

MODEL_NAME = "gpt-4o-mini"
MAX_OUTPUT_TOKENS = 400   # caps cost per call; JSON output is ~200 tokens
TEMPERATURE       = 0.0
TIMEOUT_S         = 30

SYSTEM_PROMPT = """You are a Shopping Mission Interpreter for a grocery retailer.

You read a free-text shopper request and output a STRUCTURED JSON OBJECT
describing what they want. You MUST follow the schema below exactly.

==== SCHEMA ====
Closed single-label fields (pick ONE allowed value):
  meal_occasion       : {meal_occasion}
  mission_type        : {mission_type}
  budget_sensitivity  : {budget_sensitivity}

Closed list fields (pick ZERO OR MORE allowed values):
  dietary_constraints : {dietary_constraints}   # HARD constraints only:
                                                  allergies + diet identity
  preference_constraints: {preference_constraints}
  product_categories  : {product_categories}    # coarse grocery aisles
  risk_flags          : {risk_flags}

Open list field (free strings, lowercase, the literal item being negated):
  excluded_items      : e.g. ["eggs"], ["alcohol"], ["soy"]

Model-output fields (your own description, not from a list):
  mission_summary     : ONE short sentence in your own words, capturing the
                        specific mission. e.g. "weeknight dinner that's
                        quick, plant-based, and nut-safe for a child"
  confidence          : float 0.0-1.0  -- be honest, lower it when ambiguous
  explanation         : ONE short sentence pointing at the wording in the
                        request that drove the main calls

==== RULES ====
1. dietary_constraints is for SAFETY-CRITICAL / HARD constraints only.
   Allergies (nut_allergy), intolerances (lactose_free, gluten_free),
   and identity diets (vegetarian, vegan, halal). Health goals like
   low_sugar or high_protein go in preference_constraints, NOT here.
2. Infer implicit constraints:
     "celiac" / "coeliac"      -> gluten_free
     "diabetic"                -> low_sugar  (preference) + allergy_or_safety
     "lactose intolerant"      -> lactose_free
     "no meat"                 -> vegetarian
3. Handle NEGATION carefully:
     "but not eggs"            -> excluded_items: ["eggs"], flag negation_present
     "no alcohol"              -> excluded_items: ["alcohol"], REMOVE the
                                  alcohol aisle from product_categories
4. Use risk_flags to mark linguistically tricky inputs:
     allergy_or_safety   -> any allergy, intolerance, or medical condition
     negation_present    -> explicit "no", "not", "without", "avoid"
     conflicting_goals   -> e.g. "healthy but indulgent", "cheap but premium"
     ambiguous_request   -> vague or under-specified ("the usual")
     missing_information -> not enough info to act on; ask user
5. When in doubt for closed fields, use the neutral value:
     meal_occasion -> "none"
     mission_type  -> "unclear"
     budget_sensitivity -> "not_mentioned"
6. Output VALID JSON ONLY. No prose before or after. No markdown fences.
   No fields outside the schema.
"""

def build_system_prompt():
    return SYSTEM_PROMPT.format(
        meal_occasion           = allowed("meal_occasion"),
        mission_type            = allowed("mission_type"),
        budget_sensitivity      = allowed("budget_sensitivity"),
        dietary_constraints     = allowed("dietary_constraints"),
        preference_constraints  = allowed("preference_constraints"),
        product_categories      = allowed("product_categories"),
        risk_flags              = allowed("risk_flags"),
    )

# Two few-shot examples chosen to teach the trickiest behaviors:
# (a) implicit constraint inference, (b) negation that REMOVES a category.
FEWSHOT = [
    {
        "request": "Birthday cake but my daughter is celiac",
        "output": {
            "meal_occasion": "none",
            "mission_type": "party_hosting",
            "budget_sensitivity": "not_mentioned",
            "dietary_constraints": ["gluten_free"],
            "preference_constraints": [],
            "excluded_items": [],
            "product_categories": ["bakery"],
            "risk_flags": ["allergy_or_safety"],
            "mission_summary": "a celiac-safe birthday cake for the daughter",
            "confidence": 0.9,
            "explanation": "'celiac' -> gluten_free; birthday + cake -> party_hosting + bakery",
        },
    },
    {
        "request": "Party drinks but nothing alcoholic, kids will be there",
        "output": {
            "meal_occasion": "none",
            "mission_type": "party_hosting",
            "budget_sensitivity": "not_mentioned",
            "dietary_constraints": [],
            "preference_constraints": ["kid_friendly"],
            "excluded_items": ["alcohol"],
            "product_categories": ["beverages"],
            "risk_flags": ["negation_present"],
            "mission_summary": "non-alcoholic party drinks suitable for children present",
            "confidence": 0.9,
            "explanation": "'nothing alcoholic' negates the alcohol aisle; 'kids' -> kid_friendly",
        },
    },
]


def build_messages(user_request: str):
    msgs = [{"role": "system", "content": build_system_prompt()}]
    for ex in FEWSHOT:
        msgs.append({"role": "user", "content": ex["request"]})
        msgs.append({"role": "assistant",
                     "content": json.dumps(ex["output"], ensure_ascii=False)})
    msgs.append({"role": "user", "content": user_request})
    return msgs


def interpret_llm(request: str, client=None) -> dict:
    """Call gpt-4o-mini and return a schema-coerced dict.

    Pass a pre-built OpenAI client to reuse the connection across calls
    (cheaper, faster). If client is None, one is created lazily.
    """
    if client is None:
        from openai import OpenAI
        client = OpenAI()  # reads OPENAI_API_KEY from env

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=build_messages(request),
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
            response_format={"type": "json_object"},
            timeout=TIMEOUT_S,
        )
        raw_text = resp.choices[0].message.content
        raw = json.loads(raw_text)
    except json.JSONDecodeError:
        # Model returned non-JSON despite json_object mode (rare). Fail safe.
        out = empty_output()
        out["explanation"] = "LLM returned invalid JSON; using empty fallback"
        out["confidence"] = 0.0
        return out
    except Exception as e:
        out = empty_output()
        out["explanation"] = f"LLM call failed: {type(e).__name__}: {e}"
        out["confidence"] = 0.0
        return out

    return coerce_to_schema(raw)


# ---------------------------------------------------------------------------
# Lightweight offline self-check (no API key needed).
# Verifies that the prompt builds, schema injection works, and the messages
# are well-formed. Use `python llm_interpreter.py --live` to hit the API.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if "--live" in sys.argv:
        if not os.environ.get("OPENAI_API_KEY"):
            print("ERROR: set OPENAI_API_KEY to run live tests."); sys.exit(1)
        for r in [
            "Milk, bread and eggs please",
            "Snacks for the kids, but one of them is allergic to nuts",
            "Birthday cake but my daughter is celiac",
            "Get the usual",
        ]:
            print("REQUEST:", r)
            print(json.dumps(interpret_llm(r), indent=2, ensure_ascii=False))
            print("-" * 60)
    else:
        msgs = build_messages("Quick dinner, nothing with dairy though")
        print(f"System prompt: {len(msgs[0]['content'])} chars")
        print(f"Total messages built: {len(msgs)} (1 system + {len(FEWSHOT)} few-shot pairs + 1 user)")
        print("\nFinal user message:", msgs[-1]["content"])
        print("\nFirst few-shot output (verifies schema-shaped examples):")
        print(msgs[2]["content"][:200], "...")
