"""KeywordInterpreter: the baseline.

Intentionally simple, transparent rules. This is what the LLM has to beat.
Its weaknesses are PART OF THE POINT: it cannot handle negation, implicit
constraints ('celiac' -> gluten_free), or conflicting goals. Documenting
those failures is half your evaluation story.
"""
import re
from schema_utils import coerce_to_schema, empty_output

# ---------------------------------------------------------------------------
# Rule tables. Order matters only for tie-breaking when multiple rules fire.
# We bias toward HIGH-PRECISION rules: words that clearly map one direction.
# ---------------------------------------------------------------------------
MEAL_OCCASION_RULES = {
    "breakfast": ["breakfast", "morning"],
    "lunch":     ["lunch"],
    "dinner":    ["dinner", "tonight", "evening meal"],
    "snack":     ["snack", "snacks"],
}

MISSION_RULES = {
    "party_hosting":     ["party", "hosting", "guests", "dinner party",
                          "birthday", "celebration", "date night"],
    "kids_meal":         ["kid", "kids", "children", "child", "daughter",
                          "son", "toddler"],
    "health_goal":       ["healthy", "diet", "lose weight", "high-protein",
                          "high protein", "low sugar", "low-sugar",
                          "low calorie", "low-calorie"],
    "convenience_meal":  ["quick", "fast", "easy", "no cook", "no-cook",
                          "ready meal", "ready-made"],
    "household_stock_up":["stock up", "weekly", "basics", "bulk", "for the week",
                          "for the month"],
}

DIETARY_RULES = {
    # NOTE: only EXPLICIT mentions. 'celiac', 'diabetic' are intentionally
    # absent: catching them requires inference, which is the LLM's job.
    "nut_allergy":  ["nut allergy", "allergic to nuts", "nut-free", "no nuts"],
    "lactose_free": ["lactose", "lactose-free", "lactose free"],
    "gluten_free":  ["gluten", "gluten-free", "gluten free"],
    "vegetarian":   ["vegetarian"],
    "vegan":        ["vegan"],
    "halal":        ["halal"],
}

PREFERENCE_RULES = {
    "quick":         ["quick", "fast", "in a hurry"],
    "healthy":       ["healthy", "nutritious"],
    "indulgent":     ["indulgent", "comfort food", "treat"],
    "filling":       ["filling", "hearty"],
    "light":         ["light"],
    "kid_friendly":  ["kid", "kids", "children", "child"],
    "easy_to_carry": ["portable", "on the go", "on-the-go", "at my desk"],
    "no_cooking":    ["no cook", "no-cook", "no cooking", "ready to eat"],
    "high_protein":  ["protein", "high-protein", "high protein"],
    "low_sugar":     ["low sugar", "low-sugar", "no sugar", "sugar-free"],
    "low_calorie":   ["low calorie", "low-calorie", "diet", "lose weight"],
}

BUDGET_RULES = {
    "low_budget": ["cheap", "tight budget", "budget", "affordable", "low-cost"],
    "premium":    ["premium", "high quality", "fancy", "luxury", "best"],
}

# Coarse aisle keywords (very rough — keyword baselines are not good at this)
CATEGORY_RULES = {
    "produce":        ["fruit", "vegetable", "veggies", "salad", "banana",
                       "apple", "orange", "tomato", "lettuce"],
    "meat_seafood":   ["chicken", "beef", "pork", "fish", "salmon", "meat",
                       "burger", "ground beef"],
    "dairy_eggs":     ["milk", "cheese", "yogurt", "egg", "butter", "cream"],
    "bakery":         ["bread", "toast", "baguette", "croissant", "cake",
                       "pastry"],
    "pantry_staples": ["pasta", "rice", "sauce", "olive oil", "salt", "flour",
                       "canned"],
    "breakfast":      ["cereal", "coffee", "tea", "oats", "granola"],
    "snacks_sweets":  ["chocolate", "chips", "cookies", "candy", "crackers",
                       "biscuit"],
    "frozen":         ["frozen", "ice cream"],
    "ready_meals":    ["ready meal", "ready-made", "microwave meal", "tv dinner"],
    "beverages":      ["water", "juice", "soda", "drink", "smoothie"],
    "alcohol":        ["wine", "beer", "champagne", "vodka", "whiskey", "gin"],
    "health_diet":    ["organic", "protein bar", "supplement"],
    "baby":           ["baby", "diaper", "formula"],
    "household_nonfood": ["soap", "detergent", "shampoo", "laundry", "cleaner"],
}

NEGATION_TRIGGERS = ["not ", "no ", "without", "can't", "cant", "don't",
                     "dont", "avoid", "free", "skip"]


def _match_any(text, patterns):
    return any(p in text for p in patterns)


def _all_matching_singles(text, table):
    """Return the first matching key, or None. For SINGLE-label fields."""
    for label, kws in table.items():
        if _match_any(text, kws):
            return label
    return None


def _all_matching_lists(text, table):
    """Return every matching key. For LIST fields."""
    return [label for label, kws in table.items() if _match_any(text, kws)]


def interpret_keyword(request: str) -> dict:
    text = " " + request.lower().strip() + " "
    out = empty_output()

    # singles
    occ = _all_matching_singles(text, MEAL_OCCASION_RULES)
    if occ: out["meal_occasion"] = occ
    mis = _all_matching_singles(text, MISSION_RULES)
    if mis: out["mission_type"] = mis
    bud = _all_matching_singles(text, BUDGET_RULES)
    if bud: out["budget_sensitivity"] = bud

    # lists
    out["dietary_constraints"]    = _all_matching_lists(text, DIETARY_RULES)
    out["preference_constraints"] = _all_matching_lists(text, PREFERENCE_RULES)
    out["product_categories"]     = _all_matching_lists(text, CATEGORY_RULES)

    # very crude negation detection
    has_negation = _match_any(text, NEGATION_TRIGGERS)
    if has_negation:
        out["risk_flags"].append("negation_present")
        # naive excluded_items: word right after 'not'/'no'/'without'
        for trig in ("not ", "no ", "without "):
            for m in re.finditer(re.escape(trig) + r"([a-z]+)", text):
                tok = m.group(1)
                if tok not in ("a", "an", "the"):  # stopword-ish
                    out["excluded_items"].append(tok)
        out["excluded_items"] = list(dict.fromkeys(out["excluded_items"]))

    # ambiguity flag if essentially nothing matched
    nothing_matched = (
        out["meal_occasion"] == "none"
        and out["mission_type"] == "unclear"
        and not out["dietary_constraints"]
        and not out["preference_constraints"]
        and not out["product_categories"]
    )
    if nothing_matched:
        out["risk_flags"].append("ambiguous_request")
        out["risk_flags"].append("missing_information")

    # safety flag if a dietary constraint was found
    if out["dietary_constraints"]:
        # baseline cannot infer 'celiac'->gluten_free, but DOES flag explicit ones
        out["risk_flags"].append("allergy_or_safety")

    # de-dupe flags while preserving order
    out["risk_flags"] = list(dict.fromkeys(out["risk_flags"]))

    out["mission_summary"] = ""  # baseline doesn't produce one
    out["confidence"] = 0.5 if not nothing_matched else 0.1
    out["explanation"] = "keyword-rule baseline"

    return coerce_to_schema(out)


if __name__ == "__main__":
    # quick smoke test
    import json
    for r in [
        "Milk, bread and eggs please",
        "Snacks for the kids, but one of them is allergic to nuts",
        "Birthday cake but my daughter is celiac",   # baseline should MISS gluten_free
        "Get the usual",
    ]:
        print("REQUEST:", r)
        print(json.dumps(interpret_keyword(r), indent=2, ensure_ascii=False))
        print("-" * 60)
