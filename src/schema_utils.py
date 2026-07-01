"""Shared schema utilities.

The schema (schema.json) is the single source of truth for what every
interpreter is allowed to emit. Both KeywordInterpreter and LLMInterpreter
import from here, so when the vocabulary changes we change one file.
"""
import json, os


SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "schema.json")
with open(SCHEMA_PATH, encoding="utf-8") as f:
    SCHEMA = json.load(f)

# Convenience accessors --------------------------------------------------------
def allowed(field):
    """Return the controlled vocabulary for a field (list of strings)."""
    return SCHEMA[field]["values"]

SINGLE_FIELDS = [k for k, v in SCHEMA.items()
                 if isinstance(v, dict) and v.get("type") == "single"]
LIST_FIELDS   = [k for k, v in SCHEMA.items()
                 if isinstance(v, dict) and v.get("type") == "list"]

# Fields whose vocabulary is CLOSED (we can validate values strictly)
CLOSED_LIST_FIELDS = [k for k in LIST_FIELDS if "values" in SCHEMA[k]]
# Fields whose vocabulary is OPEN (free strings, e.g. excluded_items)
OPEN_LIST_FIELDS   = [k for k in LIST_FIELDS if "values" not in SCHEMA[k]]


def empty_output():
    """A schema-shaped zero output, used as a fallback when parsing fails."""
    out = {}
    for f in SINGLE_FIELDS:
        # 'none'/'not_mentioned'/'unclear' are the natural neutral values
        vals = allowed(f)
        for neutral in ("none", "not_mentioned", "unclear"):
            if neutral in vals:
                out[f] = neutral
                break
        else:
            out[f] = vals[0]
    for f in LIST_FIELDS:
        out[f] = []
    out["mission_summary"] = ""
    out["confidence"] = 0.0
    out["explanation"] = ""
    return out


def coerce_to_schema(raw):
    """Force a possibly-noisy dict into the schema shape.

    - drops unknown keys
    - drops values outside the controlled vocabulary on closed fields
    - fills missing fields with neutral defaults
    - clamps confidence to [0, 1]
    Returns a dict that is safe to score and safe to display.
    """
    out = empty_output()
    if not isinstance(raw, dict):
        return out

    for f in SINGLE_FIELDS:
        v = raw.get(f)
        if isinstance(v, str) and v in allowed(f):
            out[f] = v

    for f in CLOSED_LIST_FIELDS:
        v = raw.get(f)
        if isinstance(v, list):
            out[f] = [x for x in v if isinstance(x, str) and x in allowed(f)]

    for f in OPEN_LIST_FIELDS:
        v = raw.get(f)
        if isinstance(v, list):
            out[f] = [x.strip().lower() for x in v if isinstance(x, str) and x.strip()]

    if isinstance(raw.get("mission_summary"), str):
        out["mission_summary"] = raw["mission_summary"].strip()
    if isinstance(raw.get("explanation"), str):
        out["explanation"] = raw["explanation"].strip()
    try:
        c = float(raw.get("confidence", 0.0))
        out["confidence"] = max(0.0, min(1.0, c))
    except (TypeError, ValueError):
        pass
    return out
