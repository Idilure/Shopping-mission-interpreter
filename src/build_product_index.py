"""build_product_index.py

One-time setup script.  Run once from the project root:
    python src/build_product_index.py

Reads:
    data/products.csv   (product_id, product_name, aisle_id, department_id)
    data/aisles.csv     (aisle_id, aisle)

Writes:
    data/product_index.json   {carrefour_aisle: [product_name, ...]}

The mapping from Instacart aisle names -> our 13 Carrefour coarse aisles
is defined here as a translation table.  Each Instacart aisle maps to
exactly one Carrefour aisle; unmapped aisles go to household_nonfood.

We keep up to MAX_PER_AISLE products per Carrefour aisle, sampled to
cover a variety of Instacart sub-aisles so the product list feels diverse.
"""
import csv, json, os, random
from collections import defaultdict

random.seed(42)
MAX_PER_AISLE = 120   # enough variety for the recommender without being huge

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")
DATA_DIR = os.path.join(ROOT_DIR, "data")

# ── Instacart aisle name → Carrefour coarse aisle ─────────────────────────────
# Key: substring match (lowercased) against the Instacart aisle name.
# Checked in order — first match wins.
AISLE_RULES = [
    # produce
    ("fresh vegetables",            "produce"),
    ("fresh fruits",                "produce"),
    ("fresh herbs",                 "produce"),
    ("packaged produce",            "produce"),
    ("salad dressing",              "produce"),    # often bought with salads
    ("prepared soups salads",       "produce"),

    # meat & seafood
    ("fresh beef",                  "meat_seafood"),
    ("fresh pork",                  "meat_seafood"),
    ("poultry counter",             "meat_seafood"),
    ("meat counter",                "meat_seafood"),
    ("beef pork",                   "meat_seafood"),
    ("fish seafood",                "meat_seafood"),
    ("packaged seafood",            "meat_seafood"),
    ("deli meats",                  "meat_seafood"),
    ("lunch meat",                  "meat_seafood"),
    ("frozen meat seafood",         "meat_seafood"),
    ("marinades meat",              "meat_seafood"),

    # dairy & eggs
    ("eggs",                        "dairy_eggs"),
    ("yogurt",                      "dairy_eggs"),
    ("butter",                      "dairy_eggs"),
    ("milk",                        "dairy_eggs"),
    ("cream",                       "dairy_eggs"),
    ("specialty cheeses",           "dairy_eggs"),
    ("packaged cheese",             "dairy_eggs"),
    ("other creams cheeses",        "dairy_eggs"),
    ("soy lactosefree",             "dairy_eggs"),

    # bakery
    ("bread",                       "bakery"),
    ("bakery",                      "bakery"),
    ("tortillas flatbreads",        "bakery"),
    ("frozen breads doughs",        "bakery"),
    ("frozen breakfast",            "bakery"),

    # pantry staples
    ("pasta sauce",                 "pantry_staples"),
    ("dry pasta",                   "pantry_staples"),
    ("grains rice",                 "pantry_staples"),
    ("canned meals beans",          "pantry_staples"),
    ("canned jarred vegetables",    "pantry_staples"),
    ("canned fruit",                "pantry_staples"),
    ("condiments",                  "pantry_staples"),
    ("oils vinegars",               "pantry_staples"),
    ("spices seasonings",           "pantry_staples"),
    ("soups broths",                "pantry_staples"),
    ("soup broth",                  "pantry_staples"),
    ("baking ingredients",          "pantry_staples"),
    ("baking supplies",             "pantry_staples"),
    ("sugar sweeteners",            "pantry_staples"),
    ("spreads",                     "pantry_staples"),
    ("pickled goods olives",        "pantry_staples"),

    # breakfast
    ("cereal",                      "breakfast"),
    ("hot cereals oatmeal",         "breakfast"),
    ("breakfast bars pastries",     "breakfast"),
    ("coffee",                      "breakfast"),
    ("tea",                         "breakfast"),
    ("energy granola bars",         "breakfast"),

    # snacks & sweets
    ("chips pretzels",              "snacks_sweets"),
    ("popcorn jerky",               "snacks_sweets"),
    ("crackers",                    "snacks_sweets"),
    ("cookies cakes",               "snacks_sweets"),
    ("candy chocolate",             "snacks_sweets"),
    ("nuts seeds dried fruit",      "snacks_sweets"),

    # frozen
    ("frozen produce",              "frozen"),
    ("frozen meals",                "frozen"),
    ("frozen pizza",                "frozen"),
    ("ice cream ice",               "frozen"),

    # ready meals
    ("instant foods",               "ready_meals"),
    ("prepared meals",              "ready_meals"),
    ("kosher foods",                "ready_meals"),

    # beverages
    ("juice nectars",               "beverages"),
    ("water seltzer",               "beverages"),
    ("soft drinks",                 "beverages"),
    ("energy sports drinks",        "beverages"),
    ("kombucha",                    "beverages"),

    # alcohol
    ("beer coolers",                "alcohol"),
    ("wines champagnes",            "alcohol"),
    ("spirits hard alcohol",        "alcohol"),

    # health & diet
    ("vitamins supplements",        "health_diet"),
    ("tofu meat alternatives",      "health_diet"),

    # baby
    ("baby food formula",           "baby"),
    ("diapers wipes",               "baby"),
]

def map_aisle(instacart_aisle_name: str) -> str:
    name = instacart_aisle_name.lower().strip()
    for pattern, carrefour in AISLE_RULES:
        if pattern in name:
            return carrefour
    return "household_nonfood"


# ── load & join ───────────────────────────────────────────────────────────────
print("Loading aisles.csv …")
aisle_lookup = {}
with open(os.path.join(DATA_DIR, "aisles.csv"), encoding="utf-8") as f:
    for row in csv.DictReader(f):
        aisle_lookup[row["aisle_id"]] = row["aisle"]

print("Loading products.csv …")
# Group products by carrefour aisle, keeping sub-aisle variety
# raw[carrefour_aisle][instacart_aisle] = [product_name, ...]
raw = defaultdict(lambda: defaultdict(list))

with open(os.path.join(DATA_DIR, "products.csv"), encoding="utf-8") as f:
    for row in csv.DictReader(f):
        instacart_aisle = aisle_lookup.get(row["aisle_id"], "other")
        carrefour_aisle = map_aisle(instacart_aisle)
        raw[carrefour_aisle][instacart_aisle].append(row["product_name"])

# ── sample for diversity ──────────────────────────────────────────────────────
index = {}
total = 0
for c_aisle, sub_aisles in raw.items():
    all_products = []
    n_subs = len(sub_aisles)
    per_sub = max(3, MAX_PER_AISLE // n_subs)
    for sub_name, products in sub_aisles.items():
        random.shuffle(products)
        all_products.extend(products[:per_sub])
    random.shuffle(all_products)
    index[c_aisle] = all_products[:MAX_PER_AISLE]
    total += len(index[c_aisle])
    print(f"  {c_aisle:<22} {len(index[c_aisle]):>4} products "
          f"(from {n_subs} Instacart aisles)")

out_path = os.path.join(DATA_DIR, "product_index.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(index, f, indent=2, ensure_ascii=False)

print(f"\n✓ product_index.json written — {total} products across "
      f"{len(index)} aisles")
print(f"  Path: {out_path}")
