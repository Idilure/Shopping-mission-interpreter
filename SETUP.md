# Setup & Replication Guide
## Shopping Mission Interpreter — NLP Group Project

This guide lets a teammate (or the grader) reproduce every result from scratch,
in the correct order. Follow the steps exactly.

---

## 0. Prerequisites

- Python 3.10 or higher
- An OpenAI API key (get one at platform.openai.com — gpt-4o-mini access required)
- Git
- The Instacart dataset files: `products.csv` and `aisles.csv` in `data/`
  (download from kaggle.com/datasets/psparks/instacart-market-basket-analysis,
  you only need those two files)

---

## 1. Clone the repository

```bash
git clone <your-github-repo-url>
cd Shopping-mission-interpreter
```

---

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Set up your API key

Create a file called `.env` in the project root (never commit this):

```
OPENAI_API_KEY=sk-...your-key-here...
```

Verify `.env` is in `.gitignore` before doing anything else.

---

## 4. Check your folder structure

Your project root should look exactly like this before running anything:

```
Shopping-mission-interpreter/
├── .env                         ← your API key (never commit)
├── .gitignore
├── README.md
├── SETUP.md                     ← this file
├── requirements.txt
│
├── src/
│   ├── schema_utils.py          ← shared vocabulary + validation
│   ├── keyword_interpreter.py   ← baseline interpreter
│   ├── llm_interpreter.py       ← LLM interpreter (gpt-4o-mini)
│   ├── evaluate.py              ← evaluation harness
│   ├── generate_report.py       ← HTML report generator
│   ├── build_product_index.py   ← one-time product index builder
│   ├── product_recommender.py   ← second-stage product recommender
│   └── app.py                   ← Streamlit app
│
├── data/
│   ├── schema.json              ← controlled vocabulary (gold standard)
│   ├── testset.json             ← 50 labelled evaluation examples
│   ├── testset.csv              ← same, human-readable
│   ├── category_mapping.json    ← Carrefour coarse aisles → subcategories
│   ├── products.csv             ← Instacart dataset (you download this)
│   ├── aisles.csv               ← Instacart dataset (you download this)
│   └── product_index.json       ← built by step 6 below
│
├── results/                     ← created automatically when you run evals
│   ├── scores.json
│   ├── report.txt
│   ├── report.html
│   ├── failures.json
│   └── user_feedback.csv
│
└── docs/
    └── figures/                 ← screenshots for the report
```

---

## 5. Verify the pipeline offline (no API calls, free)

Run these two checks first. They confirm the code loads and the schema
is wired correctly, without spending any API credit.

```bash
cd src
python keyword_interpreter.py
```

Expected: JSON output for 4 test requests. The celiac example should show
empty `dietary_constraints` — that's the baseline failing intentionally.

```bash
python llm_interpreter.py
```

Expected: "System prompt: ~3485 chars, Total messages built: 6"
No API call is made.

---

## 6. Build the product index (one-time, no API call)

```bash
cd ..                         # back to project root
python src/build_product_index.py
```

Expected: 14 aisles listed, ~1500 products total.
Writes `data/product_index.json`. Run once, never again.

---

## 7. Run a live LLM smoke test (~€0.001)

```bash
python src/llm_interpreter.py --live
```

Expected: JSON output for 4 requests. The celiac example SHOULD show
`"gluten_free"` in `dietary_constraints` and `"allergy_or_safety"` in
`risk_flags`. If it does, the pipeline is healthy.

---

## 8. Run the keyword-only evaluation (free)

```bash
python src/evaluate.py --keyword-only
```

Expected: report table with keyword baseline scores across all 50 examples.
Writes `results/scores.json`, `results/report.txt`, `results/failures.json`.

---

## 9. Run the full evaluation — keyword vs LLM (~€0.012)

```bash
python src/evaluate.py
```

Expected: side-by-side comparison table with delta column.
This is the headline result. Takes ~2-3 minutes (50 API calls).
Overwrites the files from step 8 with the full two-system results.

---

## 10. Generate the HTML report (free)

```bash
python src/generate_report.py
```

Expected: `results/report.html` — open in any browser.
To export as PDF: File → Print → Save as PDF → Landscape, no margins.

---

## 11. Run the Streamlit app

```bash
streamlit run src/app.py
```

Opens at http://localhost:8501

Three modes (selector in sidebar):
- **LLM (gpt-4o-mini)** — full pipeline with product recommendations
- **Keyword Baseline** — rule-based only, no API call
- **Side by Side** — both systems together, best for demos

Recommended demo requests (in this order):
1. `"Birthday cake but my daughter is celiac"` — implicit constraint
2. `"Party drinks but nothing alcoholic, kids will be there"` — negation
3. `"I'm diabetic, need a filling breakfast that won't spike my sugar"` — safety

Cost per LLM request in the app: ~€0.0007 (intent + recommendation call).

---

## 12. Test the product recommender directly (~€0.001)

```bash
python src/product_recommender.py
```

Expected: 8 ranked products with match scores for a diabetic breakfast mission.

---

## Cost summary

| Step | API calls | Estimated cost |
|------|-----------|----------------|
| Smoke test (step 7) | 4 | €0.001 |
| Full evaluation (step 9) | 50 | €0.012 |
| App demo session (~50 requests) | ~100 | €0.035 |
| Prompt ablation (optional) | ~150 | €0.040 |
| **Total project budget** | **~300** | **~€0.09** |

Well within the €2.50 credit budget.

---

## Key design decisions (for Q&A)

**Why a closed vocabulary for mission_type?**
Closed fields enable accuracy/F1 scoring. Open-ended mission description
lives in `mission_summary` (free text, evaluated qualitatively).

**Why two few-shot examples in the LLM prompt?**
Chosen specifically to teach implicit constraint inference (celiac→gluten_free)
and negation that removes a category. Adding more few-shots costs tokens
with diminishing returns at this task size.

**Why is dietary_constraints safety-critical only?**
Keeps the field clean for allergy/intolerance cases. Health preferences
(low_sugar, high_protein) live in preference_constraints. The separation
makes the safety story clear in Q&A.

**Why gpt-4o-mini?**
Cost-effective for structured output tasks. At ~700 input + ~250 output
tokens per call, the entire project runs for under €0.15.

**Why keep the Carrefour category mapping?**
It's a real retailer's taxonomy from prior work, making product_categories
credible rather than invented. The Instacart products map onto it as a
lookup layer.

---

## File ownership (for individual reflections)

Update this section with your team's actual contribution split.

| File | Owner |
|------|-------|
| schema_utils.py | |
| keyword_interpreter.py | |
| llm_interpreter.py | |
| evaluate.py | |
| generate_report.py | |
| build_product_index.py | |
| product_recommender.py | |
| app.py | |
| testset.json (labelling) | |
