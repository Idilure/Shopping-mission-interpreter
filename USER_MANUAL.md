# User Manual
## Shopping Mission Interpreter

This manual explains how to use the Streamlit demo after the project has been installed with `SETUP.md`.

## Purpose

The app turns a plain-language grocery shopping request into a structured shopping mission. It identifies meal occasion, mission type, dietary constraints, preferences, budget sensitivity, excluded items, suggested aisles, risk flags, confidence, and a short explanation. In LLM mode, it also recommends products from the local Instacart-derived product index.

## Start The App

From the project root:

```bash
streamlit run src/app.py
```

Open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## Choose A Mode

Use the sidebar to choose one of three modes:

- `LLM (gpt-4o-mini)`: runs the LLM interpreter and product recommender. Requires `OPENAI_API_KEY`.
- `Keyword Baseline`: runs the transparent rule-based baseline only. No API key or API cost.
- `Side by Side`: runs both interpreters and displays their outputs next to each other.

## Enter A Request

Type a natural shopping request in the main text box, then click `Interpret`.

Good demo examples:

```text
Birthday cake but my daughter is celiac
Party drinks but nothing alcoholic, kids will be there
I'm diabetic, need a filling breakfast that won't spike my sugar
Quick dinner, nothing with dairy though
Something cheap and easy for the week, I don't want to cook
```

## Read The Output

The interpretation card shows:

- `Meal Occasion`: breakfast, lunch, dinner, snack, or none.
- `Mission Type`: the main shopping mission, such as health goal, party hosting, convenience meal, or specific ingredient.
- `Dietary Constraints`: hard constraints such as gluten-free, lactose-free, vegan, halal, or nut allergy.
- `Preferences`: softer goals such as quick, filling, healthy, low sugar, or kid friendly.
- `Excluded Items`: literal items the shopper asked to avoid.
- `Suggested Aisles`: coarse product categories for downstream recommendation.
- `Risk Flags`: warnings such as allergy/safety, negation, ambiguity, conflicting goals, or missing information.
- `Confidence`: the interpreter's confidence score.
- `Explanation`: the key wording that drove the output.

In LLM mode, product cards show a product name, aisle, reason, and match score from 1 to 10.

## Submit Feedback

After a result appears, use the feedback section at the bottom:

- Choose `Wrong`, `Partial`, or `Correct`.
- Optionally describe what was wrong or missing.
- Click `Submit feedback`.

Feedback is appended to:

```text
results/user_feedback.csv
```

Use this file as supporting evidence for real user feedback in the project submission.

## Expected Demo Story

The clearest demonstration is to compare the keyword baseline and LLM on requests that require inference:

- `Birthday cake but my daughter is celiac`: the LLM should infer `gluten_free`; the keyword baseline intentionally misses it.
- `Party drinks but nothing alcoholic, kids will be there`: the LLM should recognize negation and exclude alcohol.
- `I'm diabetic, need a filling breakfast that won't spike my sugar`: the LLM should infer low-sugar preference and safety risk.

## Troubleshooting

- If LLM mode fails, check that `.env` exists and contains `OPENAI_API_KEY=...`.
- If product recommendations do not appear, run `python src/build_product_index.py` and confirm `data/product_index.json` exists.
- If imports fail, run `pip install -r requirements.txt`.
- If the app opens but only the keyword baseline works, the API key is missing, invalid, or out of credit.
