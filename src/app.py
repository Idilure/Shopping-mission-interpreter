"""app.py — Shopping Mission Interpreter
Streamlit front-end.  Run from project root:
    streamlit run src/app.py
"""
import json, os, sys, csv, datetime
import streamlit as st

# ── paths ─────────────────────────────────────────────────────────────────────
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")
DATA_DIR = os.path.join(ROOT_DIR, "data")
RES_DIR  = os.path.join(ROOT_DIR, "results")
os.makedirs(RES_DIR, exist_ok=True)

sys.path.insert(0, SRC_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT_DIR, ".env"))

from schema_utils import allowed
from keyword_interpreter    import interpret_keyword
from llm_interpreter        import interpret_llm
from product_recommender    import recommend_products

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shopping Mission Interpreter",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── category mapping for demo layer ──────────────────────────────────────────
with open(os.path.join(DATA_DIR, "category_mapping.json"), encoding="utf-8") as f:
    CAT_MAP = json.load(f)

# product index (built by build_product_index.py)
_prod_index_path = os.path.join(DATA_DIR, "product_index.json")
PRODUCT_INDEX = {}
if os.path.exists(_prod_index_path):
    with open(_prod_index_path, encoding="utf-8") as f:
        PRODUCT_INDEX = json.load(f)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── base ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.main { background: #F7F6F2; }

/* ── hero header ── */
.hero {
    background: linear-gradient(135deg, #0D2B4E 0%, #1A4A7A 100%);
    border-radius: 16px;
    padding: 2.2rem 2.5rem 1.8rem;
    margin-bottom: 1.5rem;
    color: white;
}
.hero h1 { font-size: 1.9rem; font-weight: 700; margin: 0 0 .3rem; color: white; }
.hero p  { font-size: 1rem; opacity: .8; margin: 0; color: #c8d8ea; }

/* ── input card ── */
.input-card {
    background: white;
    border-radius: 12px;
    padding: 1.4rem 1.8rem;
    box-shadow: 0 2px 8px rgba(0,0,0,.07);
    margin-bottom: 1.2rem;
}

/* ── result card ── */
.result-card {
    background: white;
    border-radius: 12px;
    padding: 1.4rem 1.8rem;
    box-shadow: 0 2px 8px rgba(0,0,0,.07);
    margin-bottom: 1rem;
}
.result-card h3 {
    font-size: .78rem; font-weight: 600; letter-spacing: .08em;
    text-transform: uppercase; color: #6B7280; margin: 0 0 .8rem;
}

/* ── tags ── */
.tag {
    display: inline-block;
    padding: 3px 11px; border-radius: 20px;
    font-size: .8rem; font-weight: 500;
    margin: 2px 3px 2px 0;
}
.tag-blue   { background:#DBEAFE; color:#1E40AF; }
.tag-green  { background:#D1FAE5; color:#065F46; }
.tag-amber  { background:#FEF3C7; color:#92400E; }
.tag-red    { background:#FEE2E2; color:#991B1B; }
.tag-purple { background:#EDE9FE; color:#5B21B6; }
.tag-gray   { background:#F3F4F6; color:#374151; }

/* ── metric row ── */
.metric-row {
    display: flex; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap;
}
.metric-box {
    flex: 1; min-width: 130px;
    background: #F7F6F2; border-radius: 10px;
    padding: .9rem 1.1rem; text-align: center;
}
.metric-box .val {
    font-size: 1.6rem; font-weight: 700; color: #0D2B4E;
}
.metric-box .lbl {
    font-size: .72rem; color: #6B7280;
    text-transform: uppercase; letter-spacing: .06em;
}

/* ── confidence bar ── */
.conf-bar-bg {
    background: #E5E7EB; border-radius: 6px;
    height: 8px; margin: .4rem 0 .2rem;
}
.conf-bar-fill {
    height: 8px; border-radius: 6px;
    background: linear-gradient(90deg, #10B981, #0D2B4E);
}

/* ── explanation box ── */
.expl-box {
    background: #F0F4FF; border-left: 3px solid #1A4A7A;
    border-radius: 0 8px 8px 0;
    padding: .7rem 1rem; font-size: .85rem;
    color: #374151; margin-top: .5rem;
}

/* ── mission summary ── */
.mission-summary {
    background: #0D2B4E; color: white;
    border-radius: 10px; padding: .9rem 1.2rem;
    font-size: .95rem; font-style: italic;
    margin-bottom: 1rem;
}

/* ── subcategory expander ── */
.subcat { font-size: .78rem; color: #6B7280; margin-top: .3rem; }

/* ── feedback ── */
.feedback-card {
    background: #F0F9FF; border: 1px solid #BAE6FD;
    border-radius: 12px; padding: 1.2rem 1.5rem;
    margin-top: 1rem;
}

/* ── side panel ── */
.side-label {
    font-size: .72rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: .07em;
    color: #6B7280; margin-bottom: .4rem;
}

/* hide streamlit chrome */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── LLM client (cached so it's created once per session) ─────────────────────
@st.cache_resource
def get_client():
    from openai import OpenAI
    return OpenAI()

# ── helpers ───────────────────────────────────────────────────────────────────
def tag(label, style="gray"):
    return f'<span class="tag tag-{style}">{label}</span>'

def tags_html(items, style="gray"):
    if not items:
        return f'<span style="color:#9CA3AF;font-size:.82rem">none detected</span>'
    return " ".join(tag(i, style) for i in items)

def confidence_bar(val):
    pct = int(val * 100)
    color = "#10B981" if val >= .8 else ("#F59E0B" if val >= .5 else "#EF4444")
    return f"""
    <div style="display:flex;align-items:center;gap:.6rem">
      <div class="conf-bar-bg" style="flex:1">
        <div class="conf-bar-fill" style="width:{pct}%;background:{color}"></div>
      </div>
      <span style="font-size:.85rem;font-weight:600;color:{color}">{pct}%</span>
    </div>"""

RISK_LABELS = {
    "allergy_or_safety":  ("🔴 Allergy / Safety",  "red"),
    "negation_present":   ("🟡 Negation detected", "amber"),
    "conflicting_goals":  ("🟡 Conflicting goals",  "amber"),
    "ambiguous_request":  ("🔵 Ambiguous",          "blue"),
    "missing_information":("🔵 Missing info",       "blue"),
}

def render_result(result, label):
    """Render one interpreter's result as a styled card."""
    st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
    st.markdown(f"<h3>{label}</h3>", unsafe_allow_html=True)

    # mission summary (LLM only)
    if result.get("mission_summary"):
        st.markdown(
            f'<div class="mission-summary">"{result["mission_summary"]}"</div>',
            unsafe_allow_html=True)

    # top metrics
    occ = result["meal_occasion"]
    mis = result["mission_type"]
    bud = result["budget_sensitivity"]
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-box">
        <div class="val">{occ if occ != "none" else "—"}</div>
        <div class="lbl">Meal Occasion</div>
      </div>
      <div class="metric-box">
        <div class="val">{mis.replace("_"," ")}</div>
        <div class="lbl">Mission Type</div>
      </div>
      <div class="metric-box">
        <div class="val">{bud.replace("_"," ")}</div>
        <div class="lbl">Budget</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # dietary constraints — special safety styling
    dc = result["dietary_constraints"]
    st.markdown("**Dietary Constraints** *(hard / safety-critical)*",
                unsafe_allow_html=False)
    st.markdown(tags_html(dc, "red" if dc else "gray"), unsafe_allow_html=True)

    # preference constraints
    st.markdown("**Preferences**")
    st.markdown(tags_html(result["preference_constraints"], "purple"),
                unsafe_allow_html=True)

    # excluded items
    ei = result["excluded_items"]
    if ei:
        st.markdown("**Excluded items**")
        st.markdown(tags_html(ei, "amber"), unsafe_allow_html=True)

    # product categories + subcategory expansion
    cats = result["product_categories"]
    st.markdown("**Suggested Aisles**")
    if cats:
        st.markdown(tags_html(cats, "blue"), unsafe_allow_html=True)
        subcats = []
        for c in cats:
            subcats.extend(CAT_MAP.get(c, [])[:3])   # top 3 per aisle
        if subcats:
            st.markdown(
                f'<div class="subcat">↳ {" · ".join(subcats[:12])}</div>',
                unsafe_allow_html=True)
    else:
        st.markdown(
            '<span style="color:#9CA3AF;font-size:.82rem">none detected</span>',
            unsafe_allow_html=True)

    # risk flags
    rf = result["risk_flags"]
    if rf:
        st.markdown("**Risk Flags**")
        flag_html = " ".join(
            tag(RISK_LABELS.get(f, (f, "gray"))[0],
                RISK_LABELS.get(f, (f, "gray"))[1])
            for f in rf)
        st.markdown(flag_html, unsafe_allow_html=True)

    # confidence + explanation
    st.markdown("**Confidence**")
    st.markdown(confidence_bar(result["confidence"]), unsafe_allow_html=True)
    if result.get("explanation"):
        st.markdown(
            f'<div class="expl-box">💬 {result["explanation"]}</div>',
            unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def save_feedback(request, result, rating, comment, system):
    """Append one feedback row to results/user_feedback.csv"""
    path = os.path.join(RES_DIR, "user_feedback.csv")
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["timestamp", "system", "request", "rating",
                        "comment", "mission_type", "dietary_constraints",
                        "risk_flags", "confidence"])
        w.writerow([
            datetime.datetime.now().isoformat(),
            system, request, rating, comment,
            result.get("mission_type", ""),
            "|".join(result.get("dietary_constraints", [])),
            "|".join(result.get("risk_flags", [])),
            result.get("confidence", ""),
        ])

# ── session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # list of (request, llm_result, kw_result)
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ── layout ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🛒 Shopping Mission Interpreter</h1>
  <p>Describe what you need in plain language — I'll understand your intent,
     flag any constraints, and suggest the right aisles.</p>
</div>
""", unsafe_allow_html=True)

# ── sidebar: mode selector + history ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    mode = st.radio(
        "Interpreter",
        ["LLM (gpt-4o-mini)", "Keyword Baseline", "Side by Side"],
        index=0,
    )
    st.markdown("---")
    st.markdown("### 🕘 Session History")
    if st.session_state.history:
        for i, (req, _, _) in enumerate(reversed(st.session_state.history[-8:])):
            st.caption(f"{i+1}. {req[:55]}{'…' if len(req)>55 else ''}")
    else:
        st.caption("No requests yet.")

# ── main input ────────────────────────────────────────────────────────────────
st.markdown('<div class="input-card">', unsafe_allow_html=True)

EXAMPLES = [
    "Quick healthy dinner for two, nothing with gluten",
    "Snacks for a kids birthday party but one child has a nut allergy",
    "I'm diabetic, need a filling breakfast that won't spike my sugar",
    "Something cheap and easy for the week, I don't want to cook",
    "Ingredients for a vegetarian dinner my meat-loving husband will also enjoy",
    "Party drinks but nothing alcoholic, kids will be there",
]

col_inp, col_ex = st.columns([3, 1])
with col_inp:
    user_input = st.text_area(
        "What are you shopping for?",
        placeholder="e.g. 'Quick healthy lunch for the office'",
        height=90, label_visibility="collapsed",
    )
with col_ex:
    st.markdown('<div class="side-label">Try an example</div>',
                unsafe_allow_html=True)
    for ex in EXAMPLES[:4]:
        if st.button(ex[:42] + "…", use_container_width=True, key=f"ex_{ex[:10]}"):
            user_input = ex

run = st.button("🔍  Interpret", type="primary", use_container_width=False)
st.markdown('</div>', unsafe_allow_html=True)

# ── run interpretation ────────────────────────────────────────────────────────
if run and user_input.strip():
    with st.spinner("Interpreting your request…"):

        kw_result  = interpret_keyword(user_input)
        llm_result = None

        if mode != "Keyword Baseline":
            try:
                llm_result = interpret_llm(user_input, client=get_client())
            except Exception as e:
                st.error(f"LLM call failed: {e}")
                llm_result = None

        st.session_state.last_result = (user_input, llm_result, kw_result)
        st.session_state.history.append((user_input, llm_result, kw_result))

elif run and not user_input.strip():
    st.warning("Please enter a shopping request first.")

# ── display results ───────────────────────────────────────────────────────────
if st.session_state.last_result:
    req, llm_res, kw_res = st.session_state.last_result

    st.markdown(f"**Request:** *{req}*")
    st.markdown("---")

    if mode == "Side by Side":
        col_llm, col_kw = st.columns(2)
        with col_llm:
            if llm_res:
                render_result(llm_res, "LLM — gpt-4o-mini")
        with col_kw:
            render_result(kw_res, "Keyword Baseline")

    elif mode == "LLM (gpt-4o-mini)":
        if llm_res:
            render_result(llm_res, "LLM — gpt-4o-mini")

    else:
        render_result(kw_res, "Keyword Baseline")

    # ── product recommendations ──────────────────────────────────────────────────
    if PRODUCT_INDEX and llm_res and mode != "Keyword Baseline":
        with st.spinner("Finding matching products…"):
            recs = recommend_products(llm_res, PRODUCT_INDEX,
                                      client=get_client())
        if recs:
            st.markdown("---")
            st.markdown("### 🛍️ Recommended Products")
            st.caption("Selected by the LLM based on your mission, "
                       "constraints, and preferences.")
            cols = st.columns(min(len(recs), 4))
            for i, rec in enumerate(recs):
                with cols[i % 4]:
                    aisle_emoji = {
                        "produce":"🥦","meat_seafood":"🥩","dairy_eggs":"🥛",
                        "bakery":"🍞","pantry_staples":"🫙","breakfast":"🥣",
                        "snacks_sweets":"🍫","frozen":"🧊","ready_meals":"🍱",
                        "beverages":"🥤","alcohol":"🍷","health_diet":"💊",
                        "baby":"👶","household_nonfood":"🏠",
                    }.get(rec["aisle"], "🛒")
                    score = rec.get("match_score", 7)
                    dot_color = (
                        "#10B981" if score >= 8 else
                        "#F59E0B" if score >= 6 else
                        "#EF4444"
                    )
                    dot_label = (
                        "Strong match" if score >= 8 else
                        "Good match"   if score >= 6 else
                        "Partial match"
                    )
                    filled = "█" * score
                    empty  = "░" * (10 - score)
                    st.markdown(f"""
<div style="background:white;border-radius:10px;padding:.9rem 1rem;
            box-shadow:0 2px 6px rgba(0,0,0,.07);margin-bottom:.5rem;
            border-top:3px solid {dot_color}">
  <div style="font-size:.95rem;font-weight:600;color:#0D2B4E">
    {aisle_emoji} {rec['product']}
  </div>
  <div style="font-size:.78rem;color:#6B7280;margin-top:.3rem">
    {rec['reason']}
  </div>
  <div style="display:flex;align-items:center;gap:.5rem;margin-top:.5rem">
    <span style="font-family:monospace;font-size:.72rem;
                 color:{dot_color};letter-spacing:1px">{filled}{empty}</span>
    <span style="font-size:.72rem;font-weight:600;color:{dot_color}">
      {score}/10</span>
    <span style="font-size:.68rem;color:#9CA3AF">{dot_label}</span>
  </div>
  <div style="font-size:.68rem;color:#9CA3AF;margin-top:.2rem">
    {rec['aisle'].replace('_',' ')}
  </div>
</div>""", unsafe_allow_html=True)

    # ── user feedback ─────────────────────────────────────────────────────────
    st.markdown('<div class="feedback-card">', unsafe_allow_html=True)
    st.markdown("#### 📝 Was this interpretation helpful?")
    st.caption("Your feedback is collected to evaluate the system. "
               "Takes 10 seconds.")

    fb_col1, fb_col2 = st.columns([1, 2])
    with fb_col1:
        rating = st.select_slider(
            "Rating",
            options=["❌ Wrong", "⚠️ Partial", "✅ Correct"],
            value="✅ Correct",
            label_visibility="collapsed",
        )
    with fb_col2:
        comment = st.text_input(
            "What was wrong or missing? (optional)",
            placeholder="e.g. 'missed the gluten-free constraint'",
            label_visibility="collapsed",
        )

    if st.button("Submit feedback", key="fb_submit"):
        active_result = llm_res if (mode != "Keyword Baseline" and llm_res) else kw_res
        active_system = "llm" if (mode != "Keyword Baseline" and llm_res) else "keyword"
        save_feedback(req, active_result, rating, comment, active_system)
        st.success("Thank you! Feedback saved to results/user_feedback.csv")

    st.markdown('</div>', unsafe_allow_html=True)

# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Shopping Mission Interpreter · NLP Group Project · "
    "Keyword Baseline + gpt-4o-mini · Carrefour category taxonomy"
)
