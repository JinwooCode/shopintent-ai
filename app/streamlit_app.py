"""ShopIntent AI: visit simulation.

Replays a completed session from the held-out test set, scores it with the main model
(no PageValues), explains what drove the score, and shows the recommended intervention.

Run from the project root:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.tools.decision import decide  # noqa: E402
from src.tools.model_tools import CATEGORICAL, engineer_features  # noqa: E402
from src.utils import load_config, resolve  # noqa: E402

# --------------------------------------------------------------------------- copy
MONTHS = {"Jan": "January", "Feb": "February", "Mar": "March", "Apr": "April", "May": "May",
          "Jun": "June", "Jul": "July", "Aug": "August", "Sep": "September", "Oct": "October",
          "Nov": "November", "Dec": "December"}
VISITOR = {"Returning_Visitor": "returning visitor", "New_Visitor": "new visitor", "Other": "visitor"}
VISITOR_FILTER = {"All": "All visitors", "Returning_Visitor": "Returning visitors",
                  "New_Visitor": "New visitors", "Other": "Other"}
SEGMENTS = {
    "buyer": {"label": "Likely buyer", "color": "#059669", "soft": "#D1FAE5",
              "meaning": "This visit already looks like a purchase. Interrupting it adds nothing."},
    "hesitant": {"label": "Hesitant", "color": "#D97706", "soft": "#FEF3C7",
                 "meaning": "A real chance to buy, but undecided. This is the window where a nudge pays off."},
    "unlikely": {"label": "Unlikely to buy", "color": "#64748B", "soft": "#E2E8F0",
                 "meaning": "Most likely just browsing. An offer here would mostly be wasted spend."},
}
ACTIONS = {
    "coupon": ("🎟️", "Offer a discount coupon",
               "Returning visitors already trust the store. A small incentive is often enough to close."),
    "chat": ("💬", "Invite to a live chat",
             "New visitors don't know the store yet. Help and reassurance come before discounts."),
    "none": ("✋", "No intervention",
             "Leave the visit alone. Spend the budget where it can change the outcome."),
}
# Factors used to explain a score: label -> columns swapped together for a "typical" visitor
FACTORS = {
    "Product browsing": ["ProductRelated", "ProductRelated_Duration"],
    "Exit rate of pages viewed": ["ExitRates"],
    "Bounce rate of pages viewed": ["BounceRates"],
    "Month of the visit": ["Month"],
    "Visitor type": ["VisitorType"],
    "Account & admin pages": ["Administrative", "Administrative_Duration"],
    "Information pages": ["Informational", "Informational_Duration"],
    "Holiday proximity": ["SpecialDay"],
    "Traffic source": ["TrafficType"],
    "Weekend visit": ["Weekend"],
}

st.set_page_config(page_title="ShopIntent AI · Visit Simulation", page_icon="🛒", layout="wide")


def html(block: str) -> None:
    """Render raw HTML. Lines are stripped so Markdown never mistakes indentation for code."""
    st.markdown("\n".join(line.strip() for line in block.splitlines() if line.strip()),
                unsafe_allow_html=True)


html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp button p {font-family: 'Inter', sans-serif;}
header[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1180px;}
@keyframes rise {from {opacity: 0; transform: translateY(8px);} to {opacity: 1; transform: none;}}
@keyframes sweep {from {transform: rotate(0deg);}}
@keyframes grow {from {width: 0;}}
.hero {background: radial-gradient(1200px 300px at 0% 0%, #6D5BFF 0%, transparent 60%),
        linear-gradient(120deg, #1E1B4B 0%, #312E81 55%, #4338CA 100%);
       border-radius: 22px; padding: 2rem 2.2rem; color: #fff; position: relative; overflow: hidden;
       box-shadow: 0 20px 45px -20px rgba(49,46,129,.6); animation: rise .5s ease-out;}
.hero:after {content: ""; position: absolute; right: -60px; top: -60px; width: 260px; height: 260px;
             border-radius: 50%; background: rgba(255,255,255,.06);}
.pill {display: inline-flex; align-items: center; gap: .45rem; font-size: .72rem; font-weight: 700;
       letter-spacing: .12em; text-transform: uppercase; padding: .32rem .75rem; border-radius: 999px;
       background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.22);}
.pill i {width: 7px; height: 7px; border-radius: 50%; background: #34D399; box-shadow: 0 0 10px #34D399;}
.hero h1 {font-size: 2.35rem; font-weight: 800; letter-spacing: -.03em; margin: .8rem 0 .35rem; color: #fff;}
.hero p {font-size: 1.05rem; color: #C7D2FE; max-width: 640px; margin: 0;}
.stats {display: flex; gap: 2.2rem; margin-top: 1.5rem; flex-wrap: wrap;}
.stat b {display: block; font-size: 1.55rem; font-weight: 800; color: #fff; letter-spacing: -.02em;}
.stat span {font-size: .8rem; color: #A5B4FC;}
.label {font-size: .72rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;
        color: #5B4CF0; margin: 1.9rem 0 .6rem; display: flex; align-items: center; gap: .55rem;}
.label em {font-style: normal; background: #5B4CF0; color: #fff; border-radius: 6px; padding: .05rem .42rem;}
.card {background: #fff; border: 1px solid #E6E8F0; border-radius: 18px; padding: 1.35rem 1.5rem;
       box-shadow: 0 1px 2px rgba(16,24,40,.04), 0 8px 24px -12px rgba(16,24,40,.08);
       animation: rise .45s ease-out both; height: 100%;}
.story {font-size: 1.12rem; line-height: 1.65; color: #1F2937; margin: 0 0 1.1rem;}
.journey {display: flex; height: 34px; border-radius: 10px; overflow: hidden; background: #EEF0F6;}
.journey div {display: flex; align-items: center; justify-content: center; font-size: .75rem;
              font-weight: 700; color: #fff; white-space: nowrap; overflow: hidden; animation: grow .8s ease-out;}
.legend {display: flex; gap: 1.1rem; font-size: .78rem; color: #6B7280; margin: .55rem 0 1.2rem; flex-wrap: wrap;}
.legend span:before {content: ""; display: inline-block; width: 9px; height: 9px; border-radius: 3px;
                     margin-right: .4rem; background: var(--c);}
.kpis {display: grid; grid-template-columns: repeat(4, 1fr); gap: .7rem;}
.kpi {background: #F8F9FC; border: 1px solid #EEF0F6; border-radius: 12px; padding: .7rem .8rem;}
.kpi b {display: block; font-size: 1.3rem; font-weight: 750; color: #111827; letter-spacing: -.02em;}
.kpi span {font-size: .74rem; color: #6B7280;}
.kpi small {display: block; font-size: .7rem; color: #9CA3AF; margin-top: .15rem;}
.gauge-wrap {display: flex; flex-direction: column; align-items: center;}
.needle {transform-box: view-box; transform-origin: 120px 120px; animation: sweep 1.1s cubic-bezier(.2,.8,.2,1);}
.score {font-size: 3rem; font-weight: 800; letter-spacing: -.04em; line-height: 1; margin-top: -.6rem;}
.sub {color: #6B7280; font-size: .92rem; text-align: center; margin-top: .3rem;}
.badge {display: inline-block; font-weight: 700; font-size: .85rem; padding: .32rem .85rem; border-radius: 999px; margin-top: .8rem;}
.facts {display: grid; grid-template-columns: 1fr 1fr; gap: .7rem; margin-top: 1.1rem; width: 100%;}
.fact {background: #F8F9FC; border-radius: 12px; padding: .65rem .8rem; text-align: center;}
.fact b {display: block; font-size: 1.15rem; font-weight: 750;}
.fact span {font-size: .74rem; color: #6B7280;}
.driver {display: grid; grid-template-columns: 1.3fr 2fr 64px; gap: .8rem; align-items: center; padding: .55rem 0;
         border-bottom: 1px dashed #EEF0F6;}
.driver:last-child {border-bottom: none;}
.driver .name {font-weight: 600; font-size: .92rem; color: #1F2937;}
.driver .name small {display: block; font-weight: 400; color: #6B7280; font-size: .76rem; margin-top: .1rem;}
.track {position: relative; height: 10px; background: #F1F2F7; border-radius: 6px;}
.track:before {content: ""; position: absolute; left: 50%; top: -4px; bottom: -4px; width: 1px; background: #CBD0DC;}
.bar {position: absolute; top: 0; height: 100%; border-radius: 6px; animation: grow .8s ease-out;}
.delta {font-weight: 750; font-size: .9rem; text-align: right;}
.action {border-radius: 18px; padding: 1.5rem; color: #fff; position: relative; overflow: hidden; height: 100%;
         animation: rise .5s ease-out both;}
.action .icon {font-size: 2.6rem;}
.action h3 {color: #fff; font-size: 1.55rem; font-weight: 800; letter-spacing: -.02em; margin: .55rem 0 .4rem;}
.action p {color: rgba(255,255,255,.88); margin: 0 0 1rem;}
.rule {background: rgba(255,255,255,.14); border-radius: 12px; padding: .75rem .9rem; font-size: .83rem; color: #fff;}
.rule b {color: #fff;}
.note {font-size: .78rem; color: #9CA3AF; margin-top: .9rem;}
div[data-testid="stExpander"] {background: #fff; border-radius: 14px; border: 1px solid #E6E8F0;}
@media (max-width: 640px) {
.kpis {grid-template-columns: 1fr 1fr;}
.hero {padding: 1.5rem;} .hero h1 {font-size: 1.9rem;}
.driver {grid-template-columns: 1fr 1fr 58px;}
}
</style>
""")


# --------------------------------------------------------------------------- data
@st.cache_resource
def load_model_and_config():
    cfg = load_config()
    path = resolve(cfg["paths"]["scientist_dir"]) / "model.pkl"
    return (joblib.load(path) if path.exists() else None), cfg


@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)
    for col in CATEGORICAL:
        df[col] = df[col].astype(str)
    train = df[df["split"] == "train"]
    typical = {c: (train[c].mode()[0] if c in CATEGORICAL else float(train[c].median()))
               for cols in FACTORS.values() for c in cols}
    test = df[df["split"] == "test"].reset_index(drop=True)
    return test, float(df["Revenue"].mean()), typical


model, cfg = load_model_and_config()
sci_dir = resolve(cfg["paths"]["scientist_dir"])
if model is None or not (sci_dir / "features.csv").exists():
    st.error("Model artifacts not found. Run `python -m src.run_data_pipeline` and "
             "`python -m src.run_model_pipeline` first.")
    st.stop()

sessions, base_rate, typical = load_data(str(sci_dir / "features.csv"))
meta_path = sci_dir / "model_meta.json"
meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
roc = meta.get("test_metrics", {}).get("main", {}).get("roc_auc", float("nan"))
th = cfg["thresholds"]
lo, hi = th["unlikely_below"], th["buyer_from"]


@st.cache_data
def score_all(_model, n: int):
    proba = _model.predict_proba(sessions)[:, 1]
    segs = ["buyer" if p >= hi else "unlikely" if p < lo else "hesitant" for p in proba]
    return proba, segs


all_proba, all_segs = score_all(model, len(sessions))
peer_rate = sessions.groupby(pd.Series(all_segs))["Revenue"].mean().to_dict()
first_hesitant = next((i for i in range(len(sessions))
                       if all_segs[i] == "hesitant" and sessions.at[i, "VisitorType"] == "Returning_Visitor"), 0)


def pct(p: float) -> str:
    return "<1%" if p < 0.01 else f"{p:.0%}"


def explain(row: pd.DataFrame, proba: float) -> list[tuple[str, float, str]]:
    """Score impact of each factor: swap it for a typical visitor's value and re-score."""
    out = []
    for name, cols in FACTORS.items():
        alt = row.copy()
        for c in cols:
            alt[c] = typical[c]
        alt = engineer_features(alt)
        delta = proba - float(model.predict_proba(alt)[:, 1][0])
        out.append((name, delta, cols))
    return sorted(out, key=lambda t: abs(t[1]), reverse=True)


def describe(cols: list[str], s: pd.Series) -> str:
    c = cols[0]
    if c == "ProductRelated":
        return (f"{int(s[c])} pages, {s['ProductRelated_Duration'] / 60:.0f} min "
                f"(typical {typical[c]:.0f} pages, {typical['ProductRelated_Duration'] / 60:.0f} min)")
    if c in ("Administrative", "Informational"):
        return f"{int(s[c])} pages (typical {typical[c]:.0f})"
    if c in ("ExitRates", "BounceRates"):
        return f"{s[c]:.1%} (typical {typical[c]:.1%})"
    if c == "Month":
        return f"{MONTHS.get(s[c], s[c])} (most common: {MONTHS.get(typical[c], typical[c])})"
    if c == "VisitorType":
        return VISITOR.get(s[c], s[c]).capitalize()
    if c == "SpecialDay":
        return "Close to a holiday" if s[c] > 0 else "Not near a holiday"
    if c == "Weekend":
        return "Yes" if s[c] else "No"
    return f"Source #{s[c]}"


def gauge_svg(p: float, color: str) -> str:
    cx, cy, r = 120, 120, 92

    def pt(v, rr=r):
        a = math.pi * (1 - v)
        return cx + rr * math.cos(a), cy - rr * math.sin(a)

    def tick(v):
        if v in (0, 1):  # ends: label just below the arc end
            x, y = pt(v)[0], cy + 16
        else:            # middle: label outside the arc
            x, y = pt(v, r + 20)
            y += 4
        return f'<text x="{x:.0f}" y="{y:.0f}" font-size="10" fill="#9CA3AF" text-anchor="middle">{v:.0%}</text>'

    def arc(v1, v2, col):
        (x1, y1), (x2, y2) = pt(v1), pt(v2)
        return (f'<path d="M {x1:.1f} {y1:.1f} A {r} {r} 0 0 1 {x2:.1f} {y2:.1f}" '
                f'stroke="{col}" stroke-width="18" fill="none" stroke-linecap="butt"/>')

    ticks = "".join(tick(v) for v in (0, lo, hi, 1))
    return f"""
<svg viewBox="0 0 240 142" width="100%" style="max-width:330px">
{arc(0, lo, "#E2E8F0")}{arc(lo, hi, "#FCD34D")}{arc(hi, 1, "#6EE7B7")}
{ticks}
<g class="needle" style="transform: rotate({min(max(p, 0), 1) * 180:.1f}deg)">
<line x1="{cx}" y1="{cy}" x2="{cx - r + 14}" y2="{cy}" stroke="#111827" stroke-width="4" stroke-linecap="round"/>
</g>
<circle cx="{cx}" cy="{cy}" r="9" fill="#111827"/><circle cx="{cx}" cy="{cy}" r="4" fill="{color}"/>
</svg>"""


# --------------------------------------------------------------------------- hero
html(f"""
<div class="hero">
<span class="pill"><i></i>Simulation mode · completed visits</span>
<h1>ShopIntent AI</h1>
<p>Predict whether an online visitor will buy, and decide in the moment whether the store should step in.</p>
<div class="stats">
<div class="stat"><b>{len(sessions):,}</b><span>unseen test visits</span></div>
<div class="stat"><b>{base_rate:.1%}</b><span>of visits end in a purchase</span></div>
<div class="stat"><b>{roc:.2f}</b><span>ROC-AUC on test data</span></div>
<div class="stat"><b>0</b><span>uses of PageValues</span></div>
</div>
</div>
""")

# --------------------------------------------------------------------------- step 1: pick
html('<div class="label"><em>1</em>Pick a visit</div>')
with st.container(border=True):
    c1, c2, c3 = st.columns([1.3, 1, 1])
    visitor_filter = c1.selectbox("Visitor type", list(VISITOR_FILTER), format_func=VISITOR_FILTER.get)
    pool = sessions if visitor_filter == "All" else sessions[sessions["VisitorType"] == visitor_filter]

    # The number input owns st.session_state.pick; the button and the filter change it via a
    # callback or before the widget is drawn, so they never overwrite each other.
    def _random_visit():
        st.session_state.pick = int(pool.sample(1).index[0])

    if "pick" not in st.session_state:
        st.session_state.pick = first_hesitant  # open on a visit where the rule actually acts
    if st.session_state.pick not in pool.index and st.session_state.get("last_filter") != visitor_filter:
        st.session_state.pick = int(pool.index[0])
    st.session_state.last_filter = visitor_filter

    c2.number_input("Visit number", min_value=0, max_value=len(sessions) - 1, step=1, key="pick",
                    help="Visits from the held-out test set. The model never saw them during training.")
    c3.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
    c3.button("🎲  Replay a random visit", width="stretch", type="primary", on_click=_random_visit)
    if st.session_state.pick not in pool.index:
        st.caption("This visit doesn't match the visitor filter. Showing it anyway.")

i = st.session_state.pick
row = sessions.loc[[i]]
s = row.iloc[0]
proba = float(all_proba[i])
result = decide(proba, s["VisitorType"], cfg)
seg = SEGMENTS[result["segment"]]
icon, action, why = ACTIONS[result["intervention"]]

# --------------------------------------------------------------------------- step 2 + 3
left, right = st.columns([1.45, 1], gap="medium")
with left:
    html('<div class="label"><em>2</em>What the visitor did</div>')
    extras = (["on a weekend"] if s["Weekend"] else []) + (["close to a holiday"] if s["SpecialDay"] > 0 else [])
    story = (f"A <b>{VISITOR.get(s['VisitorType'], 'visitor')}</b> spent <b>{s['TotalDuration'] / 60:.1f} minutes</b> "
             f"on the store in <b>{MONTHS.get(s['Month'], s['Month'])}</b>{', ' + ' and '.join(extras) if extras else ''}, "
             f"viewing <b>{int(s['TotalPages'])} pages</b>, <b>{int(s['ProductRelated'])}</b> of them product pages.")
    tot = max(s["TotalDuration"], 1e-9)
    parts = [("Account & admin", s["Administrative_Duration"], "#8B5CF6"),
             ("Information", s["Informational_Duration"], "#38BDF8"),
             ("Product", s["ProductRelated_Duration"], "#5B4CF0")]
    bars = "".join(f'<div style="width:{v / tot * 100:.2f}%;background:{c}">{f"{v / tot:.0%}" if v / tot > .08 else ""}</div>'
                   for _, v, c in parts if v > 0) or '<div style="width:100%;color:#6B7280">No time recorded</div>'
    legend = "".join(f'<span style="--c:{c}">{n} · {v / 60:.1f} min</span>' for n, v, c in parts)
    html(f"""
<div class="card">
<p class="story">{story}</p>
<div class="journey">{bars}</div>
<div class="legend">{legend}</div>
<div class="kpis">
<div class="kpi"><b>{int(s['ProductRelated'])}</b><span>Product pages</span><small>typical {typical['ProductRelated']:.0f}</small></div>
<div class="kpi"><b>{s['ProductRelated_Duration'] / 60:.1f}m</b><span>On product pages</span><small>typical {typical['ProductRelated_Duration'] / 60:.1f}m</small></div>
<div class="kpi"><b>{s['ExitRates']:.1%}</b><span>Exit rate</span><small>typical {typical['ExitRates']:.1%} · lower is better</small></div>
<div class="kpi"><b>{s['BounceRates']:.1%}</b><span>Bounce rate</span><small>typical {typical['BounceRates']:.1%} · lower is better</small></div>
</div>
</div>
""")

with right:
    html('<div class="label"><em>3</em>Purchase probability</div>')
    ratio = proba / base_rate
    if ratio >= 1.05:
        vs_big, vs_small = f"{ratio:.1f}×", "the average visitor's chance"
    elif ratio > .95:
        vs_big, vs_small = "≈ 1×", "same as the average visitor"
    elif ratio >= .1:
        vs_big, vs_small = f"{1 / ratio:.0f}× lower", "than the average visitor"
    else:
        vs_big, vs_small = "< 0.1×", "the average visitor's chance"
    html(f"""
<div class="card"><div class="gauge-wrap">
{gauge_svg(proba, seg['color'])}
<div class="score" style="color:{seg['color']}">{pct(proba)}</div>
<div class="sub">chance this visit ends in a purchase</div>
<span class="badge" style="color:{seg['color']};background:{seg['soft']}">{seg['label']}</span>
<div class="facts">
<div class="fact"><b>{vs_big}</b><span>{vs_small}</span></div>
<div class="fact"><b>{peer_rate.get(result['segment'], 0):.0%}</b><span>of {seg['label'].lower()} test visits actually bought</span></div>
</div>
</div></div>
""")

# --------------------------------------------------------------------------- step 4 + 5
left, right = st.columns([1.45, 1], gap="medium")
with left:
    html('<div class="label"><em>4</em>Why the model thinks so</div>')
    drivers = explain(row, proba)[:5]
    scale = max(abs(d) for _, d, _ in drivers) or 1
    lines = []
    for name, d, cols in drivers:
        w = abs(d) / scale * 50
        col = "#10B981" if d >= 0 else "#EF4444"
        pos = f"left:50%;width:{w:.1f}%" if d >= 0 else f"left:{50 - w:.1f}%;width:{w:.1f}%"
        lines.append(f'<div class="driver"><div class="name">{name}<small>{describe(cols, s)}</small></div>'
                     f'<div class="track"><div class="bar" style="{pos};background:{col}"></div></div>'
                     f'<div class="delta" style="color:{col}">{d * 100:+.1f} pts</div></div>')
    html(f"""
<div class="card">
{''.join(lines)}
<div class="note">Each factor was swapped for a typical visitor's value and the visit re-scored. Green raised the
purchase probability, red lowered it.</div>
</div>
""")

with right:
    html('<div class="label"><em>5</em>Recommended action</div>')
    grad = {"buyer": "linear-gradient(135deg,#047857,#10B981)",
            "hesitant": "linear-gradient(135deg,#B45309,#F59E0B)",
            "unlikely": "linear-gradient(135deg,#334155,#64748B)"}[result["segment"]]
    html(f"""
<div class="action" style="background:{grad}">
<div class="icon">{icon}</div>
<h3>{action}</h3>
<p>{why}</p>
<p>{seg['meaning']}</p>
<div class="rule"><b>Decision rule</b>, fixed before training:<br>
≥ {hi:.0%} likely buyer → no action<br>
{lo:.0%}–{hi:.0%} hesitant → coupon if returning, chat if new<br>
&lt; {lo:.0%} unlikely → no action</div>
</div>
""")

# --------------------------------------------------------------------------- check + about
html('<div class="label"><em>✓</em>Check the model</div>')
with st.expander("🔍  Reveal what this visitor actually did"):
    if bool(s["Revenue"]):
        st.success("✅ This visitor **bought**.")
    else:
        st.info("This visitor **left without buying**.")

with st.expander("ℹ️  About this simulation and the model"):
    tm = meta.get("test_metrics", {})
    st.markdown(f"""
- **Simulation, not real time.** Completed visits from the
  [UCI Online Shoppers dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
  are replayed from the test set, which the model never saw in training.
- **Model:** {meta.get('main_model', {}).get('algorithm', 'n/a').replace('_', ' ')} on session behavior.
  It does **not** use `PageValues`, a field computed from pages that led to purchases, which would leak the answer.
- **Quality on unseen visits:** ROC-AUC {tm.get('main', {}).get('roc_auc', 0):.2f}, PR-AUC
  {tm.get('main', {}).get('pr_auc', 0):.2f} versus {base_rate:.2f} for random guessing. With `PageValues` the
  comparison model reaches PR-AUC {tm.get('comparison', {}).get('pr_auc', 0):.2f}, see
  `artifacts/scientist/evaluation_report.md`.
- **Thresholds** ({lo:.0%} / {hi:.0%}) were set in advance and only checked on validation data.
""")
