"""ShopIntent AI: visit simulation.

Replays a completed session from the held-out test set, scores it with the main model
(no PageValues), and shows the segment and the recommended intervention.

Run from the project root:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.tools.decision import decide  # noqa: E402
from src.utils import load_config, resolve  # noqa: E402

MONTHS = {"Jan": "January", "Feb": "February", "Mar": "March", "Apr": "April", "May": "May",
          "Jun": "June", "Jul": "July", "Aug": "August", "Sep": "September", "Oct": "October",
          "Nov": "November", "Dec": "December"}
VISITOR = {"Returning_Visitor": "returning visitor", "New_Visitor": "new visitor", "Other": "visitor"}
SEGMENTS = {
    "buyer": {"label": "Likely buyer", "color": "#059669", "bg": "#ECFDF5",
              "meaning": "This visit looks like a purchase. No need to interrupt it."},
    "hesitant": {"label": "Hesitant", "color": "#D97706", "bg": "#FFFBEB",
                 "meaning": "A real chance to buy, but not decided. This is where a nudge can help."},
    "unlikely": {"label": "Unlikely to buy", "color": "#6B7280", "bg": "#F3F4F6",
                 "meaning": "Probably just browsing. An offer here would mostly be wasted."},
}
ACTIONS = {
    "coupon": ("🎟️", "Offer a discount coupon",
               "Returning visitors already know the store. A small incentive can tip the decision."),
    "chat": ("💬", "Invite to a live chat",
             "New visitors don't know the store yet. Help and reassurance come before discounts."),
    "none": ("✋", "No intervention",
             "Leave the visit alone. Interrupting would not change the outcome."),
}

st.set_page_config(page_title="ShopIntent AI", page_icon="🛒", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1150px;}
.hero h1 {font-size: 2.1rem; margin-bottom: .2rem;}
.hero p {color: #4B5563; font-size: 1.05rem; margin: 0;}
.steps {display: flex; gap: .75rem; margin: 1.1rem 0 .4rem; flex-wrap: wrap;}
.step {flex: 1; min-width: 200px; background: #fff; border: 1px solid #E5E7EB; border-radius: 12px;
       padding: .7rem .9rem; font-size: .92rem; color: #374151;}
.step b {color: #4F46E5; margin-right: .35rem;}
.section {font-size: .78rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
          color: #4F46E5; margin: 1.6rem 0 .4rem;}
.story {font-size: 1.15rem; line-height: 1.6; color: #1F2937; margin: .2rem 0 .8rem;}
.prob {font-size: 3.4rem; font-weight: 800; line-height: 1; margin: .2rem 0;}
.vsavg {color: #4B5563; font-size: 1rem; margin-bottom: 1rem;}
.gauge {position: relative; height: 14px; border-radius: 7px; overflow: visible; display: flex; margin: 2.2rem 0 .3rem;}
.zone {height: 100%;}
.z1 {background: #D1D5DB; border-radius: 7px 0 0 7px;} .z2 {background: #FCD34D;} .z3 {background: #6EE7B7; border-radius: 0 7px 7px 0;}
.marker {position: absolute; top: -28px; transform: translateX(-50%); text-align: center;}
.marker span {background: #1F2937; color: #fff; font-size: .78rem; font-weight: 700; padding: 2px 7px; border-radius: 6px;}
.marker i {display: block; width: 3px; height: 19px; background: #1F2937; margin: 3px auto 0; border-radius: 2px;}
.ticks {position: relative; height: 1.3rem; font-size: .75rem; color: #6B7280;}
.ticks span {position: absolute; transform: translateX(-50%);}
.zones-legend {display: flex; font-size: .78rem; color: #4B5563; margin-top: .2rem;}
.zones-legend div {text-align: center;}
.badge {display: inline-block; font-weight: 700; padding: .3rem .8rem; border-radius: 999px; font-size: .95rem;}
.card {border-radius: 14px; padding: 1.2rem 1.3rem; border: 1px solid #E5E7EB; background: #fff; height: 100%;}
.action-icon {font-size: 2.4rem; line-height: 1;}
.action-title {font-size: 1.45rem; font-weight: 750; margin: .5rem 0 .3rem;}
.muted {color: #4B5563;}
.note {font-size: .85rem; color: #6B7280;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- loading
@st.cache_resource
def load_model_and_config():
    cfg = load_config()
    model_path = resolve(cfg["paths"]["scientist_dir"]) / "model.pkl"
    return (joblib.load(model_path) if model_path.exists() else None), cfg


@st.cache_data
def load_sessions(path: str) -> tuple[pd.DataFrame, float]:
    df = pd.read_csv(path)
    base_rate = float(df["Revenue"].mean())
    test = df[df["split"] == "test"].reset_index(drop=True)
    for col in ["Month", "VisitorType", "OperatingSystems", "Browser", "Region", "TrafficType"]:
        test[col] = test[col].astype(str)
    return test, base_rate


model, cfg = load_model_and_config()
sci_dir = resolve(cfg["paths"]["scientist_dir"])
if model is None or not (sci_dir / "features.csv").exists():
    st.error("Model artifacts not found. Run the pipelines first:\n\n"
             "`python -m src.run_data_pipeline`\n\n`python -m src.run_model_pipeline`")
    st.stop()

sessions, base_rate = load_sessions(str(sci_dir / "features.csv"))
th = cfg["thresholds"]
lo, hi = th["unlikely_below"], th["buyer_from"]


@st.cache_data
def first_hesitant_visit(_model, n: int) -> int:
    p = _model.predict_proba(sessions)[:, 1]
    hits = [i for i in range(n) if lo <= p[i] < hi and sessions.at[i, "VisitorType"] == "Returning_Visitor"]
    return hits[0] if hits else 0


first_hesitant = first_hesitant_visit(model, len(sessions))


def pct_label(p: float) -> str:
    return "<1%" if p < 0.01 else f"{p:.0%}"

# --------------------------------------------------------------------------- header
st.markdown("""
<div class="hero">
  <h1>🛒 ShopIntent AI</h1>
  <p>Will this online visitor buy, and should the store step in? Replay a real visit and see what the model recommends.</p>
</div>
<div class="steps">
  <div class="step"><b>1</b>Pick a visit from past store traffic</div>
  <div class="step"><b>2</b>The model estimates the chance of a purchase</div>
  <div class="step"><b>3</b>A fixed rule turns that into an action</div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- step 1: pick a visit
st.markdown('<div class="section">Step 1 · Pick a visit</div>', unsafe_allow_html=True)
with st.container(border=True):
    c1, c2, c3 = st.columns([1.3, 1, 1])
    visitor_filter = c1.selectbox(
        "Visitor type", ["All", "Returning_Visitor", "New_Visitor", "Other"],
        format_func=lambda v: {"All": "All visitors", "Returning_Visitor": "Returning visitors",
                               "New_Visitor": "New visitors", "Other": "Other"}[v],
    )
    pool = sessions if visitor_filter == "All" else sessions[sessions["VisitorType"] == visitor_filter]

    # The number input owns st.session_state.pick; the button and the filter change it via
    # callbacks / before the widget is drawn, so they never overwrite each other.
    def _random_visit():
        st.session_state.pick = int(pool.sample(1).index[0])

    if "pick" not in st.session_state:
        st.session_state.pick = first_hesitant  # open on a visit where the rule actually acts
    if st.session_state.pick not in pool.index and st.session_state.get("last_filter") != visitor_filter:
        st.session_state.pick = int(pool.index[0])
    st.session_state.last_filter = visitor_filter

    c2.number_input("Visit number", min_value=0, max_value=len(sessions) - 1, step=1, key="pick",
                    help=f"{len(sessions):,} visits the model never saw during training.")
    c3.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
    c3.button("🎲  Random visit", width="stretch", type="primary", on_click=_random_visit)
    if st.session_state.pick not in pool.index:
        st.caption("This visit doesn't match the visitor filter. Showing it anyway.")

row = sessions.loc[[st.session_state.pick]]
s = row.iloc[0]

# --------------------------------------------------------------------------- step 2: what happened
st.markdown('<div class="section">Step 2 · What the visitor did</div>', unsafe_allow_html=True)
extras = []
if s["Weekend"]:
    extras.append("on a weekend")
if s["SpecialDay"] > 0:
    extras.append("close to a holiday")
story = (f"A <b>{VISITOR.get(s['VisitorType'], 'visitor')}</b> viewed <b>{int(s['TotalPages'])} pages</b> "
         f"(<b>{int(s['ProductRelated'])}</b> of them product pages) over <b>{s['TotalDuration'] / 60:.1f} minutes</b> "
         f"in {MONTHS.get(s['Month'], s['Month'])}{', ' + ' and '.join(extras) if extras else ''}.")
st.markdown(f'<div class="story">{story}</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Product pages", int(s["ProductRelated"]),
          help="Pages about products. More product browsing usually means more interest.")
m2.metric("Time on product pages", f"{s['ProductRelated_Duration'] / 60:.1f} min",
          help="Total time spent on product pages.")
m3.metric("Bounce rate", f"{s['BounceRates']:.0%}",
          help="How often people leave right after landing on the pages this visitor viewed. Lower is better.")
m4.metric("Exit rate", f"{s['ExitRates']:.0%}",
          help="How often the pages this visitor viewed are the last page of a visit. Lower is better.")

# --------------------------------------------------------------------------- step 3 + 4: prediction and action
proba = float(model.predict_proba(row)[:, 1][0])
result = decide(proba, s["VisitorType"], cfg)
seg = SEGMENTS[result["segment"]]
icon, action, why = ACTIONS[result["intervention"]]
ratio = proba / base_rate
compare = (f"<b>{ratio:.1f}×</b> the average visitor ({base_rate:.0%})" if ratio >= 1.05
           else f"about the same as the average visitor ({base_rate:.0%})" if ratio > 0.95
           else f"<b>{1 / ratio:.0f}× lower</b> than the average visitor ({base_rate:.0%})" if ratio >= 0.1
           else f"far below the average visitor ({base_rate:.0%})")

left, right = st.columns([1.25, 1], gap="large")
with left:
    st.markdown('<div class="section">Step 3 · Chance of a purchase</div>', unsafe_allow_html=True)
    pct = min(max(proba, 0.0), 1.0) * 100
    st.markdown(f"""
<div class="card">
  <div class="prob" style="color:{seg['color']}">{pct_label(proba)}</div>
  <div class="vsavg">chance this visit ends in a purchase, {compare}</div>
  <span class="badge" style="color:{seg['color']};background:{seg['bg']}">{seg['label']}</span>
  <div class="gauge">
    <div class="zone z1" style="width:{lo * 100:.0f}%"></div>
    <div class="zone z2" style="width:{(hi - lo) * 100:.0f}%"></div>
    <div class="zone z3" style="width:{(1 - hi) * 100:.0f}%"></div>
    <div class="marker" style="left:{pct:.1f}%"><span>{pct_label(proba)}</span><i></i></div>
  </div>
  <div class="ticks"><span style="left:0%;transform:none">0%</span><span style="left:{lo * 100:.0f}%">{lo:.0%}</span>
    <span style="left:{hi * 100:.0f}%">{hi:.0%}</span><span style="right:0;transform:none">100%</span></div>
  <div class="zones-legend">
    <div style="width:{lo * 100:.0f}%">Unlikely</div><div style="width:{(hi - lo) * 100:.0f}%">Hesitant</div>
    <div style="width:{(1 - hi) * 100:.0f}%">Likely buyer</div>
  </div>
  <p class="muted" style="margin:.9rem 0 0">{seg['meaning']}</p>
</div>
""", unsafe_allow_html=True)

with right:
    st.markdown('<div class="section">Step 4 · Recommended action</div>', unsafe_allow_html=True)
    st.markdown(f"""
<div class="card" style="border-top: 4px solid {seg['color']}">
  <div class="action-icon">{icon}</div>
  <div class="action-title">{action}</div>
  <p class="muted">{why}</p>
  <p class="note">Rule: only <b>hesitant</b> visitors ({lo:.0%}–{hi:.0%}) get an action, a coupon if returning
  and a chat if new. Thresholds were fixed in advance, not tuned to the results.</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- reveal + about
st.markdown('<div class="section">Check the model</div>', unsafe_allow_html=True)
with st.expander("🔍  Reveal what this visitor actually did"):
    if bool(s["Revenue"]):
        st.success("✅ This visitor **bought**.")
    else:
        st.info("This visitor **left without buying**.")

with st.expander("ℹ️  About this simulation and the model"):
    meta_path = sci_dir / "model_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    tm = meta.get("test_metrics", {}).get("main", {})
    st.markdown(f"""
- **Simulation, not real time.** These are completed visits from the
  [UCI Online Shoppers dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset).
  The app replays visits from the test set, which the model never saw in training.
- **Model:** {meta.get('main_model', {}).get('algorithm', 'n/a').replace('_', ' ')}, trained on session behavior.
  It does **not** use `PageValues`, a field computed from pages that led to purchases, which would leak the answer.
- **How good is it?** On the test set it ranks a real buyer above a non-buyer
  **{tm.get('roc_auc', 0):.0%}** of the time (ROC-AUC {tm.get('roc_auc', 0):.2f}). PR-AUC is {tm.get('pr_auc', 0):.2f},
  versus {base_rate:.2f} for random guessing.
- **Full comparison** with the PageValues model: `artifacts/scientist/evaluation_report.md`.
""")
