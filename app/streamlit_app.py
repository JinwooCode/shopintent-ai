"""ShopIntent AI: visit simulation.

Replays a completed session from the held-out test set, scores it with the main model
(no PageValues), and shows the segment and the recommended intervention.

Run from the project root:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.tools.decision import decide  # noqa: E402
from src.utils import load_config, resolve  # noqa: E402

SEGMENT_STYLE = {
    "buyer": ("🟢", "Likely buyer"),
    "hesitant": ("🟡", "Hesitant"),
    "unlikely": ("⚪", "Unlikely to buy"),
}
INTERVENTION_TEXT = {
    "coupon": ("🎟️ Offer a discount coupon", "Returning visitor on the fence: a coupon can tip the decision."),
    "chat": ("💬 Open a live-chat invitation", "New visitor on the fence: help and reassurance first."),
    "none": ("No intervention", "Leave the visit as is."),
}

st.set_page_config(page_title="ShopIntent AI", page_icon="🛒", layout="wide")


# --------------------------------------------------------------------------- loading
@st.cache_resource
def load_model_and_config():
    cfg = load_config()
    model_path = resolve(cfg["paths"]["scientist_dir"]) / "model.pkl"
    if not model_path.exists():
        return None, cfg
    return joblib.load(model_path), cfg


@st.cache_data
def load_test_sessions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    test = df[df["split"] == "test"].reset_index(drop=True)
    for col in ["Month", "VisitorType", "OperatingSystems", "Browser", "Region", "TrafficType"]:
        test[col] = test[col].astype(str)
    return test


model, cfg = load_model_and_config()
features_path = resolve(cfg["paths"]["scientist_dir"]) / "features.csv"

if model is None or not features_path.exists():
    st.error("Model artifacts not found. Run the pipelines first:\n\n"
             "`python -m src.run_data_pipeline`\n\n`python -m src.run_model_pipeline`")
    st.stop()

sessions = load_test_sessions(str(features_path))
th = cfg["thresholds"]

# --------------------------------------------------------------------------- header
st.title("🛒 ShopIntent AI: Visit Simulation")
st.caption(
    "Replays **completed** sessions from the held-out test set (never seen in training). "
    "This is a simulation, not real-time scoring. The model does **not** use `PageValues`."
)

# --------------------------------------------------------------------------- sidebar: pick a session
with st.sidebar:
    st.header("Choose a visit")
    visitor_filter = st.selectbox("Visitor type", ["All", "Returning_Visitor", "New_Visitor", "Other"])
    pool = sessions if visitor_filter == "All" else sessions[sessions["VisitorType"] == visitor_filter]
    if pool.empty:
        st.warning("No sessions for this filter.")
        st.stop()

    # The number input owns st.session_state.pick; the button and filter change it via
    # callbacks, which run before widgets are drawn, so the two never overwrite each other.
    def _random_visit():
        st.session_state.pick = int(pool.sample(1).index[0])

    if "pick" not in st.session_state:
        st.session_state.pick = int(pool.index[0])
    if st.session_state.pick not in pool.index and st.session_state.get("last_filter") != visitor_filter:
        st.session_state.pick = int(pool.index[0])  # filter changed: jump to a matching visit
    st.session_state.last_filter = visitor_filter

    st.button("🎲 Random visit", width="stretch", on_click=_random_visit)
    st.number_input("…or session #", min_value=0, max_value=len(sessions) - 1, step=1, key="pick")
    if st.session_state.pick not in pool.index:
        st.info("This session doesn't match the filter. Showing it anyway.")

    st.divider()
    st.subheader("Rules (fixed in advance)")
    st.markdown(
        f"- **≥ {th['buyer_from']:.0%}**: likely buyer → no intervention\n"
        f"- **{th['unlikely_below']:.0%}–{th['buyer_from']:.0%}**: hesitant → coupon (returning) / chat (new)\n"
        f"- **< {th['unlikely_below']:.0%}**: unlikely → no intervention"
    )

row = sessions.loc[[st.session_state.pick]]
s = row.iloc[0]

# --------------------------------------------------------------------------- session summary
st.subheader(f"Session #{st.session_state.pick}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Visitor", s["VisitorType"].replace("_", " "))
c2.metric("Month", s["Month"])
c3.metric("Pages viewed", int(s["TotalPages"]))
c4.metric("Time on site", f"{s['TotalDuration'] / 60:.1f} min")

with st.expander("Session details"):
    detail = pd.DataFrame({
        "Page type": ["Administrative", "Informational", "Product related"],
        "Pages": [int(s["Administrative"]), int(s["Informational"]), int(s["ProductRelated"])],
        "Seconds": [round(s["Administrative_Duration"]), round(s["Informational_Duration"]),
                    round(s["ProductRelated_Duration"])],
    })
    st.dataframe(detail, hide_index=True, width="stretch")
    st.write(f"Bounce rate **{s['BounceRates']:.3f}** · Exit rate **{s['ExitRates']:.3f}** · "
             f"Weekend **{'yes' if s['Weekend'] else 'no'}** · Special day **{s['SpecialDay']:.1f}**")

# --------------------------------------------------------------------------- prediction
proba = float(model.predict_proba(row)[:, 1][0])
result = decide(proba, s["VisitorType"], cfg)
icon, seg_label = SEGMENT_STYLE[result["segment"]]
action, why = INTERVENTION_TEXT[result["intervention"]]

st.divider()
left, right = st.columns([1, 1])
with left:
    st.markdown("#### Purchase probability")
    st.markdown(f"<h1 style='margin:0'>{proba:.0%}</h1>", unsafe_allow_html=True)
    st.progress(min(max(proba, 0.0), 1.0))
    st.markdown(f"### {icon} {seg_label}")
with right:
    st.markdown("#### Recommended action")
    with st.container(border=True):
        st.markdown(f"### {action}")
        st.write(why)

with st.expander("🔍 Reveal what actually happened"):
    if bool(s["Revenue"]):
        st.success("This visitor **bought**.")
    else:
        st.info("This visitor **did not buy**.")
