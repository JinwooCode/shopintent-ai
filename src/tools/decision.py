"""Business rules: probability -> segment -> intervention.

Thresholds and rules come from config.yaml (fixed in advance, approved by the lecturer).
Used by the training pipeline (threshold check) and by the Streamlit app.
"""
from __future__ import annotations

BUYER = "buyer"
HESITANT = "hesitant"
UNLIKELY = "unlikely"
SEGMENTS = [BUYER, HESITANT, UNLIKELY]


def segment(probability: float, thresholds: dict) -> str:
    if probability >= thresholds["buyer_from"]:
        return BUYER
    if probability < thresholds["unlikely_below"]:
        return UNLIKELY
    return HESITANT


def intervention(segment_name: str, visitor_type: str, rules: dict) -> str:
    rule = rules[segment_name]
    if isinstance(rule, dict):  # hesitant: depends on visitor type
        return rule.get(visitor_type, rule.get("Other", "none"))
    return rule


def decide(probability: float, visitor_type: str, cfg: dict) -> dict:
    seg = segment(probability, cfg["thresholds"])
    return {
        "probability": float(probability),
        "segment": seg,
        "intervention": intervention(seg, visitor_type, cfg["interventions"]),
    }
