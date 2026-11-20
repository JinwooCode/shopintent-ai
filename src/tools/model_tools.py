"""Deterministic modeling tools used by the Data Scientist crew.

feature engineering -> stratified split -> model pipelines -> evaluation -> threshold check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.tools.decision import SEGMENTS, segment

TARGET = "Revenue"

CATEGORICAL = ["Month", "VisitorType", "OperatingSystems", "Browser", "Region", "TrafficType"]
BASE_NUMERIC = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay", "Weekend",
]
ENGINEERED = ["TotalPages", "TotalDuration", "ProductPageShare", "AvgSecondsPerProductPage"]


# --------------------------------------------------------------------------- features
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add session-level features. Deterministic, row-wise (no leakage across rows)."""
    out = df.copy()
    out["Weekend"] = out["Weekend"].astype(int)
    out["TotalPages"] = out["Administrative"] + out["Informational"] + out["ProductRelated"]
    out["TotalDuration"] = (
        out["Administrative_Duration"] + out["Informational_Duration"] + out["ProductRelated_Duration"]
    )
    out["ProductPageShare"] = np.where(out["TotalPages"] > 0, out["ProductRelated"] / out["TotalPages"], 0.0)
    out["AvgSecondsPerProductPage"] = np.where(
        out["ProductRelated"] > 0, out["ProductRelated_Duration"] / out["ProductRelated"], 0.0
    )
    for col in CATEGORICAL:
        out[col] = out[col].astype(str)
    return out


def feature_lists(drop: list[str]) -> tuple[list[str], list[str]]:
    numeric = [c for c in BASE_NUMERIC + ENGINEERED if c not in drop]
    categorical = [c for c in CATEGORICAL if c not in drop]
    return numeric, categorical


def check_required_features(df: pd.DataFrame, drop: list[str]) -> None:
    numeric, categorical = feature_lists(drop)
    missing = [c for c in numeric + categorical + [TARGET] if c not in df.columns]
    if missing:
        from src.tools.data_tools import DataValidationError
        raise DataValidationError([f"Required features missing before modeling: {missing}"])


# --------------------------------------------------------------------------- split
def split(df: pd.DataFrame, test_size: float, val_size: float, seed: int) -> pd.Series:
    """Return a 'split' label (train/val/test) per row, stratified on the target."""
    idx = np.arange(len(df))
    y = df[TARGET].astype(int).to_numpy()
    rest_idx, test_idx = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    val_share = val_size / (1 - test_size)
    train_idx, val_idx = train_test_split(
        rest_idx, test_size=val_share, stratify=y[rest_idx], random_state=seed
    )
    labels = np.empty(len(df), dtype=object)
    labels[train_idx], labels[val_idx], labels[test_idx] = "train", "val", "test"
    return pd.Series(labels, index=df.index, name="split")


# --------------------------------------------------------------------------- models
def _preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    num = Pipeline([
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([("num", num, numeric), ("cat", cat, categorical)])


def build_model(algorithm: str, numeric: list[str], categorical: list[str], seed: int) -> Pipeline:
    if algorithm == "logistic_regression":
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    elif algorithm == "gradient_boosting":
        clf = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
            l2_regularization=1.0, early_stopping=False, random_state=seed,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    return Pipeline([("prep", _preprocessor(numeric, categorical)), ("clf", clf)])


# --------------------------------------------------------------------------- evaluation
def metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "precision@0.5": precision_score(y_true, y_pred, zero_division=0),
        "recall@0.5": recall_score(y_true, y_pred, zero_division=0),
        "f1@0.5": f1_score(y_true, y_pred, zero_division=0),
    }


def segment_table(y_true, proba, thresholds: dict) -> pd.DataFrame:
    """Per segment: share of sessions and actual purchase rate."""
    seg = pd.Series([segment(p, thresholds) for p in proba])
    y = pd.Series(np.asarray(y_true).astype(int))
    rows = []
    for s in SEGMENTS:
        mask = seg == s
        n = int(mask.sum())
        rows.append({
            "segment": s,
            "sessions": n,
            "share": n / len(seg),
            "actual_purchase_rate": float(y[mask].mean()) if n else float("nan"),
            "mean_predicted": float(np.asarray(proba)[mask.to_numpy()].mean()) if n else float("nan"),
        })
    return pd.DataFrame(rows)


def threshold_checks(table: pd.DataFrame, overall_rate: float) -> list[tuple[str, bool]]:
    """Sanity checks that the fixed thresholds separate sessions meaningfully (validation only)."""
    t = table.set_index("segment")
    rb, rh, ru = (t.loc[s, "actual_purchase_rate"] for s in SEGMENTS)
    return [
        ("Every segment contains sessions", bool((t["sessions"] > 0).all())),
        ("Purchase rate is ordered: buyer > hesitant > unlikely", bool(rb > rh > ru)),
        ("Buyer segment converts above the overall rate", bool(rb > overall_rate)),
        ("Unlikely segment converts below the overall rate", bool(ru < overall_rate)),
    ]
