"""Deterministic feature importance for the main model (no LLM).

Permutation importance on the VALIDATION split, scored with PR-AUC (average precision), the same metric used
for model selection. For numeric features the direction is the Spearman correlation between the feature and
the model's predicted probability on validation.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.inspection import permutation_importance

from src.tools.model_tools import CATEGORICAL, ENGINEERED, TARGET

ENGINEERED_DEFINITIONS = {
    "TotalPages": "Administrative + Informational + ProductRelated pages in the session",
    "TotalDuration": "total seconds on all three page types",
    "ProductPageShare": "ProductRelated / TotalPages (0 when no pages)",
    "AvgSecondsPerProductPage": "ProductRelated_Duration / ProductRelated (0 when no product pages)",
}


def compute_importance(model, features: pd.DataFrame, numeric: list[str], categorical: list[str],
                       seed: int, n_repeats: int = 5) -> dict:
    val = features[features["split"] == "val"].copy()
    for col in CATEGORICAL:
        val[col] = val[col].astype(str)
    cols = numeric + categorical
    X, y = val[cols], val[TARGET].astype(int)
    base_proba = model.predict_proba(X)[:, 1]
    result = permutation_importance(model, X, y, scoring="average_precision", n_repeats=n_repeats,
                                    random_state=seed, n_jobs=1)
    rows = []
    for i, col in enumerate(cols):
        direction = None
        if col in numeric:
            rho = pd.Series(X[col].astype(float).to_numpy()).corr(pd.Series(base_proba), method="spearman")
            direction = round(float(rho), 3) if pd.notna(rho) else 0.0
        rows.append({"feature": col, "kind": "categorical" if col in categorical else "numeric",
                     "engineered": col in ENGINEERED,
                     "importance": round(float(result.importances_mean[i]), 4),
                     "importance_std": round(float(result.importances_std[i]), 4),
                     "direction_rho": direction})
    rows.sort(key=lambda r: -r["importance"])
    for rank, r in enumerate(rows, 1):
        r["rank"] = rank
    return {"method": "permutation importance on the validation split, scored by PR-AUC (average precision)",
            "n_repeats": n_repeats, "seed": seed, "validation_sessions": int(len(val)),
            "features": rows, "engineered_definitions": ENGINEERED_DEFINITIONS}


def save_importance(imp: dict, out: Path) -> None:
    out.write_text(json.dumps(imp, indent=2) + "\n", encoding="utf-8", newline="\n")
