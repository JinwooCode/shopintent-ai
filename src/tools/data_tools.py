"""Deterministic data tools used by the Data Analyst crew.

1. load_raw / check_raw   - ingest the raw CSV and check its expected schema
2. clean                  - produce clean_data.csv
3. build_contract         - produce dataset_contract.json
4. validate_against_contract - check a dataset against the contract (fails loudly)

No LLM is involved here: the same input always produces the same output.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

TARGET = "Revenue"

# Expected raw schema: column -> logical type
EXPECTED_SCHEMA: dict[str, str] = {
    "Administrative": "integer",
    "Administrative_Duration": "float",
    "Informational": "integer",
    "Informational_Duration": "float",
    "ProductRelated": "integer",
    "ProductRelated_Duration": "float",
    "BounceRates": "float",
    "ExitRates": "float",
    "PageValues": "float",
    "SpecialDay": "float",
    "Month": "category",
    "OperatingSystems": "category_code",
    "Browser": "category_code",
    "Region": "category_code",
    "TrafficType": "category_code",
    "VisitorType": "category",
    "Weekend": "boolean",
    "Revenue": "boolean",
}

MONTH_FIX = {"June": "Jun"}
MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
RATE_COLUMNS = ["BounceRates", "ExitRates"]
COUNT_DURATION_PAIRS = [
    ("Administrative", "Administrative_Duration"),
    ("Informational", "Informational_Duration"),
    ("ProductRelated", "ProductRelated_Duration"),
]


class DataValidationError(Exception):
    """Raised when data does not match what the pipeline expects."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("Data validation failed:\n- " + "\n- ".join(errors))


# --------------------------------------------------------------------------- 1. ingest
def load_raw(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DataValidationError([f"Raw data file not found: {path}"])
    return pd.read_csv(path)


def check_raw(df: pd.DataFrame) -> dict:
    """Check the raw schema and return a data-quality report. Raises on schema errors."""
    errors = []
    missing_cols = sorted(set(EXPECTED_SCHEMA) - set(df.columns))
    extra_cols = sorted(set(df.columns) - set(EXPECTED_SCHEMA))
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    if extra_cols:
        errors.append(f"Unexpected columns: {extra_cols}")
    if df.empty:
        errors.append("Dataset is empty")
    if errors:
        raise DataValidationError(errors)

    numeric = df.select_dtypes("number")
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_values": {c: int(n) for c, n in df.isna().sum().items() if n > 0},
        "duplicate_rows": int(df.duplicated().sum()),
        "negative_values": {c: int(n) for c, n in (numeric < 0).sum().items() if n > 0},
        "positive_rate": round(float(df[TARGET].mean()), 4),
        "month_values": sorted(df["Month"].unique().tolist()),
    }


# --------------------------------------------------------------------------- 2. clean
def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return (clean_df, log_of_actions). The raw frame is not modified."""
    out = df.copy()
    actions: list[str] = []

    # Month: unify "June" -> "Jun" so all months use 3-letter codes
    n_fix = int(out["Month"].isin(MONTH_FIX).sum())
    out["Month"] = out["Month"].replace(MONTH_FIX)
    if n_fix:
        actions.append(f"Month: renamed 'June' -> 'Jun' in {n_fix} rows")

    # Strings: trim whitespace
    for col in ["Month", "VisitorType"]:
        out[col] = out[col].astype(str).str.strip()

    # Booleans: accept TRUE/FALSE strings as well as real bools
    for col in ["Weekend", TARGET]:
        if out[col].dtype != bool:
            out[col] = out[col].astype(str).str.upper().map({"TRUE": True, "FALSE": False})
            actions.append(f"{col}: converted to boolean")

    # Integer columns stay integers
    for col, kind in EXPECTED_SCHEMA.items():
        if kind in ("integer", "category_code"):
            out[col] = out[col].astype("int64")

    # Missing values: none expected in this dataset; drop if any appear
    n_missing = int(out.isna().any(axis=1).sum())
    if n_missing:
        out = out.dropna().reset_index(drop=True)
        actions.append(f"Dropped {n_missing} rows with missing values")

    # Duplicates are KEPT on purpose: they are single-page bounce sessions from
    # different users (one session per user), so removing them would distort the bounce rate.
    n_dup = int(out.duplicated().sum())
    actions.append(f"Kept {n_dup} identical rows (single-page bounce sessions from different users)")

    actions.append(f"Final shape: {out.shape[0]} rows x {out.shape[1]} columns")
    return out, actions


def quality_checks(df: pd.DataFrame) -> list[dict]:
    """Extra data-quality observations on the clean data. Informational: they never stop the pipeline."""
    out = []
    for count, dur in COUNT_DURATION_PAIRS:
        n = int(((df[count] == 0) & (df[dur] > 0)).sum())
        out.append({"check": f"{count}: 0 pages but time recorded", "rows": n, "ok": n == 0,
                    "detail": f"{n} sessions have {count} = 0 but {dur} > 0"})
    long_sessions = int((df[[d for _, d in COUNT_DURATION_PAIRS]].sum(axis=1) > 3 * 3600).sum())
    out.append({"check": "Very long sessions (over 3 hours)", "rows": long_sessions, "ok": True,
                "detail": f"{long_sessions} sessions last over 3 hours in total; kept, the model uses robust trees"})
    zero_time = int((df[[d for _, d in COUNT_DURATION_PAIRS]].sum(axis=1) == 0).sum())
    out.append({"check": "Sessions with no time recorded", "rows": zero_time, "ok": True,
                "detail": f"{zero_time} sessions have 0 seconds in total (single-page visits); kept"})
    rate = float(df[TARGET].mean())
    out.append({"check": "Class balance", "rows": int(df[TARGET].sum()), "ok": True,
                "detail": f"{rate * 100:.1f}% of sessions are purchases; imbalanced, so splits are stratified "
                          f"and PR-AUC is reported"})
    return out


# --------------------------------------------------------------------------- 3. contract
def build_contract(df: pd.DataFrame, main_drop: list[str]) -> dict:
    """Describe the clean dataset: schema, allowed values, assumptions, constraints."""
    columns = {}
    for col, kind in EXPECTED_SCHEMA.items():
        spec: dict = {"type": kind, "nullable": False}
        if kind in ("integer", "float"):
            spec["min"] = 0
            if col in RATE_COLUMNS:
                spec["max"] = 1
            elif col == "SpecialDay":
                spec["max"] = 1
        if kind in ("category", "category_code", "boolean"):
            values = df[col].unique().tolist()
            if col == "Month":
                values = [m for m in MONTH_ORDER if m in values]
            else:
                values = sorted(values)
            spec["allowed_values"] = [v.item() if hasattr(v, "item") else v for v in values]
        columns[col] = spec

    features = [c for c in EXPECTED_SCHEMA if c != TARGET]
    return {
        "dataset": "UCI Online Shoppers Purchasing Intention (id 468)",
        "file": "artifacts/analyst/clean_data.csv",
        "version": "1.0",
        "row_count": int(len(df)),
        "target": {
            "name": TARGET,
            "type": "boolean",
            "positive_class": True,
            "positive_rate": round(float(df[TARGET].mean()), 4),
        },
        "columns": columns,
        "required_features": features,
        "main_model_excluded_features": main_drop,
        "assumptions": [
            "Each row is one completed session; each session belongs to a different user.",
            "Sessions are complete, so predictions are a visit simulation, not real-time scoring.",
            "Identical rows are distinct single-page bounce sessions and are kept.",
            "OperatingSystems, Browser, Region and TrafficType are anonymized category codes, not quantities.",
        ],
        "constraints": [
            "No missing values in any column.",
            "Counts and durations are >= 0.",
            "BounceRates, ExitRates and SpecialDay are within [0, 1].",
            "Categorical columns contain only the listed allowed_values.",
            "PageValues is strongly tied to the purchase outcome; the main model must not use it.",
        ],
    }


def save_contract(contract: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:  # same bytes on every OS
        f.write(json.dumps(contract, indent=2) + "\n")


def load_contract(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise DataValidationError([f"Dataset contract not found: {path}"])
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- 4. validate
def validate_against_contract(df: pd.DataFrame, contract: dict) -> list[str]:
    """Check df against the contract. Returns the list of checks passed; raises on any failure."""
    errors: list[str] = []
    cols = contract["columns"]

    missing = [c for c in cols if c not in df.columns]
    if missing:
        errors.append(f"Columns in contract but missing from data: {missing}")
    extra = [c for c in df.columns if c not in cols]
    if extra:
        errors.append(f"Columns in data but not in contract: {extra}")
    missing_req = [c for c in contract["required_features"] if c not in df.columns]
    if missing_req:
        errors.append(f"Required features missing: {missing_req}")
    if contract["target"]["name"] not in df.columns:
        errors.append(f"Target column missing: {contract['target']['name']}")

    for col, spec in cols.items():
        if col not in df.columns:
            continue
        s = df[col]
        if not spec.get("nullable", True) and s.isna().any():
            errors.append(f"{col}: {int(s.isna().sum())} missing values (not allowed)")
        if "min" in spec and (s < spec["min"]).any():
            errors.append(f"{col}: {int((s < spec['min']).sum())} values below {spec['min']}")
        if "max" in spec and (s > spec["max"]).any():
            errors.append(f"{col}: {int((s > spec['max']).sum())} values above {spec['max']}")
        if "allowed_values" in spec:
            bad = sorted(set(s.dropna().unique().tolist()) - set(spec["allowed_values"]))
            if bad:
                errors.append(f"{col}: values not allowed by contract: {bad[:10]}")

    if errors:
        raise DataValidationError(errors)

    return [
        f"All {len(cols)} contract columns present, no extra columns",
        f"All {len(contract['required_features'])} required features present",
        "No missing values",
        "Numeric ranges within contract bounds",
        "Categorical values within allowed_values",
    ]
