"""Run the deterministic modeling pipeline.

clean_data.csv + dataset_contract.json
  -> validation -> features.csv -> model selection (validation set)
  -> main model (no PageValues) + comparison model (with PageValues)
  -> threshold check (validation only) -> final test evaluation
  -> model.pkl, model_with_pagevalues.pkl, model_meta.json, evaluation_report.md

Usage (from the project root):
    python -m src.run_model_pipeline
"""
from __future__ import annotations

import json
import sys

import joblib
import pandas as pd

from src.tools.data_tools import DataValidationError, load_contract, validate_against_contract
from src.tools.model_tools import (
    TARGET,
    build_model,
    check_required_features,
    engineer_features,
    feature_lists,
    metrics,
    segment_table,
    split,
    threshold_checks,
)
from src.tools.decision import segment as decide_segment
from src.utils import get_logger, load_config, resolve

ALGORITHMS = ["logistic_regression", "gradient_boosting"]


def _fmt_metrics_row(name: str, m: dict) -> str:
    return (f"| {name} | {m['roc_auc']:.3f} | {m['pr_auc']:.3f} | {m['brier']:.3f} | "
            f"{m['precision@0.5']:.3f} | {m['recall@0.5']:.3f} | {m['f1@0.5']:.3f} |")


def _fmt_segments(table: pd.DataFrame) -> list[str]:
    lines = ["| Segment | Sessions | Share | Actual purchase rate | Mean predicted |",
             "|---|---|---|---|---|"]
    for r in table.itertuples():
        lines.append(f"| {r.segment} | {r.sessions} | {r.share:.1%} | "
                     f"{r.actual_purchase_rate:.1%} | {r.mean_predicted:.1%} |")
    return lines


def main() -> int:
    log = get_logger("model_pipeline")
    cfg = load_config()
    seed = cfg["seed"]
    analyst_dir = resolve(cfg["paths"]["analyst_dir"])
    out_dir = resolve(cfg["paths"]["scientist_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    main_cfg, cmp_cfg = cfg["models"]["main"], cfg["models"]["comparison"]
    thresholds = cfg["thresholds"]

    try:
        # ---- 1. validate the handoff from the Data Analyst crew
        log.info("Step 1/6: validating clean data against the dataset contract")
        clean_path = analyst_dir / "clean_data.csv"
        if not clean_path.exists():
            raise DataValidationError([f"Clean data not found: {clean_path} (run src.run_data_pipeline first)"])
        contract = load_contract(analyst_dir / "dataset_contract.json")
        df = pd.read_csv(clean_path)
        for p in validate_against_contract(df, contract):
            log.info("  OK  %s", p)
        if contract.get("main_model_excluded_features") != main_cfg["drop_features"]:
            raise DataValidationError([
                "Contract and config disagree on features excluded from the main model: "
                f"{contract.get('main_model_excluded_features')} vs {main_cfg['drop_features']}"
            ])

        # ---- 2. features + split
        log.info("Step 2/6: feature engineering and stratified split")
        feats = engineer_features(df)
        check_required_features(feats, drop=[])
        feats["split"] = split(feats, cfg["split"]["test_size"], cfg["split"]["val_size"], seed)
        feats.to_csv(out_dir / "features.csv", index=False, lineterminator="\n")
        sizes = feats["split"].value_counts().to_dict()
        rates = feats.groupby("split")[TARGET].mean().round(4).to_dict()
        log.info("  split sizes %s | purchase rate %s", sizes, rates)

        parts = {s: feats[feats["split"] == s] for s in ("train", "val", "test")}
        y = {s: parts[s][TARGET].astype(int) for s in parts}

        # ---- 3. choose the algorithm for the main model on validation PR-AUC
        log.info("Step 3/6: model selection on validation (main model, no PageValues)")
        num_main, cat_main = feature_lists(main_cfg["drop_features"])
        selection = {}
        for algo in ALGORITHMS:
            model = build_model(algo, num_main, cat_main, seed).fit(parts["train"], y["train"])
            selection[algo] = metrics(y["val"], model.predict_proba(parts["val"])[:, 1])
            log.info("  %-20s val PR-AUC %.3f | ROC-AUC %.3f | Brier %.3f", algo,
                     selection[algo]["pr_auc"], selection[algo]["roc_auc"], selection[algo]["brier"])
        best_algo = max(ALGORITHMS, key=lambda a: selection[a]["pr_auc"])
        log.info("  selected: %s", best_algo)

        # ---- 4. train main + comparison with the same algorithm
        log.info("Step 4/6: training main and comparison models (%s)", best_algo)
        num_cmp, cat_cmp = feature_lists(cmp_cfg["drop_features"])
        main_model = build_model(best_algo, num_main, cat_main, seed).fit(parts["train"], y["train"])
        cmp_model = build_model(best_algo, num_cmp, cat_cmp, seed).fit(parts["train"], y["train"])

        # ---- 5. threshold check on validation only
        log.info("Step 5/6: checking fixed thresholds on validation data")
        val_proba = main_model.predict_proba(parts["val"])[:, 1]
        val_segments = segment_table(y["val"], val_proba, thresholds)
        checks = threshold_checks(val_segments, float(y["val"].mean()))
        for name, ok in checks:
            log.info("  %s  %s", "OK  " if ok else "WARN", name)

        # ---- 6. final evaluation on test (once)
        log.info("Step 6/6: final evaluation on the held-out test set")
        test_main_proba = main_model.predict_proba(parts["test"])[:, 1]
        test_cmp_proba = cmp_model.predict_proba(parts["test"])[:, 1]
        m_main = metrics(y["test"], test_main_proba)
        m_cmp = metrics(y["test"], test_cmp_proba)
        test_segments = segment_table(y["test"], test_main_proba, thresholds)
        log.info("  main (no PageValues)   test PR-AUC %.3f | ROC-AUC %.3f", m_main["pr_auc"], m_main["roc_auc"])
        log.info("  comparison (PageValues) test PR-AUC %.3f | ROC-AUC %.3f", m_cmp["pr_auc"], m_cmp["roc_auc"])

    except DataValidationError as e:
        for err in e.errors:
            log.error("  FAIL  %s", err)
        log.error("Model pipeline stopped: fix the problems above and run again.")
        return 1

    # ---- save artifacts
    joblib.dump(main_model, out_dir / "model.pkl")
    joblib.dump(cmp_model, out_dir / "model_with_pagevalues.pkl")
    meta = {
        "main_model": {"file": "model.pkl", "name": main_cfg["name"], "algorithm": best_algo,
                       "numeric_features": num_main, "categorical_features": cat_main},
        "comparison_model": {"file": "model_with_pagevalues.pkl", "name": cmp_cfg["name"],
                             "algorithm": best_algo, "numeric_features": num_cmp,
                             "categorical_features": cat_cmp},
        "seed": seed,
        "thresholds": thresholds,
        "test_metrics": {"main": {k: round(v, 4) for k, v in m_main.items()},
                         "comparison": {k: round(v, 4) for k, v in m_cmp.items()}},
    }
    with open(out_dir / "model_meta.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(meta, indent=2) + "\n")

    hdr = ["| Model | ROC-AUC | PR-AUC | Brier ↓ | Precision@0.5 | Recall@0.5 | F1@0.5 |",
           "|---|---|---|---|---|---|---|"]
    report = [
        "# Evaluation Report — ShopIntent AI",
        "",
        "Generated by `python -m src.run_model_pipeline` (deterministic, seed "
        f"{seed}). Target: `Revenue`. Baseline purchase rate: {feats[TARGET].mean():.1%}.",
        "",
        "## Data split",
        "",
        "| Split | Sessions | Purchase rate |", "|---|---|---|",
        *[f"| {s} | {sizes[s]} | {rates[s]:.1%} |" for s in ("train", "val", "test")],
        "",
        "Stratified on the target. Model selection and the threshold check use **validation only**; "
        "the test set is scored once, at the end.",
        "",
        "## 1. Algorithm selection (validation, main model without PageValues)",
        "",
        *hdr,
        *[_fmt_metrics_row(a, selection[a]) for a in ALGORITHMS],
        "",
        f"Selected by validation PR-AUC: **{best_algo}**. PR-AUC is used because buyers are only ~15% of sessions.",
        "",
        "## 2. Main model vs PageValues comparison (test)",
        "",
        *hdr,
        _fmt_metrics_row(f"**Main** — {main_cfg['name']}", m_main),
        _fmt_metrics_row(f"Comparison — {cmp_cfg['name']}", m_cmp),
        "",
        f"Adding `PageValues` raises test PR-AUC by {m_cmp['pr_auc'] - m_main['pr_auc']:+.3f}. "
        "`PageValues` is computed from pages that led to transactions, so it is strongly tied to the "
        "outcome itself. The app therefore uses the **main model without PageValues**; the comparison "
        "model is reported here only.",
        "",
        "## 3. Fixed thresholds — validation check",
        "",
        f"Thresholds were set in advance (buyer ≥ {thresholds['buyer_from']:.0%}, "
        f"hesitant {thresholds['unlikely_below']:.0%}–{thresholds['buyer_from']:.0%}, "
        f"unlikely < {thresholds['unlikely_below']:.0%}) and were **not tuned**. "
        "This section only checks that they separate sessions meaningfully.",
        "",
        *_fmt_segments(val_segments),
        "",
        *[f"- {'✅' if ok else '⚠️'} {name}" for name, ok in checks],
        "",
        "## 4. Segments on test (main model)",
        "",
        *_fmt_segments(test_segments),
        "",
    ]
    with open(out_dir / "evaluation_report.md", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(report))

    # Structured copy of the evaluation, read by the Model Card agent (it never parses the markdown)
    # 6 decimals: formatting to 3 decimals or 0.1% then matches evaluation_report.md (no double rounding)
    def _r(d: dict) -> dict:
        return {k: round(float(v), 6) for k, v in d.items()}

    def _seg(table: pd.DataFrame) -> list[dict]:
        return [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}
                for r in table.to_dict("records")]

    test = parts["test"].assign(_p=test_main_proba)
    by_visitor = []
    for vt, g in test.groupby("VisitorType"):
        segs = pd.Series([decide_segment(p, thresholds) for p in g["_p"]])
        by_visitor.append({"visitor_type": vt, "sessions": int(len(g)),
                           "actual_purchase_rate": round(float(g[TARGET].mean()), 6),
                           "mean_predicted": round(float(g["_p"].mean()), 6),
                           "share_hesitant": round(float((segs == "hesitant").mean()), 6)})
    evaluation = {
        "target": TARGET, "seed": seed, "baseline_purchase_rate": round(float(feats[TARGET].mean()), 6),
        "rows": int(len(feats)),
        "split": {s: {"sessions": int(sizes[s]), "purchase_rate": round(float(parts[s][TARGET].mean()), 6)}
                  for s in ("train", "val", "test")},
        "selection_metric": "pr_auc",
        "selection": {a: _r(selection[a]) for a in ALGORITHMS},
        "selected_algorithm": best_algo,
        "test_metrics": {"main": _r(m_main), "comparison": _r(m_cmp)},
        "thresholds": thresholds,
        "threshold_checks": [{"check": n, "ok": ok} for n, ok in checks],
        "val_segments": _seg(val_segments),
        "test_segments": _seg(test_segments),
        "test_by_visitor_type": by_visitor,
        "main_features": num_main + cat_main,
        "excluded_from_main": main_cfg["drop_features"],
    }
    with open(out_dir / "evaluation.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(evaluation, indent=2) + "\n")

    for name in ("features.csv", "model.pkl", "model_with_pagevalues.pkl", "model_meta.json",
                 "evaluation_report.md", "evaluation.json"):
        log.info("Saved %s", (out_dir / name).relative_to(resolve(".")))
    log.info("Model pipeline finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
