"""ShopIntent AI — CrewAI Flow that runs the whole project end to end.

    analyst_stage ──► handoff_gate ──"handoff_ok"──► scientist_stage ──► app_gate ──"ready"──► finish
                           │                                                │
                           └────────────────"failed"────────────────────────┴──► handle_failure

Stages:
  analyst_stage    Data Analyst work: raw CSV -> clean_data.csv + dataset_contract.json
  handoff_gate     validates the handoff: the contract matches the clean data and all required features exist
  scientist_stage  Data Scientist work: features, model selection, models, evaluation report
  app_gate         confirms the Streamlit app can use the artifacts (files exist, model loads and scores,
                   main model excludes PageValues)

This first version runs the deterministic Python pipelines. The agent crews are added later in the same
stages. Every run writes a summary to logs/flow_last_run.json. On failure the process exits with code 1.

Usage (from the project root):
    python -m src.flow
    python -m src.flow --plot      # also open an interactive diagram of the flow in the browser
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

# Keep runs local and non-interactive: no trace upload prompt at the end of each run.
# Set before crewai is imported; an explicit value in the environment / .env still wins.
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

import joblib
import pandas as pd
from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel, Field

from src import run_data_pipeline, run_model_pipeline
from src.tools.data_tools import DataValidationError, load_contract, validate_against_contract
from src.tools.decision import decide
from src.tools.model_tools import CATEGORICAL, check_required_features, engineer_features
from src.utils import get_logger, load_config, resolve

log = get_logger("flow")

ANALYST_OUTPUTS = ["clean_data.csv", "dataset_contract.json"]
SCIENTIST_OUTPUTS = ["features.csv", "model.pkl", "model_with_pagevalues.pkl",
                     "model_meta.json", "evaluation_report.md"]


class StepResult(BaseModel):
    name: str
    ok: bool
    seconds: float
    details: list[str] = Field(default_factory=list)


class ShopIntentState(BaseModel):
    started_at: str = ""
    status: str = "running"          # running | success | failed
    failed_step: str | None = None
    steps: list[StepResult] = Field(default_factory=list)
    test_metrics: dict = Field(default_factory=dict)


class ShopIntentFlow(Flow[ShopIntentState]):

    # ------------------------------------------------------------------ helpers
    def _record(self, name: str, ok: bool, t0: float, details: list[str] | None = None) -> bool:
        self.state.steps.append(StepResult(name=name, ok=ok, seconds=round(time.perf_counter() - t0, 2),
                                           details=details or []))
        if not ok:
            self.state.failed_step = name
        log.info("[flow] %s  %s", "OK  " if ok else "FAIL", name)
        return ok

    def _missing(self, folder_key: str, names: list[str]) -> list[str]:
        folder = resolve(load_config()["paths"][folder_key])
        return [f"Missing artifact: {folder.name}/{n}" for n in names if not (folder / n).exists()]

    # ------------------------------------------------------------------ 1. Data Analyst
    @start()
    def analyst_stage(self):
        self.state.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        log.info("[flow] ===== Stage 1/2: Data Analyst =====")
        t0 = time.perf_counter()
        try:
            rc = run_data_pipeline.main()
        except Exception as e:  # unexpected crash -> still fail gracefully
            log.exception("Data pipeline crashed")
            return self._record("analyst_stage", False, t0, [f"{type(e).__name__}: {e}"])
        missing = self._missing("analyst_dir", ANALYST_OUTPUTS)
        details = missing or [f"exit code {rc}"]
        return self._record("analyst_stage", rc == 0 and not missing, t0, details)

    # ------------------------------------------------------------------ handoff validation
    @router(analyst_stage, emit=["handoff_ok", "failed"])
    def handoff_gate(self, analyst_ok: bool):
        if not analyst_ok:
            return "failed"
        log.info("[flow] ===== Handoff validation: Data Analyst -> Data Scientist =====")
        t0 = time.perf_counter()
        cfg = load_config()
        analyst_dir = resolve(cfg["paths"]["analyst_dir"])
        try:
            df = pd.read_csv(analyst_dir / "clean_data.csv")
            contract = load_contract(analyst_dir / "dataset_contract.json")
            passed = validate_against_contract(df, contract)
            check_required_features(engineer_features(df), drop=cfg["models"]["main"]["drop_features"])
            passed.append("All required model features exist")
        except DataValidationError as e:
            for err in e.errors:
                log.error("  FAIL  %s", err)
            self._record("handoff_gate", False, t0, e.errors)
            return "failed"
        except Exception as e:
            log.exception("Handoff validation crashed")
            self._record("handoff_gate", False, t0, [f"{type(e).__name__}: {e}"])
            return "failed"
        self._record("handoff_gate", True, t0, passed)
        return "handoff_ok"

    # ------------------------------------------------------------------ 2. Data Scientist
    @listen("handoff_ok")
    def scientist_stage(self):
        log.info("[flow] ===== Stage 2/2: Data Scientist =====")
        t0 = time.perf_counter()
        try:
            rc = run_model_pipeline.main()
        except Exception as e:
            log.exception("Model pipeline crashed")
            return self._record("scientist_stage", False, t0, [f"{type(e).__name__}: {e}"])
        missing = self._missing("scientist_dir", SCIENTIST_OUTPUTS)
        details = missing or [f"exit code {rc}"]
        return self._record("scientist_stage", rc == 0 and not missing, t0, details)

    # ------------------------------------------------------------------ app readiness
    @router(scientist_stage, emit=["ready", "failed"])
    def app_gate(self, scientist_ok: bool):
        if not scientist_ok:
            return "failed"
        log.info("[flow] ===== App readiness check =====")
        t0 = time.perf_counter()
        cfg = load_config()
        sci_dir = resolve(cfg["paths"]["scientist_dir"])
        problems, passed = [], []
        try:
            meta = json.loads((sci_dir / "model_meta.json").read_text(encoding="utf-8"))
            main_feats = meta["main_model"]["numeric_features"] + meta["main_model"]["categorical_features"]
            leaked = [f for f in cfg["models"]["main"]["drop_features"] if f in main_feats]
            if leaked:
                problems.append(f"Main model uses excluded features: {leaked}")
            else:
                passed.append("Main model excludes " + ", ".join(cfg["models"]["main"]["drop_features"]))

            # Load the data exactly as the app does, then run the full chain the app shows:
            # probability -> segment -> intervention.
            model = joblib.load(sci_dir / "model.pkl")
            feats = pd.read_csv(sci_dir / "features.csv")
            for col in CATEGORICAL:
                feats[col] = feats[col].astype(str)
            sample = feats[feats["split"] == "test"].head(50)
            proba = model.predict_proba(sample)[:, 1]
            if not ((proba >= 0) & (proba <= 1)).all():
                problems.append("Model returned probabilities outside [0, 1]")
            else:
                decisions = [decide(p, v, cfg) for p, v in zip(proba, sample["VisitorType"])]
                seg_counts = pd.Series([d["segment"] for d in decisions]).value_counts().to_dict()
                passed.append(f"model.pkl loads and scores {len(sample)} test sessions "
                              f"(segments: {seg_counts})")
            self.state.test_metrics = meta.get("test_metrics", {})
        except Exception as e:
            problems.append(f"{type(e).__name__}: {e}")

        if problems:
            for p in problems:
                log.error("  FAIL  %s", p)
            self._record("app_gate", False, t0, problems)
            return "failed"
        for p in passed:
            log.info("  OK  %s", p)
        self._record("app_gate", True, t0, passed)
        return "ready"

    # ------------------------------------------------------------------ endings
    @listen("ready")
    def finish(self):
        self.state.status = "success"
        self._write_summary()
        m = self.state.test_metrics.get("main", {})
        log.info("[flow] Flow finished successfully. Main model test PR-AUC %s | ROC-AUC %s",
                 m.get("pr_auc"), m.get("roc_auc"))
        log.info("[flow] Next: streamlit run app/streamlit_app.py")
        return self.state.status

    @listen("failed")
    def handle_failure(self):
        self.state.status = "failed"
        self._write_summary()
        log.error("[flow] Flow stopped at '%s'. Later stages were skipped, so no half-built artifacts "
                  "were handed on. Fix the errors above and run again.", self.state.failed_step)
        return self.state.status

    def _write_summary(self) -> None:
        out = resolve(load_config()["paths"]["log_dir"]) / "flow_last_run.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.state.model_dump_json(indent=2) + "\n", encoding="utf-8")
        log.info("[flow] Run summary saved to %s", out.relative_to(resolve(".")))


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    flow = ShopIntentFlow()
    if "--plot" in argv:
        path = flow.plot("shopintent_flow.html", show=True)  # CrewAI writes it to a temp folder
        log.info("[flow] Flow diagram opened in the browser (%s)", path)
    flow.kickoff()
    return 0 if flow.state.status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
