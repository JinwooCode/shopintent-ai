"""ShopIntent AI — CrewAI Flow that runs the whole project end to end.

    analyst_stage ──► insights_stage ──► insights_gate ──"insights_ok"──►
                                              │
    handoff_gate ──"handoff_ok"──► scientist_stage ──► app_gate ──"ready"──►
         │                                               │
    intervention_stage ──► intervention_gate ──"agent_ok"──► finish
                                  │                      │
    ──────────────────────────────┴──────"failed"────────┴──► handle_failure   (any gate can fail)

Stages:
  analyst_stage    Data Analyst work: raw CSV -> clean_data.csv + dataset_contract.json
  insights_stage   EDA tool -> eda_stats.json + eda_report.html; Insights Analyst agent -> insights.md
                   (src/agents/insights_agent.py; reuses a saved agent draft when the stats are unchanged)
  insights_gate    all analyst outputs exist; every number in insights.md is traced to the EDA facts;
                   PageValues stays out; the gate catches an invented number; the fallback is valid
  handoff_gate     validates the handoff: the contract matches the clean data and all required features exist
  scientist_stage  Data Scientist work: features, model selection, models, evaluation report
  app_gate         confirms the Streamlit app can use the artifacts (files exist, model loads and scores,
                   main model excludes PageValues)
  intervention_stage  the intervention agent writes one coupon and one chat message for hesitant test
                      visits (src/agents/intervention_agent.py). It acts only after the rule has decided.
  intervention_gate   the agent contract holds: messages pass the output gate, buyer/unlikely visits and
                      inputs that contradict the rule are refused, bad drafts are caught, fallback works.
                      If the LLM is unavailable the fallback is used and recorded; that is not a failure.

The analyst and scientist stages run the deterministic Python pipelines; their crews are added later.
Every run writes a summary to logs/flow_last_run.json. On failure the process exits with code 1.

Usage (from the project root):
    python -m src.flow
    python -m src.flow --no-llm    # no API calls: agents reuse a saved draft or use their fallback
    python -m src.flow --refresh-insights   # ask the insights agent again even if the stats are unchanged
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
from src.agents.insights_agent import fallback_draft as insights_fallback
from src.agents.insights_agent import save as save_insights
from src.agents.insights_agent import validate_insights, write_insights
from src.tools.eda_tools import build_eda_report, compute_eda_stats, save_stats, stats_to_facts
from src.agents.intervention_agent import (ACTIONS, AgentDraft, AgentInputError, fallback_draft,
                                           input_from_row, validate_message, write_message)
from src.tools.data_tools import DataValidationError, load_contract, validate_against_contract
from src.tools.decision import decide
from src.tools.model_tools import CATEGORICAL, check_required_features, engineer_features
from src.utils import get_logger, load_config, resolve

log = get_logger("flow")

ANALYST_OUTPUTS = ["clean_data.csv", "dataset_contract.json", "eda_stats.json", "eda_report.html",
                   "insights.md", "insights.json"]
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
    agent_sources: dict = Field(default_factory=dict)   # coupon/chat -> "agent" | "fallback"


class ShopIntentFlow(Flow[ShopIntentState]):
    use_llm: bool = True            # False (--no-llm): agents use their fallback only, no API calls
    refresh_insights: bool = False  # True (--refresh-insights): call the insights agent even if a draft exists

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
        missing = self._missing("analyst_dir", ANALYST_OUTPUTS[:2])
        details = missing or [f"exit code {rc}"]
        return self._record("analyst_stage", rc == 0 and not missing, t0, details)

    # ------------------------------------------------------------------ 1b. EDA + Insights Analyst agent
    @listen(analyst_stage)
    def insights_stage(self, analyst_ok: bool):
        """EDA tool (Python) -> eda_stats.json + eda_report.html, then the agent -> insights.md."""
        if not analyst_ok:
            return False
        log.info("[flow] ===== Stage 1b: EDA + Insights Analyst agent (%s) =====",
                 "LLM" if self.use_llm else "LLM off, fallback only")
        t0 = time.perf_counter()
        cfg = load_config()
        a_dir = resolve(cfg["paths"]["analyst_dir"])
        try:
            df = pd.read_csv(a_dir / "clean_data.csv")
            self._eda_stats = compute_eda_stats(df)
            save_stats(self._eda_stats, a_dir / "eda_stats.json")
            build_eda_report(self._eda_stats, load_contract(a_dir / "dataset_contract.json"),
                             a_dir / "eda_report.html")
            self._insights = write_insights(self._eda_stats, cfg, use_llm=self.use_llm,
                                            refresh=self.refresh_insights)
            save_insights(self._insights, cfg)
        except Exception as e:
            log.exception("Insights stage crashed")
            return self._record("insights_stage", False, t0, [f"{type(e).__name__}: {e}"])
        r = self._insights
        detail = f"insights: source={r.source}" + (" (reused, no LLM call)" if r.reused else "") + \
                 (f" ({'; '.join(r.problems)})" if r.problems else "")
        return self._record("insights_stage", True, t0, ["eda_report.html and eda_stats.json written", detail])

    @router(insights_stage, emit=["insights_ok", "failed"])
    def insights_gate(self, stage_ok: bool):
        """The analyst outputs are complete and the insights keep to their contract."""
        if not stage_ok:
            return "failed"
        log.info("[flow] ===== Insights validation =====")
        t0 = time.perf_counter()
        cfg = load_config()
        problems, passed = self._missing("analyst_dir", ANALYST_OUTPUTS), []
        try:
            facts = stats_to_facts(self._eda_stats)
            errs = validate_insights(self._insights.draft, facts)
            if errs:
                problems += [f"insights.md fails the gate: {e}" for e in errs]
            else:
                passed.append(f"insights.md ({self._insights.source}): {len(self._insights.draft.insights)} "
                              f"insights, every number traced to the EDA facts")
            if any("pagevalues" in f.lower() for f in facts):
                problems.append("PageValues leaked into the facts given to the agent")
            else:
                passed.append("PageValues is not in the agent's facts")
            # the gate must catch an invented number and the fallback must be valid
            bad = self._insights.draft.model_copy(deep=True)
            bad.insights[0].finding += " The rate was 99.9%."
            if validate_insights(bad, facts):
                passed.append("Gate rejects an invented number")
            else:
                problems.append("Gate let an invented number through")
            if validate_insights(insights_fallback(self._eda_stats), facts):
                problems.append("Fallback insights fail the gate")
            else:
                passed.append("Fallback insights pass the gate")
            self.state.agent_sources["insights"] = self._insights.source
        except Exception as e:
            problems.append(f"{type(e).__name__}: {e}")
        if problems:
            for p in problems:
                log.error("  FAIL  %s", p)
            self._record("insights_gate", False, t0, problems)
            return "failed"
        for p in passed:
            log.info("  OK  %s", p)
        self._record("insights_gate", True, t0, passed)
        return "insights_ok"

    # ------------------------------------------------------------------ handoff validation
    @router("insights_ok", emit=["handoff_ok", "failed"])
    def handoff_gate(self, *_):
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

    # ------------------------------------------------------------------ 3. Intervention agent
    @listen("ready")
    def intervention_stage(self):
        """Run the agent on one coupon visit and one chat visit from the test set."""
        log.info("[flow] ===== Stage 3: Intervention agent (%s) =====",
                 "LLM" if self.use_llm else "LLM off, fallback only")
        t0 = time.perf_counter()
        cfg = load_config()
        sci_dir = resolve(cfg["paths"]["scientist_dir"])
        try:
            model = joblib.load(sci_dir / "model.pkl")
            feats = pd.read_csv(sci_dir / "features.csv")
            for col in CATEGORICAL:
                feats[col] = feats[col].astype(str)
            test = feats[feats["split"] == "test"].reset_index(drop=True)
            proba = model.predict_proba(test)[:, 1]
            self._agent_cases = {}   # first test visit the rule sends to coupon, and to chat
            for i, p in enumerate(proba):
                action = decide(p, test.at[i, "VisitorType"], cfg)["intervention"]
                if action in ACTIONS and action not in self._agent_cases:
                    self._agent_cases[action] = input_from_row(i, test.loc[i], p, cfg)
            todo = [a for a in ACTIONS if a in self._agent_cases]
            self._agent_outputs = [write_message(self._agent_cases[a], cfg, use_llm=self.use_llm) for a in todo]
        except Exception as e:
            log.exception("Intervention stage crashed")
            return self._record("intervention_stage", False, t0, [f"{type(e).__name__}: {e}"])
        details = [f"{m.intervention}: source={m.source}" + (f" ({'; '.join(m.problems)})" if m.problems else "")
                   for m in self._agent_outputs]
        return self._record("intervention_stage", len(self._agent_outputs) == len(ACTIONS), t0,
                            details or ["No hesitant coupon/chat visits found in the test set"])

    @router(intervention_stage, emit=["agent_ok", "failed"])
    def intervention_gate(self, stage_ok: bool):
        """The agent contract holds: it acts only after the rule, every shown message passes the gate,
        bad drafts are caught, and the fallback works."""
        if not stage_ok:
            return "failed"
        log.info("[flow] ===== Intervention agent validation =====")
        t0 = time.perf_counter()
        cfg = load_config()
        problems, passed = [], []
        try:
            # 1. Every message the app would show passes the output gate
            for m in self._agent_outputs:
                code = m.coupon_code
                errs = validate_message(AgentDraft(headline=m.headline, message=m.message, cta=m.cta),
                                        m.intervention, cfg, code)
                if errs:
                    problems.append(f"{m.intervention} message fails the gate: {errs}")
                elif m.intervention == "coupon" and m.discount_pct != cfg["intervention_agent"]["coupon"]["discount_pct"]:
                    problems.append("Coupon discount differs from config")
                else:
                    passed.append(f"{m.intervention} message ({m.source}) passes the output gate")

            # 2. The agent refuses visits the rule did not send to it
            base = next(iter(self._agent_cases.values()))
            for p, label in ((min(0.99, cfg["thresholds"]["buyer_from"] + 0.1), "buyer"),
                             (max(0.0, cfg["thresholds"]["unlikely_below"] - 0.1), "unlikely")):
                bad = base.model_copy(update={"probability": p})
                try:
                    write_message(bad, cfg, use_llm=False)
                    problems.append(f"Agent accepted a {label} visit")
                except AgentInputError:
                    passed.append(f"Agent refuses {label} visits")
            wrong = base.model_copy(update={"intervention": "chat" if base.intervention == "coupon" else "coupon"})
            try:
                write_message(wrong, cfg, use_llm=False)
                problems.append("Agent accepted an input that contradicts the rule")
            except AgentInputError:
                passed.append("Agent refuses input that contradicts the rule")

            # 3. The gate catches bad drafts
            pct = cfg["intervention_agent"]["coupon"]["discount_pct"]
            bad_coupon = AgentDraft(headline="Big sale", message=f"Take {pct + 15}% off now!", cta="Shop")
            bad_chat = AgentDraft(headline="Hi", message="Use code WELCOME for 20% off", cta="Chat")
            if validate_message(bad_coupon, "coupon", cfg, "SHOP10-TEST") and \
                    validate_message(bad_chat, "chat", cfg, None):
                passed.append("Gate rejects a wrong discount and a discount inside a chat")
            else:
                problems.append("Gate let a bad draft through")

            # 4. The fallback is valid for both actions
            for a in ACTIONS:
                code = "SHOP10-TEST" if a == "coupon" else None
                if validate_message(fallback_draft(a, cfg, code), a, cfg, code):
                    problems.append(f"Fallback template for {a} fails the gate")
            passed.append("Fallback templates pass the gate")

            out = resolve(cfg["paths"]["agents_dir"]) / "intervention_samples.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps([m.model_dump() for m in self._agent_outputs], indent=2,
                                      ensure_ascii=False) + "\n", encoding="utf-8")
            self.state.agent_sources = {m.intervention: m.source for m in self._agent_outputs}
        except Exception as e:
            problems.append(f"{type(e).__name__}: {e}")

        if problems:
            for p in problems:
                log.error("  FAIL  %s", p)
            self._record("intervention_gate", False, t0, problems)
            return "failed"
        for p in passed:
            log.info("  OK  %s", p)
        self._record("intervention_gate", True, t0, passed)
        return "agent_ok"

    # ------------------------------------------------------------------ endings
    @listen("agent_ok")
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
    flow.use_llm = "--no-llm" not in argv
    flow.refresh_insights = "--refresh-insights" in argv
    if "--plot" in argv:
        path = flow.plot("shopintent_flow.html", show=True)  # CrewAI writes it to a temp folder
        log.info("[flow] Flow diagram opened in the browser (%s)", path)
    flow.kickoff()
    return 0 if flow.state.status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
