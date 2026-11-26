"""ShopIntent AI — CrewAI Flow that runs the whole project end to end.

    analyst_stage ──► validator_stage ──► validator_gate ──"data_ok"──►
    contract_stage ──► contract_gate ──"contract_ok"──►
    insights_stage ──► insights_gate ──"insights_ok"──►
    handoff_gate ──"handoff_ok"──► scientist_stage ──► feature_stage ──► feature_gate ──"features_ok"──►
    evaluator_stage ──► evaluator_gate ──"eval_ok"──►
    model_card_stage ──► model_card_gate ──"card_ok"──►
    app_gate ──"ready"──►
    intervention_stage ──► intervention_gate ──"agent_ok"──► finish

    every gate can return "failed" ──► handle_failure

Stages:
  analyst_stage    Data Analyst work: raw CSV -> clean_data.csv + dataset_contract.json + data_quality.json
  validator_stage  Data Validator agent explains data_quality.json in data_quality.md (also when the
                   pipeline failed, so the failure is explained); src/agents/validator_agent.py
  validator_gate   the tools say the data is ready; the report's verdict matches the tools, explains every
                   cleaning action and invents no numbers; the agent cannot overrule the verdict
  contract_stage   Contract Author agent documents every column (meaning, how to use) -> column_docs in
                   dataset_contract.json + dataset_contract.md; src/agents/contract_agent.py
  contract_gate    the machine-checked part of the contract is unchanged by the agent and still matches the
                   data; every column is documented; code columns, PageValues and the target are described
                   correctly; the gate catches a missing column; the fallback is valid
  insights_stage   EDA tool -> eda_stats.json + eda_report.html; Insights Analyst agent -> insights.md
                   (src/agents/insights_agent.py; reuses a saved agent draft when the stats are unchanged)
  insights_gate    all analyst outputs exist; every number in insights.md is traced to the EDA facts;
                   PageValues stays out; the gate catches an invented number; the fallback is valid
  handoff_gate     validates the handoff: the contract matches the clean data and all required features exist
  scientist_stage  Data Scientist work: features, model selection, models, evaluation report + evaluation.json
  feature_stage    permutation importance on validation (Python) -> feature_importance.json; Feature Analyst
                   agent explains it -> features.md (src/agents/feature_agent.py)
  feature_gate     importance is reproducible (recomputed) and excludes PageValues; the agent's top 5, order,
                   engineered verdicts and effect directions match the code; numbers traced; fallback valid
  evaluator_stage  Model Evaluator agent explains evaluation.json -> evaluation_interpretation.md
                   (src/agents/evaluator_agent.py)
  evaluator_gate   a fixed rule recommends keeping the main model (threshold checks pass, PR-AUC >= 2x the
                   random baseline); the agent's recommendation and algorithm match the code; it never claims
                   the thresholds were tuned; numbers traced; fallback valid
  model_card_stage Model Card agent writes purpose, uses, training data, metrics summary, limitations and
                   ethics from evaluation.json -> model_card.md (tables by code); src/agents/model_card_agent.py
  model_card_gate  all brief sections present; the main model's PR-AUC is quoted (never the PageValues model's);
                   the simulation limit and intervention ethics are covered; numbers traced; fallback valid
  app_gate         confirms the Streamlit app can use the artifacts (files exist, model loads and scores,
                   main model excludes PageValues)
  intervention_stage  the intervention agent writes one coupon and one chat message for hesitant test
                      visits (src/agents/intervention_agent.py). It acts only after the rule has decided.
  intervention_gate   the agent contract holds: messages pass the output gate, buyer/unlikely visits and
                      inputs that contradict the rule are refused, bad drafts are caught, fallback works.
                      If the LLM is unavailable the fallback is used and recorded; that is not a failure.

Python tools do all computing; agents explain and write, and each agent has a contract, a gate and a fallback.
Every run writes a summary to logs/flow_last_run.json. On failure the process exits with code 1.

Usage (from the project root):
    python -m src.flow
    python -m src.flow --no-llm          # no API calls: agents reuse a saved draft or use their fallback
    python -m src.flow --refresh-agents  # ask the analyst agents again even if their inputs are unchanged
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
from src.agents.common import sha_of
from src.agents.contract_agent import fallback_docs as contract_fallback
from src.agents.contract_agent import machine_part as contract_machine_part
from src.agents.contract_agent import make_input as contract_input
from src.agents.contract_agent import save as save_contract_docs
from src.agents.contract_agent import validate_docs, write_contract_docs
from src.agents.feature_agent import fallback_draft as features_fallback
from src.agents.feature_agent import make_input as feature_input
from src.agents.feature_agent import save as save_features
from src.agents.feature_agent import validate_features, write_feature_analysis
from src.tools.importance_tools import compute_importance, save_importance
from src.agents.evaluator_agent import fallback_draft as evaluator_fallback
from src.agents.evaluator_agent import make_input as evaluator_input
from src.agents.evaluator_agent import save as save_interpretation
from src.agents.evaluator_agent import validate_evaluation, write_interpretation
from src.agents.model_card_agent import fallback_draft as model_card_fallback
from src.agents.model_card_agent import make_input as model_card_input
from src.agents.model_card_agent import save as save_model_card
from src.agents.model_card_agent import validate_card, write_model_card
from src.agents.validator_agent import fallback_draft as quality_fallback
from src.agents.validator_agent import make_input as validator_input
from src.agents.validator_agent import save as save_quality
from src.agents.validator_agent import validate_quality, write_quality_report
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
PAGEVALUES_NAME = "PageValues"

ANALYST_OUTPUTS = ["clean_data.csv", "dataset_contract.json", "data_quality.json", "data_quality.md",
                   "data_quality_report.json", "contract_docs.json", "dataset_contract.md",
                   "eda_stats.json", "eda_report.html", "insights.md", "insights.json"]
SCIENTIST_OUTPUTS = ["features.csv", "model.pkl", "model_with_pagevalues.pkl",
                     "model_meta.json", "evaluation_report.md", "evaluation.json"]


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
    refresh_agents: bool = False    # True (--refresh-agents): call the analyst agents even if a saved draft fits

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

    # ------------------------------------------------------------------ 1a. Data Validator agent
    @listen(analyst_stage)
    def validator_stage(self, analyst_ok: bool):
        """The agent explains data_quality.json in data_quality.md. Runs even when the pipeline failed,
        so the failure is explained too; the gate then stops the flow."""
        log.info("[flow] ===== Stage 1a: Data Validator agent (%s) =====",
                 "LLM" if self.use_llm else "LLM off, fallback only")
        t0 = time.perf_counter()
        cfg = load_config()
        q_path = resolve(cfg["paths"]["analyst_dir"]) / "data_quality.json"
        # never leave a report from an earlier run behind if this one breaks
        (q_path.parent / "data_quality.md").unlink(missing_ok=True)
        if not q_path.exists():
            return self._record("validator_stage", False, t0, ["data_quality.json was not written"])
        try:
            self._quality = json.loads(q_path.read_text(encoding="utf-8"))
            self._quality_report = write_quality_report(self._quality, cfg, use_llm=self.use_llm,
                                                        refresh=self.refresh_agents)
            save_quality(self._quality_report, self._quality, cfg)
        except Exception as e:
            log.exception("Validator stage crashed")
            return self._record("validator_stage", False, t0, [f"{type(e).__name__}: {e}"])
        r = self._quality_report
        detail = f"data_quality.md: source={r.source}, verdict={r.draft.verdict}" + \
                 (" (reused, no LLM call)" if r.reused else "") + (f" ({'; '.join(r.problems)})" if r.problems else "")
        # a failed pipeline is recorded as failed by analyst_stage; this stage only reports its own work
        return self._record("validator_stage", True, t0, [detail]) if analyst_ok else True

    @router(validator_stage, emit=["data_ok", "failed"])
    def validator_gate(self, stage_ok: bool):
        """The data is ready (decided by the tools) and the report keeps to its contract."""
        if not stage_ok or self.state.failed_step:
            return "failed"
        log.info("[flow] ===== Data quality validation =====")
        t0 = time.perf_counter()
        problems, passed = [], []
        try:
            q, r = self._quality, self._quality_report
            inp = validator_input(q)
            if q["status"] != "ready":
                problems.append("Data is not ready, see artifacts/analyst/data_quality.md")
            errs = validate_quality(r.draft, inp)
            if errs:
                problems += [f"data_quality.md fails the gate: {e}" for e in errs]
            else:
                passed.append(f"data_quality.md ({r.source}): verdict matches the tools, one explanation per "
                              f"cleaning action, every number traced")
            flipped = r.draft.model_copy(update={"verdict": "not_ready" if r.draft.verdict == "ready" else "ready"})
            if validate_quality(flipped, inp):
                passed.append("Gate rejects a verdict that contradicts the tools")
            else:
                problems.append("Gate let the agent overrule the tools' verdict")
            if validate_quality(quality_fallback(q), inp):
                problems.append("Fallback data quality report fails the gate")
            else:
                passed.append("Fallback report passes the gate")
            self.state.agent_sources["data_quality"] = r.source
        except Exception as e:
            problems.append(f"{type(e).__name__}: {e}")
        if problems:
            for p in problems:
                log.error("  FAIL  %s", p)
            self._record("validator_gate", False, t0, problems)
            return "failed"
        for p in passed:
            log.info("  OK  %s", p)
        self._record("validator_gate", True, t0, passed)
        return "data_ok"

    # ------------------------------------------------------------------ 1b. Contract Author agent
    @listen("data_ok")
    def contract_stage(self, *_):
        """The agent documents every column; the docs are added to dataset_contract.json + dataset_contract.md."""
        log.info("[flow] ===== Stage 1b: Contract Author agent (%s) =====",
                 "LLM" if self.use_llm else "LLM off, fallback only")
        t0 = time.perf_counter()
        cfg = load_config()
        a_dir = resolve(cfg["paths"]["analyst_dir"])
        try:
            contract = load_contract(a_dir / "dataset_contract.json")
            self._contract_sha_before = sha_of(contract_machine_part(contract))
            self._contract_docs = write_contract_docs(contract, cfg, use_llm=self.use_llm,
                                                      refresh=self.refresh_agents)
            save_contract_docs(self._contract_docs, contract, cfg)
        except Exception as e:
            log.exception("Contract stage crashed")
            return self._record("contract_stage", False, t0, [f"{type(e).__name__}: {e}"])
        r = self._contract_docs
        detail = f"column docs: source={r.source}" + (" (reused, no LLM call)" if r.reused else "") + \
                 (f" ({'; '.join(r.problems)})" if r.problems else "")
        return self._record("contract_stage", True, t0, [detail])

    @router(contract_stage, emit=["contract_ok", "failed"])
    def contract_gate(self, stage_ok: bool):
        """The agent only added documentation: the machine part is unchanged and still matches the data."""
        if not stage_ok:
            return "failed"
        log.info("[flow] ===== Contract validation =====")
        t0 = time.perf_counter()
        cfg = load_config()
        a_dir = resolve(cfg["paths"]["analyst_dir"])
        problems, passed = [], []
        try:
            contract = load_contract(a_dir / "dataset_contract.json")
            inp = contract_input(contract)
            if sha_of(contract_machine_part(contract)) != self._contract_sha_before:
                problems.append("The machine-checked part of the contract changed during the agent stage")
            else:
                passed.append("Machine-checked part of the contract unchanged by the agent")
            validate_against_contract(pd.read_csv(a_dir / "clean_data.csv"), contract)
            passed.append("Clean data still matches the documented contract")
            docs = self._contract_docs.docs
            errs = validate_docs(docs, inp)
            if errs or set(contract.get("column_docs", {})) != set(inp.columns):
                problems += [f"Column docs fail the gate: {e}" for e in errs] or ["column_docs incomplete in the json"]
            else:
                passed.append(f"Every one of the {len(inp.columns)} columns documented ({self._contract_docs.source})")
            broken = docs.model_copy(update={"columns": docs.columns[1:]})
            if validate_docs(broken, inp):
                passed.append("Gate rejects docs with a missing column")
            else:
                problems.append("Gate let docs with a missing column through")
            if validate_docs(contract_fallback(contract), inp):
                problems.append("Fallback column docs fail the gate")
            else:
                passed.append("Fallback docs pass the gate")
            self.state.agent_sources["contract_docs"] = self._contract_docs.source
        except DataValidationError as e:
            problems += e.errors
        except Exception as e:
            problems.append(f"{type(e).__name__}: {e}")
        if problems:
            for p in problems:
                log.error("  FAIL  %s", p)
            self._record("contract_gate", False, t0, problems)
            return "failed"
        for p in passed:
            log.info("  OK  %s", p)
        self._record("contract_gate", True, t0, passed)
        return "contract_ok"

    # ------------------------------------------------------------------ 1c. EDA + Insights Analyst agent
    @listen("contract_ok")
    def insights_stage(self, *_):
        """EDA tool (Python) -> eda_stats.json + eda_report.html, then the agent -> insights.md."""
        log.info("[flow] ===== Stage 1c: EDA + Insights Analyst agent (%s) =====",
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
                                            refresh=self.refresh_agents)
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

    # ------------------------------------------------------------------ 2a. Feature Analyst agent
    @listen(scientist_stage)
    def feature_stage(self, scientist_ok: bool):
        """Permutation importance (Python) -> feature_importance.json; the agent explains it in features.md."""
        if not scientist_ok:
            return False
        log.info("[flow] ===== Stage 2a: Feature Analyst agent (%s) =====",
                 "LLM" if self.use_llm else "LLM off, fallback only")
        t0 = time.perf_counter()
        cfg = load_config()
        s_dir = resolve(cfg["paths"]["scientist_dir"])
        (s_dir / "features.md").unlink(missing_ok=True)
        try:
            m = json.loads((s_dir / "model_meta.json").read_text(encoding="utf-8"))["main_model"]
            self._importance_args = (s_dir, m["numeric_features"], m["categorical_features"], cfg["seed"])
            self._importance = compute_importance(joblib.load(s_dir / "model.pkl"),
                                                  pd.read_csv(s_dir / "features.csv"), *self._importance_args[1:])
            save_importance(self._importance, s_dir / "feature_importance.json")
            self._features = write_feature_analysis(self._importance, cfg, use_llm=self.use_llm,
                                                    refresh=self.refresh_agents)
            save_features(self._features, self._importance, cfg)
        except Exception as e:
            log.exception("Feature stage crashed")
            return self._record("feature_stage", False, t0, [f"{type(e).__name__}: {e}"])
        r = self._features
        detail = f"features.md: source={r.source}, top 3 {[t.feature for t in r.draft.top_features[:3]]}" + \
                 (" (reused, no LLM call)" if r.reused else "") + (f" ({'; '.join(r.problems)})" if r.problems else "")
        return self._record("feature_stage", True, t0, ["feature_importance.json written", detail])

    @router(feature_stage, emit=["features_ok", "failed"])
    def feature_gate(self, stage_ok: bool):
        """Importance is reproducible, excludes PageValues, and the analysis keeps to its contract."""
        if not stage_ok:
            return "failed"
        log.info("[flow] ===== Feature analysis validation =====")
        t0 = time.perf_counter()
        problems, passed = [], []
        try:
            s_dir, num, cat, seed = self._importance_args
            again = compute_importance(joblib.load(s_dir / "model.pkl"), pd.read_csv(s_dir / "features.csv"),
                                       num, cat, seed)
            if again != self._importance:
                problems.append("Feature importance is not reproducible (two runs differ)")
            else:
                passed.append("Feature importance is reproducible (recomputed, identical)")
            if any(PAGEVALUES_NAME == r["feature"] for r in self._importance["features"]):
                problems.append("PageValues appears among the main model's features")
            inp = feature_input(self._importance)
            d = self._features.draft
            errs = validate_features(d, inp)
            if errs:
                problems += [f"features.md fails the gate: {e}" for e in errs]
            else:
                passed.append(f"features.md ({self._features.source}): top {len(inp.top)} and engineered verdicts "
                              f"match the code, directions agree, every number traced")
            swapped = d.model_copy(update={"top_features": d.top_features[::-1]})
            if validate_features(swapped, inp):
                passed.append("Gate rejects a wrong feature ranking")
            else:
                problems.append("Gate let a wrong feature ranking through")
            if validate_features(features_fallback(self._importance), inp):
                problems.append("Fallback feature analysis fails the gate")
            else:
                passed.append("Fallback feature analysis passes the gate")
            self.state.agent_sources["features"] = self._features.source
        except Exception as e:
            problems.append(f"{type(e).__name__}: {e}")
        if problems:
            for p in problems:
                log.error("  FAIL  %s", p)
            self._record("feature_gate", False, t0, problems)
            return "failed"
        for p in passed:
            log.info("  OK  %s", p)
        self._record("feature_gate", True, t0, passed)
        return "features_ok"

    # ------------------------------------------------------------------ 2b. Model Evaluator agent
    @listen("features_ok")
    def evaluator_stage(self, *_):
        """The agent explains evaluation.json in evaluation_interpretation.md."""
        log.info("[flow] ===== Stage 2b: Model Evaluator agent (%s) =====",
                 "LLM" if self.use_llm else "LLM off, fallback only")
        t0 = time.perf_counter()
        cfg = load_config()
        s_dir = resolve(cfg["paths"]["scientist_dir"])
        (s_dir / "evaluation_interpretation.md").unlink(missing_ok=True)
        try:
            self._evaluation = json.loads((s_dir / "evaluation.json").read_text(encoding="utf-8"))
            self._interpretation = write_interpretation(self._evaluation, cfg, use_llm=self.use_llm,
                                                        refresh=self.refresh_agents)
            save_interpretation(self._interpretation, cfg)
        except Exception as e:
            log.exception("Evaluator stage crashed")
            return self._record("evaluator_stage", False, t0, [f"{type(e).__name__}: {e}"])
        r = self._interpretation
        detail = f"evaluation_interpretation.md: source={r.source}, recommendation={r.draft.recommendation}" + \
                 (" (reused, no LLM call)" if r.reused else "") + (f" ({'; '.join(r.problems)})" if r.problems else "")
        return self._record("evaluator_stage", True, t0, [detail])

    @router(evaluator_stage, emit=["eval_ok", "failed"])
    def evaluator_gate(self, stage_ok: bool):
        """The rule recommends keeping the main model and the interpretation keeps to its contract."""
        if not stage_ok:
            return "failed"
        log.info("[flow] ===== Evaluation interpretation validation =====")
        t0 = time.perf_counter()
        problems, passed = [], []
        try:
            inp = evaluator_input(self._evaluation)
            d = self._interpretation.draft
            if inp.expected_recommendation != "keep_main_model":
                problems.append("The recommendation rule says 'investigate': a threshold check failed or the main "
                                "model is not at least twice the random baseline. See evaluation_interpretation.md")
            errs = validate_evaluation(d, inp)
            if errs:
                problems += [f"evaluation_interpretation.md fails the gate: {e}" for e in errs]
            else:
                passed.append(f"evaluation_interpretation.md ({self._interpretation.source}): recommendation and "
                              f"algorithm match the code, every number traced")
            flipped = d.model_copy(update={"recommendation": "investigate" if d.recommendation == "keep_main_model"
                                           else "keep_main_model"})
            tuned = d.model_copy(update={"thresholds_and_segments": "The thresholds were tuned on test data."})
            if validate_evaluation(flipped, inp) and validate_evaluation(tuned, inp):
                passed.append("Gate rejects an overruled recommendation and a claim that thresholds were tuned")
            else:
                problems.append("Gate let an overruled recommendation or a 'tuned thresholds' claim through")
            if validate_evaluation(evaluator_fallback(self._evaluation), inp):
                problems.append("Fallback interpretation fails the gate")
            else:
                passed.append("Fallback interpretation passes the gate")
            self.state.agent_sources["evaluation"] = self._interpretation.source
        except Exception as e:
            problems.append(f"{type(e).__name__}: {e}")
        if problems:
            for p in problems:
                log.error("  FAIL  %s", p)
            self._record("evaluator_gate", False, t0, problems)
            return "failed"
        for p in passed:
            log.info("  OK  %s", p)
        self._record("evaluator_gate", True, t0, passed)
        return "eval_ok"

    # ------------------------------------------------------------------ 2b. Model Card agent
    @listen("eval_ok")
    def model_card_stage(self, *_):
        """The agent writes the judgement sections of model_card.md from evaluation.json."""
        log.info("[flow] ===== Stage 2c: Model Card agent (%s) =====",
                 "LLM" if self.use_llm else "LLM off, fallback only")
        t0 = time.perf_counter()
        cfg = load_config()
        s_dir = resolve(cfg["paths"]["scientist_dir"])
        (s_dir / "model_card.md").unlink(missing_ok=True)   # never leave an outdated card behind
        try:
            self._evaluation = json.loads((s_dir / "evaluation.json").read_text(encoding="utf-8"))
            meta = json.loads((s_dir / "model_meta.json").read_text(encoding="utf-8"))
            self._model_card = write_model_card(self._evaluation, cfg, use_llm=self.use_llm,
                                                refresh=self.refresh_agents)
            save_model_card(self._model_card, self._evaluation, meta, cfg)
        except Exception as e:
            log.exception("Model card stage crashed")
            return self._record("model_card_stage", False, t0, [f"{type(e).__name__}: {e}"])
        r = self._model_card
        detail = f"model_card.md: source={r.source}" + (" (reused, no LLM call)" if r.reused else "") + \
                 (f" ({'; '.join(r.problems)})" if r.problems else "")
        return self._record("model_card_stage", True, t0, [detail])

    @router(model_card_stage, emit=["card_ok", "failed"])
    def model_card_gate(self, stage_ok: bool):
        """The model card keeps to its contract and matches the evaluation."""
        if not stage_ok:
            return "failed"
        log.info("[flow] ===== Model card validation =====")
        t0 = time.perf_counter()
        problems, passed = self._missing("scientist_dir", ["evaluation.json", "model_card.md", "model_card.json"]), []
        try:
            inp = model_card_input(self._evaluation)
            d = self._model_card.draft
            errs = validate_card(d, inp)
            if errs:
                problems += [f"model_card.md fails the gate: {e}" for e in errs]
            else:
                passed.append(f"model_card.md ({self._model_card.source}): all brief sections present (purpose, "
                              f"training data, metrics, limitations, ethics), main PR-AUC {inp.main_pr_auc} quoted, "
                              f"every number traced")
            wrong = d.model_copy(update={"metrics_summary": d.metrics_summary.replace(inp.main_pr_auc, "0.753")})
            if validate_card(wrong, inp):
                passed.append("Gate rejects a card that reports the PageValues model's score as the main score")
            else:
                problems.append("Gate let a card with the wrong main score through")
            if validate_card(model_card_fallback(self._evaluation), inp):
                problems.append("Fallback model card fails the gate")
            else:
                passed.append("Fallback model card passes the gate")
            self.state.agent_sources["model_card"] = self._model_card.source
        except Exception as e:
            problems.append(f"{type(e).__name__}: {e}")
        if problems:
            for p in problems:
                log.error("  FAIL  %s", p)
            self._record("model_card_gate", False, t0, problems)
            return "failed"
        for p in passed:
            log.info("  OK  %s", p)
        self._record("model_card_gate", True, t0, passed)
        return "card_ok"

    # ------------------------------------------------------------------ app readiness
    @router("card_ok", emit=["ready", "failed"])
    def app_gate(self, *_):
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
    flow.refresh_agents = "--refresh-agents" in argv
    if "--plot" in argv:
        path = flow.plot("shopintent_flow.html", show=True)  # CrewAI writes it to a temp folder
        log.info("[flow] Flow diagram opened in the browser (%s)", path)
    flow.kickoff()
    return 0 if flow.state.status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
