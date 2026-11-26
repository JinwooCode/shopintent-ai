"""Model Card agent (Data Scientist crew): evaluation.json + model_meta.json -> model_card.md.

    run_model_pipeline (Python: split, select, train, evaluate) ──► evaluation.json + model_meta.json
                                                                           │
                                              facts ──► [this agent] ──► gate ──► model_card.md (+ model_card.json)
                                                                           │
                                                                           └─ fail ──► fallback card from the facts

Python computes every metric; the agent writes the parts of a model card that need judgement.
The metric tables in model_card.md are rendered by code from evaluation.json, not by the agent.

Contract
  Input   ModelCardInput: plain-language facts from evaluation.json (split, model selection, test metrics of
          the main and comparison models, fixed thresholds and their checks, segments, results by visitor type).
  Output  ModelCardDraft: purpose, intended uses, out-of-scope uses, training data summary, metrics summary,
          3-6 limitations, 3-6 ethical considerations.
  Gate    validate_card(): counts and lengths; every number traced to the facts; the metrics summary quotes
          the MAIN model's test PR-AUC and does not present the PageValues model as the one in use;
          limitations say this is a simulation on completed sessions; ethics address who gets the
          interventions (coupon / chat).
  Failure deterministic card from the same facts, marked as fallback.
  Reuse   while evaluation.json is unchanged, the saved agent draft is re-checked and reused (no LLM call).
"""
from __future__ import annotations

import os
import re
import time
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.agents.common import invented_problem, make_llm, sha_of, short_error
from src.utils import get_logger, load_config, resolve

load_dotenv(resolve(".env"))
log = get_logger("model_card_agent")

LIMITS = {"purpose": 500, "item": 350, "training_data": 900, "metrics_summary": 900}
SIM_WORDS = re.compile(r"\b(simulat\w*|completed sessions?|not real[- ]time|historical)\b", re.I)
INTERVENTION_WORDS = re.compile(r"\b(coupon\w*|discount\w*|intervention\w*|offer\w*|chat)\b", re.I)
PAGEVALUES = re.compile(r"page\s*values?", re.I)
VT_LABEL = {"Returning_Visitor": "returning visitors", "New_Visitor": "new visitors", "Other": "other visitors"}


# --------------------------------------------------------------------------- contract
class ModelCardInput(BaseModel):
    facts: list[str] = Field(min_length=5)
    main_pr_auc: str
    evaluation_sha: str


class ModelCardDraft(BaseModel):
    purpose: str
    intended_uses: list[str]
    out_of_scope: list[str]
    training_data: str
    metrics_summary: str
    limitations: list[str]
    ethical_considerations: list[str]


class ModelCardResult(BaseModel):
    draft: ModelCardDraft
    source: Literal["agent", "fallback"]
    model: str | None = None
    evaluation_sha: str
    reused: bool = False
    problems: list[str] = Field(default_factory=list)
    seconds: float = 0.0


def _p(x: float) -> str:
    return f"{x * 100:.1f}%"


def evaluation_to_facts(ev: dict) -> list[str]:
    m, c = ev["test_metrics"]["main"], ev["test_metrics"]["comparison"]
    th = ev["thresholds"]
    facts = [
        f"Task: predict whether an online shopping session ends in a purchase (target {ev['target']}); "
        f"{ev['rows']:,} sessions, {_p(ev['baseline_purchase_rate'])} end in a purchase.",
        *[f"{s.capitalize()} split: {v['sessions']:,} sessions, purchase rate {_p(v['purchase_rate'])}."
          for s, v in ev["split"].items()],
        "The split is stratified on the target; model selection and the threshold check use the validation "
        "split only; the test split is scored once at the end.",
        *[f"Validation PR-AUC of {a}: {v['pr_auc']:.3f} (ROC-AUC {v['roc_auc']:.3f})."
          for a, v in ev["selection"].items()],
        f"Selected algorithm (highest validation PR-AUC): {ev['selected_algorithm']}.",
        f"Main model (in use, without {', '.join(ev['excluded_from_main'])}) on test: PR-AUC {m['pr_auc']:.3f}, "
        f"ROC-AUC {m['roc_auc']:.3f}, Brier {m['brier']:.3f}, precision at 0.5 {m['precision@0.5']:.3f}, "
        f"recall at 0.5 {m['recall@0.5']:.3f}.",
        f"Comparison model (with PageValues, report only, not used in the app) on test: PR-AUC {c['pr_auc']:.3f}, "
        f"ROC-AUC {c['roc_auc']:.3f}.",
        "PageValues is computed from pages that led to transactions, so it leaks the outcome.",
        f"A random model would have PR-AUC about {ev['baseline_purchase_rate']:.3f} (the purchase rate).",
        f"Fixed thresholds set in advance, not tuned: buyer at or above {_p(th['buyer_from'])}, hesitant from "
        f"{_p(th['unlikely_below'])} to {_p(th['buyer_from'])}, unlikely below {_p(th['unlikely_below'])}.",
        "Interventions: hesitant returning visitors get a coupon, hesitant new visitors get a chat invitation, "
        "everyone else gets no intervention.",
        *[f"Threshold check on validation: {t['check']}: {'passed' if t['ok'] else 'failed'}."
          for t in ev["threshold_checks"]],
        *[f"Test segment {s['segment']}: {s['sessions']:,} sessions ({_p(s['share'])}), actual purchase rate "
          f"{_p(s['actual_purchase_rate'])}, mean predicted {_p(s['mean_predicted'])}." for s in ev["test_segments"]],
        *[s for v in ev["test_by_visitor_type"] for s in (
            f"On test there are {v['sessions']:,} sessions from {VT_LABEL.get(v['visitor_type'], v['visitor_type'])}; "
            f"their actual purchase rate is {_p(v['actual_purchase_rate'])} and the model's mean predicted "
            f"probability for them is {_p(v['mean_predicted'])}.",
            f"Of all {VT_LABEL.get(v['visitor_type'], v['visitor_type'])} on test, {_p(v['share_hesitant'])} are "
            f"classified hesitant and would receive "
            f"{ {'Returning_Visitor': 'a coupon', 'New_Visitor': 'a chat invitation'}.get(v['visitor_type'], 'no intervention') }.")],
        "Data: UCI Online Shoppers Purchasing Intention dataset; one year of sessions from one online store; "
        "each session is a different user; sessions are complete, so the app is a visit simulation.",
        f"Main model features: {', '.join(ev['main_features'])}.",
    ]
    return facts


def make_input(ev: dict) -> ModelCardInput:
    return ModelCardInput(facts=evaluation_to_facts(ev), main_pr_auc=f"{ev['test_metrics']['main']['pr_auc']:.3f}",
                          evaluation_sha=sha_of(ev))


# --------------------------------------------------------------------------- gate
def validate_card(d: ModelCardDraft, inp: ModelCardInput) -> list[str]:
    problems = []
    for field in ("purpose", "training_data", "metrics_summary"):
        text = getattr(d, field).strip()
        if not text or len(text) > LIMITS[field]:
            problems.append(f"{field} must be 1-{LIMITS[field]} chars")
    for field, lo, hi in (("intended_uses", 2, 5), ("out_of_scope", 2, 5), ("limitations", 3, 6),
                          ("ethical_considerations", 3, 6)):
        items = getattr(d, field)
        if not lo <= len(items) <= hi:
            problems.append(f"{field}: need {lo}-{hi} items, got {len(items)}")
        for i, x in enumerate(items, 1):
            if not x.strip() or len(x) > LIMITS["item"]:
                problems.append(f"{field} item {i} must be 1-{LIMITS['item']} chars")
    if inp.main_pr_auc not in d.metrics_summary:
        problems.append(f"metrics_summary must quote the main model's test PR-AUC ({inp.main_pr_auc})")
    if PAGEVALUES.search(d.metrics_summary) and not re.search(r"\b(not used|comparison|report|exclud\w*|leak\w*)\b",
                                                               d.metrics_summary, re.I):
        problems.append("metrics_summary mentions PageValues without saying that model is only a comparison")
    if not SIM_WORDS.search(" ".join(d.limitations)):
        problems.append("limitations must say the app is a simulation on completed sessions, not real time")
    if not INTERVENTION_WORDS.search(" ".join(d.ethical_considerations)):
        problems.append("ethical_considerations must address who receives the coupon / chat interventions")
    text = " ".join([d.purpose, d.training_data, d.metrics_summary, *d.intended_uses, *d.out_of_scope,
                     *d.limitations, *d.ethical_considerations])
    problems += invented_problem(text, inp.facts)
    return problems


# --------------------------------------------------------------------------- fallback
def fallback_draft(ev: dict) -> ModelCardDraft:
    m, c = ev["test_metrics"]["main"], ev["test_metrics"]["comparison"]
    vt = {v["visitor_type"]: v for v in ev["test_by_visitor_type"]}
    new, ret = vt.get("New_Visitor"), vt.get("Returning_Visitor")
    return ModelCardDraft(
        purpose="Estimate the probability that an online shopping session ends in a purchase, so that fixed "
                "rules can decide whether a hesitant visitor gets a coupon or a chat invitation.",
        intended_uses=["Visit simulation and teaching: replaying completed sessions to show how scoring and "
                       "fixed intervention rules interact.",
                       "Comparing modelling choices, such as the effect of leaving out PageValues."],
        out_of_scope=["Real-time decisions on live visitors without new validation on live data.",
                      "Decisions about individuals beyond a light on-site nudge (pricing, credit, eligibility)."],
        training_data=f"{ev['split']['train']['sessions']:,} training sessions from the UCI Online Shoppers dataset "
                      f"(one store, one year), split stratified on the target; "
                      f"{_p(ev['baseline_purchase_rate'])} of sessions end in a purchase.",
        metrics_summary=f"On the test split the main model reaches PR-AUC {m['pr_auc']:.3f} and ROC-AUC "
                        f"{m['roc_auc']:.3f}, against about {ev['baseline_purchase_rate']:.3f} for a random model. "
                        f"The comparison model with PageValues reaches PR-AUC {c['pr_auc']:.3f} but is only "
                        f"reported, because PageValues leaks the outcome.",
        limitations=["The data are completed sessions, so the app is a visit simulation, not real-time scoring.",
                     f"Recall at 0.5 is low ({m['recall@0.5']:.3f}): most buyers are not flagged as buyers.",
                     "One store over one year: patterns may not transfer to other stores or seasons.",
                     "Browser, region and traffic source are anonymized codes, so their effects cannot be explained."],
        ethical_considerations=[
            f"Who gets a coupon depends on visitor type: {_p(ret['share_hesitant'])} of returning and "
            f"{_p(new['share_hesitant'])} of new visitors are classified hesitant on test." if new and ret else
            "Who gets a coupon depends on visitor type.",
            "Interventions must stay light (a coupon or a chat), never higher prices or withheld offers.",
            "No personal data is used; sessions are anonymous, but live use would need consent and privacy review."])


# --------------------------------------------------------------------------- the agent
def _run_crew(inp: ModelCardInput, cfg: dict) -> ModelCardDraft:
    from crewai import Agent, Crew, Process, Task

    agent = Agent(
        role="Model documentation specialist",
        goal="Write an honest, specific model card that a reviewer can trust",
        backstory="You work in the Data Scientist crew. Every metric was computed by Python and is correct. "
                  "You explain what the model is for, where it fails and what risks it carries. You never "
                  "invent, round or derive numbers.",
        llm=make_llm(cfg, temperature=0.2), allow_delegation=False, max_iter=2, verbose=False,
        max_execution_time=cfg["llm"].get("timeout_seconds", 30) * 3,
    )

    def guardrail(output):
        draft = output.pydantic
        if draft is None:
            return False, "Return JSON with all model card fields."
        problems = validate_card(draft, inp)
        return (True, output) if not problems else (False, "Fix these problems: " + "; ".join(problems))

    facts = "\n".join(f"- {f}" for f in inp.facts)
    task = Task(
        description=(
            "Write the judgement sections of a model card from these verified facts:\n"
            f"{facts}\n\n"
            "Rules:\n"
            f"- purpose (<= {LIMITS['purpose']} chars); training_data summary (<= {LIMITS['training_data']}); "
            f"metrics_summary (<= {LIMITS['metrics_summary']}) that quotes the MAIN model's test PR-AUC "
            f"{inp.main_pr_auc} and compares it with the random baseline.\n"
            f"- intended_uses and out_of_scope: 2-5 items each; limitations and ethical_considerations: 3-6 items "
            f"each; every item <= {LIMITS['item']} chars.\n"
            "- limitations must include that the data are completed sessions, so the app is a simulation, not "
            "real time.\n"
            "- ethical_considerations must address who receives coupons and chats (use the results by visitor type).\n"
            "- The PageValues model is only a comparison; never present it as the model in use.\n"
            "- Copy every number exactly as written in the facts; do not calculate new numbers.\n"
            "- Be concise: 2-4 sentences per text field. Respect every character limit."),
        expected_output="JSON with purpose, intended_uses, out_of_scope, training_data, metrics_summary, "
                        "limitations, ethical_considerations",
        agent=agent, output_pydantic=ModelCardDraft, guardrail=guardrail, guardrail_max_retries=2)
    result = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff()
    if result.pydantic is None:
        raise ValueError("Agent returned no structured output")
    return result.pydantic


def write_model_card(ev: dict, cfg: dict | None = None, use_llm: bool = True,
                     refresh: bool = False) -> ModelCardResult:
    """Always returns a card that passes the gate (agent, reused agent draft, or fallback)."""
    cfg = cfg or load_config()
    t0 = time.perf_counter()
    inp = make_input(ev)
    saved = resolve(cfg["paths"]["scientist_dir"]) / "model_card.json"
    problems: list[str] = []

    if not refresh and saved.exists():
        try:
            prev = ModelCardResult.model_validate_json(saved.read_text(encoding="utf-8"))
            if prev.source == "agent" and prev.evaluation_sha == inp.evaluation_sha:
                errs = validate_card(prev.draft, inp)
                if not errs:
                    log.info("[agent] model card: reusing the agent draft for %s (no LLM call)", inp.evaluation_sha)
                    return prev.model_copy(update={"reused": True})
                problems.append(f"Saved draft no longer passes the gate: {errs}")
        except Exception as e:
            problems.append(f"Could not read the saved draft: {short_error(e)}")

    draft = None
    if not use_llm:
        problems.append("Agent disabled (use_llm=False)")
    elif not os.getenv("OPENAI_API_KEY"):
        problems.append("OPENAI_API_KEY is not set")
    else:
        try:
            draft = _run_crew(inp, cfg)
            errs = validate_card(draft, inp)
            if errs:
                problems.extend(errs)
                draft = None
        except Exception as e:
            problems.append(short_error(e))

    source = "agent" if draft is not None else "fallback"
    if draft is None:
        draft = fallback_draft(ev)
        log.warning("[agent] model card: fallback used (%s)", "; ".join(problems))
    else:
        log.info("[agent] model card written in %.1fs", time.perf_counter() - t0)
    return ModelCardResult(draft=draft, source=source, model=cfg["llm"]["model"] if source == "agent" else None,
                           evaluation_sha=inp.evaluation_sha, problems=problems if source == "fallback" else [],
                           seconds=round(time.perf_counter() - t0, 2))


def render_markdown(res: ModelCardResult, ev: dict, meta: dict) -> str:
    d = res.draft
    m, c = ev["test_metrics"]["main"], ev["test_metrics"]["comparison"]
    th = ev["thresholds"]
    who = f"Model Card agent ({res.model})" if res.source == "agent" else "fallback template (agent unavailable)"
    bullets = lambda xs: [f"- {x.strip()}" for x in xs]  # noqa: E731
    row = lambda name, x: (f"| {name} | {x['roc_auc']:.3f} | {x['pr_auc']:.3f} | {x['brier']:.3f} | "  # noqa: E731
                           f"{x['precision@0.5']:.3f} | {x['recall@0.5']:.3f} | {x['f1@0.5']:.3f} |")
    lines = [
        "# Model card · ShopIntent AI purchase-intent model", "",
        f"_Sections in prose written by the {who} from `evaluation.json` ({res.evaluation_sha}). "
        f"All tables are generated by code from the same file._", "",
        "## Model details", "",
        "| | |", "|---|---|",
        f"| Model in use | `{meta['main_model']['file']}` ({meta['main_model']['name']}) |",
        f"| Algorithm | {ev['selected_algorithm'].replace('_', ' ')} (selected on validation {ev['selection_metric']}) |",
        f"| Features | {len(ev['main_features'])}; excluded: {', '.join(f'`{x}`' for x in ev['excluded_from_main'])} |",
        f"| Target | `{ev['target']}` (session ended in a purchase) |",
        f"| Seed | {ev['seed']} (fully deterministic pipeline) |",
        f"| Comparison only | `{meta['comparison_model']['file']}` (with PageValues, never used by the app) |", "",
        "## Purpose", "", d.purpose.strip(), "",
        "### Intended uses", "", *bullets(d.intended_uses), "",
        "### Out of scope", "", *bullets(d.out_of_scope), "",
        "## Training data", "", d.training_data.strip(), "",
        "| Split | Sessions | Purchase rate |", "|---|---|---|",
        *[f"| {s} | {v['sessions']:,} | {_p(v['purchase_rate'])} |" for s, v in ev["split"].items()], "",
        "## Metrics", "", d.metrics_summary.strip(), "",
        "| Test split | ROC-AUC | PR-AUC | Brier ↓ | Precision@0.5 | Recall@0.5 | F1@0.5 |",
        "|---|---|---|---|---|---|---|",
        row("**Main model (in use)**", m), row("Comparison with PageValues (report only)", c), "",
        f"Random baseline PR-AUC ≈ {ev['baseline_purchase_rate']:.3f}.", "",
        "### Fixed decision thresholds (set in advance, checked on validation only)", "",
        f"Buyer ≥ {_p(th['buyer_from'])} → no action · hesitant {_p(th['unlikely_below'])}–{_p(th['buyer_from'])} → "
        f"coupon (returning) / chat (new) · unlikely < {_p(th['unlikely_below'])} → no action", "",
        "| Test segment | Sessions | Share | Actual purchase rate | Mean predicted |", "|---|---|---|---|---|",
        *[f"| {s['segment']} | {s['sessions']:,} | {_p(s['share'])} | {_p(s['actual_purchase_rate'])} | "
          f"{_p(s['mean_predicted'])} |" for s in ev["test_segments"]], "",
        "| Test, by visitor type | Sessions | Actual purchase rate | Mean predicted | Classified hesitant |",
        "|---|---|---|---|---|",
        *[f"| {VT_LABEL.get(v['visitor_type'], v['visitor_type'])} | {v['sessions']:,} | "
          f"{_p(v['actual_purchase_rate'])} | {_p(v['mean_predicted'])} | {_p(v['share_hesitant'])} |"
          for v in ev["test_by_visitor_type"]], "",
        "## Limitations", "", *bullets(d.limitations), "",
        "## Ethical considerations", "", *bullets(d.ethical_considerations), "",
    ]
    return "\n".join(lines)


def save(res: ModelCardResult, ev: dict, meta: dict, cfg: dict) -> None:
    out_dir = resolve(cfg["paths"]["scientist_dir"])
    (out_dir / "model_card.md").write_text(render_markdown(res, ev, meta), encoding="utf-8", newline="\n")
    keep = res.model_copy(update={"reused": False, "seconds": 0.0})
    (out_dir / "model_card.json").write_text(keep.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
