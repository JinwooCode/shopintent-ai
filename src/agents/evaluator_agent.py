"""Model Evaluator agent (Data Scientist crew): evaluation.json -> evaluation_interpretation.md.

    run_model_pipeline (Python) ──► evaluation.json ──► facts ──► [this agent] ──► gate ──► evaluation_interpretation.md
                                                                                    │
                                                                                    └─ fail ──► fallback from the facts

evaluation_report.md (code) holds the numbers; this agent explains what they mean: why the algorithm was
chosen, what leaving out PageValues costs, how the fixed thresholds behave, and what it means for the store.

Contract
  Input   EvaluatorInput: the facts from the model card agent plus validation segments, the PageValues gap,
          and the result of a fixed recommendation rule computed by code.
  Output  EvaluationDraft: recommendation, selected algorithm, headline, four explanation sections, next steps.
  Gate    validate_evaluation(): the recommendation equals the code's rule (the agent cannot overrule it);
          the selected algorithm matches evaluation.json; numbers traced to the facts; never claims the
          thresholds were tuned; the PageValues section says why it is excluded; counts and lengths.
  Failure deterministic interpretation from the same facts, marked as fallback.
  Reuse   while evaluation.json is unchanged, the saved agent draft is re-checked and reused (no LLM call).

Recommendation rule (code): keep_main_model when every threshold check passed on validation and the main
model's test PR-AUC is at least twice the random baseline; otherwise investigate.
"""
from __future__ import annotations

import os
import re
import time
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.agents.common import invented_problem, make_llm, sha_of, short_error
from src.agents.model_card_agent import evaluation_to_facts
from src.utils import get_logger, load_config, resolve

load_dotenv(resolve(".env"))
log = get_logger("evaluator_agent")

LIMITS = {"headline": 200, "section": 700, "step": 250}
SECTIONS = ["why_this_algorithm", "pagevalues_tradeoff", "thresholds_and_segments", "meaning_for_the_store"]
TUNED = re.compile(r"\b(?:re-?)?tun(?:ed|ing)\b", re.I)
NOT_TUNED = re.compile(r"\b(?:not|never|no|without)\s+(?:\w+\s+){0,2}(?:re-?)?tun(?:ed|ing)\b", re.I)
LEAK = re.compile(r"\b(leak\w*|tied to the outcome|computed from pages that led)\b", re.I)


# --------------------------------------------------------------------------- contract
class EvaluatorInput(BaseModel):
    facts: list[str]
    selected_algorithm: str
    expected_recommendation: Literal["keep_main_model", "investigate"]
    evaluation_sha: str


class EvaluationDraft(BaseModel):
    recommendation: Literal["keep_main_model", "investigate"]
    selected_algorithm: str
    headline: str
    why_this_algorithm: str
    pagevalues_tradeoff: str
    thresholds_and_segments: str
    meaning_for_the_store: str
    next_steps: list[str]


class EvaluationResult(BaseModel):
    draft: EvaluationDraft
    source: Literal["agent", "fallback"]
    model: str | None = None
    evaluation_sha: str
    reused: bool = False
    problems: list[str] = Field(default_factory=list)
    seconds: float = 0.0


def recommendation_rule(ev: dict) -> tuple[str, bool, bool]:
    checks_ok = all(t["ok"] for t in ev["threshold_checks"])
    beats = ev["test_metrics"]["main"]["pr_auc"] >= 2 * ev["baseline_purchase_rate"]
    return ("keep_main_model" if checks_ok and beats else "investigate"), checks_ok, beats


def make_input(ev: dict) -> EvaluatorInput:
    rec, checks_ok, beats = recommendation_rule(ev)
    m, c = ev["test_metrics"]["main"], ev["test_metrics"]["comparison"]
    p = lambda x: f"{x * 100:.1f}%"  # noqa: E731
    facts = evaluation_to_facts(ev) + [
        f"Adding PageValues raises test PR-AUC by {c['pr_auc'] - m['pr_auc']:.3f}.",
        *[f"Validation segment {s['segment']}: {s['sessions']:,} sessions ({p(s['share'])}), actual purchase rate "
          f"{p(s['actual_purchase_rate'])}, mean predicted {p(s['mean_predicted'])}." for s in ev["val_segments"]],
        f"Rule check: all threshold checks passed on validation: {'yes' if checks_ok else 'no'}.",
        f"Rule check: main model test PR-AUC is at least twice the random baseline: {'yes' if beats else 'no'}.",
        f"Recommendation decided by the rule: {rec}.",
    ]
    return EvaluatorInput(facts=facts, selected_algorithm=ev["selected_algorithm"], expected_recommendation=rec,
                          evaluation_sha=sha_of(ev))


# --------------------------------------------------------------------------- gate
def validate_evaluation(d: EvaluationDraft, inp: EvaluatorInput) -> list[str]:
    problems = []
    if d.recommendation != inp.expected_recommendation:
        problems.append(f"recommendation must be '{inp.expected_recommendation}' (decided by the rule), "
                        f"got '{d.recommendation}'")
    if d.selected_algorithm.strip() != inp.selected_algorithm:
        problems.append(f"selected_algorithm must be '{inp.selected_algorithm}', got '{d.selected_algorithm}'")
    if not d.headline.strip() or len(d.headline) > LIMITS["headline"]:
        problems.append(f"headline must be 1-{LIMITS['headline']} chars")
    for s in SECTIONS:
        text = getattr(d, s).strip()
        if not text or len(text) > LIMITS["section"]:
            problems.append(f"{s} must be 1-{LIMITS['section']} chars")
    if not 2 <= len(d.next_steps) <= 4:
        problems.append(f"next_steps: need 2-4 items, got {len(d.next_steps)}")
    for i, x in enumerate(d.next_steps, 1):
        if not x.strip() or len(x) > LIMITS["step"]:
            problems.append(f"next step {i} must be 1-{LIMITS['step']} chars")
    if not LEAK.search(d.pagevalues_tradeoff):
        problems.append("pagevalues_tradeoff must say why PageValues is excluded (it leaks the outcome)")
    text = " ".join([d.headline, *[getattr(d, s) for s in SECTIONS], *d.next_steps])
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if re.search(r"\bcheck", sentence, re.I) and re.search(r"\bpass", sentence, re.I) and \
                re.search(r"\btest\b", sentence, re.I):
            problems.append("The threshold checks ran on validation only; do not say they passed on test")
            break
    if len(TUNED.findall(text)) > len(NOT_TUNED.findall(text)):
        problems.append("Do not say the thresholds were tuned: they were fixed in advance and only checked")
    problems += invented_problem(text, inp.facts)
    return problems


# --------------------------------------------------------------------------- fallback
def fallback_draft(ev: dict) -> EvaluationDraft:
    rec, _, _ = recommendation_rule(ev)
    m, c = ev["test_metrics"]["main"], ev["test_metrics"]["comparison"]
    sel = ev["selection"]
    algo = ev["selected_algorithm"]
    others = ", ".join(f"{a} {v['pr_auc']:.3f}" for a, v in sel.items() if a != algo)
    seg = {s["segment"]: s for s in ev["test_segments"]}
    p = lambda x: f"{x * 100:.1f}%"  # noqa: E731
    return EvaluationDraft(
        recommendation=rec, selected_algorithm=algo,
        headline=f"The main model ({algo}) reaches test PR-AUC {m['pr_auc']:.3f} without PageValues; "
                 f"recommendation: {rec}.",
        why_this_algorithm=f"{algo} had the highest validation PR-AUC ({sel[algo]['pr_auc']:.3f}; {others}). "
                           f"PR-AUC is used because only {p(ev['baseline_purchase_rate'])} of sessions are purchases.",
        pagevalues_tradeoff=f"With PageValues the test PR-AUC would be {c['pr_auc']:.3f} instead of {m['pr_auc']:.3f}, "
                            f"but PageValues is computed from pages that led to transactions, so it leaks the outcome. "
                            f"The app uses the model without it.",
        thresholds_and_segments=f"The thresholds were fixed in advance and not tuned. On test, the buyer segment buys "
                                f"at {p(seg['buyer']['actual_purchase_rate'])}, hesitant at "
                                f"{p(seg['hesitant']['actual_purchase_rate'])} and unlikely at "
                                f"{p(seg['unlikely']['actual_purchase_rate'])}, so the segments are ordered as intended.",
        meaning_for_the_store=f"Interventions go to the hesitant segment ({p(seg['hesitant']['share'])} of test "
                              f"sessions), where a nudge can change the outcome; most sessions are left alone.",
        next_steps=["Validate on live sessions before any real-time use.",
                    "Measure the effect of coupons and chats with a controlled experiment."])


# --------------------------------------------------------------------------- the agent
def _run_crew(inp: EvaluatorInput, cfg: dict) -> EvaluationDraft:
    from crewai import Agent, Crew, Process, Task

    agent = Agent(
        role="Model evaluator",
        goal="Explain what the evaluation results mean and whether the main model is fit for its purpose",
        backstory="You work in the Data Scientist crew. Python computed every metric and a fixed rule made the "
                  "recommendation; you explain the results honestly. You never invent, round or derive numbers.",
        llm=make_llm(cfg, temperature=0.2), allow_delegation=False, max_iter=2, verbose=False,
        max_execution_time=cfg["llm"].get("timeout_seconds", 30) * 3,
    )

    def guardrail(output):
        draft = output.pydantic
        if draft is None:
            return False, "Return JSON with all fields."
        problems = validate_evaluation(draft, inp)
        return (True, output) if not problems else (False, "Fix these problems: " + "; ".join(problems))

    facts = "\n".join(f"- {f}" for f in inp.facts)
    task = Task(
        description=(
            "Interpret this model evaluation for the project team:\n"
            f"{facts}\n\n"
            "Rules:\n"
            f"- recommendation: '{inp.expected_recommendation}' (decided by the rule; do not change it). "
            f"selected_algorithm: '{inp.selected_algorithm}'.\n"
            f"- headline: one sentence (<= {LIMITS['headline']} chars).\n"
            f"- Four sections, each 2-4 sentences (<= {LIMITS['section']} chars): why_this_algorithm, "
            "pagevalues_tradeoff (say why PageValues is excluded: it leaks the outcome), thresholds_and_segments "
            "(the thresholds were fixed in advance and not tuned; the threshold checks ran on VALIDATION only; "
            "the test segment table shows how the segments behave on unseen data), "
            "meaning_for_the_store.\n"
            f"- next_steps: 2-4 items (<= {LIMITS['step']} chars each).\n"
            "- Copy every number exactly as written in the facts; do not calculate new numbers."),
        expected_output="JSON with recommendation, selected_algorithm, headline, why_this_algorithm, "
                        "pagevalues_tradeoff, thresholds_and_segments, meaning_for_the_store, next_steps",
        agent=agent, output_pydantic=EvaluationDraft, guardrail=guardrail, guardrail_max_retries=2)
    result = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff()
    if result.pydantic is None:
        raise ValueError("Agent returned no structured output")
    return result.pydantic


def write_interpretation(ev: dict, cfg: dict | None = None, use_llm: bool = True,
                         refresh: bool = False) -> EvaluationResult:
    """Always returns an interpretation that passes the gate (agent, reused agent draft, or fallback)."""
    cfg = cfg or load_config()
    t0 = time.perf_counter()
    inp = make_input(ev)
    saved = resolve(cfg["paths"]["scientist_dir"]) / "evaluation_interpretation.json"
    problems: list[str] = []

    if not refresh and saved.exists():
        try:
            prev = EvaluationResult.model_validate_json(saved.read_text(encoding="utf-8"))
            if prev.source == "agent" and prev.evaluation_sha == inp.evaluation_sha:
                errs = validate_evaluation(prev.draft, inp)
                if not errs:
                    log.info("[agent] evaluation: reusing the agent draft for %s (no LLM call)", inp.evaluation_sha)
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
            errs = validate_evaluation(draft, inp)
            if errs:
                problems.extend(errs)
                draft = None
        except Exception as e:
            problems.append(short_error(e))

    source = "agent" if draft is not None else "fallback"
    if draft is None:
        draft = fallback_draft(ev)
        log.warning("[agent] evaluation: fallback used (%s)", "; ".join(problems))
    else:
        log.info("[agent] evaluation interpretation written in %.1fs", time.perf_counter() - t0)
    return EvaluationResult(draft=draft, source=source, model=cfg["llm"]["model"] if source == "agent" else None,
                            evaluation_sha=inp.evaluation_sha, problems=problems if source == "fallback" else [],
                            seconds=round(time.perf_counter() - t0, 2))


def render_markdown(res: EvaluationResult) -> str:
    d = res.draft
    who = f"Model Evaluator agent ({res.model})" if res.source == "agent" else "fallback template (agent unavailable)"
    rec = "✅ Keep the main model" if d.recommendation == "keep_main_model" else "⚠️ Investigate before use"
    titles = {"why_this_algorithm": "Why this algorithm", "pagevalues_tradeoff": "The PageValues trade-off",
              "thresholds_and_segments": "Thresholds and segments", "meaning_for_the_store": "What it means for the store"}
    lines = ["# ShopIntent AI · Evaluation interpretation", "",
             f"**Recommendation: {rec}** (decided by a fixed rule: all threshold checks pass and test PR-AUC ≥ 2× "
             f"the random baseline)", "",
             f"_Written by the {who} from `evaluation.json` ({res.evaluation_sha}). "
             f"All numbers: `evaluation_report.md`._", "", f"> {d.headline.strip()}", ""]
    for s in SECTIONS:
        lines += [f"## {titles[s]}", "", getattr(d, s).strip(), ""]
    lines += ["## Next steps", ""] + [f"- {x.strip()}" for x in d.next_steps] + [""]
    return "\n".join(lines)


def save(res: EvaluationResult, cfg: dict) -> None:
    out_dir = resolve(cfg["paths"]["scientist_dir"])
    (out_dir / "evaluation_interpretation.md").write_text(render_markdown(res), encoding="utf-8", newline="\n")
    keep = res.model_copy(update={"reused": False, "seconds": 0.0})
    (out_dir / "evaluation_interpretation.json").write_text(keep.model_dump_json(indent=2) + "\n", encoding="utf-8",
                                                            newline="\n")
