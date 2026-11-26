"""Data Validator agent (Data Analyst crew): data_quality.json -> data_quality.md.

    raw CSV ──► data_tools (Python: check, clean, contract, validate) ──► data_quality.json
                                                                              │
                                                     facts ──► [this agent] ──► gate ──► data_quality.md
                                                                                 │
                                                                                 └─ fail ──► fallback report

The Python tools decide whether the data is usable. The agent explains that decision; it cannot change it.

Contract
  Input   ValidatorInput: the pipeline status (ready / failed) and plain-language facts built from
          data_quality.json (raw checks, cleaning actions, contract checks, extra checks, errors).
  Output  QualityDraft: verdict, summary, 3-7 findings (what was found and why it matters), and one plain
          explanation per cleaning action, in the same order. Code renders data_quality.md and always
          includes the tool's own record verbatim next to the agent's explanation.
  Gate    validate_quality(): the verdict must equal the tool's status; one explanation per cleaning action;
          every number must appear in the facts; counts and lengths. Runs as a CrewAI guardrail (one retry
          with feedback) and again after the run.
  Failure no API key, API error, timeout, or output that still fails the gate -> a deterministic report
          from the same facts, marked as fallback with the reason.
  Reuse   while data_quality.json is unchanged, the saved agent draft is re-checked and reused (no LLM call).
"""
from __future__ import annotations

import os
import time
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.agents.common import invented_problem, make_llm, sha_of, short_error
from src.utils import get_logger, load_config, resolve

load_dotenv(resolve(".env"))
log = get_logger("validator_agent")

LIMITS = {"summary": 500, "title": 90, "detail": 350, "why": 300, "cleaning": 300}
VERDICT = {"ready": "ready", "failed": "not_ready"}


# --------------------------------------------------------------------------- contract
class ValidatorInput(BaseModel):
    status: Literal["ready", "failed"]
    facts: list[str] = Field(min_length=1)   # an early failure leaves only the status and the error
    n_cleaning_actions: int
    quality_sha: str


class Finding(BaseModel):
    title: str
    detail: str
    why_it_matters: str


class QualityDraft(BaseModel):
    verdict: Literal["ready", "not_ready"]
    summary: str
    findings: list[Finding]
    cleaning_explained: list[str]


class QualityResult(BaseModel):
    draft: QualityDraft
    source: Literal["agent", "fallback"]
    model: str | None = None
    quality_sha: str
    reused: bool = False
    problems: list[str] = Field(default_factory=list)
    seconds: float = 0.0


def quality_to_facts(q: dict) -> list[str]:
    r = q.get("raw", {})
    facts = [f"Pipeline status: {q['status']}" + (f" (stopped at step '{q['failed_step']}')" if q.get("failed_step") else "")]
    if r:
        facts += [f"Raw data: {r['rows']:,} rows and {r['columns']} columns.",
                  f"Missing values in the raw data: {sum(r['missing_values'].values()) if r['missing_values'] else 0}.",
                  f"Identical (duplicate) rows in the raw data: {r['duplicate_rows']}.",
                  f"Negative numeric values in the raw data: {sum(r['negative_values'].values()) if r['negative_values'] else 0}.",
                  f"Share of sessions that ended in a purchase: {r['positive_rate'] * 100:.1f}%.",
                  f"Month values found: {', '.join(r['month_values'])}."]
    facts += [f"Cleaning action {i}: {a}" for i, a in enumerate(q.get("cleaning_actions", []), 1)]
    facts += [f"Contract check passed: {c}" for c in q.get("contract_checks", [])]
    facts += [f"Extra check '{c['check']}': {c['detail']}" + ("" if c["ok"] else " (warning)")
              for c in q.get("extra_checks", [])]
    facts += [f"Error: {e}" for e in q.get("errors", [])]
    return facts


def make_input(q: dict) -> ValidatorInput:
    return ValidatorInput(status=q["status"], facts=quality_to_facts(q),
                          n_cleaning_actions=len(q.get("cleaning_actions", [])), quality_sha=sha_of(q))


# --------------------------------------------------------------------------- gate
def validate_quality(draft: QualityDraft, inp: ValidatorInput) -> list[str]:
    problems = []
    if draft.verdict != VERDICT[inp.status]:
        problems.append(f"verdict must be '{VERDICT[inp.status]}' (decided by the validation tools), "
                        f"got '{draft.verdict}'")
    if not 3 <= len(draft.findings) <= 7:
        problems.append(f"Need 3-7 findings, got {len(draft.findings)}")
    if len(draft.cleaning_explained) != inp.n_cleaning_actions:
        problems.append(f"Need exactly one explanation per cleaning action ({inp.n_cleaning_actions}), "
                        f"got {len(draft.cleaning_explained)}")
    if not draft.summary.strip() or len(draft.summary) > LIMITS["summary"]:
        problems.append(f"summary must be 1-{LIMITS['summary']} chars")
    for i, f in enumerate(draft.findings, 1):
        for field, key in (("title", "title"), ("detail", "detail"), ("why_it_matters", "why")):
            text = getattr(f, field).strip()
            if not text or len(text) > LIMITS[key]:
                problems.append(f"finding {i}: {field} must be 1-{LIMITS[key]} chars")
    for i, c in enumerate(draft.cleaning_explained, 1):
        if not c.strip() or len(c) > LIMITS["cleaning"]:
            problems.append(f"cleaning explanation {i} must be 1-{LIMITS['cleaning']} chars")
    text = " ".join([draft.summary, *draft.cleaning_explained] +
                    [f"{f.title} {f.detail} {f.why_it_matters}" for f in draft.findings])
    problems += invented_problem(text, inp.facts)
    return problems


# --------------------------------------------------------------------------- fallback
def fallback_draft(q: dict) -> QualityDraft:
    r = q.get("raw", {})
    findings = []
    if r:
        findings.append(Finding(title="Schema matches what the pipeline expects",
                                detail=f"The raw file has {r['rows']:,} rows and {r['columns']} columns.",
                                why_it_matters="Every later step can rely on the same columns."))
        findings.append(Finding(title="Identical rows are kept",
                                detail=f"There are {r['duplicate_rows']} identical rows in the raw data.",
                                why_it_matters="They are separate single-page visits; removing them would "
                                               "distort the bounce rate."))
    for c in q.get("extra_checks", [])[:4]:
        findings.append(Finding(title=c["check"], detail=c["detail"] + ".",
                                why_it_matters="Recorded so the Data Scientist crew knows about it."))
    for e in q.get("errors", [])[:2]:
        findings.append(Finding(title="Validation error", detail=e[:LIMITS["detail"]],
                                why_it_matters="The pipeline stops so no bad data is handed on."))
    while len(findings) < 3:
        findings.append(Finding(title="No further issues", detail="No other problems were recorded.",
                                why_it_matters="Nothing else needs action."))
    status = q["status"]
    return QualityDraft(
        verdict=VERDICT[status],
        summary=("The data passed every check and is ready for the Data Scientist crew." if status == "ready"
                 else f"The data is not ready: the pipeline stopped at '{q.get('failed_step')}'."),
        findings=findings[:7],
        cleaning_explained=[f"Done by the cleaning tool: {a}" for a in q.get("cleaning_actions", [])])


# --------------------------------------------------------------------------- the agent
def _run_crew(inp: ValidatorInput, cfg: dict) -> QualityDraft:
    from crewai import Agent, Crew, Process, Task

    agent = Agent(
        role="Data quality validator",
        goal="Explain clearly whether the cleaned dataset can be trusted and what was done to it",
        backstory="You work in the Data Analyst crew. Python tools ran every check and decided the status; "
                  "you explain the results to the team. You never invent, round or derive numbers.",
        llm=make_llm(cfg, temperature=0.2), allow_delegation=False, max_iter=2, verbose=False,
        max_execution_time=cfg["llm"].get("timeout_seconds", 30) * 2,
    )

    def guardrail(output):
        draft = output.pydantic
        if draft is None:
            return False, "Return JSON with verdict, summary, findings and cleaning_explained."
        problems = validate_quality(draft, inp)
        return (True, output) if not problems else (False, "Fix these problems: " + "; ".join(problems))

    facts = "\n".join(f"- {f}" for f in inp.facts)
    task = Task(
        description=(
            "Write a data-quality report for the team from these results of the validation tools:\n"
            f"{facts}\n\n"
            "Rules:\n"
            f"- verdict: '{VERDICT[inp.status]}' (the tools decided this; do not change it).\n"
            f"- summary: 2-3 sentences (<= {LIMITS['summary']} chars).\n"
            f"- 3 to 7 findings: title (<= {LIMITS['title']}), detail citing the facts (<= {LIMITS['detail']}), "
            f"why_it_matters for modelling or the business (<= {LIMITS['why']}).\n"
            f"- cleaning_explained: exactly {inp.n_cleaning_actions} items, one per cleaning action in the same "
            f"order, each saying in plain words what was done and why (<= {LIMITS['cleaning']} chars).\n"
            "- Copy every number exactly as written in the facts. Do not calculate new numbers."),
        expected_output="JSON with verdict, summary, findings (title, detail, why_it_matters), cleaning_explained",
        agent=agent, output_pydantic=QualityDraft, guardrail=guardrail, guardrail_max_retries=1)
    result = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff()
    if result.pydantic is None:
        raise ValueError("Agent returned no structured output")
    return result.pydantic


def write_quality_report(q: dict, cfg: dict | None = None, use_llm: bool = True,
                         refresh: bool = False) -> QualityResult:
    """Always returns a result that passes the gate (agent, reused agent draft, or fallback)."""
    cfg = cfg or load_config()
    t0 = time.perf_counter()
    inp = make_input(q)
    saved = resolve(cfg["paths"]["analyst_dir"]) / "data_quality_report.json"
    problems: list[str] = []

    if not refresh and saved.exists():
        try:
            prev = QualityResult.model_validate_json(saved.read_text(encoding="utf-8"))
            if prev.source == "agent" and prev.quality_sha == inp.quality_sha:
                errs = validate_quality(prev.draft, inp)
                if not errs:
                    log.info("[agent] data quality: reusing the agent draft for %s (no LLM call)", inp.quality_sha)
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
            errs = validate_quality(draft, inp)
            if errs:
                problems.extend(errs)
                draft = None
        except Exception as e:
            problems.append(short_error(e))

    source = "agent" if draft is not None else "fallback"
    if draft is None:
        draft = fallback_draft(q)
        log.warning("[agent] data quality: fallback used (%s)", "; ".join(problems))
    else:
        log.info("[agent] data quality: report written in %.1fs", time.perf_counter() - t0)
    return QualityResult(draft=draft, source=source, model=cfg["llm"]["model"] if source == "agent" else None,
                         quality_sha=inp.quality_sha, problems=problems if source == "fallback" else [],
                         seconds=round(time.perf_counter() - t0, 2))


def render_markdown(res: QualityResult, q: dict) -> str:
    d = res.draft
    who = f"Data Validator agent ({res.model})" if res.source == "agent" else "fallback template (agent unavailable)"
    badge = "✅ Ready for the Data Scientist crew" if d.verdict == "ready" else "❌ Not ready"
    lines = ["# ShopIntent AI · Data quality report", "",
             f"**Verdict: {badge}** (decided by the validation tools)", "",
             f"_Explained by the {who} from `data_quality.json` ({res.quality_sha})._", "",
             "## Summary", "", d.summary.strip(), "", "## Findings", ""]
    for i, f in enumerate(d.findings, 1):
        lines += [f"### {i}. {f.title.strip()}", "", f.detail.strip(), "", f"**Why it matters:** {f.why_it_matters.strip()}", ""]
    lines += ["## Cleaning steps", "", "| Tool record | Explanation |", "|---|---|"]
    for a, e in zip(q.get("cleaning_actions", []), d.cleaning_explained):
        lines.append(f"| {a} | {e.strip().replace('|', '/')} |")
    lines += ["", "## Checks run by the tools", ""]
    lines += [f"- ✅ {c}" for c in q.get("contract_checks", [])]
    lines += [f"- {'✅' if c['ok'] else '⚠️'} {c['check']}: {c['detail']}" for c in q.get("extra_checks", [])]
    lines += [f"- ❌ {e}" for e in q.get("errors", [])]
    return "\n".join(lines) + "\n"


def save(res: QualityResult, q: dict, cfg: dict) -> None:
    out_dir = resolve(cfg["paths"]["analyst_dir"])
    (out_dir / "data_quality.md").write_text(render_markdown(res, q), encoding="utf-8", newline="\n")
    keep = res.model_copy(update={"reused": False, "seconds": 0.0})
    (out_dir / "data_quality_report.json").write_text(keep.model_dump_json(indent=2) + "\n", encoding="utf-8",
                                                      newline="\n")
