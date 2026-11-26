"""Insights Analyst agent (Data Analyst crew): EDA facts -> insights.md.

    clean_data.csv ──► eda_tools (Python, no LLM) ──► eda_stats.json + eda_report.html
                                     │
                                     └── facts ──► [this agent] ──► gate ──► insights.md (+ insights.json)
                                                                     │
                                                                     └─ fail ──► fallback insights from the facts

Contract
  Input   InsightsInput: the plain-language facts from stats_to_facts() and a hash of eda_stats.json.
          PageValues is never in the facts (it leaks the answer).
  Output  InsightsDraft: a summary, 4-6 insights (title, finding with evidence, business action), caveats.
          Code renders it to insights.md, so the structure is always the same.
  Gate    validate_insights(): counts and lengths; every number the agent writes must appear in the facts
          (no invented or derived numbers; small counts up to 10 are allowed); every finding cites at least
          one number; PageValues is not mentioned. Runs as a CrewAI task guardrail (one retry with
          feedback) and again after the run.
  Failure no API key, API error, timeout, or output that still fails the gate -> deterministic insights built
          from the same facts, marked source="fallback" with the reason. The flow continues.
  Reuse   if insights.json already holds an agent draft for the same stats hash, it is re-validated and reused
          instead of calling the LLM again, so reruns are reproducible and free. --refresh-insights forces a call.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.agents.common import make_llm, short_error
from src.tools.eda_tools import stats_to_facts
from src.utils import get_logger, load_config, resolve

load_dotenv(resolve(".env"))
log = get_logger("insights_agent")

NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
LEAK = re.compile(r"page\s*values?", re.I)
LIMITS = {"summary": 600, "title": 90, "finding": 420, "action": 300, "caveat": 250}
FIXED_CAVEATS = [
    "PageValues is left out of these insights and of the main model: it is computed from pages that led "
    "to a purchase, so it leaks the answer.",
    "These are associations in one store's data over one year, not proof of cause and effect.",
]


# --------------------------------------------------------------------------- contract
class InsightsInput(BaseModel):
    facts: list[str] = Field(min_length=5)
    stats_sha: str


class Insight(BaseModel):
    title: str
    finding: str
    action: str


class InsightsDraft(BaseModel):
    summary: str
    insights: list[Insight]
    caveats: list[str] = Field(default_factory=list)


class InsightsResult(BaseModel):
    draft: InsightsDraft
    source: Literal["agent", "fallback"]
    model: str | None = None
    stats_sha: str
    reused: bool = False
    problems: list[str] = Field(default_factory=list)
    seconds: float = 0.0


def stats_sha(stats: dict) -> str:
    return hashlib.sha256(json.dumps(stats, sort_keys=True).encode()).hexdigest()[:12]


def make_input(stats: dict) -> InsightsInput:
    return InsightsInput(facts=stats_to_facts(stats), stats_sha=stats_sha(stats))


# --------------------------------------------------------------------------- gate
def _numbers(text: str) -> set[float]:
    return {float(n.replace(",", "")) for n in NUMBER.findall(text)}


def validate_insights(draft: InsightsDraft, facts: list[str]) -> list[str]:
    problems = []
    if not 4 <= len(draft.insights) <= 6:
        problems.append(f"Need 4-6 insights, got {len(draft.insights)}")
    if not draft.summary.strip():
        problems.append("summary is empty")
    elif len(draft.summary) > LIMITS["summary"]:
        problems.append(f"summary is {len(draft.summary)} chars, limit {LIMITS['summary']}")
    for i, ins in enumerate(draft.insights, 1):
        for field in ("title", "finding", "action"):
            text = getattr(ins, field).strip()
            if not text:
                problems.append(f"insight {i}: {field} is empty")
            elif len(text) > LIMITS[field]:
                problems.append(f"insight {i}: {field} is {len(text)} chars, limit {LIMITS[field]}")
        if not _numbers(ins.finding) - set(range(11)):
            problems.append(f"insight {i}: the finding must cite at least one number from the facts")
    if len(draft.caveats) > 4:
        problems.append(f"At most 4 caveats, got {len(draft.caveats)}")
    for c in draft.caveats:
        if len(c) > LIMITS["caveat"]:
            problems.append(f"caveat is {len(c)} chars, limit {LIMITS['caveat']}")

    allowed = _numbers(" ".join(facts)) | set(map(float, range(11)))
    text = " ".join([draft.summary, *draft.caveats] + [f"{x.title} {x.finding} {x.action}" for x in draft.insights])
    invented = sorted(n for n in _numbers(text) if n not in allowed)
    if invented:
        shown = ", ".join(f"{n:g}" for n in invented[:8])
        problems.append(f"Numbers not found in the facts (invented or derived): {shown}. "
                        f"Use numbers exactly as written in the facts")
    if LEAK.search(text):
        problems.append("Do not mention PageValues (it leaks the answer and is excluded)")
    return problems


# --------------------------------------------------------------------------- fallback
def fallback_draft(stats: dict) -> InsightsDraft:
    """Deterministic insights from the same stats. Numbers are formatted exactly like the facts."""
    pct = lambda x: f"{x * 100:.1f}%"  # noqa: E731
    d = stats["dataset"]
    months = sorted(stats["by_month"], key=lambda r: -r["rate"])
    best, worst = months[0], months[-1]
    vt = {r["group"]: r for r in stats["by_visitor_type"]}
    new, ret = vt.get("New visitors"), vt.get("Returning visitors")
    cmp = {r["feature"]: r for r in stats["buyer_vs_non_buyer"]}
    pr, ex = cmp["ProductRelated"], cmp["ExitRates"]
    wk = {r["group"]: r for r in stats["by_weekend"]}
    insights = [
        Insight(title=f"{best['group']} is the strongest month",
                finding=f"The purchase rate in {best['group']} is {pct(best['rate'])} ({best['sessions']:,} sessions), "
                        f"against {pct(worst['rate'])} in {worst['group']}.",
                action=f"Concentrate campaign budget and stock planning on {best['group']}."),
        Insight(title=("New visitors convert better than returning ones" if new["rate"] > ret["rate"]
                       else "Returning visitors convert better than new ones"),
                finding=f"New visitors buy at {pct(new['rate'])}, returning visitors at {pct(ret['rate'])}.",
                action="Make a strong first impression for new visitors (help, trust signals) and give returning "
                       "visitors a reason to come back and finish."),
        Insight(title="Buyers browse many more product pages",
                finding=f"Buyers view a median of {pr['buyer_median']} product pages, non-buyers {pr['non_buyer_median']}.",
                action="Make it easy to keep browsing: related products, good search and clear categories."),
        Insight(title="Buyers stay on pages that rarely end a visit",
                finding=f"The median exit rate is {ex['buyer_median']}% for buyers and {ex['non_buyer_median']}% "
                        f"for non-buyers.",
                action="Review the pages with the highest exit rates first."),
        Insight(title=("Weekends convert slightly better" if wk["Weekend"]["rate"] > wk["Weekday"]["rate"]
                       else "Weekdays convert slightly better"),
                finding=f"The purchase rate is {pct(wk['Weekend']['rate'])} on a weekend and "
                        f"{pct(wk['Weekday']['rate'])} on a weekday.",
                action="Schedule promotions and support staffing with the weekend in mind."),
    ]
    return InsightsDraft(
        summary=f"Of {d['sessions']:,} sessions, {d['buyers']:,} ended in a purchase ({pct(d['purchase_rate'])}). "
                f"The month, the visitor type and how much a visitor browses products separate buyers from "
                f"non-buyers most clearly.",
        insights=insights, caveats=[])


# --------------------------------------------------------------------------- the agent
def _run_crew(inp: InsightsInput, cfg: dict) -> InsightsDraft:
    from crewai import Agent, Crew, Process, Task

    agent = Agent(
        role="E-commerce insights analyst",
        goal="Turn verified EDA facts into a few clear business insights for the store's managers",
        backstory="You work in the Data Analyst crew. The numbers were computed by Python tools and are correct. "
                  "You never invent, round or derive numbers; you quote them exactly as given.",
        llm=make_llm(cfg, temperature=0.2), allow_delegation=False, max_iter=2, verbose=False,
        max_execution_time=cfg["llm"].get("timeout_seconds", 30) * 2,
    )

    def guardrail(output):
        draft = output.pydantic
        if draft is None:
            return False, "Return JSON with summary, insights (title, finding, action) and caveats."
        problems = validate_insights(draft, inp.facts)
        return (True, output) if not problems else (False, "Fix these problems: " + "; ".join(problems))

    facts = "\n".join(f"- {f}" for f in inp.facts)
    task = Task(
        description=(
            "Write business insights about an online store's visitors from these verified facts:\n"
            f"{facts}\n\n"
            "Rules:\n"
            f"- 4 to 6 insights. Each has a short title (<= {LIMITS['title']} chars), a finding (<= {LIMITS['finding']} "
            f"chars) that cites at least one number from the facts, and a practical action for the store "
            f"(<= {LIMITS['action']} chars).\n"
            f"- A summary of 2-3 sentences (<= {LIMITS['summary']} chars) and 0-2 caveats (<= {LIMITS['caveat']} chars).\n"
            "- Copy every number exactly as written in the facts (same decimals, same % sign). Do not calculate "
            "differences, ratios or 'times more'; describe them in words instead.\n"
            "- Prefer the strongest patterns: month, visitor type, browsing behaviour, exit and bounce rates.\n"
            "- Plain English for managers. Associations, not causes.\n"
            "- Caveats are only for limits of THESE facts (for example small groups). Do not write caveats about "
            "correlation vs causation or about PageValues: both are added automatically."),
        expected_output="JSON with summary, insights (list of title, finding, action) and caveats",
        agent=agent, output_pydantic=InsightsDraft, guardrail=guardrail, guardrail_max_retries=1)
    result = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff()
    if result.pydantic is None:
        raise ValueError("Agent returned no structured output")
    return result.pydantic


def _load_previous(path: Path, sha: str) -> InsightsResult | None:
    try:
        prev = InsightsResult.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return prev if prev.source == "agent" and prev.stats_sha == sha else None


def write_insights(stats: dict, cfg: dict | None = None, use_llm: bool = True, refresh: bool = False) -> InsightsResult:
    """Always returns a result that passes the gate (agent output, a reused agent output, or the fallback)."""
    cfg = cfg or load_config()
    t0 = time.perf_counter()
    inp = make_input(stats)
    out_dir = resolve(cfg["paths"]["analyst_dir"])
    problems: list[str] = []

    prev = None if refresh else _load_previous(out_dir / "insights.json", inp.stats_sha)
    if prev is not None:
        errs = validate_insights(prev.draft, inp.facts)
        if not errs:
            log.info("[agent] insights: reusing the agent draft for stats %s (no LLM call)", inp.stats_sha)
            return prev.model_copy(update={"reused": True, "seconds": round(time.perf_counter() - t0, 2)})
        problems.append(f"Saved draft no longer passes the gate: {errs}")

    draft = None
    if not use_llm:
        problems.append("Agent disabled (use_llm=False)")
    elif not os.getenv("OPENAI_API_KEY"):
        problems.append("OPENAI_API_KEY is not set")
    else:
        try:
            draft = _run_crew(inp, cfg)
            errs = validate_insights(draft, inp.facts)
            if errs:
                problems.extend(errs)
                draft = None
        except Exception as e:
            problems.append(short_error(e))

    source = "agent" if draft is not None else "fallback"
    if draft is None:
        draft = fallback_draft(stats)
        log.warning("[agent] insights: fallback used (%s)", "; ".join(problems))
    else:
        log.info("[agent] insights: %d insights written in %.1fs", len(draft.insights), time.perf_counter() - t0)
    return InsightsResult(draft=draft, source=source, model=cfg["llm"]["model"] if source == "agent" else None,
                          stats_sha=inp.stats_sha, problems=problems if source == "fallback" else [],
                          seconds=round(time.perf_counter() - t0, 2))


def render_markdown(res: InsightsResult) -> str:
    d = res.draft
    who = f"Insights Analyst agent ({res.model})" if res.source == "agent" else "fallback template (agent unavailable)"
    lines = ["# ShopIntent AI · Business insights", "",
             f"_Written by the {who} from the verified facts in `eda_stats.json` (stats {res.stats_sha}). "
             f"Charts and tables: `eda_report.html`._", "", "## Summary", "", d.summary.strip(), "", "## Insights", ""]
    for i, ins in enumerate(d.insights, 1):
        lines += [f"### {i}. {ins.title.strip()}", "", ins.finding.strip(), "",
                  f"**What the store can do:** {ins.action.strip()}", ""]
    lines += ["## Caveats", ""] + [f"- {c.strip()}" for c in [*d.caveats, *FIXED_CAVEATS]] + [""]
    return "\n".join(lines)


def save(res: InsightsResult, cfg: dict) -> None:
    out_dir = resolve(cfg["paths"]["analyst_dir"])
    (out_dir / "insights.md").write_text(render_markdown(res), encoding="utf-8", newline="\n")
    keep = res.model_copy(update={"reused": False, "seconds": 0.0})   # stable bytes on reuse
    (out_dir / "insights.json").write_text(keep.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
