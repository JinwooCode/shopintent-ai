"""Feature Analyst agent (Data Scientist crew): feature_importance.json -> features.md.

    model.pkl + features.csv ──► importance_tools (Python: permutation importance on validation) ──► feature_importance.json
                                                                                                        │
                                                      facts ──► [this agent] ──► gate ──► features.md (+ features_analysis.json)
                                                                                  │
                                                                                  └─ fail ──► fallback from the facts

Contract
  Input   FeatureInput: one fact per feature (rank, importance, spread, direction), the engineered-feature
          definitions, the method, and a verdict per engineered feature computed by code
          (useful = importance > 0 and more than twice its spread across repeats).
  Output  FeaturesDraft: summary; exactly the top 5 features in the code's order (what it measures, the
          direction of its effect, business meaning); one verdict + explanation per engineered feature;
          a note naming every feature with no measurable effect; caveats.
  Gate    validate_features(): top-5 names and order match the code; engineered verdicts match the code; the
          low-impact note names every feature with importance <= 0; direction words agree with the sign of the
          correlation for numeric top features; numbers traced; PageValues never presented as a feature in use.
  Failure deterministic analysis from the same facts, marked as fallback.
  Reuse   while feature_importance.json is unchanged, the saved agent draft is re-checked and reused.
"""
from __future__ import annotations

import os
import re
import time
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.agents.common import invented_problem, make_llm, sha_of, short_error
from src.agents.contract_agent import SOURCE_NOTES
from src.utils import get_logger, load_config, resolve

load_dotenv(resolve(".env"))
log = get_logger("feature_agent")

TOP_N = 5
LIMITS = {"summary": 600, "field": 300, "note": 400, "caveat": 250}
UP = re.compile(r"\b(higher|more|longer|increase\w*|raise\w*|boost\w*|positive\w*)\b", re.I)
DOWN = re.compile(r"\b(lower|fewer|less|shorter|decrease\w*|reduce\w*|negative\w*|drop\w*)\b", re.I)
PAGEVALUES = re.compile(r"page\s*values?", re.I)


# --------------------------------------------------------------------------- contract
class FeatureInput(BaseModel):
    facts: list[str]
    top: list[str]
    directions: dict[str, float | None]
    engineered_verdicts: dict[str, Literal["useful", "weak"]]
    no_effect: list[str]
    importance_sha: str


class TopFeature(BaseModel):
    feature: str
    what_it_measures: str
    effect: str
    business_meaning: str


class EngineeredVerdict(BaseModel):
    feature: str
    verdict: Literal["useful", "weak"]
    explanation: str


class FeaturesDraft(BaseModel):
    summary: str
    top_features: list[TopFeature]
    engineered: list[EngineeredVerdict]
    low_impact_note: str
    caveats: list[str]


class FeaturesResult(BaseModel):
    draft: FeaturesDraft
    source: Literal["agent", "fallback"]
    model: str | None = None
    importance_sha: str
    reused: bool = False
    problems: list[str] = Field(default_factory=list)
    seconds: float = 0.0


def engineered_verdict(r: dict) -> Literal["useful", "weak"]:
    return "useful" if r["importance"] > 0 and r["importance"] > 2 * r["importance_std"] else "weak"


def _fact(r: dict) -> str:
    s = (f"Rank {r['rank']}: {r['feature']} ({r['kind']}{', engineered' if r['engineered'] else ''}): "
         f"importance {r['importance']} (spread {r['importance_std']})")
    if r["importance"] <= 0:
        s += ", no measurable effect"
    if r["direction_rho"] is not None:
        way = "higher values go with a HIGHER predicted purchase probability" if r["direction_rho"] > 0 else \
              "higher values go with a LOWER predicted purchase probability"
        s += f"; correlation with the predicted probability {r['direction_rho']} ({way})"
    return s + "."


def make_input(imp: dict) -> FeatureInput:
    rows = imp["features"]
    eng = {r["feature"]: engineered_verdict(r) for r in rows if r["engineered"]}
    facts = [f"Method: {imp['method']}, {imp['n_repeats']} repeats, {imp['validation_sessions']:,} validation "
             f"sessions. Importance = how much PR-AUC drops when the feature is shuffled.",
             "Correlated features (for example the page counts and their durations) can share importance, so a "
             "low rank does not always mean the information is useless.",
             "PageValues is not a feature of the main model (it leaks the outcome)."]
    facts += [_fact(r) for r in rows]
    facts += [f"Meaning of {f}: {SOURCE_NOTES[f]}" for f in [r["feature"] for r in rows] if f in SOURCE_NOTES]
    facts += [f"Engineered feature {k} = {v}." for k, v in imp["engineered_definitions"].items()]
    facts.append("For categorical features (Month, VisitorType, ...) the importance does not say which values "
                 "raise or lower the probability; say only that the effect depends on the value.")
    facts += [f"Code verdict for engineered feature {k}: {v} (useful = importance above 0 and more than twice "
              f"its spread)." for k, v in eng.items()]
    return FeatureInput(facts=facts, top=[r["feature"] for r in rows[:TOP_N]],
                        directions={r["feature"]: r["direction_rho"] for r in rows[:TOP_N]},
                        engineered_verdicts=eng, no_effect=[r["feature"] for r in rows if r["importance"] <= 0],
                        importance_sha=sha_of(imp))


# --------------------------------------------------------------------------- gate
def validate_features(d: FeaturesDraft, inp: FeatureInput) -> list[str]:
    problems = []
    names = [t.feature for t in d.top_features]
    if names != inp.top:
        problems.append(f"top_features must be exactly {inp.top} in this order, got {names}")
    for t in d.top_features:
        for field in ("what_it_measures", "effect", "business_meaning"):
            text = getattr(t, field).strip()
            if not text or len(text) > LIMITS["field"]:
                problems.append(f"{t.feature}: {field} must be 1-{LIMITS['field']} chars")
        rho = inp.directions.get(t.feature)
        if rho is not None and abs(rho) >= 0.2:
            up, down = bool(UP.search(t.effect)), bool(DOWN.search(t.effect))
            if rho < 0 and not down:
                problems.append(f"{t.feature}: effect must say that higher values LOWER the predicted probability")
            if rho > 0 and not up:
                problems.append(f"{t.feature}: effect must say that higher values RAISE the predicted probability")
        if PAGEVALUES.search(f"{t.feature} {t.effect} {t.business_meaning}"):
            problems.append("Do not present PageValues as a feature of the main model")
    got = {e.feature: e.verdict for e in d.engineered}
    if set(got) != set(inp.engineered_verdicts):
        problems.append(f"engineered must cover exactly {sorted(inp.engineered_verdicts)}")
    for f, v in inp.engineered_verdicts.items():
        if f in got and got[f] != v:
            problems.append(f"{f}: verdict must be '{v}' (decided by the code), got '{got[f]}'")
    for e in d.engineered:
        if not e.explanation.strip() or len(e.explanation) > LIMITS["field"]:
            problems.append(f"{e.feature}: explanation must be 1-{LIMITS['field']} chars")
    missing = [f for f in inp.no_effect if f not in d.low_impact_note]
    if missing:
        problems.append(f"low_impact_note must name every feature with no measurable effect; missing {missing}")
    if len(d.low_impact_note) > LIMITS["note"]:
        problems.append(f"low_impact_note must be <= {LIMITS['note']} chars")
    if not d.summary.strip() or len(d.summary) > LIMITS["summary"]:
        problems.append(f"summary must be 1-{LIMITS['summary']} chars")
    if not 1 <= len(d.caveats) <= 3 or any(len(c) > LIMITS["caveat"] for c in d.caveats):
        problems.append(f"caveats: 1-3 items, each <= {LIMITS['caveat']} chars")
    text = " ".join([d.summary, d.low_impact_note, *d.caveats, *[e.explanation for e in d.engineered]] +
                    [f"{t.what_it_measures} {t.effect} {t.business_meaning}" for t in d.top_features])
    problems += invented_problem(text, inp.facts)
    return problems


# --------------------------------------------------------------------------- fallback
MEANING = {
    "ExitRates": "how often the pages the visitor viewed end a visit",
    "BounceRates": "how often the pages viewed are left without another request",
    "Month": "the month of the visit",
    "VisitorType": "whether the visitor is new or returning",
    "ProductRelated_Duration": "time spent on product pages",
    "ProductRelated": "number of product pages viewed",
    "Administrative_Duration": "time spent on account pages",
    "Administrative": "number of account pages viewed",
    "AvgSecondsPerProductPage": "average time per product page",
    "ProductPageShare": "share of viewed pages that are product pages",
    "TrafficType": "the traffic source code",
    "TotalPages": "total pages viewed",
    "TotalDuration": "total time on the site",
}


def fallback_draft(imp: dict) -> FeaturesDraft:
    inp = make_input(imp)
    rows = {r["feature"]: r for r in imp["features"]}
    tops = []
    for f in inp.top:
        r = rows[f]
        if r["direction_rho"] is None:
            effect = "Categorical: some values raise and others lower the predicted probability."
        elif r["direction_rho"] > 0:
            effect = f"Higher values go with a higher predicted probability (correlation {r['direction_rho']})."
        else:
            effect = f"Higher values go with a lower predicted probability (correlation {r['direction_rho']})."
        tops.append(TopFeature(feature=f, what_it_measures=MEANING.get(f, f).capitalize() + ".", effect=effect,
                               business_meaning=f"Shuffling it lowers PR-AUC by {r['importance']}, so the model relies "
                                                f"on it."))
    eng = [EngineeredVerdict(feature=f, verdict=v,
                             explanation=f"Importance {rows[f]['importance']} with spread {rows[f]['importance_std']}.")
           for f, v in inp.engineered_verdicts.items()]
    return FeaturesDraft(
        summary=f"The model relies most on {', '.join(inp.top[:3])}. Importance is measured by how much PR-AUC "
                f"drops on validation when a feature is shuffled.",
        top_features=tops, engineered=eng,
        low_impact_note=("No measurable effect: " + ", ".join(inp.no_effect) + ".") if inp.no_effect else
                        "Every feature has a measurable effect.",
        caveats=["Correlated features can share importance, so a low rank does not always mean useless information."])


# --------------------------------------------------------------------------- the agent
def _run_crew(inp: FeatureInput, cfg: dict) -> FeaturesDraft:
    from crewai import Agent, Crew, Process, Task

    agent = Agent(
        role="Feature analyst",
        goal="Explain which session signals drive the purchase model and what they mean for the store",
        backstory="You work in the Data Scientist crew. Python computed permutation importance and the verdicts; "
                  "you explain them in plain words. You never invent, round or derive numbers.",
        llm=make_llm(cfg, temperature=0.2), allow_delegation=False, max_iter=2, verbose=False,
        max_execution_time=cfg["llm"].get("timeout_seconds", 30) * 3,
    )

    def guardrail(output):
        draft = output.pydantic
        if draft is None:
            return False, "Return JSON with all fields."
        problems = validate_features(draft, inp)
        return (True, output) if not problems else (False, "Fix these problems: " + "; ".join(problems))

    facts = "\n".join(f"- {f}" for f in inp.facts)
    verdicts = ", ".join(f"{k}: {v}" for k, v in inp.engineered_verdicts.items())
    task = Task(
        description=(
            "Explain the feature importance of a purchase-prediction model from these verified facts:\n"
            f"{facts}\n\n"
            "Rules:\n"
            f"- top_features: exactly these {TOP_N}, in this order: {', '.join(inp.top)}. For each: what_it_measures, "
            "effect (for numeric features say whether higher values raise or lower the predicted probability, "
            "following the correlation sign in the facts), business_meaning for the store. Each field <= "
            f"{LIMITS['field']} chars.\n"
            f"- engineered: one entry per engineered feature with the code's verdict ({verdicts}) and a short "
            "explanation.\n"
            f"- low_impact_note: name every feature with no measurable effect ({', '.join(inp.no_effect) or 'none'}) "
            f"and say what that means (<= {LIMITS['note']} chars).\n"
            f"- summary: 2-3 sentences (<= {LIMITS['summary']} chars); caveats: 1-3 items.\n"
            "- Copy every number exactly as written in the facts; do not calculate new numbers."),
        expected_output="JSON with summary, top_features, engineered, low_impact_note, caveats",
        agent=agent, output_pydantic=FeaturesDraft, guardrail=guardrail, guardrail_max_retries=2)
    result = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff()
    if result.pydantic is None:
        raise ValueError("Agent returned no structured output")
    return result.pydantic


def write_feature_analysis(imp: dict, cfg: dict | None = None, use_llm: bool = True,
                           refresh: bool = False) -> FeaturesResult:
    """Always returns an analysis that passes the gate (agent, reused agent draft, or fallback)."""
    cfg = cfg or load_config()
    t0 = time.perf_counter()
    inp = make_input(imp)
    saved = resolve(cfg["paths"]["scientist_dir"]) / "features_analysis.json"
    problems: list[str] = []

    if not refresh and saved.exists():
        try:
            prev = FeaturesResult.model_validate_json(saved.read_text(encoding="utf-8"))
            if prev.source == "agent" and prev.importance_sha == inp.importance_sha:
                errs = validate_features(prev.draft, inp)
                if not errs:
                    log.info("[agent] features: reusing the agent draft for %s (no LLM call)", inp.importance_sha)
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
            errs = validate_features(draft, inp)
            if errs:
                problems.extend(errs)
                draft = None
        except Exception as e:
            problems.append(short_error(e))

    source = "agent" if draft is not None else "fallback"
    if draft is None:
        draft = fallback_draft(imp)
        log.warning("[agent] features: fallback used (%s)", "; ".join(problems))
    else:
        log.info("[agent] feature analysis written in %.1fs", time.perf_counter() - t0)
    return FeaturesResult(draft=draft, source=source, model=cfg["llm"]["model"] if source == "agent" else None,
                          importance_sha=inp.importance_sha, problems=problems if source == "fallback" else [],
                          seconds=round(time.perf_counter() - t0, 2))


def render_markdown(res: FeaturesResult, imp: dict) -> str:
    d = res.draft
    who = f"Feature Analyst agent ({res.model})" if res.source == "agent" else "fallback template (agent unavailable)"
    lines = ["# ShopIntent AI · What drives the purchase model", "",
             f"_Prose by the {who} from `feature_importance.json` ({res.importance_sha}). "
             f"The table is generated by code._", "", d.summary.strip(), "", "## Top features", ""]
    for i, t in enumerate(d.top_features, 1):
        lines += [f"### {i}. `{t.feature}`", "", f"**What it measures:** {t.what_it_measures.strip()}", "",
                  f"**Effect:** {t.effect.strip()}", "", f"**For the store:** {t.business_meaning.strip()}", ""]
    lines += ["## Engineered features", "", "| Feature | Definition | Verdict | Why |", "|---|---|---|---|"]
    for e in d.engineered:
        lines.append(f"| `{e.feature}` | {imp['engineered_definitions'].get(e.feature, '')} | "
                     f"{'✅ useful' if e.verdict == 'useful' else '➖ weak'} | {e.explanation.strip().replace('|', '/')} |")
    lines += ["", "## Features with little or no effect", "", d.low_impact_note.strip(), "",
              "## All features", "",
              f"_{imp['method'].capitalize()}, {imp['n_repeats']} repeats, seed {imp['seed']}. Direction = Spearman "
              f"correlation with the predicted probability (numeric features only)._", "",
              "| Rank | Feature | Type | Importance (PR-AUC drop) | Spread | Direction |", "|---|---|---|---|---|---|"]
    for r in imp["features"]:
        dirn = "" if r["direction_rho"] is None else f"{r['direction_rho']:+.3f}"
        lines.append(f"| {r['rank']} | `{r['feature']}` | {r['kind']}{' (engineered)' if r['engineered'] else ''} | "
                     f"{r['importance']:.4f} | {r['importance_std']:.4f} | {dirn} |")
    lines += ["", "## Caveats", ""] + [f"- {c.strip()}" for c in d.caveats] + [""]
    return "\n".join(lines)


def save(res: FeaturesResult, imp: dict, cfg: dict) -> None:
    out_dir = resolve(cfg["paths"]["scientist_dir"])
    (out_dir / "features.md").write_text(render_markdown(res, imp), encoding="utf-8", newline="\n")
    keep = res.model_copy(update={"reused": False, "seconds": 0.0})
    (out_dir / "features_analysis.json").write_text(keep.model_dump_json(indent=2) + "\n", encoding="utf-8",
                                                    newline="\n")
