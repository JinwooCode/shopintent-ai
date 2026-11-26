"""Contract Author agent (Data Analyst crew): documents every column of the dataset contract.

    clean_data.csv ──► build_contract (Python) ──► dataset_contract.json   (schema, ranges, allowed values,
                                                          │                  assumptions, constraints: code only)
                          column facts + source notes ──► [this agent] ──► gate ──► column_docs
                                                                            │       (added to the contract as
                                                                            │        documentation) + dataset_contract.md
                                                                            └─ fail ──► docs from the source notes

The machine-checked part of the contract never comes from the LLM. The agent only adds what each column
means and how the Data Scientist crew should use it; validate_against_contract() ignores column_docs.

Contract
  Input   ContractInput: one fact line per column (type, range or allowed values, role) plus the dataset's
          documented meaning of each column (SOURCE_NOTES, from the UCI dataset description).
  Output  ContractDocs: summary + one ColumnDoc (meaning, how_to_use) per column.
  Gate    validate_docs(): exactly one doc per contract column, no unknown columns; numbers traced to the
          facts; anonymized code columns are described as categories; PageValues says it is excluded from
          the main model; the target is described as the target; lengths.
  Failure deterministic docs built from SOURCE_NOTES, marked as fallback.
  Reuse   while the contract's machine part is unchanged, the saved agent docs are re-checked and reused.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.agents.common import invented_problem, make_llm, sha_of, short_error
from src.utils import get_logger, load_config, resolve

load_dotenv(resolve(".env"))
log = get_logger("contract_agent")

LIMITS = {"summary": 600, "meaning": 260, "how_to_use": 260}

# Documented meaning of each column (UCI Online Shoppers Purchasing Intention dataset description).
SOURCE_NOTES = {
    "Administrative": "Number of account-management pages the visitor viewed in the session.",
    "Administrative_Duration": "Total seconds spent on account-management pages in the session.",
    "Informational": "Number of informational pages (about the site, contact, policies) viewed in the session.",
    "Informational_Duration": "Total seconds spent on informational pages in the session.",
    "ProductRelated": "Number of product pages viewed in the session.",
    "ProductRelated_Duration": "Total seconds spent on product pages in the session.",
    "BounceRates": "Average bounce rate (Google Analytics) of the pages viewed: share of visitors who entered "
                   "on that page and left without another request.",
    "ExitRates": "Average exit rate (Google Analytics) of the pages viewed: share of page views that were the "
                 "last of their session.",
    "PageValues": "Average Google Analytics page value of the pages viewed before a completed transaction.",
    "SpecialDay": "Closeness of the visit date to a special day such as Mother's Day or Valentine's Day, "
                  "from 0 (not close) to 1 (closest).",
    "Month": "Month of the visit.",
    "OperatingSystems": "Anonymized code of the visitor's operating system.",
    "Browser": "Anonymized code of the visitor's browser.",
    "Region": "Anonymized code of the visitor's region.",
    "TrafficType": "Anonymized code of the traffic source that brought the visitor (e.g. direct, banner, SMS).",
    "VisitorType": "Whether the visitor is new, returning or other.",
    "Weekend": "Whether the visit happened on a weekend.",
    "Revenue": "Whether the session ended in a purchase.",
}
CATEGORY_WORDS = re.compile(r"\b(categor\w*|code|label|identifier)\b", re.I)
EXCLUDED_WORDS = re.compile(r"\b(exclud\w*|not used|must not|leak\w*)\b", re.I)
TARGET_WORDS = re.compile(r"\b(target|label|outcome|predict\w*)\b", re.I)


# --------------------------------------------------------------------------- contract
class ContractInput(BaseModel):
    columns: list[str]
    facts: list[str]
    code_columns: list[str]
    excluded: list[str]
    target: str
    contract_sha: str


class ColumnDoc(BaseModel):
    column: str
    meaning: str
    how_to_use: str


class ContractDocs(BaseModel):
    summary: str
    columns: list[ColumnDoc]


class ContractResult(BaseModel):
    docs: ContractDocs
    source: Literal["agent", "fallback"]
    model: str | None = None
    contract_sha: str
    reused: bool = False
    problems: list[str] = Field(default_factory=list)
    seconds: float = 0.0


def machine_part(contract: dict) -> dict:
    """The code-built part of the contract (everything except the agent's documentation)."""
    return {k: v for k, v in contract.items() if k not in ("column_docs", "column_docs_source")}


def _column_fact(col: str, spec: dict, contract: dict) -> str:
    role = ("target" if col == contract["target"]["name"] else
            "excluded from the main model" if col in contract.get("main_model_excluded_features", []) else
            "feature")
    parts = [f"type {spec['type']}"]
    if "min" in spec or "max" in spec:
        parts.append(f"range {spec.get('min', '-')} to {spec.get('max', 'no upper limit')}")
    if "allowed_values" in spec:
        vals = spec["allowed_values"]
        parts.append(f"{len(vals)} allowed values: {', '.join(map(str, vals[:12]))}" + (" ..." if len(vals) > 12 else ""))
    return f"{col}: {', '.join(parts)}; role: {role}. Source note: {SOURCE_NOTES.get(col, 'n/a')}"


def make_input(contract: dict) -> ContractInput:
    cols = contract["columns"]
    facts = [f"The clean dataset has {contract['row_count']:,} rows; {contract['target']['positive_rate'] * 100:.1f}% "
             f"are purchases."] + [_column_fact(c, s, contract) for c, s in cols.items()]
    facts += [f"Assumption: {a}" for a in contract.get("assumptions", [])]
    facts += [f"Constraint: {c}" for c in contract.get("constraints", [])]
    return ContractInput(columns=list(cols), facts=facts,
                         code_columns=[c for c, s in cols.items() if s["type"] == "category_code"],
                         excluded=contract.get("main_model_excluded_features", []),
                         target=contract["target"]["name"], contract_sha=sha_of(machine_part(contract)))


# --------------------------------------------------------------------------- gate
def validate_docs(docs: ContractDocs, inp: ContractInput) -> list[str]:
    problems = []
    names = [d.column for d in docs.columns]
    missing = [c for c in inp.columns if c not in names]
    unknown = [c for c in names if c not in inp.columns]
    dupes = sorted({c for c in names if names.count(c) > 1})
    if missing:
        problems.append(f"Missing docs for columns: {missing}")
    if unknown:
        problems.append(f"Docs for columns that are not in the contract: {unknown}")
    if dupes:
        problems.append(f"Columns documented more than once: {dupes}")
    if not docs.summary.strip() or len(docs.summary) > LIMITS["summary"]:
        problems.append(f"summary must be 1-{LIMITS['summary']} chars")
    for d in docs.columns:
        for field in ("meaning", "how_to_use"):
            text = getattr(d, field).strip()
            if not text or len(text) > LIMITS[field]:
                problems.append(f"{d.column}: {field} must be 1-{LIMITS[field]} chars")
        both = f"{d.meaning} {d.how_to_use}"
        if d.column in inp.code_columns and not CATEGORY_WORDS.search(both):
            problems.append(f"{d.column}: say that it is a category code, not a quantity")
        if d.column in inp.excluded and not EXCLUDED_WORDS.search(both):
            problems.append(f"{d.column}: say that it is excluded from the main model (it leaks the answer)")
        if d.column == inp.target and not TARGET_WORDS.search(both):
            problems.append(f"{d.column}: say that it is the prediction target")
    text = " ".join([docs.summary] + [f"{d.meaning} {d.how_to_use}" for d in docs.columns])
    problems += invented_problem(text, inp.facts)
    return problems


# --------------------------------------------------------------------------- fallback
def fallback_docs(contract: dict) -> ContractDocs:
    inp = make_input(contract)
    docs = []
    for col, spec in contract["columns"].items():
        if col == inp.target:
            use = "Prediction target. Never use it as a feature."
        elif col in inp.excluded:
            use = "Excluded from the main model: it is computed from purchases, so it leaks the answer."
        elif spec["type"] == "category_code":
            use = "Category code, not a quantity: encode as a category."
        elif spec["type"] in ("category", "boolean"):
            use = "Categorical feature: encode as a category."
        else:
            use = "Numeric feature, use as is."
        docs.append(ColumnDoc(column=col, meaning=SOURCE_NOTES.get(col, col), how_to_use=use))
    return ContractDocs(summary="One row per completed session. The machine-checked schema, ranges and allowed "
                                "values below are enforced by the validation tools.", columns=docs)


# --------------------------------------------------------------------------- the agent
def _run_crew(inp: ContractInput, cfg: dict) -> ContractDocs:
    from crewai import Agent, Crew, Process, Task

    agent = Agent(
        role="Dataset contract author",
        goal="Document every column of the dataset contract so the Data Scientist crew uses it correctly",
        backstory="You work in the Data Analyst crew. The schema, ranges and allowed values were computed by "
                  "Python and are enforced by validation tools; you write the documentation around them.",
        llm=make_llm(cfg, temperature=0.2), allow_delegation=False, max_iter=2, verbose=False,
        max_execution_time=cfg["llm"].get("timeout_seconds", 30) * 3,
    )

    def guardrail(output):
        docs = output.pydantic
        if docs is None:
            return False, "Return JSON with summary and columns (column, meaning, how_to_use)."
        problems = validate_docs(docs, inp)
        return (True, output) if not problems else (False, "Fix these problems: " + "; ".join(problems))

    facts = "\n".join(f"- {f}" for f in inp.facts)
    task = Task(
        description=(
            "Write the documentation part of a dataset contract from these facts:\n"
            f"{facts}\n\n"
            "Rules:\n"
            f"- One entry for EVERY column, exactly these names: {', '.join(inp.columns)}.\n"
            f"- meaning: what the column measures, in plain words (<= {LIMITS['meaning']} chars).\n"
            f"- how_to_use: how the Data Scientist crew should treat it in modelling (<= {LIMITS['how_to_use']} chars).\n"
            f"- Say explicitly that {', '.join(inp.code_columns)} are category codes, not quantities.\n"
            f"- Say explicitly that {', '.join(inp.excluded)} is excluded from the main model because it leaks the answer.\n"
            f"- Say explicitly that {inp.target} is the prediction target.\n"
            f"- summary: 2-3 sentences about the dataset and how to use the contract (<= {LIMITS['summary']} chars).\n"
            "- Copy numbers exactly as in the facts; do not calculate new ones."),
        expected_output="JSON with summary and columns: list of {column, meaning, how_to_use}",
        agent=agent, output_pydantic=ContractDocs, guardrail=guardrail, guardrail_max_retries=1)
    result = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff()
    if result.pydantic is None:
        raise ValueError("Agent returned no structured output")
    return result.pydantic


def write_contract_docs(contract: dict, cfg: dict | None = None, use_llm: bool = True,
                        refresh: bool = False) -> ContractResult:
    """Always returns docs that pass the gate (agent, reused agent docs, or fallback)."""
    cfg = cfg or load_config()
    t0 = time.perf_counter()
    inp = make_input(contract)
    saved = resolve(cfg["paths"]["analyst_dir"]) / "contract_docs.json"
    problems: list[str] = []

    if not refresh and saved.exists():
        try:
            prev = ContractResult.model_validate_json(saved.read_text(encoding="utf-8"))
            if prev.source == "agent" and prev.contract_sha == inp.contract_sha:
                errs = validate_docs(prev.docs, inp)
                if not errs:
                    log.info("[agent] contract docs: reusing the agent docs for %s (no LLM call)", inp.contract_sha)
                    return prev.model_copy(update={"reused": True})
                problems.append(f"Saved docs no longer pass the gate: {errs}")
        except Exception as e:
            problems.append(f"Could not read the saved docs: {short_error(e)}")

    docs = None
    if not use_llm:
        problems.append("Agent disabled (use_llm=False)")
    elif not os.getenv("OPENAI_API_KEY"):
        problems.append("OPENAI_API_KEY is not set")
    else:
        try:
            docs = _run_crew(inp, cfg)
            errs = validate_docs(docs, inp)
            if errs:
                problems.extend(errs)
                docs = None
        except Exception as e:
            problems.append(short_error(e))

    source = "agent" if docs is not None else "fallback"
    if docs is None:
        docs = fallback_docs(contract)
        log.warning("[agent] contract docs: fallback used (%s)", "; ".join(problems))
    else:
        log.info("[agent] contract docs: %d columns documented in %.1fs", len(docs.columns), time.perf_counter() - t0)
    # keep the contract's column order
    order = {c: i for i, c in enumerate(inp.columns)}
    docs = docs.model_copy(update={"columns": sorted(docs.columns, key=lambda d: order.get(d.column, 99))})
    return ContractResult(docs=docs, source=source, model=cfg["llm"]["model"] if source == "agent" else None,
                          contract_sha=inp.contract_sha, problems=problems if source == "fallback" else [],
                          seconds=round(time.perf_counter() - t0, 2))


def render_markdown(res: ContractResult, contract: dict) -> str:
    who = f"Contract Author agent ({res.model})" if res.source == "agent" else "fallback (dataset source notes)"
    lines = ["# ShopIntent AI · Dataset contract", "",
             f"_Machine-checked part: `dataset_contract.json`, built by code and enforced by the validation tools. "
             f"Column documentation written by the {who} ({res.contract_sha})._", "",
             res.docs.summary.strip(), "",
             f"**Rows:** {contract['row_count']:,} · **Target:** `{contract['target']['name']}` "
             f"(positive rate {contract['target']['positive_rate'] * 100:.1f}%) · "
             f"**Excluded from the main model:** {', '.join(f'`{c}`' for c in contract.get('main_model_excluded_features', []))}",
             "", "## Columns", "", "| Column | Type | Allowed values / range | Meaning | How to use |", "|---|---|---|---|---|"]
    docs = {d.column: d for d in res.docs.columns}
    for col, spec in contract["columns"].items():
        if "allowed_values" in spec:
            rng = ", ".join(map(str, spec["allowed_values"]))
        else:
            rng = f"{spec.get('min', '')} to {spec.get('max', '∞')}"
        d = docs[col]
        cell = lambda s: s.strip().replace("|", "/")  # noqa: E731
        lines.append(f"| `{col}` | {spec['type']} | {rng} | {cell(d.meaning)} | {cell(d.how_to_use)} |")
    lines += ["", "## Assumptions", ""] + [f"- {a}" for a in contract.get("assumptions", [])]
    lines += ["", "## Constraints (enforced)", ""] + [f"- {c}" for c in contract.get("constraints", [])]
    return "\n".join(lines) + "\n"


def save(res: ContractResult, contract: dict, cfg: dict) -> dict:
    """Write contract_docs.json + dataset_contract.md and add column_docs to dataset_contract.json."""
    out_dir = resolve(cfg["paths"]["analyst_dir"])
    keep = res.model_copy(update={"reused": False, "seconds": 0.0})
    (out_dir / "contract_docs.json").write_text(keep.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
    (out_dir / "dataset_contract.md").write_text(render_markdown(res, contract), encoding="utf-8", newline="\n")
    full = machine_part(contract)
    full["column_docs"] = {d.column: {"meaning": d.meaning.strip(), "how_to_use": d.how_to_use.strip()}
                           for d in res.docs.columns}
    full["column_docs_source"] = res.source
    with open(out_dir / "dataset_contract.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(full, indent=2) + "\n")
    return full
