"""Intervention agent: writes the coupon / chat message AFTER the rule has decided.

    model ──► probability ──► decide() (fixed thresholds + rules) ──► [this agent] ──► gate ──► app
                                    │                                      │            │
                                    └── buyer / unlikely: agent never runs  └─ fail ─────┴──► fallback template

Contract
  Input   InterventionInput: one hesitant visit whose decision is already coupon (returning) or chat (new).
          check_input() refuses anything else, and re-runs decide() so the input cannot contradict the rule.
  Output  InterventionMessage: headline, message, call to action. The discount, coupon code and validity
          come from config.yaml and are attached by code, never chosen by the LLM.
  Gate    validate_message(): lengths, non-empty, coupon text carries the exact code and discount and no
          other percentage, chat text offers no discount and leaks no model score, no leftover {placeholders}.
          The same check runs as a CrewAI task guardrail (one retry with feedback) and again after the run.
  Failure no API key, API error, timeout, or output that still fails the gate -> the fixed template from
          config.yaml is used, source="fallback", and the reason is logged. The decision itself never changes.

Every call is appended to logs/intervention_agent.jsonl.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Literal

os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from dotenv import load_dotenv  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src.agents.common import make_llm  # noqa: E402
from src.tools.decision import HESITANT, decide  # noqa: E402
from src.utils import get_logger, load_config, resolve  # noqa: E402

load_dotenv(resolve(".env"))
log = get_logger("intervention_agent")

ACTIONS = ("coupon", "chat")
VISITOR_TEXT = {"Returning_Visitor": "a returning visitor", "New_Visitor": "a first-time visitor"}
SCORE_WORDS = re.compile(r"\b(probabilit\w*|score\w*|model|predict\w*|hesitant|segment)\b", re.I)
DISCOUNT_WORDS = re.compile(r"(%|\bdiscount\w*|\bcoupon\w*|\boff\b|\bpromo\w*|\bcode\b)", re.I)


# --------------------------------------------------------------------------- contract
class InterventionInput(BaseModel):
    """What the agent receives. Built from one scored visit plus the rule's decision."""
    session_id: int
    probability: float = Field(ge=0, le=1)
    segment: str
    intervention: str
    visitor_type: str
    month: str
    weekend: bool
    near_holiday: bool
    product_pages: int = Field(ge=0)
    product_minutes: float = Field(ge=0)


class AgentDraft(BaseModel):
    """What the LLM is asked to return (wording only)."""
    headline: str
    message: str
    cta: str


class InterventionMessage(BaseModel):
    """What the agent stage returns to the app. Always valid: either agent output or the fallback."""
    session_id: int
    intervention: Literal["coupon", "chat"]
    headline: str
    message: str
    cta: str
    coupon_code: str | None = None
    discount_pct: int | None = None
    valid_hours: int | None = None
    source: Literal["agent", "fallback"]
    model: str | None = None
    problems: list[str] = Field(default_factory=list)   # why the fallback was used, if it was
    seconds: float = 0.0


class AgentInputError(ValueError):
    """The agent was asked to act on a visit the rule did not send to it."""


# --------------------------------------------------------------------------- input guard
def check_input(inp: InterventionInput, cfg: dict) -> None:
    """Refuse to run unless the rule itself decided coupon/chat for this visit."""
    expected = decide(inp.probability, inp.visitor_type, cfg)
    problems = []
    if expected["segment"] != inp.segment or expected["intervention"] != inp.intervention:
        problems.append(f"Input contradicts the rule: rule says {expected['segment']}/{expected['intervention']}, "
                        f"input says {inp.segment}/{inp.intervention}")
    if inp.segment != HESITANT:
        problems.append(f"Agent only runs for hesitant visits, got '{inp.segment}'")
    if inp.intervention not in ACTIONS:
        problems.append(f"Agent only runs for {ACTIONS}, got '{inp.intervention}'")
    if problems:
        raise AgentInputError("; ".join(problems))


def coupon_code(session_id: int, cfg: dict) -> str:
    """Deterministic code per visit, so the same visit always shows the same code."""
    prefix = cfg["intervention_agent"]["coupon"]["code_prefix"]
    return f"{prefix}-{hashlib.sha1(str(session_id).encode()).hexdigest()[:5].upper()}"


# --------------------------------------------------------------------------- output gate
def validate_message(draft: AgentDraft, intervention: str, cfg: dict, code: str | None) -> list[str]:
    """Return a list of problems; empty means the message may be shown."""
    ac = cfg["intervention_agent"]
    lim = ac["limits"]
    problems = []
    for field, limit in (("headline", lim["headline_max_chars"]), ("message", lim["message_max_chars"]),
                         ("cta", lim["cta_max_chars"])):
        text = getattr(draft, field).strip()
        if not text:
            problems.append(f"{field} is empty")
        elif len(text) > limit:
            problems.append(f"{field} is {len(text)} chars, limit {limit}")
    full = f"{draft.headline} {draft.message} {draft.cta}"
    if re.search(r"[{}]", full):
        problems.append("Text contains an unfilled {placeholder}")
    if SCORE_WORDS.search(full):
        problems.append(f"Text mentions the model or score ('{SCORE_WORDS.search(full).group(0)}')")

    if intervention == "coupon":
        pct = ac["coupon"]["discount_pct"]
        if code and code not in draft.message:
            problems.append(f"Coupon message must contain the exact code {code}")
        if f"{pct}%" not in full.replace(" %", "%"):
            problems.append(f"Coupon message must state the configured discount {pct}%")
        other = sorted({int(n) for n in re.findall(r"(\d+)\s*%", full)} - {pct})
        if other:
            problems.append(f"Coupon message promises a different discount: {other}%")
    else:  # chat: new visitors get help, not a discount
        hit = DISCOUNT_WORDS.search(full)
        if hit:
            problems.append(f"Chat message must not offer a discount or code ('{hit.group(0)}')")
    return problems


# --------------------------------------------------------------------------- fallback
def fallback_draft(intervention: str, cfg: dict, code: str | None) -> AgentDraft:
    ac = cfg["intervention_agent"]
    t = ac["fallback"][intervention]
    fill = dict(discount=ac["coupon"]["discount_pct"], code=code, hours=ac["coupon"]["valid_hours"],
                store=ac["store_name"])
    return AgentDraft(**{k: t[k].format(**fill) for k in ("headline", "message", "cta")})


# --------------------------------------------------------------------------- the agent
def _visit_brief(inp: InterventionInput) -> str:
    when = [f"in {inp.month}"] + (["on a weekend"] if inp.weekend else []) + \
           (["close to a holiday"] if inp.near_holiday else [])
    return (f"{VISITOR_TEXT.get(inp.visitor_type, 'a visitor')} who viewed {inp.product_pages} product pages "
            f"for about {inp.product_minutes:.0f} minutes, {' '.join(when)}, and has not bought yet")


def _task_text(inp: InterventionInput, cfg: dict, code: str | None) -> str:
    ac = cfg["intervention_agent"]
    lim = ac["limits"]
    common = (f"Store: {ac['store_name']}. Visitor: {_visit_brief(inp)}.\n"
              f"Limits: headline <= {lim['headline_max_chars']} chars, message <= {lim['message_max_chars']} chars, "
              f"cta (button text) <= {lim['cta_max_chars']} chars. Plain text, English, friendly, no pressure "
              f"or urgency words (no 'hurry', 'last chance', 'only'), no emojis, no placeholders in braces. Never mention models, scores, probabilities or predictions.")
    if inp.intervention == "coupon":
        c = ac["coupon"]
        return (f"Write a short on-site pop-up offering a coupon to a returning visitor.\n{common}\n"
                f"The offer is FIXED: exactly {c['discount_pct']}% off, code {code}, valid {c['valid_hours']} hours. "
                f"The message must contain the code {code} exactly and the text '{c['discount_pct']}%'. "
                f"Do not mention any other percentage, price or free gift.")
    return (f"Write the opening line of a live-chat invitation for a first-time visitor.\n{common}\n"
            f"Offer help (products, sizing, shipping, returns) and reassurance. Do NOT offer any discount, "
            f"coupon, code or price, and do not use the % sign.")


def _run_crew(inp: InterventionInput, cfg: dict, code: str | None) -> AgentDraft:
    from crewai import Agent, Crew, Process, Task

    llm_cfg = cfg["llm"]
    llm = make_llm(cfg)
    agent = Agent(
        role="Customer engagement copywriter",
        goal="Write one short, honest message that helps an undecided shopper finish their visit",
        backstory="You write on-site messages for an online store. The decision to show a message and the "
                  "offer itself are already made by the store's rules; you only choose the words.",
        llm=llm, allow_delegation=False, max_iter=2, verbose=False,
        max_execution_time=llm_cfg.get("timeout_seconds", 30),
    )

    def guardrail(output):  # CrewAI retries once with this feedback if the draft breaks the contract
        draft = output.pydantic
        if draft is None:
            return False, "Return JSON with headline, message and cta."
        problems = validate_message(draft, inp.intervention, cfg, code)
        return (True, output) if not problems else (False, "Fix these problems: " + "; ".join(problems))

    task = Task(description=_task_text(inp, cfg, code),
                expected_output="JSON with fields headline, message, cta",
                agent=agent, output_pydantic=AgentDraft, guardrail=guardrail, guardrail_max_retries=1)
    result = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff()
    if result.pydantic is None:
        raise ValueError("Agent returned no structured output")
    return result.pydantic


def write_message(inp: InterventionInput, cfg: dict | None = None, use_llm: bool = True) -> InterventionMessage:
    """Run the agent for one visit. Raises AgentInputError if the rule did not choose coupon/chat;
    otherwise always returns a valid message (agent output, or the fallback with the reason)."""
    cfg = cfg or load_config()
    check_input(inp, cfg)
    t0 = time.perf_counter()
    code = coupon_code(inp.session_id, cfg) if inp.intervention == "coupon" else None
    problems: list[str] = []
    draft = None

    if not use_llm or not cfg["intervention_agent"].get("enabled", True):
        problems.append("Agent disabled (use_llm=False or intervention_agent.enabled=false)")
    elif not os.getenv("OPENAI_API_KEY"):
        problems.append("OPENAI_API_KEY is not set")
    else:
        try:
            draft = _run_crew(inp, cfg, code)
            problems = validate_message(draft, inp.intervention, cfg, code)  # final gate, after the guardrail
            if problems:
                draft = None
        except Exception as e:  # API error, timeout, guardrail exhausted, bad JSON
            problems.append(f"{type(e).__name__}: {str(e)[:200]}")

    source = "agent" if draft is not None else "fallback"
    if draft is None:
        draft = fallback_draft(inp.intervention, cfg, code)
    c = cfg["intervention_agent"]["coupon"]
    msg = InterventionMessage(
        session_id=inp.session_id, intervention=inp.intervention, **draft.model_dump(),
        coupon_code=code, discount_pct=c["discount_pct"] if code else None,
        valid_hours=c["valid_hours"] if code else None, source=source,
        model=cfg["llm"]["model"] if source == "agent" else None,
        problems=problems, seconds=round(time.perf_counter() - t0, 2))
    _log_call(inp, msg)
    return msg


def _log_call(inp: InterventionInput, msg: InterventionMessage) -> None:
    if msg.source == "fallback":
        log.warning("[agent] session %s: fallback used (%s)", inp.session_id, "; ".join(msg.problems))
    else:
        log.info("[agent] session %s: %s message written in %.1fs", inp.session_id, msg.intervention, msg.seconds)
    out = resolve(load_config()["paths"]["log_dir"]) / "intervention_agent.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "input": inp.model_dump(), "output": msg.model_dump()}
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def input_from_row(session_id: int, row, probability: float, cfg: dict) -> InterventionInput:
    """Build the agent input from one row of features.csv and its model probability."""
    d = decide(probability, row["VisitorType"], cfg)
    return InterventionInput(
        session_id=int(session_id), probability=float(probability), segment=d["segment"],
        intervention=d["intervention"], visitor_type=str(row["VisitorType"]), month=str(row["Month"]),
        weekend=str(row["Weekend"]).lower() in ("true", "1"), near_holiday=float(row["SpecialDay"]) > 0,
        product_pages=int(row["ProductRelated"]), product_minutes=float(row["ProductRelated_Duration"]) / 60)
