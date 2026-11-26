# ShopIntent AI

Predicting online shopper purchase intent with CrewAI and Streamlit.

A **CrewAI Flow** coordinates two agent crews:

| Crew | Role | Outputs (`artifacts/`) |
|---|---|---|
| **Data Analyst** | Ingest, validate, clean, EDA, business insights, dataset contract | `clean_data.csv`, `eda_report.html`, `insights.md`, `dataset_contract.json` |
| **Data Scientist** | Validate contract, feature engineering, train & compare models, model card | `features.csv`, `model.pkl`, `evaluation_report.md`, `model_card.md` |

A **Streamlit app** runs a *visit simulation*: it scores a completed session from the dataset and shows the purchase probability, the segment, and the recommended intervention.

## Dataset

[UCI Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset), 12,330 sessions, 15.5% purchases. Licensed CC BY 4.0. Target: `Revenue`.

## Key design decisions

- **Simulation, not real time.** The data are completed sessions.
- **Main model excludes `PageValues`.** The model that includes it is only a comparison in the evaluation report.
- **Fixed thresholds** (`src/config.yaml`), validated on validation data only:

| Purchase probability | Segment | Intervention |
|---|---|---|
| ≥ 50% | Buyer | None |
| 20–50% | Hesitant | Coupon (returning visitor) / Chat (new visitor) |
| < 20% | Unlikely | None |

## Project structure

```
data/raw/             original dataset
artifacts/analyst/    Data Analyst crew outputs
artifacts/scientist/  Data Scientist crew outputs
artifacts/agents/     sample intervention agent messages
src/config.yaml       seed, paths, thresholds, rules, agent settings
src/tools/            deterministic Python tools (cleaning, validation, training)
src/agents/           intervention agent (acts after the rule decides)
src/crews/            CrewAI crews
app/                  Streamlit app
logs/                 run logs
```

## Setup

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # then add your LLM API key
```

## Run

```bash
python -m src.flow                  # whole pipeline: raw CSV -> artifacts, with validation gates
streamlit run app/streamlit_app.py  # visit simulation at http://localhost:8501
```

`python -m src.flow --plot` also opens an interactive diagram of the flow.

### The flow

```
analyst_stage ──► validator_stage ──► validator_gate ──data_ok──►
insights_stage ──► insights_gate ──insights_ok──►
handoff_gate ──handoff_ok──► scientist_stage ──► app_gate ──ready──►
intervention_stage ──► intervention_gate ──agent_ok──► finish

every gate can return "failed" ──► handle_failure
```

| Step | What it does |
|---|---|
| `analyst_stage` | Raw CSV → `clean_data.csv` + `dataset_contract.json` + `data_quality.json` |
| `validator_stage` | Data Validator agent explains `data_quality.json` in `data_quality.md` (also when the pipeline failed) |
| `validator_gate` | The tools say the data is ready; the report's verdict matches the tools, explains every cleaning action and invents no numbers |
| `insights_stage` | EDA tool → `eda_stats.json` + `eda_report.html`; Insights Analyst agent → `insights.md` |
| `insights_gate` | All analyst outputs exist; every number in `insights.md` is traced to the EDA facts; PageValues stays out; the fallback is valid |
| `handoff_gate` | Checks that the contract matches the clean data and that all required model features exist |
| `scientist_stage` | Features, model selection, main + comparison models, `evaluation_report.md` |
| `app_gate` | Checks that the app can use the artifacts: the model loads and scores sessions end to end, and the main model excludes `PageValues` |
| `intervention_stage` | The intervention agent writes one coupon and one chat message for hesitant test visits → `artifacts/agents/intervention_samples.json` |
| `intervention_gate` | Checks the agent contract: every message passes the output gate, buyer/unlikely visits and inputs that contradict the rule are refused, bad drafts are caught, the fallback works |

If any check fails, the flow stops at that step, skips the later stages, logs the reason and exits with code 1.
Each run writes a summary to `logs/flow_last_run.json` and a full log to `logs/pipeline.log`.
All steps are deterministic (seed in `src/config.yaml`), so rerunning reproduces the artifacts byte for byte.

`python -m src.flow --no-llm` runs everything without API calls (agents reuse a saved draft or use their fallback).

### The Insights Analyst agent (Data Analyst crew)

Python computes every number (`src/tools/eda_tools.py`); the agent only interprets them (`src/agents/insights_agent.py`).

| | |
|---|---|
| **Input** | The plain-language EDA facts (purchase rate by month, visitor type, weekend, holidays, traffic source; buyer vs non-buyer medians; correlations). PageValues is never included. |
| **Output** | Summary, 4–6 insights (title, finding with evidence, business action), caveats. Code renders them to `insights.md`. |
| **Check** | Every number the agent writes must appear in the facts (no invented or derived numbers), every finding cites a number, lengths and counts, no PageValues. Runs as a CrewAI guardrail (one retry with feedback) and again after the run. |
| **On failure** | Deterministic insights built from the same facts, marked as fallback with the reason. |
| **Reproducibility** | The agent draft is saved in `insights.json` with a hash of the stats. While the stats are unchanged it is re-checked and reused without an LLM call. `--refresh-agents` asks the agent again. |

### The Data Validator agent (Data Analyst crew)

The Python tools decide whether the data is usable (`src/tools/data_tools.py`); the agent explains that decision and cannot change it (`src/agents/validator_agent.py`).

| | |
|---|---|
| **Input** | The pipeline status (ready / failed) and facts from `data_quality.json`: raw checks, cleaning actions, contract checks, extra checks, errors. |
| **Output** | Verdict, summary, 3–7 findings (what and why it matters), one explanation per cleaning action. `data_quality.md` always shows the tool's own record next to the agent's explanation. |
| **Check** | The verdict must equal the tools' status, one explanation per cleaning action, every number traced to the facts, lengths and counts. |
| **On failure** | Deterministic report from the same facts, marked as fallback. If the data itself is broken, the agent still explains why and the flow stops at the gate. |
| **Reproducibility** | Same reuse as the insights agent: re-checked and reused while `data_quality.json` is unchanged. |

### The intervention agent

The agent does **not** replace the model or the thresholds. The model gives a probability, the fixed rule decides
whether to intervene, and only then does the agent write the wording (`src/agents/intervention_agent.py`).

| | |
|---|---|
| **Input** | `InterventionInput`: one hesitant visit whose rule decision is coupon (returning) or chat (new). The agent re-runs the rule and refuses any other visit. |
| **Output** | `InterventionMessage`: headline, message, button text. Discount, coupon code and validity come from `config.yaml`, never from the LLM. |
| **Check** | Lengths, coupon text has the exact code and discount and no other percentage, chat text offers no discount, no mention of the model or score. It runs as a CrewAI task guardrail (one retry with feedback) and again after the run. |
| **On failure** | No API key, API error, timeout, or output that still fails the check → the fixed template from `config.yaml` is shown, marked as fallback, with the reason. The decision never changes. |

Every call is logged to `logs/intervention_agent.jsonl`. In the app, hesitant visits get a **Write the message** button.

The individual stages can still be run on their own:

```bash
python -m src.run_data_pipeline     # Data Analyst stage only
python -m src.run_model_pipeline    # Data Scientist stage only
```
