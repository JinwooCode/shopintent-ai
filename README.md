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
src/config.yaml       seed, paths, thresholds, rules
src/tools/            deterministic Python tools (cleaning, validation, training)
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
analyst_stage ──► handoff_gate ──handoff_ok──► scientist_stage ──► app_gate ──ready──► finish
                       │                                              │
                       └───────────────────failed─────────────────────┴──► handle_failure
```

| Step | What it does |
|---|---|
| `analyst_stage` | Raw CSV → `clean_data.csv` + `dataset_contract.json` |
| `handoff_gate` | Checks that the contract matches the clean data and that all required model features exist |
| `scientist_stage` | Features, model selection, main + comparison models, `evaluation_report.md` |
| `app_gate` | Checks that the app can use the artifacts: the model loads and scores sessions end to end, and the main model excludes `PageValues` |

If any check fails, the flow stops at that step, skips the later stages, logs the reason and exits with code 1.
Each run writes a summary to `logs/flow_last_run.json` and a full log to `logs/pipeline.log`.
All steps are deterministic (seed in `src/config.yaml`), so rerunning reproduces the artifacts byte for byte.

The individual stages can still be run on their own:

```bash
python -m src.run_data_pipeline     # Data Analyst stage only
python -m src.run_model_pipeline    # Data Scientist stage only
```
