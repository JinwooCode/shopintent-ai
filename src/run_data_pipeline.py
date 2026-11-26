"""Run the deterministic data pipeline: raw CSV -> clean_data.csv + dataset_contract.json.

Also writes data_quality.json (on success AND on failure): what was checked, what was cleaned, and any
errors. The Data Validator agent explains that record in data_quality.md.

Usage (from the project root):
    python -m src.run_data_pipeline
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from src.tools.data_tools import (
    DataValidationError,
    build_contract,
    check_raw,
    clean,
    load_contract,
    load_raw,
    quality_checks,
    save_contract,
    validate_against_contract,
)
from src.utils import get_logger, load_config, resolve

STEPS = ["load raw data", "clean", "build contract", "validate against contract"]


def main() -> int:
    log = get_logger("data_pipeline")
    cfg = load_config()
    out_dir = resolve(cfg["paths"]["analyst_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_path = out_dir / "clean_data.csv"
    contract_path = out_dir / "dataset_contract.json"
    quality = {"status": "failed", "failed_step": None, "raw": {}, "cleaning_actions": [],
               "contract_checks": [], "extra_checks": [], "errors": []}
    step = STEPS[0]

    try:
        log.info("Step 1/4: loading raw data")
        raw = load_raw(resolve(cfg["paths"]["raw_data"]))
        report = check_raw(raw)
        quality["raw"] = report
        log.info("Raw data OK: %s", report)

        step = STEPS[1]
        log.info("Step 2/4: cleaning")
        df, actions = clean(raw)
        quality["cleaning_actions"] = actions
        for a in actions:
            log.info("  - %s", a)
        df.to_csv(clean_path, index=False, lineterminator="\n")  # same bytes on every OS
        log.info("Saved %s", clean_path.relative_to(resolve(".")))
        quality["extra_checks"] = quality_checks(df)

        step = STEPS[2]
        log.info("Step 3/4: building dataset contract")
        contract = build_contract(df, cfg["models"]["main"]["drop_features"])
        save_contract(contract, contract_path)
        log.info("Saved %s", contract_path.relative_to(resolve(".")))

        step = STEPS[3]
        log.info("Step 4/4: validating clean data against the contract")
        # Re-read from disk to validate exactly what was saved
        passed = validate_against_contract(pd.read_csv(clean_path), load_contract(contract_path))
        quality["contract_checks"] = passed
        for p in passed:
            log.info("  OK  %s", p)
        quality["status"] = "ready"

    except DataValidationError as e:
        quality["failed_step"], quality["errors"] = step, e.errors
        for err in e.errors:
            log.error("  FAIL  %s", err)
        log.error("Data pipeline stopped: fix the problems above and run again.")
        return 1
    finally:
        with open(out_dir / "data_quality.json", "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(quality, indent=2) + "\n")

    log.info("Data pipeline finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
