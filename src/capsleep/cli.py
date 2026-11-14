from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cohort import verify_manifest
from .config import load_settings
from .analysis_cohort import train_analysis_cohort, validate_analysis_cohort
from .posthoc import run_posthoc_primary
from .pipeline import (
    analyze_cap,
    extract_cohort,
    record_inventory,
    diagnose_annotation_cohort,
    train_cap,
    train_primary,
    validate_annotation_cohort,
    validate_feature_schema,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="CAP Sleep Database analysis pipeline")
    parser.add_argument(
        "command",
        choices=(
            "verify-cohort", "record-inventory", "extract", "validate-schema",
            "validate-annotations", "diagnose-annotations", "train-primary",
            "validate-analysis-cohort", "train-analysis-cohort",
            "posthoc-primary", "analyze-cap", "train-cap",
        ),
    )
    parser.add_argument("--config", help="Pipeline TOML configuration path")
    parser.add_argument(
        "--cohort",
        type=Path,
        help="Versioned analysis cohort JSON (for analysis-cohort commands)",
    )
    arguments = parser.parse_args()
    settings = load_settings(arguments.config)
    if arguments.command == "verify-cohort":
        result = verify_manifest(settings.manifest_path)
    elif arguments.command == "record-inventory":
        result = record_inventory(settings)
    elif arguments.command == "extract":
        result = extract_cohort(settings)
    elif arguments.command == "validate-schema":
        result = validate_feature_schema(settings)
    elif arguments.command == "validate-annotations":
        result = validate_annotation_cohort(settings)
    elif arguments.command == "diagnose-annotations":
        result = diagnose_annotation_cohort(settings)
    elif arguments.command == "train-primary":
        result = train_primary(settings)
    elif arguments.command == "validate-analysis-cohort":
        result = validate_analysis_cohort(settings, arguments.cohort)
    elif arguments.command == "train-analysis-cohort":
        result = train_analysis_cohort(settings, arguments.cohort)
    elif arguments.command == "posthoc-primary":
        result = run_posthoc_primary(settings)
    elif arguments.command == "analyze-cap":
        result = analyze_cap(settings)
    else:
        result = train_cap(settings)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
