from collections import Counter
import json
from pathlib import Path

from capsleep.cohort import EXPECTED_COUNTS, load_manifest, selected_records


ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_manifest_has_verified_class_counts():
    manifest = load_manifest(ROOT / "config" / "cohort_manifest.json")
    assert manifest["inventory_record_count"] == 108
    assert manifest["inventory_sha256"] == (
        "b794825833c98285c6e9f6052af79837afc1ac1939f7e410fdfc88dded52f945"
    )
    assert Counter(row["label"] for row in manifest["records"]) == EXPECTED_COUNTS


def test_selected_cohort_matches_requested_official_prefixes():
    inventory = (
        [f"n{index}.edf" for index in range(1, 17)]
        + [f"nfle{index}.edf" for index in range(1, 41)]
        + [f"rbd{index}.edf" for index in range(1, 23)]
        + [f"plm{index}.edf" for index in range(1, 11)]
        + ["narco1.edf", "ins1.edf", "brux1.edf", "sdb1.edf"]
    )
    selected = selected_records(inventory)
    assert len(selected) == 88
    assert Counter(row["label"] for row in selected) == EXPECTED_COUNTS
    assert all(not row["record_id"].startswith("narco") for row in selected)
