from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from urllib.request import urlopen


INVENTORY_URL = "https://physionet.org/files/capslpdb/1.0.0/RECORDS"
CLASS_PATTERNS = (
    ("control", re.compile(r"n(1[0-6]|[1-9])\.edf$", re.IGNORECASE)),
    ("nfle", re.compile(r"nfle([1-9]|[1-3][0-9]|40)\.edf$", re.IGNORECASE)),
    ("rbd", re.compile(r"rbd([1-9]|1[0-9]|2[0-2])\.edf$", re.IGNORECASE)),
    ("plm", re.compile(r"plm([1-9]|10)\.edf$", re.IGNORECASE)),
)
EXPECTED_COUNTS = {"control": 16, "nfle": 40, "rbd": 22, "plm": 10}


def fetch_inventory(url: str = INVENTORY_URL, timeout: int = 30) -> tuple[list[str], str]:
    with urlopen(url, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"PhysioNet RECORDS returned HTTP {response.status}")
        content = response.read()
    names = [line.strip() for line in content.decode("ascii").splitlines() if line.strip()]
    if len(names) != len(set(names)):
        raise ValueError("PhysioNet RECORDS contains duplicate filenames")
    return names, hashlib.sha256(content).hexdigest()


def selected_records(inventory: list[str]) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for filename in inventory:
        matches = [(label, pattern.fullmatch(filename)) for label, pattern in CLASS_PATTERNS]
        matched = [(label, match) for label, match in matches if match]
        if not matched:
            continue
        if len(matched) != 1:
            raise ValueError(f"Ambiguous cohort label for {filename}")
        label, match = matched[0]
        assert match is not None
        record_id = filename[:-4]
        if record_id in seen:
            raise ValueError(f"Duplicate record ID: {record_id}")
        seen.add(record_id)
        selected.append({"record_id": record_id, "filename": filename, "label": label})
    counts = Counter(row["label"] for row in selected)
    if dict(counts) != EXPECTED_COUNTS:
        raise ValueError(f"Official inventory cohort counts {dict(counts)} != {EXPECTED_COUNTS}")
    return sorted(selected, key=lambda row: (row["label"], int(re.search(r"\d+$", row["record_id"]).group())))


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records = manifest.get("records", [])
    if len(records) != 88 or len({row.get("record_id") for row in records}) != 88:
        raise ValueError("Cohort manifest must contain exactly 88 unique recordings")
    counts = Counter(row.get("label") for row in records)
    if dict(counts) != EXPECTED_COUNTS:
        raise ValueError(f"Manifest class counts {dict(counts)} != {EXPECTED_COUNTS}")
    for row in records:
        if row.get("filename") != f"{row.get('record_id')}.edf":
            raise ValueError(f"Invalid manifest filename: {row}")
    return manifest


def verify_manifest(path: Path, url: str = INVENTORY_URL) -> dict:
    manifest = load_manifest(path)
    inventory, inventory_hash = fetch_inventory(url)
    current = selected_records(inventory)
    saved = sorted(
        manifest["records"],
        key=lambda row: (row["label"], int(re.search(r"\d+$", row["record_id"]).group())),
    )
    if current != saved:
        raise ValueError("Selected cohort manifest differs from the official RECORDS inventory")
    if manifest["inventory_sha256"] != inventory_hash:
        raise ValueError(
            "Official inventory contents changed since manifest verification; "
            "review the cohort before updating the manifest"
        )
    return {
        "inventory_record_count": len(inventory),
        "selected_record_count": len(current),
        "class_counts": dict(Counter(row["label"] for row in current)),
        "inventory_sha256": inventory_hash,
        "status": "verified",
    }


def create_manifest(path: Path) -> dict:
    inventory, digest = fetch_inventory()
    records = selected_records(inventory)
    manifest = {
        "database": "CAP Sleep Database",
        "physionet_version": "1.0.0",
        "inventory_url": INVENTORY_URL,
        "inventory_sha256": digest,
        "inventory_record_count": len(inventory),
        "selected_record_count": len(records),
        "expected_class_counts": EXPECTED_COUNTS,
        "records": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return verify_manifest(path)
