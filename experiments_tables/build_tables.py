"""
Reads all experiment_results.jsonl files, parses them with pydantic,
flattens every nested field into individual columns, and writes CSVs.

Run from the project root:
  python3 experiments_tables/build_tables.py
"""

import json
import csv
from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict


# ── Pydantic models ──────────────────────────────────────────────────────────

class ExperimentRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    uuid: str
    task: str
    best_dev_acc: Optional[float] = None
    best_chrf: Optional[float] = None
    dev_acc_scores: Optional[list] = None
    dev_f1_scores: Optional[list] = None
    chrf_scores: Optional[list] = None
    job_id: Optional[str] = None
    method: Optional[str] = None
    args: dict = {}
    training_meta_data: Optional[dict] = None

    def to_flat_row(self) -> dict:
        row = {
            "uuid": self.uuid,
            "task": self.task,
            "job_id": self.job_id,
            "method": self.method,
            "best_dev_acc": self.best_dev_acc,
            "best_chrf": self.best_chrf,
            "dev_acc_scores": self.dev_acc_scores,
            "dev_f1_scores": self.dev_f1_scores,
            "chrf_scores": self.chrf_scores,
        }
        for k, v in self.args.items():
            row[f"args.{k}"] = v
        if self.training_meta_data:
            for k, v in self.training_meta_data.items():
                row[f"train_meta.{k}"] = v
        return row


# ── Processing ────────────────────────────────────────────────────────────────

def process_jsonl(jsonl_path: Path) -> list[dict]:
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = ExperimentRecord.model_validate(json.loads(line))
            rows.append(record.to_flat_row())
    return rows


def write_csv(rows: list[dict], out_path: Path):
    if not rows:
        print(f"  No rows, skipping {out_path.name}")
        return
    all_keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in all_keys})
    print(f"  Wrote {len(rows)} rows → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

ROOT = Path(".")
OUT_DIR = ROOT / "experiments_tables"

SOURCES = [
    ("paraphasing_exp_result",  "paraphrase_results.csv"),
    #("paraphasing_exp_result2", "paraphrase_results2.csv"),
    #("sonnet_exp_result",       "sonnet_results.csv"),
]

for folder, out_name in SOURCES:
    jsonl = ROOT / folder / "experiment_results.jsonl"
    if not jsonl.exists():
        print(f"Skipping {folder} (not found)")
        continue
    print(f"Processing {folder}...")
    rows = process_jsonl(jsonl)
    write_csv(rows, OUT_DIR / out_name)

print("Done.")
