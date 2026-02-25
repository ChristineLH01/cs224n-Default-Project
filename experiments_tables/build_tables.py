"""
Reads all experiment_results.jsonl files, parses them with pydantic,
flattens every nested field into individual columns, and writes CSVs.

Run from the project root:
  python3 experiments_tables/build_tables.py
"""

import json
import csv
import traceback
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, ConfigDict


class ExperimentRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    uuid: str
    task: str
    best_dev_acc: Optional[float] = None
    best_chrf: Optional[float] = None
    best_dev_chrf: Optional[float] = None
    dev_acc_scores: Optional[list] = None
    dev_f1_scores: Optional[list] = None
    chrf_scores: Optional[list] = None
    job_id: Optional[str] = None
    method: Optional[str] = None
    args: dict = {}
    training_meta_data: Optional[dict] = None

    def to_flat_row(self) -> dict:
        lora_rank = self.args.get("lora_rank", 0)
        method = "LoRA" if lora_rank and lora_rank > 0 else self.method
        row = {
            "uuid": self.uuid,
            "task": self.task,
            "job_id": self.job_id,
            "method": method,
            "best_dev_acc": self.best_dev_acc,
            "best_chrf": self.best_chrf or self.best_dev_chrf,
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


ROOT = Path(".")
OUT_DIR = ROOT / "experiments_tables"

SOURCES = [
    ("paraphasing_exp_result",  "paraphrase_results.csv"),
    ("paraphasing_exp_result2", "paraphrase_results2.csv"),
    ("sonnet_exp_result",       "sonnet_results.csv"),
]

for folder, out_name in SOURCES:
    try:
        jsonl = ROOT / folder / "experiment_results.jsonl"
        if not jsonl.exists():
            print(f"Skipping {folder} (not found)")
            continue
        print(f"Processing {folder}...")
        rows = []
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(ExperimentRecord.model_validate(json.loads(line)).to_flat_row())
        all_keys = list(dict.fromkeys(k for row in rows for k in row))
        out_path = OUT_DIR / out_name
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in all_keys})
        print(f"  Wrote {len(rows)} rows → {out_path}")
    except Exception as e:
        print(f"ERROR in {folder}: {e}")
        traceback.print_exc()

print("Done.")
