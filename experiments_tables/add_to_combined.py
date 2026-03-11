"""
Add records from any experiment_results.jsonl into the matching combined CSVs.

Handles mixed files (sonnet + paraphrase in same JSONL), deduplicates by UUID,
and preserves all existing rows in the CSV.

Usage (from project root):
    python3 experiments_tables/add_to_combined.py <path/to/experiment_results.jsonl>

    # Multiple files at once:
    python3 experiments_tables/add_to_combined.py rs_lora_ranking/experiment_results.jsonl final_model_results2/experiment_results.jsonl
"""

import sys
import json
import csv
from pathlib import Path

# Inline ExperimentRecord to avoid side-effects from importing build_tables.py
# (build_tables runs table-building code at module level)

from typing import ClassVar, Optional
from pydantic import BaseModel, ConfigDict


class ExperimentRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    uuid: str
    task: str
    additional_data_used: Optional[bool] = None
    best_dev_acc: Optional[float] = None
    best_chrf: Optional[float] = None
    best_dev_chrf: Optional[float] = None
    dev_acc_scores: Optional[list] = None
    dev_f1_scores: Optional[list] = None
    chrf_scores: Optional[list] = None
    train_losses: Optional[list] = None
    job_id: Optional[str] = None
    method: Optional[str] = None
    method_metadata: Optional[dict] = None
    args: dict = {}
    training_meta_data: Optional[dict] = None

    def resolve_method(self, source_folder: str) -> str:
        fine_tune_mode = self.args.get("fine_tune_mode", "")
        use_rslora = self.args.get("use_rslora", False)
        if source_folder.endswith("_adalora") or fine_tune_mode == "adalora":
            return "adalora"
        if source_folder.endswith("_dora") or fine_tune_mode in ("dora", "dora_mlp"):
            return fine_tune_mode if fine_tune_mode in ("dora", "dora_mlp") else "dora"
        if source_folder.endswith("_rslora") or use_rslora:
            return "rslora"
        if source_folder.endswith("_full") or self.method == "full_parameter_tuning":
            return "full"
        if source_folder.endswith("_lora"):
            return "lora"
        lora_rank = self.args.get("lora_rank", 0)
        if lora_rank and lora_rank > 0:
            return "lora"
        return self.method or "unknown"

    PATH_ARGS: ClassVar[set] = {
        "sonnet_path", "held_out_sonnet_path", "held_out_sonnet_dev_path",
        "true_held_out_sonnet_dev_path", "sonnet_out", "filepath",
        "para_train", "para_dev", "para_test", "para_dev_out", "para_test_out",
    }
    HPARAM_ARGS: ClassVar[set] = {
        "model_size", "epochs", "lr", "batch_size", "seed",
        "lora_rank", "lora_alpha", "lora_dropout", "use_rslora",
        "fine_tune_mode", "apply_lora_to_attention", "apply_lora_to_mlp",
        "apply_lora_to_all_linear", "rank_pattern",
        "adalora_alpha", "adalora_init_r", "adalora_target_r",
        "adalora_tinit_ratio", "adalora_delta_t",
        "use_flash_attention", "temperature", "top_p",
        "d", "l", "num_heads", "use_gpu",
    }

    def to_flat_row(self, source_folder: str) -> dict:
        method = self.resolve_method(source_folder)
        row = {
            "uuid": self.uuid,
            "task": self.task,
            "method": method,
            "additional_data_used": self.additional_data_used,
            "job_id": self.job_id,
            "best_dev_acc": self.best_dev_acc,
            "best_chrf": self.best_chrf or self.best_dev_chrf,
        }
        for k in self.HPARAM_ARGS:
            if k in self.args:
                row[k] = self.args[k]
        if self.training_meta_data:
            for k, v in self.training_meta_data.items():
                row[f"train_meta.{k}"] = v
        for k in self.PATH_ARGS:
            if k in self.args:
                row[f"path.{k}"] = self.args[k]
        row["train_losses"] = self.train_losses
        row["dev_acc_scores"] = self.dev_acc_scores
        row["dev_f1_scores"] = self.dev_f1_scores
        row["chrf_scores"] = self.chrf_scores
        return row


ROOT = Path(".")
OUT_DIR = ROOT / "experiments_tables"

TASK_CONFIG = {
    "paraphrase_detection": {
        "csv": OUT_DIR / "paraphrase_combined.csv",
        "label": "paraphrase",
    },
    "sonnet_generation": {
        "csv": OUT_DIR / "sonnet_combined.csv",
        "label": "sonnet",
    },
}

PRIORITY_COLS = [
    "uuid", "task", "method", "additional_data_used", "job_id",
    "best_dev_acc", "best_chrf",
    "model_size", "epochs", "lr", "batch_size", "seed",
    "lora_rank", "lora_alpha", "lora_dropout", "use_rslora",
    "fine_tune_mode",
    "apply_lora_to_attention", "apply_lora_to_mlp",
    "apply_lora_to_all_linear", "rank_pattern",
    "adalora_alpha", "adalora_init_r", "adalora_target_r",
    "adalora_tinit_ratio", "adalora_delta_t",
    "use_flash_attention", "temperature", "top_p",
    "d", "l", "num_heads", "use_gpu",
]
TAIL_COLS = ["train_losses", "dev_acc_scores", "dev_f1_scores", "chrf_scores"]


def load_csv(csv_path: Path) -> tuple[list[dict], list[str]]:
    """Load existing CSV, return (rows, fieldnames). Empty if file missing."""
    if not csv_path.exists():
        return [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return rows, list(fieldnames)


def write_csv(csv_path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def merge_fieldnames(existing: list[str], new_keys: set) -> list[str]:
    """Extend fieldnames preserving priority/tail order."""
    existing_set = set(existing)
    extra_new = new_keys - existing_set - set(PRIORITY_COLS) - set(TAIL_COLS)

    priority = [c for c in PRIORITY_COLS if c in existing_set or c in new_keys]
    tail = [c for c in TAIL_COLS if c in existing_set or c in new_keys]

    existing_middle = [c for c in existing if c not in set(priority) and c not in set(tail)]
    new_middle = sorted(extra_new, key=lambda k: (0 if k.startswith("train_meta.") else 1, k))
    middle = existing_middle + [c for c in new_middle if c not in existing_middle]

    return priority + middle + tail


def add_jsonl_to_combined(jsonl_path: str | Path) -> None:
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found.")
        return

    # Use parent folder name as source_folder for method resolution
    source_folder = jsonl_path.parent.name

    # Parse all records, split by task
    new_by_task: dict[str, list[dict]] = {"paraphrase_detection": [], "sonnet_generation": []}
    skipped = 0

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                record = ExperimentRecord.model_validate(raw)
                if record.args.get("model_size") not in ("gpt2", "gpt2-medium"):
                    skipped += 1
                    continue
                task = record.task
                if task in new_by_task:
                    new_by_task[task].append(record.to_flat_row(source_folder))
            except Exception as e:
                print(f"  WARNING: skipping bad record: {e}")

    if skipped:
        print(f"  Skipped {skipped} non-gpt2/gpt2-medium records.")

    for task, new_rows in new_by_task.items():
        if not new_rows:
            continue

        cfg = TASK_CONFIG[task]
        csv_path = cfg["csv"]
        label = cfg["label"]

        existing_rows, existing_fields = load_csv(csv_path)
        existing_uuids = {r["uuid"] for r in existing_rows}

        # Deduplicate: only add rows not already present
        to_add = [r for r in new_rows if r["uuid"] not in existing_uuids]
        duplicates = len(new_rows) - len(to_add)

        if not to_add:
            print(f"[{label}] All {len(new_rows)} records already in {csv_path.name}, nothing added.")
            continue

        # Merge fieldnames
        new_keys = set()
        for r in to_add:
            new_keys.update(r.keys())
        all_fields = merge_fieldnames(existing_fields, new_keys)

        all_rows = existing_rows + to_add
        write_csv(csv_path, all_rows, all_fields)

        print(f"[{label}] Added {len(to_add)} new rows to {csv_path.name} "
              f"(total: {len(all_rows)}"
              + (f", skipped {duplicates} duplicates" if duplicates else "") + ")")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 experiments_tables/add_to_combined.py <jsonl_path> [jsonl_path2 ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        print(f"\nProcessing: {path}")
        add_jsonl_to_combined(path)

    print("\nDone.")
