"""
Reads all experiment_results.jsonl files, parses them with pydantic,
flattens every nested field into individual columns, and writes CSVs.

Produces two combined tables:
  - paraphrase_combined.csv  (full + lora + rslora + dora + adalora)
  - sonnet_combined.csv      (full + lora + rslora + dora + adalora)

Run from the project root:
  python3 experiments_tables/build_tables.py
"""

import json
import csv
import traceback
from pathlib import Path
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
OUT_DIR.mkdir(exist_ok=True)

METHOD_SUFFIXES = ["_full", "_lora", "_rslora", "_dora", "_adalora"]

TASKS = [
    {
        "task_name": "paraphrase",
        "folder_prefix": "paraphrase_exp_results",
        "out_csv": "paraphrase_combined.csv",
    },
    {
        "task_name": "sonnet",
        "folder_prefix": "sonnet_exp_results",
        "out_csv": "sonnet_combined.csv",
    },
]

for task_cfg in TASKS:
    all_rows = []
    folders_found = []

    for suffix in METHOD_SUFFIXES:
        folder = task_cfg["folder_prefix"] + suffix
        jsonl = ROOT / folder / "experiment_results.jsonl"
        if not jsonl.exists():
            print(f"  Skipping {folder} (not found)")
            continue
        folders_found.append(folder)
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = ExperimentRecord.model_validate(json.loads(line))
                    all_rows.append(record.to_flat_row(folder))
                except Exception as e:
                    print(f"  WARNING: bad record in {folder}: {e}")

    if not all_rows:
        print(f"No data for {task_cfg['task_name']}, skipping.")
        continue

    all_keys_set = set()
    for row in all_rows:
        all_keys_set.update(row.keys())

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
    TAIL_COLS = [
        "train_losses", "dev_acc_scores", "dev_f1_scores", "chrf_scores",
    ]
    priority = [c for c in PRIORITY_COLS if c in all_keys_set]
    tail = [c for c in TAIL_COLS if c in all_keys_set]
    middle = sorted(
        all_keys_set - set(priority) - set(tail),
        key=lambda k: (0 if k.startswith("train_meta.") else 1, k),
    )
    all_keys = priority + middle + tail

    out_path = OUT_DIR / task_cfg["out_csv"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in all_keys})

    print(
        f"[{task_cfg['task_name']}] {len(all_rows)} rows from "
        f"{', '.join(folders_found)} → {out_path}"
    )

print("Done.")
