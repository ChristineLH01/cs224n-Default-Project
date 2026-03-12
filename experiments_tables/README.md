# Experiment Tables & Plots

Scripts for collecting experiment results, building combined CSVs, and generating LaTeX tables and publication-quality plots.

## Scripts

| Script | Description |
|---|---|
| `build_tables.py` | Scans all `*_exp_results/experiment_results.jsonl` folders, parses records, and produces `paraphrase_combined.csv` and `sonnet_combined.csv` |
| `add_to_combined.py` | Incrementally adds new records from any `experiment_results.jsonl` into the existing combined CSVs (deduplicates by UUID) |
| `generate_paper_tables.py` | Reads the combined CSVs and generates LaTeX tables: main results, LoRA ablation, learning rate ablation, and comprehensive appendix tables |
| `generate_paper_plots.py` | Generates accuracy-vs-GPU-hours tradeoff scatter plots for each task |
| `generate_paper_ablation_plots.py` | Generates module ablation, data scaling, rank scaling, and rsLoRA sweep plots |

## Combined CSVs

| File | Description |
|---|---|
| `paraphrase_combined.csv` | All paraphrase detection experiment results (full, LoRA, rsLoRA, DoRA, AdaLoRA) |
| `sonnet_combined.csv` | All sonnet generation experiment results (full, LoRA, rsLoRA, DoRA, AdaLoRA) |

## Usage

Build combined CSVs from scratch (requires `*_exp_results/` folders in the project root):

```bash
python experiments_tables/build_tables.py
```

Add new results incrementally:

```bash
python experiments_tables/add_to_combined.py <path/to/experiment_results.jsonl>
```

Generate tables and plots (run from inside this directory):

```bash
cd experiments_tables
python generate_paper_tables.py
python generate_paper_plots.py
python generate_paper_ablation_plots.py
```

## `paper_tables/`

Contains the generated LaTeX tables (`.tex`) and plot images (`.png`) used in the paper.
