# Memory Conflict Review Artifact

This repository contains the code and saved results for an anonymous workshop
submission about conflict resolution in long-term LLM memory. The main question
is simple: if the newest fact is already retrieved, can an older competing fact
still make the model give the wrong answer?

The experiments use the single-hop FactConsolidation task from
MemoryAgentBench. We test different BM25 evidence budgets and also make a
controlled comparison where one unrelated statement is replaced with an older
version of the target fact.

The saved outputs used in the paper are already included. Reviewers do not need
to rerun all model calls to inspect the results.

## Repository structure

```text
memory-conflict-review/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/       Python experiment and analysis scripts
├── results/       Raw JSONL outputs, summaries, and CSV files
└── plots/         Figures used in the paper
```

Please run commands from the repository root. The scripts use relative paths
such as `results/...`, so running them from inside the `scripts` folder will not
work correctly.

## How to check the paper numbers without rerunning models

The primary claims can be verified from these summary files:

| Paper result | File |
|---|---|
| 6K BM25 budgets 64 / 128 / 256 / 512 | `results/12_bm25_budget_audit_summary.json` |
| 6K paired budget comparisons | `results/13_bm25_budget_pairwise_summary.json` |
| 6K conflict vs. no-conflict split | `results/15_conflict_stratification_summary.json` |
| Final Qwen2.5 equal-size comparison | `results/19_final_controlled_qwen25_summary.json` |
| Final Qwen3 equal-size comparison | `results/20b_final_controlled_qwen3_summary.json` |
| 32K BM25-64 vs. BM25-512 | `results/21b_sh32k_bm25_replication_summary.json` |
| Manual audit and sensitivity analysis | `results/22_manual_audit_adjustment.json` |

Row-level model answers are in the matching `.jsonl` files. The two paper
figures are `plots/figure1_controlled_models.png` and
`plots/figure2_budget_accuracy.png`.

The qualitative 20-case manual audit and the metric-sensitivity notes are
reported in the paper appendix. 
The qualitative 23-case manual audit and metric-sensitivity analysis are
available in:

- `results/22_manual_audit_cases.csv`
- `results/22_manual_audit_adjustment.json`
- `results/22_manual_audit_report.md`

The underlying model answers are in the `19`, `20b`, and `21b` JSONL files.

## Requirements

- Python 3.10 or newer is recommended.
- An internet connection is needed for the first dataset and tokenizer download.
- Ollama, or another server compatible with the Ollama chat API, is needed only
  when rerunning model inference.
- Enough memory is required to run Qwen2.5-7B-Instruct and Qwen3-14B.

Create a virtual environment and install the Python packages.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The main Python dependencies are `datasets`, `rank-bm25`, `requests`, and
`transformers`.

## Dataset

The complete benchmark dataset is not copied into this repository. The scripts
load the public dataset automatically:

```python
load_dataset(
    "ai-hyz/MemoryAgentBench",
    split="Conflict_Resolution",
)
```

The experiments use these two sources:

```text
factconsolidation_sh_6k
factconsolidation_sh_32k
```

`results/factconsolidation_inventory.json` records the dataset structure that
was observed during the experiments.

## Ollama setup

Install Ollama separately, then download the models:

```bash
ollama pull qwen2.5:7b-instruct
ollama pull qwen3:14b
```

The scripts use local Ollama by default:

```text
http://localhost:11434
```

For local Ollama, no username or password is required. If a remote protected
server is used, set the endpoint and username before running a model script.
The script will then request the password without storing it in the repository.

### Windows PowerShell

```powershell
$env:OLLAMA_BASE_URL="https://your-server.example"
$env:OLLAMA_USERNAME="your_username"
```

### Linux or macOS

```bash
export OLLAMA_BASE_URL="https://your-server.example"
export OLLAMA_USERNAME="your_username"
```

Do not commit a real server address, username, or password.

## Quick inspection without model calls

The repository already contains the frozen results. The following scripts only
read saved files and rebuild comparisons or summaries:

```bash
python scripts/07_compare_full_vs_question_only.py
python scripts/12_audit_bm25_budgets.py
python scripts/13_compare_bm25_budgets.py
python scripts/15_stratify_results_by_conflict.py
python scripts/17_audit_controlled_failure_modes.py
```

Some audit scripts also download the public dataset, but they do not call the
language models.

## Main experiment order

The numbered filenames show the original experiment order. Not every earlier
diagnostic is needed to reproduce the main claims.

### 1. Inspect the dataset

```bash
python scripts/01_inspect_dataset.py
```

### 2. Run and audit the 6K BM25-512 experiment

Script 08 runs the 512-token BM25 condition. Script 09 audits its retrieved
evidence. Script 14 then uses that audit output to reconstruct the update
chains. Therefore, these scripts must be run in this order:

```bash
python scripts/08_run_bm25_sh6k.py
python scripts/09_audit_bm25_retrieval.py
python scripts/14_audit_update_chains.py
```

Script 08 calls the language model. Scripts 09 and 14 analyze the saved results
and do not make additional model calls.

The generated update chain file is used by conflict stratified and
controlled experiments.

### 3. Run the remaining 6K BM25 budget experiments

The 512-token condition was produced in Step 2. Script 11 reads the evidence
budget from the `EVIDENCE_TOKEN_BUDGET` environment variable. Run it once for
each of the remaining budgets: 64, 128, and 256 tokens.

Windows PowerShell:

```powershell
$env:EVIDENCE_TOKEN_BUDGET="64"
python scripts/11_run_bm25_budget.py

$env:EVIDENCE_TOKEN_BUDGET="128"
python scripts/11_run_bm25_budget.py

$env:EVIDENCE_TOKEN_BUDGET="256"
python scripts/11_run_bm25_budget.py
```

Linux or macOS:

```bash
EVIDENCE_TOKEN_BUDGET=64 python scripts/11_run_bm25_budget.py
EVIDENCE_TOKEN_BUDGET=128 python scripts/11_run_bm25_budget.py
EVIDENCE_TOKEN_BUDGET=256 python scripts/11_run_bm25_budget.py
```

Then rebuild the audits and comparisons:

```bash
python scripts/12_audit_bm25_budgets.py
python scripts/13_compare_bm25_budgets.py
python scripts/15_stratify_results_by_conflict.py
```

### 4. Run the final equal-size controlled experiments

```bash
python scripts/19_run_final_controlled_qwen25.py
python scripts/20_run_final_controlled_qwen3.py
```

Script 19 writes `results/19_final_controlled_qwen25.*`. Script 20 writes
`results/20b_final_controlled_qwen3.*`. The `20b` prefix marks the valid
Qwen3 run with thinking disabled. The earlier thinking-mode run is not
included.

Both conditions contain four statements. Condition A contains the newest target
statement and three unrelated statements. Condition B replaces one unrelated
statement with the obsolete target statement.

### 5. Run the 32K replication

The complete saved results are already included. To rerun the full replication:

```bash
python scripts/21b_run_sh32k_bm25_replication.py

The 32K experiment compares BM25-64 and BM25-512 on the same 100 questions.

## Results reported in the paper

| Experiment | Smaller/latest-only condition | Larger/conflict condition |
|---|---:|---:|
| 6K BM25, 64 vs. 512 tokens | 45/100 | 32/100 |
| Qwen2.5 controlled comparison | 74/74 | 33/74 |
| Qwen3 controlled comparison | 74/74 | 65/74 |
| 32K BM25, 64 vs. 512 tokens | 69/100 | 58/100 |

For the final controlled comparison, all 41 harmful Qwen2.5 flips and all nine
harmful Qwen3 flips select the obsolete value. In the official 32K comparison,
18 questions are correct only with BM25-64 and seven are correct only with
BM25-512.

The detailed model answers, retrieved evidence, scoring decisions, and paired
statistics are available in `results/`.

## Important notes about rerunning

- Model outputs can depend on the Ollama version, model build, hardware, and
  tokenizer version. The saved outputs are the exact outputs used for the paper.
- Scripts 19, 20, and 21b support continuing from completed output rows.
- Script 18 is a 10-question frozen-prompt pilot with 40 calls. It opens its raw
  output file in write mode. Make a backup or rename the existing result before
  running script 18 again.
- Gold-assisted support checks use the benchmark answer only for post-hoc
  diagnosis. They are not presented as a deployable retrieval method.
- Exact McNemar tests treat questions as paired benchmark units. Because the
  questions share a benchmark history, the p-values are diagnostic rather than
  population-level claims.

## Saved-result files

- `02` to `10`: preliminary baselines and diagnostics.
- `11` to `15`: 6K budget sweep, pairwise analysis, and conflict stratification.
- `16` and `17`: earlier exploratory controlled experiment and failure audit.
- `18`: frozen-prompt pilot.
- `19`: final Qwen2.5 controlled experiment.
- `20b`: valid Qwen3 controlled experiment with thinking disabled. Produced by
  `scripts/20_run_final_controlled_qwen3.py`.
- `21b`: corrected 32K BM25 replication and update chains.
- `22`: manual audit of 23 selected cases and sensitivity calculations.

`context_check.json`, Python cache files, private server details, and excluded
invalid runs are intentionally not part of the review artifact.

## Reproducibility scope

The code is provided to reproduce the main analyses and model calls. Running
all inference from the beginning can take time and requires the two Ollama
models. For review of the numerical claims, the included row-level JSONL files
and summaries are sufficient; they contain all predictions used to calculate
the reported accuracies and paired comparisons.
