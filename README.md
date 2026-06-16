# RETEM - Relationally Enriched Transformer-based Entity Matching

RETEM is an end-to-end entity matching pipeline designed for HPC environments using SLURM. 
This repository includes a fork of the **Ditto** entity matcher, extended to support GPU enabled parallel computing and updates of deprecated 
pytorch functions. 

Results from all experiments are automatically logged in a SQLite database **cem_results.db**.

## Clone repository

git clone repo 

---
## Install Ditto Submodule 

git submodule update --init --recursive

conda activate <env>

python -m pip install -e ./models/ditto

---

## Run Experiment: 

If you have a GPU enabled local machine you can directly run the python script: 

python exp_main.py -<dataset> 

Note: Local execution has not been tested. 

If you have access to a HPC cluster with a Slurm workload management: 

sbatch experiment.sh <dataset>

## Pipeline Stages

1. **Dataset Preparation:** Utilities to ingest, clean, and split raw entity data into train/validation/test sets.
2. **Preprocessing:** * **Relational Enrichment:** Injecting contextual/relational graph information into the entity profiles.
   * **Serialization:** Converting enriched entity pairs into Ditto-compliant text sequences.
3. **2-Phase Finetuning (Curriculum Learning):** Training the underlying language model sequentially, starting with easy matches and progressing to hard/noisy pairs.
4. **Iterative Matching:** A feedback loop that dynamically updates match classifications and propagates constraints across the dataset.

---

## Important directories and files

```text
├── data/                               # Raw, interim and processed datasets
├── src/                                # Source code
│   ├── exp_baseline.py/                # Baseline Experiment
│   └── exp_main.py/                    # Main experiment
│   ├── models/ditto/                   # Customized fork of the Ditto entity matcher
│   ├── normalize_with_new_ids.py       # Dataset normalization and assignment of new identifieres to avoid leakage
│   ├── pollution.py                    # Dataset pollution
│   ├── split_generation.py             # ML dataset split generation 
│   ├── split_generation_naive.py       # ML dataset split generation without mitigating relational leakage
├── baseline.sh                         # Slurm batch script for baseline experiment
├── cem_results.db/                     # SQLite DB results
├── experiment.sh                       # Slurm batch script for main experiment
└── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies

```
---

<!-- ## Pipeline Step-by-Step Breakdown

If you need to run steps manually or debug locally on a small subset, you can trigger individual scripts:

### Step 1: Dataset Preparation

```bash
python src/preparation/prepare_data.py --input data/raw/ --output data/interim/

```

### Step 2: Preprocessing & Serialization

```bash
python src/preprocessing/enrich_and_serialize.py --config config/enrichment_rules.json

```

### Step 3: Two-Phase Curriculum Finetuning

```bash
# Phase 1: Train on easy pairs
python src/training/train.py --phase 1 --config config/train_phase1.json
# Phase 2: Train on hard pairs (Curriculum Learning)
python src/training/train.py --phase 2 --config config/train_phase2.json

```

### Step 4: Iterative Matching

```bash
python src/matching/iterative_match.py --checkpoint checkpoints/best_model.pt

``` -->

<!-- ---

## 🗄️ Results Tracking (SQL Database)

At the end of every pipeline run (both Baseline and Custom), performance metrics are committed to your configured SQL database.

### Metrics Tracked:

| Column Name | Description |
| --- | --- |
| `run_id` | Unique identifier for the experiment run |
| `timestamp` | Date and time of execution |
| `experiment_type` | `baseline` or `custom_pipeline` |
| `precision` | Precision score on the test set |
| `recall` | Recall score on the test set |
| `f1_score` | F1-Score achievement |
| `epoch_count` | Total epochs completed across training phases |
| `slurm_job_id` | Associated HPC Job ID for log cross-referencing |

To view results, you can query your database directly or use the quick summary script:

```bash
python src/utils/fetch_results.py --limit 10

``` -->



```

```
