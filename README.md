# RETEM: Relationally Enriched Transformers-based Entity Matching

RETEM is an end-to-end entity matching pipeline optimized for High-Performance Computing (HPC) environments running SLURM. This repository 
features an extended fork of the **Ditto** entity matcher, updated to support modern GPU-enabled parallel computing and recent PyTorch API changes.

All experiment metrics, parameters, and results are automatically tracked and logged inside a local SQLite database (`cem_results.db`).
Relevant results discussed in the thesis can be explored in notebooks/results_DATASET.ipynb.

---

## Getting Started

### 1. Clone the Repository
```bash
git clone [https://github.com/oskar-ong/retem.git](https://github.com/oskar-ong/retem.git)
cd retem

```

### 2. Environment Setup & Submodule Installation

Ensure you have Conda installed, then run the following sequence to build the environment and install the customized Ditto submodule:

```bash
# Create and activate the environment
conda create -y -n retem_env python=3.11
conda activate retem_env

# Configure channel priority and install dependencies
conda config --env --set channel_priority strict 
CONDA_OVERRIDE_CUDA=12.4 conda install -y \
  -c pytorch -c nvidia -c conda-forge -c defaults \
  pytorch pytorch-cuda=12.4 \
  pandas scikit-learn jsonlines tqdm transformers tensorboardX nltk spacy

# Initialize and pull the Ditto submodule
git submodule update --init --recursive

# Install the customized Ditto fork in editable mode
python -m pip install -e ./models/ditto

```

---

## Running Experiments

### HPC Cluster (SLURM)

To submit the main experiment to a SLURM queue, pass the target dataset name as an argument:

```bash
sbatch experiment.sh <dataset>

```

* **Baselines:** Run baseline experiments using `sbatch baseline.sh <dataset>`.
* **Configuration:** Adjust random seeds and data pollution levels directly inside `experiment.sh`. To tweak Ditto's hyperparameters, modify `experiment_config.py` or `baseline_config.py`.

### Local Execution (Testing)

If your local machine has a compatible GPU, you can run the pipeline script directly:

```bash
python src/exp_main.py <dataset>

```

> **Note:** Local execution is unverified. Running via the SLURM workload manager is highly recommended.

---
## Datasets

Available Datasets are 

- imdb
- music
- pokemon

Adding custom datasets requires: 

1) normalization via src/normalize_with_new_ids.py 
2) creation of new entity_config.py entry 
3) creation of ML splits via split_generation.py or split_generation_naive.py
4) creation of experiment_config.py entry 
5) reference to the ml splits filepath in models/ditto/configs.json 

---

## Pipeline Architecture

The core execution pipeline progresses through four major stages:

1. **Dataset Preparation:** Creation of ML splits
2. **Preprocessing:**
* **Relational Enrichment:** Injects relational graph information directly into the entity profiles to boost matching accuracy.
* **Serialization:** Converts the enriched entity pairs into text sequences compliant with Ditto’s input format.

1. **2-Phase Fine-Tuning:** Trains the underlying Transformer model sequentially: first attribute only with empty relationship scores, then with relationship scores.
2. **Iterative Matching:** Executes a feedback loop that dynamically updates match classifications and propagates matching decisions across related entity profiles.

---

## Repository Structure

```text
├── data/                               # Raw, interim, and processed datasets
├── src/                                # Pipeline source code
│   ├── baseline_config.py              # Configuration for baseline experiment
│   ├── experiment_config.py            # Configuration for main experiment
│   ├── exp_baseline.py                 # Baseline experiment workflow
│   ├── exp_main.py                     # Main RETEM experiment pipeline
│   ├── normalize_with_new_ids.py       # Dataset normalization & ID re-assignment to prevent leakage
│   ├── pollution.py                    # Utilities for simulating data pollution/noise
│   ├── split_generation.py             # ML split generation with relational leakage mitigation
│   └── split_generation_naive.py       # Naive ML split generation (Independent, random sampling)
├── models/
│   └── ditto/                          # Customized fork of the Ditto entity matcher
├── baseline.sh                         # SLURM batch script for baseline runs
├── experiment.sh                       # SLURM batch script for main RETEM runs
├── cem_results.db                      # Local SQLite database tracking experiment results
├── requirements.txt                    # Python package dependencies
└── README.md                           # Project documentation

```
