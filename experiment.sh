#!/usr/bin/env bash

# Request one Task (unless using mpi4py)
#SBATCH --job-name=e2e-sweep-cem
#SBATCH --partition=epyc-gpu
#SBATCH --output=./slurm_logs/e2e-%j-%a.out
# Request memory per CPU
#SBATCH --mem-per-cpu=32G
# Request n CPUs for your task.
#SBATCH --cpus-per-task=1
# Request GPU Ressources (model:number)
#SBATCH --gpus=a100:1
#SBATCH --mail-type=all
#SBATCH --mail-user=ongoskar@proton.me

# --- How long will it take per task? 
# lets say 1 model takes 15 min to train (5 epochs) -> 10 models = 2h30min
#SBATCH --time=04:00:00
# NEED TO ADJUST THIS EVERYTIME EXPERIMENT NUMBER CHANGES!!!
# TOTAL EXPERIMENTS: POLLUTION * SEEDS * SIZES 
#SBATCH --array=0-2%3

# Clear all interactively loaded modules
module purge

# Load a python package manager
module load anaconda # or micromamba or anaconda

# Activate a certain environment
conda activate deepcem2

# Slurm array math to map 1D ID to 3D parameters

NUM_POLLUTION=1
POLLUTION_LEVELS=("high")
#POLLUTION_LEVELS=("source" "low" "medium" "high")

NUM_SEEDS=3
SEEDS=(10 20 30)
# SEEDS=(42)

NUM_SIZES=1
TRAIN_SIZES=(1) # '1' represents 100% or full
#TRAIN_SIZES=(125 625 3125 1) # '1' represents 100% or full

# Calculate indices using integer division and modulo
# This ensures every unique Task ID maps to a unique combination
idx_p=$((SLURM_ARRAY_TASK_ID / (NUM_SEEDS * NUM_SIZES) % NUM_POLLUTION))
idx_s=$((SLURM_ARRAY_TASK_ID / NUM_SIZES % NUM_SEEDS))
idx_t=$((SLURM_ARRAY_TASK_ID % NUM_SIZES))


# Pick the values for this specific task
P=${POLLUTION_LEVELS[$idx_p]}
S=${SEEDS[$idx_s]}
T=${TRAIN_SIZES[$idx_t]}
 
# set number of OpenMP threads (i.e. for numpy, etc...)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
# if you are adding your own level of parallelzation, you
# probably want to set OMP_NUM_THREADS=1 instead, in order 
# to prevent the creation of too many threads (massive slowdown!)

# export cache
export HF_HOME=/hpc/gpfs2/scratch/u/zeru47vu/hf
export HF_HUB_CACHE=/hpc/gpfs2/scratch/u/zeru47vu/hf/hub

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# DATASET=$1
# if [ -z "$DATASET" ]; then
#     echo "Error: No dataset provided. Position 1"
#     exit 1
# fi


DATASET=""
BINNING=false

# 2. Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -b|--binning) BINNING=true; shift ;;
        *) DATASET="$1"; shift ;; 
    esac
done

# 3. Validation logic
if [ -z "$DATASET" ]; then
    echo "Error: No dataset provided."
    echo "Usage: ./script.sh [DATASET] [--binning]"
    exit 1
fi

BIN_ARG=""
if [ "$BINNING" = true ]; then
    BIN_ARG="--binning"
fi

if [ "$T" == "1" ]; then
    SUFFIX=""
else
    SUFFIX="$T"
fi
echo "--- STARTING JOB ID: ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} ---"
echo "--- STARTING TASK ID $SLURM_ARRAY_TASK_ID ---"
echo "--- BASELINE ---"
echo "Dataset: $DATASET | Pollution: $P | Seed: $S | Size: $T "

srun python src/exp_baseline.py \
    --dataset "$DATASET" \
    --pollution "$P" \
    --seed "$S" \
    --train_suffix "$SUFFIX"

echo "--- EXPERIMENT ---"
echo "Dataset: $DATASET | Pollution: $P | Seed: $S | Size: $T | Binning: $BINNING"

srun python src/exp_main.py --dataset "$DATASET" --pollution "$P" \
    --seed "$S" \
    --train_suffix "$SUFFIX"
echo "--- COMPLETED JOB $SLURM_ARRAY_TASK_ID ---"
