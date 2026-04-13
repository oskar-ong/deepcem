#!/usr/bin/env bash

# Request one Task (unless using mpi4py)
#SBATCH --job-name=cem-baseline-sweep
#SBATCH --partition=epyc-gpu
#SBATCH --output=./slurm_logs/baseline-%j-%a.out
# Request memory per CPU
#SBATCH --mem-per-cpu=32G
# Request n CPUs for your task.
#SBATCH --cpus-per-task=1
# Request GPU Ressources (model:number)
#SBATCH --gpus=a100:1
# lets say 1 model takes 15 min to train (5 epochs) -> 10 models = 2h30min
#SBATCH --time=04:00:00
# NEED TO ADJUST THIS EVERYTIME EXPERIMENT NUMBER CHANGES!!!
# TOTAL EXPERIMENTS: POLLUTION * SEEDS * SIZES 
# 2 * 1 * 2 = 4
#SBATCH --array=0-3%4

# Clear all interactively loaded modules
module purge

# Load a python package manager
module load anaconda # or micromamba or anaconda

# Activate a certain environment
conda activate deepcem2

# Slurm array math to map 1D ID to 3D parameters
# Index logic:
NUM_POLLUTION=2
NUM_SEEDS=1
NUM_SIZES=2

# Calculate indices using integer division and modulo
# This ensures every unique Task ID maps to a unique combination
idx_p=$((SLURM_ARRAY_TASK_ID / (NUM_SEEDS * NUM_SIZES) % NUM_POLLUTION))
idx_s=$((SLURM_ARRAY_TASK_ID / NUM_SIZES % NUM_SEEDS))
idx_t=$((SLURM_ARRAY_TASK_ID % NUM_SIZES))

# Define your parameter arrays
#POLLUTION_LEVELS=("source" "low" "medium" "high")
POLLUTION_LEVELS=("source" "high")
#SEEDS=(0 42 1337)
SEEDS=(42)
#TRAIN_SIZES=(125 625 3125 1) # '1' represents 100% or full
TRAIN_SIZES=(125 1) # '1' represents 100% or full

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

DATASET=$1
if [ -z "$DATASET" ]; then
    echo "Error: No dataset provided. Position 1"
    exit 1
fi

if [ "$T" == "1" ]; then
    SUFFIX=""
else
    SUFFIX="_$T"
fi

echo "--- STARTING JOB $SLURM_ARRAY_TASK_ID ---"
echo "--- BASELINE ---"
echo "Dataset: $DATASET | Pollution: $P | Seed: $S | Size: $T"

srun python src/exp_baseline.py \
    --dataset "$DATASET" \
    --pollution "$P" \
    --seed "$S" \
    --train_suffix "$SUFFIX"

echo "--- COMPLETED JOB $SLURM_ARRAY_TASK_ID ---"
