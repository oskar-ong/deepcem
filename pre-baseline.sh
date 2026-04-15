#!/usr/bin/env bash

# Request one Task (unless using mpi4py)
#SBATCH --job-name=cem-pre-baseline
#SBATCH --partition=epyc-gpu
#SBATCH --output=./slurm_logs/pre-baseline-%j.out
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


# Clear all interactively loaded modules
module purge

# Load a python package manager
module load anaconda # or micromamba or anaconda

# Activate a certain environment
conda activate deepcem2

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
    SUFFIX="$T"
fi

echo "--- STARTING JOB $SLURM_ARRAY_TASK_ID ---"
echo "--- PRE BASELINE ---"
echo "Dataset: $DATASET | Pollution: $P | Seed: $S | Size: $T"

srun python src/exp_baseline.py \
    --dataset "$DATASET" \
    --pollution "source" \
    --seed "0" 

echo "--- COMPLETED JOB $SLURM_ARRAY_TASK_ID ---"
