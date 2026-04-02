#!/usr/bin/env bash

# Request one Task (unless using mpi4py)
#SBATCH --job-name=cem-e2e-sweep
#SBATCH --partition=epyc-gpu-test
#SBATCH --time=02:00:00
#SBATCH --output=./slurm_logs/e2e-%j-%a.out
# Request memory per CPU
#SBATCH --mem-per-cpu=32G
# Request n CPUs for your task.
#SBATCH --cpus-per-task=1
# Request GPU Ressources (model:number)
#SBATCH --gpus=a100:1
#SBATCH --array=0-3%1

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
    echo "Error: No dataset provided."
    echo "Usage: sbatch experiment.sh <dataset>"
    exit 1
fi

POLLUTION_LEVELS=("source" "low" "medium" "high")
P=${POLLUTION_LEVELS[$SLURM_ARRAY_TASK_ID]}

echo "Running full pipeline for dataset / pollution level: $DATASET - $P"

# No need to pass number of tasks to srun
srun python src/exp_main_finetune.py --dataset "$DATASET" --pollution "$P"
srun python src/exp_main_match.py --dataset "$DATASET" --pollution "$P"
echo "Completed full pipeline for dataset / pollution level: $DATASET - $P"
