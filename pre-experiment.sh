#!/usr/bin/env bash

# Request one Task (unless using mpi4py)
#SBATCH --job-name=cem_signal_only
#SBATCH --partition=epyc-gpu
#SBATCH --output=./slurm_logs/signal_only_cem_-%j.out
# Request memory per CPU
#SBATCH --mem-per-cpu=32G
# Request n CPUs for your task.
#SBATCH --cpus-per-task=4
# Request GPU Ressources (model:number)
#SBATCH --gpus=a100:1
#SBATCH --mail-type=all
#SBATCH --mail-user=ongoskar@proton.me

# --- How long will it take per task? 
# lets say 1 model takes 15 min to train (5 epochs) -> 10 models = 2h30min
#SBATCH --time=04:00:00
# NEED TO ADJUST THIS EVERYTIME EXPERIMENT NUMBER CHANGES!!!
# TOTAL EXPERIMENTS: POLLUTION * SEEDS * SIZES 
# 2 * 1 * 2  = 4

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

echo "--- STARTING JOB $SLURM_ARRAY_TASK_ID ---"
echo "--- EXPERIMENT ---"
echo "Dataset: $DATASET | Pollution: $P | Seed: $S | Size: $T | Binning: $BINNING"

# No need to pass number of tasks to srun
srun python src/exp_main.py --dataset "$DATASET" --pollution "source" \
    --seed "0" 
echo "--- COMPLETED JOB $SLURM_ARRAY_TASK_ID ---"
