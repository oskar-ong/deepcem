#!/usr/bin/env bash

# Request one Task (unless using mpi4py)
#SBATCH --job-name=deepcem-test
#SBATCH --partition=epyc-gpu
#SBATCH --time=1-0

# Request memory per CPU
#SBATCH --mem-per-cpu=32G
# Request n CPUs for your task.
#SBATCH --cpus-per-task=1
# Request GPU Ressources (model:number)
#SBATCH --gpus=a100:1

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

# No need to pass number of tasks to srun
srun python run_base_ditto.py