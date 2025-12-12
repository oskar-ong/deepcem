#!/usr/bin/env bash
 
#SBATCH --job-name=deepcem-test
#SBATCH --partition=test
#SBATCH --time=1-0

# Request one Task (unless using mpi4py)
#SBATCH --ntasks=1
# Request memory per CPU
#SBATCH --mem-per-cpu=1G
# Request n CPUs for your task.
#SBATCH --cpus-per-task=n

# Clear all interactively loaded modules
module purge

# Load a python package manager
module load anaconda # or micromamba or anaconda

# Activate a certain environment
conda activate deepcem
 
# set number of OpenMP threads (i.e. for numpy, etc...)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
# if you are adding your own level of parallelzation, you
# probably want to set OMP_NUM_THREADS=1 instead, in order 
# to prevent the creation of too many threads (massive slowdown!)

# No need to pass number of tasks to srun
srun python run_base_ditto.py