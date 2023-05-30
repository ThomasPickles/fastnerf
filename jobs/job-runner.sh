#!/bin/sh

# SLURM options:

#SBATCH --partition=gpu               

#SBATCH --job-name=mulitple_jobs
#SBATCH --output=mulitple_jobs_%j.log

#SBATCH --ntasks=2
#SBATCH --mem=16G                    # Memory in MB per default
#SBATCH --time=0-00:05:00             # d-hh:mm:ss
#SBATCH --gres=gpu:1

#SBATCH --mail-user=tpickles@liris.cnrs.fr   # Where to send mail
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)

# Commands to be submitted:
module load python
pip install -q --user -r requirements.txt

srun -n 1 --exclusive jobs/tiny.sh &
srun -n 1 --exclusive jobs/tiny2.sh &
wait