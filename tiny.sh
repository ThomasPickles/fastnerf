#!/bin/sh

# SLURM options:

#SBATCH --job-name=tiny    # Job name
#SBATCH --output=tiny_%j.log   # Standard output and error log

#SBATCH --partition=gpu               

#SBATCH --ntasks=1                    # Run a single task
#SBATCH --mem=16G                    # Memory in MB per default
#SBATCH --time=0-00:05:00             # d-hh:mm:ss
#SBATCH --gres=gpu:1

#SBATCH --mail-user=tpickles@liris.cnrs.fr   # Where to send mail
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)

# Commands to be submitted:
module load python
pip install -q --user -r requirements.txt

python main.py --px 20 --epochs 3 --lr 0.00005 --video --slice --test_device cuda
