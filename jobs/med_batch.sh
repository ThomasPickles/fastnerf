#!/bin/sh

# SLURM options:

#SBATCH --job-name=med_batch    # Job name
#SBATCH --output=job_%j.log   # Standard output and error log

#SBATCH --partition=gpu               

#SBATCH --ntasks=1                    # Run a single task
#SBATCH --mem=16G                    # Memory in MB per default
#SBATCH --time=0-11:00:00             # d-hh:mm:ss
#SBATCH --gres=gpu:1

#SBATCH --mail-user=tpickles@liris.cnrs.fr   # Where to send mail
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)

# Commands to be submitted:
module load python
pip install -q --user -r requirements.txt

python main.py --px 250 --batchsize 2048 --epochs 20 --lr 0.00005 --video --slice --test_device cuda
