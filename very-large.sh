#!/bin/sh

# SLURM options:

#SBATCH --job-name=1000px    # Job name
#SBATCH --output=1000px_%j.log   # Standard output and error log

#SBATCH --partition=gpu               

#SBATCH --ntasks=1                    # Run a single task
#SBATCH --mem=16G                    # Memory in MB per default
#SBATCH --time=4-00:00:00             # d-hh:mm:ss
#SBATCH --gres=gpu:1

#SBATCH --mail-user=tpickles@liris.cnrs.fr   # Where to send mail
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)

# Commands to be submitted:
module load python
python -m pip install -U scikit-image
pip install -q --user -r requirements.txt

python main.py config/very-large.json
