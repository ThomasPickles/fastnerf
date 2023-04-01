#!/bin/sh

# SLURM options:

#SBATCH --job-name=nerf_train    # Job name
#SBATCH --output=nerf_train_%j.log   # Standard output and error log

#SBATCH --partition=gpu               

#SBATCH --ntasks=1                    # Run a single task
#SBATCH --mem=32G                    # Memory in MB per default
#SBATCH --time=0-03:00:00             # d-hh:mm:ss
#SBATCH --gres=gpu:1

#SBATCH --mail-user=tpickles@liris.cnrs.fr   # Where to send mail
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)

# Commands to be submitted:
module load python
pip install --user -r requirements.txt
python train.py --px 275 --epochs 2 --loss L2
python train.py --px 275 --epochs 2 --loss L1
python train.py --px 275 --epochs 2 --loss Huber
python train.py --px 275 --epochs 2 --layers 4
python train.py --px 275 --epochs 2 --layers 4 --loss L2
python train.py --px 275 --epochs 2 --layers 4 --loss L1
python train.py --px 275 --epochs 2 --layers 4 --loss Huber
