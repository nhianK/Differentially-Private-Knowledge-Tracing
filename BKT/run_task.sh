#!/bin/bash
#SBATCH --job-name=GPUjob
#SBATCH --partition=lowd
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --account=uoml
#SBATCH --ntasks-per-node=1
#SBATCH --mem=150G
#SBATCH --error=hostname.err


module load miniconda
conda activate /projects/uoml/anbin/miniconda3/envs/priv-bert


/projects/uoml/anbin/miniconda3/envs/priv-bert/bin/python -u train.py
