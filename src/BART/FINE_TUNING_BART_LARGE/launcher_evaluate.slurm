#!/bin/bash

#SBATCH --job-name=evaluation                                   # job name
#SBATCH --output=./outputslurm/%x_job_%j                        # output logs
#SBATCH --error=./outputslurm/%x_job_%j.err                     # error logs
#SBATCH --qos=short
#SBATCH --time=20:00:00                                       # maximum wall time allocated for the job (D-H:MM:SS)

## CPU allocation
#SBATCH --cpus-per-task=50
#SBATCH --nodes=1
#SBATCH --mem=32G 
#SBATCH --ntasks-per-node=1     

# GPU allocation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                                            # Number of GPUs per node

## CONTACT
#SBATCH --mail-user=i.botcazou@gmx.fr
#SBATCH --mail-type=ALL                                         #END, BEGIN, FAIL 

# Nettoyage des modules chargés
module purge

# Initialisation correcte de micromamba
eval "$(micromamba shell hook --shell=bash)"

micromamba activate torch_venv

# Check GPU availability
nvidia-smi || { echo "No GPU detected"; exit 1; }

path_to_load='/LAB-DATA/GLiCID/users/ibotca@univ-angers.fr/datasets/cnn_dailymail_test'

# Programme à exécuter
srun python CNN_processing_and_evaluation.py $SLURM_JOB_ID $path_to_load 
