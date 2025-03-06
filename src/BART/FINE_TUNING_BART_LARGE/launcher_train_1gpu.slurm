#!/bin/bash

#SBATCH --job-name=finetune_bart_large_cnn_single_gpu                                   # job name
#SBATCH --output=./outputslurm/job_%j/%x                       # output logs
#SBATCH --error=./outputslurm/job_%j/%x.err                     # error logs
#SBATCH --qos=medium
#SBATCH --time=2-20:00:00                                       # maximum wall time allocated for the job (D-H:MM:SS)

## CPU allocation
#SBATCH --cpus-per-task=50
#SBATCH --nodes=1
#SBATCH --mem=128G 
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

checkpoint_path='/LAB-DATA/GLiCID/users/ibotca@univ-angers.fr/checkpoints'

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" # to avoid fragmentation

# Programme à exécuter
srun python finetune_bart_large_cnn_single_gpu.py $SLURM_JOB_ID $checkpoint_path 
