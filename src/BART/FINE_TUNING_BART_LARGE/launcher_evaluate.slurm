#!/bin/bash

#SBATCH --job-name=evaluation                                   # job name
#SBATCH --output=./outputslurm/job_%j/%x                       # output logs
#SBATCH --error=./outputslurm/job_%j/%x.err                     # error logs
#SBATCH --qos=short
#SBATCH --time=2:00:00                                       # maximum wall time allocated for the job (D-H:MM:SS)

## CPU allocation
#SBATCH --nodes=1
#SBATCH --mem=32G 
#SBATCH --cpus-per-task=50
# #SBATCH --ntasks-per-node=1     

# GPU allocation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2                                            # Number of GPUs per node

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

checkpoints_path=$SCRATCHDIRNAUTILUS/checkpoints/finetune_Bart_large/Bart-large-2025-03-07_12h44-6473173/
config_path="./config_finetune_bart_large.json"
results_path="./evaluations"

# Programme à exécuter
srun python evaluate_bart_large_finetune_Data_parallel.py $SLURM_JOB_ID $checkpoints_path $config_path $results_path
