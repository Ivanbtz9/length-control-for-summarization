#!/bin/bash

#SBATCH --job-name=cnn_processing                                  # job name
#SBATCH --output=./outputslurm/%x.job_%j                        # output logs
#SBATCH --error=./outputslurm/%x_job_%j.err                     # error logs
#SBATCH --qos=short
#SBATCH --time=10:00:00                                       # maximum wall time allocated for the job (D-H:MM:SS)

## CPU allocation
#SBATCH --cpus-per-task=50
#SBATCH --nodes=1
#SBATCH --mem=32G 
#SBATCH --ntasks-per-node=1     

## CONTACT
#SBATCH --mail-user=i.botcazou@gmx.fr
#SBATCH --mail-type=ALL                                         #END, BEGIN, FAIL 

# Nettoyage des modules chargés
module purge

# Initialisation correcte de micromamba
eval "$(micromamba shell hook --shell=bash)"

micromamba activate torch_venv


# Programme à exécuter
srun python processing_cnn_daily.py $SLURM_JOB_ID

