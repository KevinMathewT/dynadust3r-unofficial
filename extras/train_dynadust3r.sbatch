#!/bin/bash
#SBATCH --job-name=dynadust3r
#SBATCH --account=pr_96_general
#SBATCH --time=6:00:00
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --requeue
#SBATCH --output=job_output_%j.txt
#SBATCH --error=job_error_%j.txt 

cd /scratch/km6748/vision-experiments/
$(poetry env activate)
accelerate launch -m train
