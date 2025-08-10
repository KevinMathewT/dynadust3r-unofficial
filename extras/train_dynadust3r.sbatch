#!/bin/bash
#SBATCH --job-name=dynadust3r
#SBATCH --account=pr_96_general
#SBATCH --time=4:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --output=job_output_%j.txt
#SBATCH --error=job_error_%j.txt 

cd /scratch/km6748/vision-experiments/
$(poetry env activate)
accelerate launch -m train
