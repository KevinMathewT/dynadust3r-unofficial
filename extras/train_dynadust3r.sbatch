#!/bin/bash
#SBATCH --job-name=dynadust3r
#SBATCH --account=pr_96_general
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --requeue
#SBATCH --output=job_output_%j.txt
#SBATCH --error=job_error_%j.txt 

cd /scratch/km6748/vision-experiments/
$(poetry env activate)
accelerate launch -m train
