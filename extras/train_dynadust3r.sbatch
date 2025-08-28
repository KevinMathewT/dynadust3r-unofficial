#!/bin/bash
#SBATCH --job-name=dynadust3r
#SBATCH --account=pr_96_general
#SBATCH --time=48:00:00
#SBATCH --mem=750G                   # all ram on the node
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=64            # 16 CPUs/GPU; enough for 12 workers + overhead
#SBATCH --requeue
#SBATCH --output=job_output_%j.txt
#SBATCH --error=job_error_%j.txt

cd /scratch/km6748/vision-experiments/

# activate your poetry env (keeping your original pattern)
$(poetry env activate)

# one process per gpu
accelerate launch -m train
