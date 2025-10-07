#!/bin/bash
#SBATCH --job-name=s4d_preproc_cpu
#SBATCH --account=pr_96_general
# #SBATCH --account=pr_96_tandon_advanced
# #SBATCH --account=pr_96_tandon_priority
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=32
#SBATCH --requeue
#SBATCH --output=job_output_%j.txt
#SBATCH --error=job_error_%j.txt

cd /scratch/km6748/vision-experiments/

# activate poetry env (consistent with training script)
$(poetry env activate)

poetry run python -m extras.preprocess_stereo4d \
  +preproc.split=train \
  dataset.stereo4d.wds_dir=/scratch/projects/fouheylab/km6748/stereo4d-data/wds/train \
  +preproc.debug=false +preproc.num_triplets=1568000 \
  +preproc.verify=false \
  +preproc.num_workers=32 \
  +preproc.image_format=npy \
  +preproc.shard_size_gb=8.0 +preproc.samples_per_shard=20000
  +preproc.clean_output=true


poetry run python -m extras.preprocess_stereo4d \
  +preproc.split=test \
  dataset.stereo4d.wds_dir=/scratch/projects/fouheylab/km6748/stereo4d-data/wds/test \
  +preproc.debug=false +preproc.num_triplets=16000 \
  +preproc.verify=false \
  +preproc.num_workers=32 \
  +preproc.image_format=npy \
  +preproc.shard_size_gb=8.0 +preproc.samples_per_shard=20000
  +preproc.clean_output=true
