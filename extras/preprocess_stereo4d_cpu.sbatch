#!/bin/bash
#SBATCH --job-name=s4d_preproc_cpu
# #SBATCH --account=pr_96_general
# #SBATCH --account=pr_96_tandon_advanced
#SBATCH --account=pr_96_tandon_priority
#SBATCH --time=16:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=32
#SBATCH --requeue
#SBATCH --output=job_output_%j.txt
#SBATCH --error=job_error_%j.txt

cd /scratch/km6748/vision-experiments/

# activate poetry env (consistent with training script)
$(poetry env activate)

# CPU-only pre-processing (no GPU decode)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Tune via env as needed; keep shards clean on rerun
export PRE_S4D_USE_ALL_VIDEOS=1
export PRE_S4D_SPLIT=train
export PRE_S4D_NUM_WORKERS=${SLURM_CPUS_PER_TASK:-$(nproc)}
export PRE_S4D_DECORD_GPU=0
export PRE_S4D_WRITE_MODE=per_process
export PRE_S4D_MAX_SHARD_SIZE_GB=50
export PRE_S4D_MAX_SAMPLES_PER_SHARD=8000,
export PRE_S4D_CLOSE_EACH=0
export PRE_S4D_VERIFY_N=0
export PRE_S4D_CLEAN_SHARDS=1

# Run preprocessor for both splits (pass wds_dir explicitly since config uses ???)
WDS_DIR=/scratch/projects/fouheylab/km6748/stereo4d-data/wds-full
for SPLIT in train test; do
  export PRE_S4D_SPLIT=${SPLIT}
  poetry run python -m extras.preprocess_stereo4d \
    dataset.stereo4d.wds_dir=${WDS_DIR}
done


