#!/bin/bash
#SBATCH --job-name=dynadust3r
# #SBATCH --account=pr_96_general
# #SBATCH --account=pr_96_tandon_advanced
# #SBATCH --account=pr_96_tandon_priority
#SBATCH --time=36:00:00
#SBATCH --mem=750G                   # all ram on the node
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=64            # 64 CPUs/GPU; enough for overhead
#SBATCH --requeue
#SBATCH --output=job_output_%j.txt
#SBATCH --error=job_error_%j.txt

# Prefer on-node transports; avoid flaky IB path on single-node jobs
# export NCCL_IB_DISABLE=1

# Surface comm errors early; let PyTorch/NCCL abort instead of hanging
# export NCCL_ASYNC_ERROR_HANDLING=1
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Better diagnostics for hangs/timeouts
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,COLL
# export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576

# Optional: make collectives block instead of spinning (fails faster, a bit slower)
# export NCCL_BLOCKING_WAIT=1

# Avoid CPU oversubscription from underlying libs across many workers
# export OMP_NUM_THREADS=4

# Optional: richer PyTorch DDP logs (very verbose)
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

cd /scratch/km6748/vision-experiments/

# activate your poetry env (keeping your original pattern)
$(poetry env activate)

# one process per gpu
accelerate launch -m train dataset.stereo4d.wds_dir=/scratch/projects/fouheylab/km6748/stereo4d-data/wds debug=False
