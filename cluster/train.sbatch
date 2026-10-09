#!/usr/bin/env bash
#SBATCH --job-name=agent-course
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=runs/slurm-%j.out
set -euo pipefail
source configs/cluster.env
export COURSE_NNODES="$SLURM_NNODES"
export COURSE_MASTER_ADDR
COURSE_MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COURSE_MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export COURSE_RUN_ID="$SLURM_JOB_ID"
cd "$COURSE_ROOT"
# One launcher per node; each launcher starts eight GPU processes.
srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash -c '
  exec "$COURSE_PYTHON" -m torch.distributed.run \
    --nnodes="$SLURM_NNODES" --nproc-per-node="$COURSE_GPUS_PER_NODE" \
    --node-rank="$SLURM_PROCID" --master-addr="$COURSE_MASTER_ADDR" \
    --master-port="$COURSE_MASTER_PORT" --max-restarts=0 --module "$@"
' bash "$@"
