#!/usr/bin/env bash
set -euo pipefail
source configs/cluster.env
: "${COURSE_NODE_RANK:?set COURSE_NODE_RANK to 0,1,... on each allocated node}"
: "${COURSE_MASTER_ADDR:?}"
cd "$COURSE_ROOT"
exec "$COURSE_PYTHON" -m torch.distributed.run \
  --nnodes="$COURSE_NNODES" --nproc-per-node="$COURSE_GPUS_PER_NODE" \
  --node-rank="$COURSE_NODE_RANK" --master-addr="$COURSE_MASTER_ADDR" \
  --master-port="$COURSE_MASTER_PORT" --max-restarts=0 \
  --module "$@"
