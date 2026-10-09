#!/usr/bin/env bash
set -euo pipefail
source configs/cluster.env
export RAY_ADDRESS="$COURSE_MASTER_ADDR:$COURSE_RAY_PORT"
exec vllm serve "$COURSE_MODEL" --served-model-name course-model \
  --host "$COURSE_MASTER_ADDR" --port 8000 \
  --tensor-parallel-size "$COURSE_GPUS_PER_NODE" --pipeline-parallel-size "$COURSE_NNODES" \
  --distributed-executor-backend ray --max-model-len 8192 "$@"
