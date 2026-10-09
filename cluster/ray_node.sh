#!/usr/bin/env bash
set -euo pipefail
source configs/cluster.env
export PYTHONPATH="$COURSE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export COURSE_DATA_DIR="$COURSE_ROOT/data"
: "${COURSE_NODE_RANK:?}"
: "${COURSE_NODE_IP:?set the reachable private IP of this allocated node}"
if [[ "$COURSE_NODE_RANK" == 0 ]]; then
  exec ray start --head --node-ip-address="$COURSE_NODE_IP" --port="$COURSE_RAY_PORT" \
    --num-gpus="$COURSE_GPUS_PER_NODE" --dashboard-host=127.0.0.1 \
    --dashboard-port="$COURSE_DASHBOARD_PORT" --block
else
  exec ray start --address="$COURSE_MASTER_ADDR:$COURSE_RAY_PORT" \
    --node-ip-address="$COURSE_NODE_IP" --num-gpus="$COURSE_GPUS_PER_NODE" --block
fi
