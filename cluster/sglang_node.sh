#!/usr/bin/env bash
set -euo pipefail
source configs/cluster.env
: "${COURSE_NODE_RANK:?}"
cd "$COURSE_ROOT"
# Run inside the SGLang environment, one copy per allocated node.
# Qwen2.5-7B has 28 attention heads, not divisible by TP8/16. Default scale-serving model is 72B.
TP=$((COURSE_NNODES * COURSE_GPUS_PER_NODE))
exec python -m sglang.launch_server --model-path "$COURSE_MODEL" \
  --served-model-name course-model --host "${COURSE_NODE_IP:-0.0.0.0}" --port 8000 \
  --tp "$TP" --nnodes "$COURSE_NNODES" --node-rank "$COURSE_NODE_RANK" \
  --dist-init-addr "$COURSE_MASTER_ADDR:$COURSE_MASTER_PORT" "$@"
