#!/usr/bin/env bash
set -euo pipefail
source configs/cluster.env
: "${COURSE_WORKER_RANK:?}"
: "${COURSE_WORKERS:?total number of worker processes, not threads}"
cd "$COURSE_ROOT"
exec python -m src.agent --backend http --tasks data/test.jsonl \
  --base-url "http://$COURSE_MASTER_ADDR:8000/v1" --workers 16 \
  --shards "$COURSE_WORKERS" --shard "$COURSE_WORKER_RANK" \
  --db "runs/worker-$COURSE_WORKER_RANK.sqlite" \
  --out "runs/worker-$COURSE_WORKER_RANK.jsonl" "$@"
