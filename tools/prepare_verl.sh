#!/usr/bin/env bash
set -euo pipefail
# Source snapshot inspected for this course. Do not silently move to main.
source configs/cluster.env
expected=$(cat configs/verl.commit)
if [[ ! -d "$COURSE_VERL_ROOT/.git" ]]; then
  git clone --no-checkout https://github.com/verl-project/verl.git "$COURSE_VERL_ROOT"
  git -C "$COURSE_VERL_ROOT" checkout "$expected"
fi
if [[ ! -f configs/verl.commit ]]; then
  echo 'Existing checkout has no course lock. Record its reviewed commit into configs/verl.commit first.' >&2
  exit 2
fi
actual=$(git -C "$COURSE_VERL_ROOT" rev-parse HEAD)
[[ "$actual" == "$expected" ]] || { echo 'verl commit differs from course lock' >&2; exit 2; }
[[ -f "$COURSE_VERL_ROOT/verl/tools/function_tool.py" ]] || {
  echo 'This checkout lacks the documented function-tool API. Use the documented current version, then lock it.' >&2; exit 2;
}
# Separate env. Prefer your company's approved matching image if already supplied.
python3 -m venv .venv-verl
source .venv-verl/bin/activate
python -m pip install --upgrade pip
python -m pip install -e "$COURSE_VERL_ROOT[vllm]"
python -m pip install ninja packaging psutil 'liger-kernel>=0.8.2'
if ! python -c 'import flash_attn' 2>/dev/null; then
  command -v nvcc >/dev/null || { echo 'Use the company-approved CUDA development image with nvcc and compatible FlashAttention.' >&2; exit 2; }
  MAX_JOBS=8 python -m pip install flash-attn --no-build-isolation
fi
python -m pip check
mkdir -p runs
python -m pip freeze > runs/verl.freeze.txt
python -m src.preflight > runs/verl.preflight.json
