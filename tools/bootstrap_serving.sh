#!/usr/bin/env bash
set -euo pipefail
engine=${1:?usage: bash tools/bootstrap_serving.sh vllm OR sglang}
case "$engine" in
  vllm|sglang) ;;
  *) exit 2 ;;
esac
python3 -m venv ".venv-$engine"
source ".venv-$engine/bin/activate"
python -m pip install --upgrade pip
if [[ "$engine" == vllm ]]; then
  python -m pip install vllm 'ray[default]'
  vllm serve --help > /dev/null
else
  python -m pip install 'sglang[all]' 'ray[default]'
  python -m sglang.launch_server --help > /dev/null
fi
python -m pip check
mkdir -p runs
python -m pip freeze > "runs/$engine.freeze.txt"
python -m src.preflight > "runs/$engine.preflight.json"
