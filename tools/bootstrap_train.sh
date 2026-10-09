#!/usr/bin/env bash
set -euo pipefail
# Run on a CPU/interactive node where package installation is allowed. Never replace system CUDA.
python3 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 transformers==4.51.3 peft==0.15.2 accelerate==1.6.0 safetensors==0.5.3
python -m pip check
mkdir -p runs
python -m pip freeze > runs/train.freeze.txt
python -m src.preflight > runs/train.preflight.json
