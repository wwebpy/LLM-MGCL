#!/usr/bin/env bash
# Reproduces all experiments of the paper (3 seeds each). Expect several GPU hours.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${CONFIG:-configs/default.yaml}

python scripts/build_graphs.py --config "$CONFIG"

# LLM-MGCL and ablations
for variant in full wo_cl wo_sem wo_geo; do
  python scripts/train.py --model llm_mgcl --variant "$variant" --config "$CONFIG"
done

# Trainable baselines
for model in lightgcn sgl ngcf gcmc neumf mf_bpr; do
  python scripts/train.py --model "$model" --config "$CONFIG"
done

# Evaluation (ItemKNN / UserKNN are computed here; they need no training)
python scripts/compare_models.py --config "$CONFIG"
python scripts/cold_start.py --config "$CONFIG"
python scripts/make_figures.py --config "$CONFIG"
