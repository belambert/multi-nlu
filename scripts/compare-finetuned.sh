#!/usr/bin/env bash
# Train a LoRA adapter per model/format, then evaluate it, same split and sample.
set -euo pipefail

# keep nohup logs small: redraw bars once a minute, in ascii, and hide HF download bars
export TQDM_MININTERVAL="${TQDM_MININTERVAL:-60}" TQDM_ASCII="${TQDM_ASCII:-1}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"

models=(
    Qwen/Qwen3.5-0.8B
    Qwen/Qwen3.5-2B
    Qwen/Qwen3.5-4B
    Qwen/Qwen3.5-9B
    Qwen/Qwen3.5-27B
    Qwen/Qwen3.5-35B-A3B
)

for model in "${models[@]}"; do
    for format in xml dict; do
        name="$(basename "$model" | tr '[:upper:]' '[:lower:]')-$format"
        run="runs/$name"
        echo "== $model $format: train =="
        uv run multi-nlu train -m "$model" --format "$format" -o "$run" --wandb-project multi-nlu-finetune
        echo "== $model $format: eval =="
        uv run multi-nlu predict -m "$model" --format "$format" --adapter "$run" --split test
        echo
    done
done
