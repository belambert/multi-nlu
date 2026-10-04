#!/usr/bin/env bash
# Compare xml/dict on each instruct Qwen3.5 model under 40B, same split and sample.
set -euo pipefail

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
        echo "== $model $format =="
        uv run multi-nlu predict -m "$model" --format "$format" --shots 8 --split test \
            -o "predictions/few-shot/$name.jsonl"
        echo
    done
done
