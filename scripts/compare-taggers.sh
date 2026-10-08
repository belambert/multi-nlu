#!/usr/bin/env bash
# Train a token tagger per backbone, then evaluate it on the full test split.
# Qwen is a decoder: it tags with left context only, the point of comparison.
set -euo pipefail

models=(
    answerdotai/ModernBERT-base
    google-bert/bert-base-uncased
    FacebookAI/roberta-base
    Qwen/Qwen3.5-0.8B
)

for model in "${models[@]}"; do
    run="runs/$(basename "$model" | tr '[:upper:]' '[:lower:]')-tagger"
    echo "== $model: train =="
    uv run multi-nlu tag-train -m "$model" -o "$run" --wandb-project multi-nlu-tagger
    echo "== $model: eval =="
    uv run multi-nlu tag-predict -t "$run" --split test -o "$run/preds.jsonl"
    echo
done
