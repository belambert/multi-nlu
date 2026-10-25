#!/usr/bin/env bash
# Train a token tagger per backbone, then evaluate it on the full test split.
# Qwen is a decoder: it tags with left context only, the point of comparison.
# Ettin pairs encoders and decoders trained alike, isolating that difference.
# Backbones whose run already has predictions are skipped.
set -euo pipefail

# commented out: full fine-tuning with AdamW needs ~16 bytes per parameter,
# more than the DGX Spark's 128 GB above roughly 7B
models=(
    answerdotai/ModernBERT-base
    answerdotai/ModernBERT-large
    google-bert/bert-base-uncased
    google-bert/bert-large-uncased
    FacebookAI/roberta-base
    FacebookAI/roberta-large
    microsoft/deberta-v3-base
    microsoft/deberta-v3-large
    jhu-clsp/ettin-encoder-150m
    jhu-clsp/ettin-decoder-150m
    jhu-clsp/ettin-encoder-400m
    jhu-clsp/ettin-decoder-400m
    jhu-clsp/ettin-encoder-1b
    jhu-clsp/ettin-decoder-1b
    EuroBERT/EuroBERT-210m
    EuroBERT/EuroBERT-610m
    EuroBERT/EuroBERT-2.1B
    google/flan-t5-large
    google/flan-t5-xl
    google/flan-t5-xxl # encoder only, about 4.8B
    facebook/xlm-roberta-xl
    # facebook/xlm-roberta-xxl
    Qwen/Qwen3.5-0.8B
    Qwen/Qwen3.5-2B
    Qwen/Qwen3.5-4B
    # Qwen/Qwen3.5-9B
)

for model in "${models[@]}"; do
    run="runs/$(basename "$model" | tr '[:upper:]' '[:lower:]')-tagger"
    if [[ -f "$run/preds.jsonl" ]]; then
        echo "== $model: done, skipping =="
        continue
    fi
    echo "== $model: train =="
    uv run multi-nlu tag-train -m "$model" -o "$run" --wandb-project multi-nlu-tagger
    echo "== $model: eval =="
    uv run multi-nlu tag-predict -t "$run" --split test -o "$run/preds.jsonl"
    echo
done
