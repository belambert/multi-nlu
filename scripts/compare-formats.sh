#!/usr/bin/env bash
# Compare xml/offset/dict on the same model, split and sample.
set -euo pipefail

model="Qwen/Qwen3-1.7B"

for format in xml offset dict; do
    echo "== $format =="
    uv run multi-nlu predict -m "$model" --format "$format" --shots 8 --split test -n 100
    echo
done
