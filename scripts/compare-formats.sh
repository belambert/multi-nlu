#!/usr/bin/env bash
# Compare xml/dict on the same model, split and sample.
set -euo pipefail

model="Qwen/Qwen3.5-2B"

for format in xml dict; do
    echo "== $format =="
    uv run multi-nlu predict -m "$model" --format "$format" --shots 8 --split test
    echo
done
