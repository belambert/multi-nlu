#!/usr/bin/env bash
# Compare xml/dict on each instruct Qwen3.5 model under 40B, same split and sample.
# Pass model ids to run only those. A run whose predictions exist is skipped, so a
# rerun resumes; failures are listed at the end instead of stopping the rest.
set -uo pipefail

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
(($#)) && models=("$@")

failed=()
for model in "${models[@]}"; do
    for format in xml dict; do
        name="$(basename "$model" | tr '[:upper:]' '[:lower:]')-$format"
        preds="runs/few-shot/$name.jsonl"
        if [[ -f $preds ]]; then
            echo "== $model $format: skipped, $preds exists =="
            continue
        fi

        echo "== $model $format =="
        uv run multi-nlu predict -m "$model" --format "$format" --shots 8 --split test \
            --out "$preds" || failed+=("$model $format")
        echo
    done
done

if ((${#failed[@]})); then
    printf 'failed: %s\n' "${failed[@]}" >&2
    exit 1
fi
