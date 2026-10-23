#!/usr/bin/env bash
# Train a LoRA adapter per model/format, then evaluate it, same split and sample.
# Pass model ids to run only those. An existing adapter is reused rather than
# retrained (delete its run directory to retrain), and a run whose predictions
# exist is skipped, so a rerun resumes; failures are listed at the end instead of
# stopping the rest.
set -uo pipefail

# keep nohup logs small: redraw bars once a minute, in ascii, and hide HF download bars
export TQDM_MININTERVAL="${TQDM_MININTERVAL:-60}"
# tqdm reads this as the bar's characters, so "1" or "True" break or garble the bar
export TQDM_ASCII="${TQDM_ASCII:- 123456789#}"
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
        run="runs/$name"
        if [[ -f $run/preds.jsonl ]]; then
            echo "== $model $format: skipped, $run/preds.jsonl exists =="
            continue
        fi

        # the adapter is only saved once training finishes, so its config marks a complete run
        if [[ -f $run/adapter_config.json ]]; then
            echo "== $model $format: reusing the adapter in $run =="
        else
            echo "== $model $format: train =="
            if ! uv run multi-nlu train -m "$model" --format "$format" -o "$run" \
                --wandb-project multi-nlu-finetune; then
                failed+=("$model $format (train)")
                continue
            fi
        fi

        echo "== $model $format: eval =="
        uv run multi-nlu predict -m "$model" --format "$format" --adapter "$run" --split test \
            --out "$run/preds.jsonl" || failed+=("$model $format (eval)")
        echo
    done
done

if ((${#failed[@]})); then
    printf 'failed: %s\n' "${failed[@]}" >&2
    exit 1
fi
