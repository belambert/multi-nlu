#!/usr/bin/env bash
# Run `multi-nlu predict` once per registered format, against the same model and
# sampled examples, and print a side-by-side metrics table.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: compare-formats.sh -m MODEL [options]

Required:
  -m, --model MODEL      Hugging Face model id

Options:
  -s, --schema SCHEMA    Built-in schema name or YAML path (default: mixsnips)
      --split SPLIT      Dataset split to run on (default: test)
  -n, --limit N          Random subsample size (default: all examples)
      --shots N          Few-shot demonstrations from the train split (default: 0)
  -b, --batch-size N     Batch size (default: 8)
      --device DEVICE    cpu, mps or cuda; auto by default
      --seed N           Random seed, shared across formats for a fair sample (default: 0)
      --formats LIST     Comma-separated formats to compare (default: all of `multi-nlu formats`)
  -o, --out-dir DIR      Where to write each format's predictions and logs (default: runs/compare)
  -h, --help             Show this help

Example:
  ./scripts/compare-formats.sh -m Qwen/Qwen3-1.7B --shots 8 --split test -n 200

Predictions land at <out-dir>/<format>.jsonl; re-inspect them any time with:
  uv run multi-nlu score <out-dir>/<format>.jsonl -d errors
EOF
}

model=""
schema="mixsnips"
split="test"
limit=""
shots="0"
batch_size="8"
device=""
seed="0"
formats_opt=""
out_dir="runs/compare"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model) model="$2"; shift 2 ;;
        -s|--schema) schema="$2"; shift 2 ;;
        --split) split="$2"; shift 2 ;;
        -n|--limit) limit="$2"; shift 2 ;;
        --shots) shots="$2"; shift 2 ;;
        -b|--batch-size) batch_size="$2"; shift 2 ;;
        --device) device="$2"; shift 2 ;;
        --seed) seed="$2"; shift 2 ;;
        --formats) formats_opt="$2"; shift 2 ;;
        -o|--out-dir) out_dir="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [[ -z "$model" ]]; then
    echo "error: -m/--model is required" >&2
    usage >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "error: 'uv' not found on PATH; install it from https://docs.astral.sh/uv/" >&2
    exit 1
fi

# resolve the repo root from this script's own location, not the caller's cwd
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "$root/pyproject.toml" ]]; then
    echo "error: no pyproject.toml found at $root; is this script still inside the multi-nlu repo?" >&2
    exit 1
fi

run() { uv run --project "$root" multi-nlu "$@"; }

# discover formats fresh each run, so new ones (e.g. `dict`) are picked up without editing this script
available=()
while IFS= read -r line; do
    available+=("$line")
done < <(run formats | cut -d: -f1)
if [[ ${#available[@]} -eq 0 ]]; then
    echo "error: 'multi-nlu formats' listed no formats" >&2
    exit 1
fi

if [[ -n "$formats_opt" ]]; then
    IFS=',' read -r -a targets <<< "$formats_opt"
    for fmt in "${targets[@]}"; do
        if ! printf '%s\n' "${available[@]}" | grep -qx "$fmt"; then
            echo "error: unknown format '$fmt'; available: ${available[*]}" >&2
            exit 1
        fi
    done
else
    targets=("${available[@]}")
fi

mkdir -p "$out_dir"
echo "comparing formats: ${targets[*]}"
echo "model: $model   schema: $schema   split: $split   shots: $shots   seed: $seed   limit: ${limit:-all}"
echo

for fmt in "${targets[@]}"; do
    args=(predict --model "$model" --schema "$schema" --format "$fmt" --split "$split"
          --shots "$shots" --batch-size "$batch_size" --seed "$seed"
          --out "$out_dir/$fmt.jsonl")
    [[ -n "$limit" ]] && args+=(--limit "$limit")
    [[ -n "$device" ]] && args+=(--device "$device")

    echo "== $fmt =="
    # stdout only: the score summary (predict's progress bars go to stderr, left on screen live)
    run "${args[@]}" > "$out_dir/$fmt.log"
    cat "$out_dir/$fmt.log"
    echo
done

# Build the comparison table.
#
# Each <fmt>.log ends with lines from Scores.summary() (metrics.py): a "n = N"
# line, then one "<name>  <value>" row per metric, name left-padded to a fixed
# width and value a right-aligned percentage. This depends on that layout: we
# pull the metric name as everything but the last whitespace-delimited field,
# and the value as that last field, which also makes the table robust to the
# padding width changing if a longer metric name is ever added.
# skip any line whose last field isn't a percentage (the "wrote N predictions..."
# and "n = N" lines that precede the metric rows in predict's output)
metrics_file() {
    awk '{ v = $NF; if (v !~ /%$/) next; sub(" *" v "$", "", $0); print $0 "\t" v }' "$1"
}

names_file="$out_dir/.metric-names"
metrics_file "$out_dir/${targets[0]}.log" | cut -f1 > "$names_file"

col_width=12
header="metric"
printf -v header_pad "%-28s" "$header"
row="$header_pad"
for fmt in "${targets[@]}"; do
    printf -v cell "%${col_width}s" "$fmt"
    row+="$cell"
done
echo "$row"

while IFS= read -r name; do
    printf -v row "%-28s" "$name"
    for fmt in "${targets[@]}"; do
        value="$(metrics_file "$out_dir/$fmt.log" | awk -F'\t' -v n="$name" '$1==n{print $2}')"
        printf -v cell "%${col_width}s" "$value"
        row+="$cell"
    done
    echo "$row"
done < "$names_file"

rm -f "$names_file"
