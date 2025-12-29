#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Self-refine runs used in Homework 2.
# cfg1 -> draft 0.7, feedback 1.0, refine 1.5
# cfg2 -> draft 0.1, feedback 1.0, refine 0.7
# -----------------------------------------------------------------------------

set -euo pipefail

cd "$(dirname "$0")"

mkdir -p results

run_experiment() {
  local description="$1"
  local handler="$2"
  local output_path="$3"
  local model_path="$4"
  local draft_temp="$5"
  local feedback_temp="$6"
  local refine_temp="$7"

  echo "Running ${description}..."
  python self_refine.py \
    --handler "${handler}" \
    --output "${output_path}" \
    --model-path "${model_path}" \
    --num-refine-steps 3 \
    --batch-size 4 \
    --temperature 0.0 \
    --draft-temperature "${draft_temp}" \
    --feedback-temperature "${feedback_temp}" \
    --refine-temperature "${refine_temp}" \
    --trust-remote-code
}

# MMLU experiments
run_experiment "MMLU (cfg1) with Qwen3 4B" \
  mmlu_med \
  results/qwen3-4b_mmlu_cfg1.jsonl \
  Qwen/Qwen3-4B \
  0.7 1.0 1.5

run_experiment "MMLU (cfg2) with Qwen3 4B" \
  mmlu_med \
  results/qwen3-4b_mmlu_cfg2.jsonl \
  Qwen/Qwen3-4B \
  0.1 1.0 0.7

run_experiment "MMLU (cfg1) with Qwen3 0.6B" \
  mmlu_med \
  results/qwen3-0.6B_mmlu_cfg1.jsonl \
  Qwen/Qwen3-0.6B \
  0.7 1.0 1.5

run_experiment "MMLU (cfg2) with Qwen3 0.6B" \
  mmlu_med \
  results/qwen3-0.6B_mmlu_cfg2.jsonl \
  Qwen/Qwen3-0.6B \
  0.1 1.0 0.7

# Graph experiments
run_experiment "Graph (cfg1) with Qwen3 4B" \
  graph \
  results/qwen3-4b_graph_cfg1.jsonl \
  Qwen/Qwen3-4B \
  0.7 1.0 1.5

run_experiment "Graph (cfg2) with Qwen3 4B" \
  graph \
  results/qwen3-4b_graph_cfg2.jsonl \
  Qwen/Qwen3-4B \
  0.1 1.0 0.7

run_experiment "Graph (cfg1) with Qwen3 0.6B" \
  graph \
  results/qwen3-0.6b_graph_cfg1.jsonl \
  Qwen/Qwen3-0.6B \
  0.7 1.0 1.5

run_experiment "Graph (cfg2) with Qwen3 0.6B" \
  graph \
  results/qwen3-0.6b_graph_cfg2.jsonl \
  Qwen/Qwen3-0.6B \
  0.1 1.0 0.7

echo "All runs completed."
