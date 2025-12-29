# Self-Refine Pipeline

This directory contains a lightweight implementation of the self-refinement workflow used in Homework 2. It lets you generate reasoning drafts with a causal LLM, collect feedback, iteratively refine answers, and analyze how refinement affects accuracy.

## Repository Layout

- `self_refine.py` – main pipeline entry point (draft → feedback → refine).  
- `dataset.py` – dataset handler abstractions for graph shortest-path and MMLU-med QA.  
- `refine_modal.py` – Modal deployment helper for running the pipeline on GPU.  
- `analyze_accuracy.py`, `analyze_conditional.py` – scripts for Section 3.2 analysis questions.  
- `results/` - Stores all the responses and graphs

## Prerequisites

1. Python 3.10+ recommended.  
2. Install dependencies:
   ```bash
   cd self_refine
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. (Optional) Set Hugging Face credentials if you plan to pull gated models:
   ```bash
   export HF_HOME=~/.cache/huggingface
   export HF_TOKEN=your_hf_token
   ```
   The current script hard-codes placeholder values—override them via your environment before running in production.

## Running the Pipeline Locally

1. **Prepare data** – Provide a JSONL file with examples compatible with the handler:
   - Graph handler expects entries with keys like `edges`, `N`, and ground-truth paths.
   - MMLU-med handler expects `question`, `choices`, and `answer`.

2. **Invoke the CLI**:
   ```bash
   python src/self_refine.py \
     --handler graph \
     --output runs/graph_self_refine.jsonl \
     --model-path Qwen/Qwen3-4B \
     --num-refine-steps 3 \
     --batch-size 4 \
     --temperature 0.1 \
     --draft-temperature 0.7 \
     --feedback-temperature 1.0 \
     --refine-temperature 1.5 \
     --trust-remote-code
   ```

   Key arguments:
   - `--handler`: `graph` or `mmlu_med`.
   - `--input`: JSONL file with evaluation examples.
   - `--model-path`: Hugging Face model identifier or local checkpoint.
   - `--num-refine-steps`: number of draft → feedback → refine cycles.
   - Temperatures can be tuned per stage.

   Temperature presets tracked:
   - `cfg1`: draft 0.7, feedback 1.0, refine 1.5
   - `cfg2`: draft 0.1, feedback 1.0, refine 0.7

## Reproducing Reported Runs

All commands submitted for Homework 2 are captured in `run_self_refine.sh`. Make it executable and launch the full suite:

```bash
chmod +x run_self_refine.sh
./run_self_refine.sh
```

The script records eight runs (graph + MMLU × cfg1/cfg2 × {Qwen-2.5 4B, Qwen-2.5 0.6B}) and writes their outputs under `src/jsonl_results/`.

3. Outputs are written as JSONL with draft, refinement attempts, and correctness metadata.

## Modal Deployment

Use `refine_modal.py` to schedule a GPU job on Modal:

```bash
modal run src/refine_modal.py \
  --handler graph \
  --model-path Qwen/Qwen3-4B \
  --output modal_runs/graph.jsonl \
  --num-refine-steps 3
```

The app image mirrors `requirements.txt` and mounts `/results` for persisted outputs.

## Analysis & Reporting

After generating results:

1. **Accuracy curves**  
   ```bash
   python src/analyze_accuracy.py \
     --results runs/graph_self_refine.jsonl \
     --output plots/accuracy_vs_iteration.png \
     --num-iterations 4
   ```

2. **Conditional probabilities**  
   ```bash
   python src/analyze_conditional.py \
     --results runs/graph_self_refine.jsonl \
     --output plots/conditional_probabilities.png \
     --num-iterations 4
   ```
