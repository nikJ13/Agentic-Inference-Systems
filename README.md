# Agentic Inference Systems

This project implements advanced agentic systems for complex inference tasks, focusing on Deep Research capabilities and Self-Refining generation strategies.

## Project Structure

The repository focuses on two main agentic workflows:

### 1. Deep Research Agent (`deep_research_agent/`)
A comprehensive research agent capable of executing multi-step research tasks.
- **Core capabilities:**
  - Automated information gathering and synthesis
  - Multi-step reasoning and planning
  - Integration with external tools (Search, Browser)
- **Key components:**
  - `react_agent.py`: Implementation of the ReAct (Reasoning + Acting) paradigm.
  - `mcp_agents/`: Modular Component Protocol (MCP) agents for extensible tool use.
  - `graph/`: Graph-based reasoning utilities.

### 2. Self-Refining Agent (`self_refine/`)
An agentic system that iteratively improves its own outputs through self-correction.
- **Core capabilities:**
  - Self-evaluation of generated content
  - Iterative refinement loops
  - Performance analysis on benchmarks (MMLU, Graph tasks)
- **Key components:**
  - `self_refine.py`: Main logic for the self-refining loop.
  - `refine_modal.py`: Modal integration for scalable execution.
  - `analyze_accuracy.py`: Tools for evaluating refinement performance.

### 3. Reranking & Evaluation (`rerank_outputs.py`)
Tools for evaluating and selecting the best generations from multiple candidates.
- Implements various scoring mechanisms:
  - **Scalar Reward Models** using Skywork/Reward-Llama
  - **Pairwise Reward Models** using LLM-Blender (PairRM)
  - **MBR (Minimum Bayes Risk)** decoding with BLEU and BERTScore
  - **Log-probability analysis** using Qwen models

## Setup & Installation

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables:**
   Create a `.env` file with the following keys:
   ```bash
   OPENAI_API_KEY=your_key
   ANTHROPIC_API_KEY=your_key
   # Add other provider keys as needed
   ```

## Usage

### Running the Deep Research Agent
Navigate to the `deep_research_agent` directory and configure the agent in `react_agent.yaml`.
```bash
python deep_research_agent/react_agent.py
```

### Running Self-Refinement
To run the self-refinement experiments:
```bash
python self_refine/self_refine.py --task mmlu --model qwen3-4b
```

### Reranking Outputs
To evaluate generated outputs using the reranking system:
```bash
python rerank_outputs.py
```
This will process `all_results_processed.json` and compute scores for all candidates.

## Analysis

Use `calculate_stats_reranking.py` to generate statistical analysis and plots comparing different reranking strategies against gold-standard evaluations.
