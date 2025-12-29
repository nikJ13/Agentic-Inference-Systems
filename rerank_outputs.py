import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from typing import List, Optional
from sacrebleu.metrics import BLEU
from bert_score import score as bert_score
from tqdm import tqdm
import gc
import json
import os
from pathlib import Path


class OutputReranker:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.models = {}
        self.tokenizers = {}

    def _clear_memory(self):
        gc.collect()
        torch.cuda.empty_cache()

    def _load_model(self, model_name: str):
        if model_name not in self.models:
            print(f"Loading {model_name}...")
            self.models[model_name] = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map='auto'
            )
            self.models[model_name].eval()
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
        return self.models[model_name], self.tokenizers[model_name]

    def _load_model_reward(self, model_name):
        if model_name not in self.models:
            self.models[model_name] = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map='auto',
                num_labels=1
            )
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
        return self.models[model_name], self.tokenizers[model_name]

    def compute_model_prob(
        self,
        outputs: List[str],
        prompt: str,
        model: str = "Qwen/Qwen3-4B"
    ) -> List[float]:
        model_obj, tokenizer = self._load_model(model)
        log_probs = []
        try:
            for output in tqdm(outputs, desc=f"Computing {model} log-probs"):
                chat = [{"role": "user", "content": prompt}]
                full_chat = chat + [{"role": "assistant", "content": output}]
                full_text = tokenizer.apply_chat_template(
                    full_chat,
                    tokenize=False,
                    add_generation_prompt=False
                )
                prompt_text = tokenizer.apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=True
                )
                full_tokens = tokenizer(full_text, return_tensors="pt")["input_ids"].to(self.device)
                prompt_tokens = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(self.device)
                with torch.no_grad():
                    logits = model_obj(full_tokens).logits
                    log_probs_all = F.log_softmax(logits[0], dim=-1)
                prompt_len = prompt_tokens.shape[1]
                output_tokens = full_tokens[0, prompt_len:]
                cumulative_log_prob = 0.0
                for i, token_id in enumerate(output_tokens):
                    if i + prompt_len - 1 < log_probs_all.shape[0]:
                        cumulative_log_prob += log_probs_all[i + prompt_len - 1, token_id].item()
                log_probs.append(cumulative_log_prob)
        finally:
            pass
        return log_probs

    def compute_scalar_reward(
        self,
        outputs: List[str],
        prompt: str
    ) -> List[float]:
        model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
        model_obj, tokenizer = self._load_model_reward(model_name)
        rewards = []
        try:
            for output in tqdm(outputs, desc="Computing scalar rewards"):
                chat = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": output}
                ]
                input_ids = tokenizer.apply_chat_template(
                    chat,
                    return_tensors="pt"
                ).to(self.device)
                with torch.no_grad():
                    output_model = model_obj(input_ids)
                    reward = output_model.logits[0][0].item()
                rewards.append(reward)
        finally:
            pass
        return rewards

    def compute_pairwise_reward(
        self,
        outputs: List[str],
        prompt: str
    ) -> List[float]:
        import llm_blender
        print("Loading PairRM reranker")
        blender = llm_blender.Blender()
        # blender.loadranker("llm-blender/PairRM") # Use default or specific model ID
        blender.loadranker("llm-blender/PairRM")
        n = len(outputs)
        wins = [0] * n
        try:
            for i in tqdm(range(n), desc="Computing pairwise comparisons"):
                for j in range(i + 1, n):
                    try:
                        ranks = blender.rank(
                            [prompt],
                            [[outputs[i], outputs[j]]],
                            return_scores=True,
                            batch_size=1
                        )
                        if ranks[0][0] > ranks[0][1]:
                            wins[i] += 1
                        else:
                            wins[j] += 1
                    except Exception as e:
                        print(f"\nWarning: Error comparing outputs {i} vs {j}: {e}")
                        continue
        finally:
            del blender
            self._clear_memory()
        return [float(w) for w in wins]

    def mbr_bleu(
        self,
        outputs: List[str],
        prompt: str = None
    ) -> List[float]:
        bleu = BLEU(effective_order=True)
        n = len(outputs)
        mbr_scores = []
        for i in tqdm(range(n), desc="Computing MBR-BLEU"):
            total_score = 0.0
            for j in range(n):
                if i != j:
                    score = bleu.sentence_score(
                        outputs[i],
                        [outputs[j]]
                    ).score
                    total_score += score
            mbr_scores.append(total_score / (n - 1) if n > 1 else 0.0)
        return mbr_scores

    def mbr_bertscore(
        self,
        outputs: List[str],
        prompt: str = None
    ) -> List[float]:
        n = len(outputs)
        mbr_scores = []
        print("Computing BERTScores")
        for i in tqdm(range(n), desc="Computing MBR-BERTScore"):
            candidates = [outputs[i]] * (n - 1)
            references = [outputs[j] for j in range(n) if j != i]
            if len(references) > 0:
                P, R, F1 = bert_score(
                    candidates,
                    references,
                    lang='en',
                    model_type='roberta-large-mnli',
                    device=self.device,
                    batch_size=32
                )
                avg_score = F1.mean().item()
                mbr_scores.append(avg_score)
            else:
                mbr_scores.append(0.0)
        return mbr_scores


def check_scores_complete(candidate: dict, score_names: List[str]) -> bool:
    scores = candidate.get('scores', {})
    return all(scores.get(name) is not None for name in score_names)


def save_checkpoint(all_results: dict, output_file: str):
    temp_file = output_file + '.tmp'
    backup_file = output_file + '.backup'
    try:
        with open(temp_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        if os.path.exists(output_file):
            if os.path.exists(backup_file):
                os.remove(backup_file)
            os.rename(output_file, backup_file)
        os.rename(temp_file, output_file)
        print(f"Checkpoint saved to {output_file}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        if os.path.exists(backup_file) and not os.path.exists(output_file):
            os.rename(backup_file, output_file)
            print("Restored from backup")
        raise


def load_results_safely(file_path: str) -> Optional[dict]:
    if not os.path.exists(file_path):
        return None
    try:
        print(f"Attempting to load {file_path}...")
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded {file_path}")
        return data
    except json.JSONDecodeError as e:
        print(f"Error loading {file_path}: {e}")
        backup_file = file_path + '.backup'
        if os.path.exists(backup_file):
            print(f"Attempting to load backup file {backup_file}...")
            try:
                with open(backup_file, 'r') as f:
                    data = json.load(f)
                print("Successfully loaded backup file")
                return data
            except json.JSONDecodeError as e2:
                print(f"Backup file also corrupted: {e2}")
        print("Could not load any saved progress. Will start fresh.")
        return None


def process_with_checkpointing(
    input_file: str = "all_results_processed.json",
    output_file: str = "all_results_with_scores.json",
    num_questions: Optional[int] = None
):
    all_results = load_results_safely(output_file)
    if all_results is None:
        print(f"Loading fresh data from {input_file}...")
        with open(input_file) as f:
            all_results = json.load(f)
        print("Starting from scratch...")
    else:
        print("Resuming from previous progress...")
    reranker = OutputReranker()
    score_names = ['qwen3_4b', 'qwen3_14b', 'r_scalar', 'r_pairwise', 'mbr_bleu', 'mbr_bert']
    results_to_process = all_results['results'][:num_questions] if num_questions else all_results['results']
    for idx, result in enumerate(results_to_process):
        question_id = result['question_id']
        all_complete = all(
            check_scores_complete(c, score_names)
            for c in result['candidates']
        )
        if all_complete:
            print(f"\nQuestion {question_id} ({idx+1}/{len(results_to_process)}): All scores already computed. Skipping...")
            continue
        print(f"Processing question {question_id} ({idx+1}/{len(results_to_process)})\n")
        prompt = result['prompt']['intermediate_prompt']
        candidates = result['candidates']
        outputs = [c['generated_text'] for c in candidates]
        for candidate in candidates:
            if 'scores' not in candidate:
                candidate['scores'] = {}
        try:
            if any(c['scores'].get('qwen3_4b') is None for c in candidates):
                print("\n1. Computing Qwen3-4B log probs...")
                qwen4b_scores = reranker.compute_model_prob(outputs, prompt, "Qwen/Qwen3-4B")
                for i, candidate in enumerate(candidates):
                    candidate['scores']['qwen3_4b'] = qwen4b_scores[i]
                save_checkpoint(all_results, output_file)
            else:
                print("\n1. Qwen3-4B log probs already computed. Skipping...")
            if any(c['scores'].get('qwen3_14b') is None for c in candidates):
                print("\n2. Computing Qwen3-14B log probs...")
                qwen14b_scores = reranker.compute_model_prob(outputs, prompt, "Qwen/Qwen3-14B")
                for i, candidate in enumerate(candidates):
                    candidate['scores']['qwen3_14b'] = qwen14b_scores[i]
                save_checkpoint(all_results, output_file)
            else:
                print("\n2. Qwen3-14B log probs already computed. Skipping...")
            if any(c['scores'].get('r_scalar') is None for c in candidates):
                print("\n3. Computing scalar rewards...")
                scalar_scores = reranker.compute_scalar_reward(outputs, prompt)
                for i, candidate in enumerate(candidates):
                    candidate['scores']['r_scalar'] = scalar_scores[i]
                save_checkpoint(all_results, output_file)
            else:
                print("\n3. Scalar rewards already computed. Skipping...")
            if any(c['scores'].get('r_pairwise') is None for c in candidates):
                print("\n4. Computing pairwise rewards...")
                pairwise_scores = reranker.compute_pairwise_reward(outputs, prompt)
                for i, candidate in enumerate(candidates):
                    candidate['scores']['r_pairwise'] = pairwise_scores[i]
                save_checkpoint(all_results, output_file)
            else:
                print("\n4. Pairwise rewards already computed. Skipping...")
            if any(c['scores'].get('mbr_bleu') is None for c in candidates):
                print("\n5. Computing MBR-BLEU...")
                mbr_bleu_scores = reranker.mbr_bleu(outputs, prompt)
                for i, candidate in enumerate(candidates):
                    candidate['scores']['mbr_bleu'] = mbr_bleu_scores[i]
                save_checkpoint(all_results, output_file)
            else:
                print("\n5. MBR-BLEU already computed. Skipping...")
            if any(c['scores'].get('mbr_bert') is None for c in candidates):
                print("\n6. Computing MBR-BERTScore...")
                mbr_bert_scores = reranker.mbr_bertscore(outputs, prompt)
                for i, candidate in enumerate(candidates):
                    candidate['scores']['mbr_bert'] = mbr_bert_scores[i]
                save_checkpoint(all_results, output_file)
            else:
                print("\n6. MBR-BERTScore already computed. Skipping...")
        except Exception as e:
            print(f"\n!!! Error processing question {question_id}: {e}")
            print("Saving current progress before continuing...")
            save_checkpoint(all_results, output_file)
            print("Continuing to next question...")
            continue
    print("\nProcessing complete! Saving final results...")
    save_checkpoint(all_results, output_file)
    print(f"All results saved to {output_file}")


def analyze_generation_statistics(input_file: str = "all_results_processed.json"):
    with open(input_file) as f:
        all_results = json.load(f)
    
    unique_generations_per_prompt = []
    for result in all_results['results']:
        generated_texts = [candidate['generated_text'] for candidate in result['candidates']]
        unique_texts = set(generated_texts)
        unique_generations_per_prompt.append(len(unique_texts))
    
    mean_unique_generations = np.mean(unique_generations_per_prompt)
    median_unique_generations = np.median(unique_generations_per_prompt)
    std_unique_generations = np.std(unique_generations_per_prompt)
    
    infobench_score_differences = []
    for result in all_results['results']:
        infobench_scores = [
            candidate['scores']['infobench'] 
            for candidate in result['candidates'] 
            if candidate['scores'].get('infobench') is not None
        ]
        if infobench_scores:
            max_score = max(infobench_scores)
            min_score = min(infobench_scores)
            score_difference = max_score - min_score
            infobench_score_differences.append(score_difference)
    
    average_infobench_score_difference = np.mean(infobench_score_differences)
    
    print(f"\nUnique Generations per Prompt:")
    print(f"  Mean:   {mean_unique_generations:.4f}")
    print(f"  Median: {median_unique_generations:.4f}")
    print(f"  Std:    {std_unique_generations:.4f}")
    print(f"\nInfoBench Score Analysis:")
    print(f"  Average difference (best - worst): {average_infobench_score_difference:.4f}")
    
    num_questions_to_examine = 5
    print(f"\nExamining {num_questions_to_examine} prompts:")
    for i in range(min(num_questions_to_examine, len(all_results['results']))):
        question = all_results['results'][i]
        print(f"\n--- Question {i+1} ---")
        print(f"Prompt: {question['prompt']['intermediate_prompt'][:100]}...")
        print(f"Number of unique generations: {unique_generations_per_prompt[i]}")
        
        candidates_to_examine = question['candidates'][:5]
        for j, candidate in enumerate(candidates_to_examine):
            print(f"\n  Candidate {j+1}:")
            print(f"    Length: {candidate['generation_len']}")
            print(f"    Finish: {candidate['finish_reason']}")
            print(f"    Text: {candidate['generated_text'][:300]}...")
    
    return {
        'mean_unique_generations': mean_unique_generations,
        'median_unique_generations': median_unique_generations,
        'std_unique_generations': std_unique_generations,
        'average_infobench_score_difference': average_infobench_score_difference
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--clean':
        print("Cleaning up corrupted output files...")
        output_file = "all_results_with_scores.json"
        backup_file = output_file + '.backup'
        temp_file = output_file + '.tmp'
        for f in [output_file, backup_file, temp_file]:
            if os.path.exists(f):
                print(f"Removing {f}...")
                os.remove(f)
        print("Cleanup complete. Run without --clean to start fresh.")
        sys.exit(0)
    
    if len(sys.argv) > 1 and sys.argv[1] == '--stats':
        print("Running statistics analysis...")
        stats = analyze_generation_statistics("all_results_processed.json")
        sys.exit(0)
    
    process_with_checkpointing(
        input_file="all_results_processed.json",
        output_file="all_results_with_scores.json",
        num_questions=None
    )
