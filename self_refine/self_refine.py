import os, json, time, argparse, logging
from dataclasses import dataclass
from pathlib import Path
import torch
import dataset
from typing import List, Dict, Any, Tuple, Optional
import re
torch.manual_seed(42)
GraphHandler = dataset.GraphHandler 
MMLUMedHandler = dataset.MMLUMedHandler

os.environ["HF_HOME"] = ""
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

"""NOTE: The repo includes a bare-bones scaffolds. 
It exists to help you start quickly. 
Please feel free to change your structure. 
Any clean, reproducible solution is acceptable.
"""

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

@dataclass
class RefineConfig:
    """Configuration for self-refine process."""
    # Adjust as needed
    model_path: str = None
    dtype: str = "bfloat16"
    max_new_tokens: int = 256
    temperature: float = 0.0
    batch_size: int = 1
    num_refine_steps: int = 3
    device: Optional[str] = None
    stop_sequences: Tuple[str, ...] = ()
    draft_temperature: Optional[float] = None
    feedback_temperature: Optional[float] = None
    refine_temperature: Optional[float] = None
    trust_remote_code: bool = False
    # chat_template_kwargs: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.draft_temperature is None:
            self.draft_temperature = self.temperature
        if self.feedback_temperature is None:
            self.feedback_temperature = self.temperature
        if self.refine_temperature is None:
            self.refine_temperature = self.temperature

def chat(role: str, content: str) -> str:
    """Format chat messages - adjust for your model's chat template"""
    return {"role": role, "content": content}

# These are for abstractions you can make them dataset specific or agnostic based on your design
def draft_prompt(question: str, handler_type: str) -> str:
    # NOTE: This model is the solver 
    # You might wanna experiment with prompts and instruction for each dataset
    if handler_type == "graph":
        k = 1
        match = re.search(r'top (\d+)', question.lower())
        if match:
            k = int(match.group(1))
        elif "single shortest path" in question.lower():
            k = 1
        
        if k == 1:
            requirement_1 = "Return exactly 1 path (the shortest one)"
            format_spec = """{"paths":[{"path":[node1,node2,...,nodeN],"weight":total_weight}]}

    Where:
    - node1, node2, ..., nodeN are the actual node IDs in your path
    - total_weight is the sum of all edge weights"""
        else:
            requirement_1 = f"Return exactly {k} paths (ordered shortest to longest)"
            format_spec = """{"paths":[{"path":[...],"weight":...},{"path":[...],"weight":...}]}

    Where each object contains the actual path nodes and calculated weight."""
        
        prompt = f"""{question}

    Requirements:
    1. {requirement_1}
    2. Use integer node IDs: [0, 3, 5] NOT ["0", "3", "5"]
    3. Only use edges from the graph above
    4. Calculate weight = sum of all edge weights in path
    5. Each path must be valid (consecutive edges must exist)

    Output format:
    {format_spec}

    CRITICAL: 
    - Use actual node IDs and weights from YOUR problem, not placeholders
    - Ensure all JSON brackets are properly closed

    Respond with ONLY valid JSON and an explanation in a single para."""
        
        return prompt
    elif handler_type == "mmlu_med":
        return (
            "You are answering a medical multiple choice question.\n"
            f"Question: {question}\n"
            "Explain your reasoning and end with 'Answer: X' where X is the letter choice."
        )
    else:
        return f"Question: {question}\nProvide a detailed answer."


def feedback_prompt(question: str, attempt: str, handler_type: str) -> str:
    if handler_type == "graph":
        prompt = f"""Review the following shortest-path solution for correctness.

            Problem: {question}

            Attempted Solution: {attempt}

            Check these specific issues:
            1. Is the JSON valid? Check that all brackets are properly closed: {{ }}, [ ]
            2. Does the path exist in the graph (all edges valid)?
            3. Is the weight correctly calculated (sum of edge weights)?
            4. Is this the shortest possible path?

            If there are ANY errors, output ONLY the corrected JSON with an explanation in a single para.
            If correct, output the same JSON.

            CRITICAL: Ensure your JSON has all closing brackets. For example:
            {{"paths":[{{"path":[0,3,5],"weight":472}}]}}
                                                      ^^^ These closing brackets are required!"""
    
        return prompt
    
    elif handler_type == "mmlu_med":
        return f"""Review this medical answer for accuracy and reasoning.

Question: {question}

Answer Attempt: {attempt}

Provide constructive feedback on any errors or gaps in reasoning:"""
    
    else:
        return f"""Review this answer and identify any issues.

Question: {question}

Answer: {attempt}

Feedback:"""

def refine_prompt(question: str, attempt: str, feedback: str, handler_type: str) -> str:
    if handler_type == "graph":
        format_hint = '{"paths": [[node1, node2, ...]], "weights": [total_weight1, ...]}'
        return (
            "Update your previous shortest-path solution using the feedback below.\n"
            f"Output only valid JSON in this format: {format_hint}.\n"
            "Do not include explanations or code fences.\n"
            f"Problem: {question}\n"
            f"Previous Solution: {attempt}\n"
            f"Feedback: {feedback}\n"
            "Output only the revised JSON object."
        )
    
    elif handler_type == "mmlu_med":
        return f"""Revise your answer based on the feedback.

Question: {question}

Previous Answer: {attempt}

Feedback: {feedback}

Provide your revised answer in the format (PLEASE STRICTLY FOLLOW THIS FORMAT): Answer: [letter]
Revised Response:"""
    
    else:
        return f"""Improve your answer based on this feedback.

Question: {question}

Previous Answer: {attempt}

Feedback: {feedback}

Improved Answer:"""


class Generator:
    "LLM Engine for generation, feedback, and refinement"
    # You can use transformers, hf piepeline, vllm, etc.
    def __init__(self, cfg: RefineConfig):
        self.cfg = cfg
        logger.info(f"Loading model from {cfg.model_path}")
        tokenizer_kwargs = {"trust_remote_code": cfg.trust_remote_code}
        if cfg.trust_remote_code:
            tokenizer_kwargs.setdefault("use_fast", False)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, **tokenizer_kwargs)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(cfg.dtype, torch.bfloat16)
        
        model_kwargs = {"torch_dtype": dtype}
        if cfg.device is None:
            model_kwargs["device_map"] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            trust_remote_code=cfg.trust_remote_code,
            **model_kwargs,
        )

        if cfg.device is not None:
            self.model.to(cfg.device)
        
        self.model.eval()
        logger.info("Model loaded successfully")

    def _apply_chat_template(self, prompt: str) -> str:
        """Apply chat template if available"""
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [chat("user", prompt)]
            return self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        return prompt

    def _gen(self, prompts: List[str], temperature: Optional[float] = None) -> List[str]:
        outputs = []
        stage_temp = self.cfg.temperature if temperature is None else temperature
        do_sample = stage_temp is not None and stage_temp > 0
        
        for i in range(0, len(prompts), self.cfg.batch_size):
            batch = prompts[i:i + self.cfg.batch_size]
            
            formatted = [self._apply_chat_template(p) for p in batch]
            
            inputs = self.tokenizer(
                formatted,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            if self.model.device.type != "cpu":
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generate_kwargs = {
                    **inputs,
                    "max_new_tokens": self.cfg.max_new_tokens,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "do_sample": do_sample,
                }
                if do_sample:
                    generate_kwargs["temperature"] = stage_temp
                generated = self.model.generate(**generate_kwargs)
            
            for idx in range(len(batch)):
                prompt_len = inputs['attention_mask'][idx].sum().item()
                gen_tokens = generated[idx][prompt_len:]
                text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                
                if self.cfg.stop_sequences:
                    for stop in self.cfg.stop_sequences:
                        if stop in text:
                            text = text[:text.index(stop)]
                
                outputs.append(text.strip())
        
        return outputs

    def draft(self, qs: List[str], handler_type) -> List[str]:
        """Generate initial drafts for questions"""
        prompts = [draft_prompt(q, handler_type) for q in qs]
        return self._gen(prompts, self.cfg.draft_temperature)

    def feedback(self, qs_attempts: List[Tuple[str, str]], handler_type) -> List[str]:
        """Generate feedback for question-attempt pairs"""
        prompts = [feedback_prompt(q, attempt, handler_type) for q, attempt in qs_attempts]
        return self._gen(prompts, self.cfg.feedback_temperature)

    def refine(self, qs_attempts_feedback: List[Tuple[str, str, str]], handler_type) -> List[str]:
        """Generate refinements for the attempts based on feedback"""
        prompts = [
            refine_prompt(q, attempt, feedback, handler_type)
            for q, attempt, feedback in qs_attempts_feedback
        ]
        return self._gen(prompts, self.cfg.refine_temperature)


def run_self_refine(
    examples: List[Dict[str, Any]],
    handler: dataset.DatasetHandler,
    generator: Generator,
    config: RefineConfig,
) -> List[Dict[str, Any]]:
    """
    Implement the self-refinement algorithm.
    
    Args:
        examples: List of dataset examples
        handler: Dataset handler
        generator: Generator instance for model inference
        config: Your configuration
    
    Returns:
        - You might want to keep track of outputs at different stages so you can do interesting analysis later
    """
    handler_type = getattr(handler, 'name', handler.__class__.__name__.lower().replace('handler', ''))
    questions = [handler.format_question(ex) for ex in examples]
    
    logger.info(f"Starting self-refine with {config.num_refine_steps} refinement steps")
    
    logger.info("Generating drafts...")
    start_time = time.time()
    drafts = generator.draft(questions, handler_type)
    draft_time = time.time() - start_time
    logger.info(f"Drafts generated in {draft_time:.2f}s")
    
    all_attempts = [drafts]
    all_feedback = []
    current_attempts = drafts
    
    for step in range(config.num_refine_steps):
        logger.info(f"Step {step + 2}: Refinement iteration {step + 1}/{config.num_refine_steps}")
        
        start_time = time.time()
        feedback_list = generator.feedback(
            list(zip(questions, current_attempts)), 
            handler_type
        )
        feedback_time = time.time() - start_time
        logger.info(f"  Feedback generated in {feedback_time:.2f}s")
        all_feedback.append(feedback_list)
        
        start_time = time.time()
        refined = generator.refine(
            list(zip(questions, current_attempts, feedback_list)),
            handler_type
        )
        refine_time = time.time() - start_time
        logger.info(f"  Refinements generated in {refine_time:.2f}s")
        
        all_attempts.append(refined)
        current_attempts = refined
    
    results = []
    for idx, example in enumerate(examples):
        ground_truth = handler.get_ground_truth(example)
        
        draft_text = drafts[idx]
        try:
            draft_parsed = handler.parse_answer(draft_text)
            draft_correct = handler.verify_answer(draft_parsed, ground_truth)
        except Exception as e:
            draft_parsed = None
            draft_correct = False
            logger.warning(f"Draft parsing failed for example {idx}: {e}")
        
        refinements = []
        for step_idx in range(config.num_refine_steps):
            attempt = all_attempts[step_idx + 1][idx]
            feedback = all_feedback[step_idx][idx]
            
            try:
                parsed = handler.parse_answer(attempt)
                correct = handler.verify_answer(parsed, ground_truth)
            except Exception as e:
                parsed = None
                correct = False
                logger.warning(f"Refinement {step_idx + 1} parsing failed for example {idx}: {e}")
            
            refinements.append({
                "step": step_idx + 1,
                "feedback": feedback,
                "answer": attempt,
                "parsed": parsed,
                "correct": correct,
            })
        
        result = {
            "id": example.get("id", idx),
            "question": questions[idx],
            "ground_truth": ground_truth,
            "draft": {
                "text": draft_text,
                "parsed": draft_parsed,
                "correct": draft_correct,
            },
            "refinements": refinements,
            "final_answer": {
                "text": refinements[-1]["answer"] if refinements else draft_text,
                "parsed": refinements[-1]["parsed"] if refinements else draft_parsed,
                "correct": refinements[-1]["correct"] if refinements else draft_correct,
            }
        }
        results.append(result)
    
    return results

def _load_examples(handler_type: str = None) -> List[Dict[str, Any]]:
    logger.info(f"Loading from HuggingFace dataset")
    
    if handler_type == "graph":
        config_name = "graph_dev"
    elif handler_type == "mmlu_med":
        config_name = "mmlu_med"
    else:
        raise ValueError(f"Unknown handler type for HF loading: {handler_type}")
    
    dataset = load_dataset(
        "vashistht/11763_datasets",
        config_name,
        split="dev"
    )
    
    examples = [dict(example) for example in dataset][:150]
    logger.info(f"Loaded {len(examples)} examples from HuggingFace")
    return examples

def _save_results(path: Path, results: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    logger.info(f"Results saved to {path}")


def _analyze_results(results: List[Dict[str, Any]]) -> None:
    total = len(results)
    
    draft_correct = sum(1 for r in results if r['draft']['correct'])
    draft_acc = draft_correct / total if total > 0 else 0
    
    final_correct = sum(1 for r in results if r['final_answer']['correct'])
    final_acc = final_correct / total if total > 0 else 0
    
    num_steps = len(results[0]['refinements']) if results and results[0]['refinements'] else 0
    step_accuracies = []
    for step in range(num_steps):
        step_correct = sum(1 for r in results if r['refinements'][step]['correct'])
        step_acc = step_correct / total if total > 0 else 0
        step_accuracies.append(step_acc)
    
    logger.info("\n")
    logger.info("RESULTS SUMMARY")
    logger.info("\n")
    logger.info(f"Total examples: {total}")
    logger.info(f"Draft accuracy: {draft_acc:.3f} ({draft_correct}/{total})")
    for i, acc in enumerate(step_accuracies):
        correct = int(acc * total)
        logger.info(f"Step {i+1} accuracy: {acc:.3f} ({correct}/{total})")
    logger.info(f"Final accuracy: {final_acc:.3f} ({final_correct}/{total})")
    logger.info("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--handler", choices=["graph", "mmlu_med"], required=True)
    parser.add_argument("--output", type=Path, default=Path("self_refine_output.jsonl"))
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-refine-steps", type=int, default=3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--stop-sequences", type=str, nargs="*", default=())
    parser.add_argument("--draft-temperature", type=float, default=0.7)
    parser.add_argument("--feedback-temperature", type=float, default=1.0)
    parser.add_argument("--refine-temperature", type=float, default=1.5)
    parser.add_argument("--trust-remote-code", action="store_true")

    args = parser.parse_args()
    HANDLERS = {
    "graph": GraphHandler,
    "mmlu_med": MMLUMedHandler,
    }

    handler = HANDLERS[args.handler]()
    # TODO: Initialize your generator and config
    config = RefineConfig(
        model_path=args.model_path,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
        num_refine_steps=args.num_refine_steps,
        stop_sequences=tuple(args.stop_sequences),
        device=args.device,
        draft_temperature=args.draft_temperature,
        feedback_temperature=args.feedback_temperature,
        refine_temperature=args.refine_temperature,
        trust_remote_code=args.trust_remote_code,
    )
    generator = Generator(config)
    # TODO: Load dataset
    examples = _load_examples(handler_type=args.handler)
    logger.info("Loaded %d examples", len(examples))
    # TODO: Run self-refine
    results = run_self_refine(examples, handler, generator, config)

    # TODO: Analyze results
    _analyze_results(results)
    # TODO: Save outputs
    _save_results(args.output, results)
    # TODO: Analyse the results

if __name__ == "__main__":
    main()
