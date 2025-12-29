import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple


def load_results(results_path: str) -> List[Dict]:
    results = []
    
    with open(results_path, 'r') as f:
        first_line = f.readline()
        f.seek(0)
        
        try:
            json.loads(first_line)
            is_jsonl = True
        except:
            is_jsonl = False
        
        if is_jsonl:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        else:
            data = json.load(f)
            results = data if isinstance(data, list) else data.get('results', [])
    
    return results


def extract_correctness_sequence(result: Dict) -> List[bool]:
    sequence = [result['draft']['correct']]
    
    for refinement in result.get('refinements', []):
        sequence.append(refinement['correct'])
    
    return sequence


def compute_conditional_probabilities(results: List[Dict], num_iterations: int = 4) -> Tuple[List[float], List[float]]:
    p_correct_given_correct = []
    p_correct_given_incorrect = []
    
    for i in range(num_iterations - 1):
        count_correct_to_correct = 0      
        count_correct_to_incorrect = 0
        count_incorrect_to_correct = 0
        count_incorrect_to_incorrect = 0
        
        for result in results:
            sequence = extract_correctness_sequence(result)
            
            if len(sequence) <= i + 1:
                continue
            
            current_correct = sequence[i]
            next_correct = sequence[i + 1]
            
            if current_correct:
                if next_correct:
                    count_correct_to_correct += 1
                else:
                    count_correct_to_incorrect += 1
            else:
                if next_correct:
                    count_incorrect_to_correct += 1
                else:
                    count_incorrect_to_incorrect += 1
        
        total_correct = count_correct_to_correct + count_correct_to_incorrect
        if total_correct > 0:
            p_correct_given_correct.append(count_correct_to_correct / total_correct)
        else:
            p_correct_given_correct.append(np.nan)
        
        total_incorrect = count_incorrect_to_correct + count_incorrect_to_incorrect
        if total_incorrect > 0:
            p_correct_given_incorrect.append(count_incorrect_to_correct / total_incorrect)
        else:
            p_correct_given_incorrect.append(np.nan)
    
    return p_correct_given_correct, p_correct_given_incorrect


def plot_conditional_probabilities(p_correct_given_correct: List[float], p_correct_given_incorrect: List[float], output_path: str = 'conditional_probabilities.png'):
    iterations = list(range(len(p_correct_given_correct)))
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(iterations, p_correct_given_correct, 
             marker='o', linewidth=2.5, markersize=10,
             label='P(correct$_{i+1}$ | correct$_i$)', color='green')
    
    plt.plot(iterations, p_correct_given_incorrect, 
             marker='s', linewidth=2.5, markersize=10,
             label='P(correct$_{i+1}$ | incorrect$_i$)', color='red')
    
    plt.xlabel('Transition (i → i+1)', fontsize=13)
    plt.ylabel('Probability', fontsize=13)
    plt.title('Conditional Probabilities Across Self-Refine Iterations', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    
    transition_labels = [f'{i}→{i+1}' for i in iterations]
    plt.xticks(iterations, transition_labels)
    
    for i, (p_cc, p_ic) in enumerate(zip(p_correct_given_correct, p_correct_given_incorrect)):
        if not np.isnan(p_cc):
            plt.text(i, p_cc + 0.03, f'{p_cc:.3f}', ha='center', va='bottom', fontsize=9, color='green')
        if not np.isnan(p_ic):
            plt.text(i, p_ic - 0.03, f'{p_ic:.3f}', ha='center', va='top', fontsize=9, color='red')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results', 
        type=str, 
        required=True,
        help='Path to self-refine results JSON file'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='conditional_probabilities.png',
        help='Output path for the plot'
    )
    parser.add_argument(
        '--num-iterations',
        type=int,
        default=4,
        help='Total number of iterations (including draft)'
    )
    
    args = parser.parse_args()
    
    print(f"Loading results from {args.results}...")
    results = load_results(args.results)
    print(f"Loaded {len(results)} examples\n")
    
    print("Computing conditional probabilities...")
    p_correct_given_correct, p_correct_given_incorrect = compute_conditional_probabilities(
        results, 
        num_iterations=args.num_iterations
    )
    
    print("Creating plot...")
    plot_conditional_probabilities(p_correct_given_correct, p_correct_given_incorrect, output_path=args.output)


if __name__ == "__main__":
    main()