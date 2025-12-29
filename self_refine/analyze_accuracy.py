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


def compute_accuracy_metrics(results: List[Dict], num_iterations: int = 4) -> Tuple[List[float], List[float]]:
    accuracy_per_iteration = []
    best_accuracy_so_far = []
    
    for i in range(num_iterations):
        correct_at_i = 0
        correct_at_any_up_to_i = 0
        total = 0
        
        for result in results:
            sequence = extract_correctness_sequence(result)
            
            if len(sequence) <= i:
                continue
            
            total += 1
            
            if sequence[i]:
                correct_at_i += 1
            
            if any(sequence[:i+1]):
                correct_at_any_up_to_i += 1
        
        if total > 0:
            accuracy_per_iteration.append(correct_at_i / total)
            best_accuracy_so_far.append(correct_at_any_up_to_i / total)
        else:
            accuracy_per_iteration.append(np.nan)
            best_accuracy_so_far.append(np.nan)
    
    return accuracy_per_iteration, best_accuracy_so_far


def plot_accuracy_metrics(
    accuracy_per_iteration: List[float],
    best_accuracy_so_far: List[float],
    output_path: str = 'accuracy_vs_iteration.png'
):
    iterations = list(range(len(accuracy_per_iteration)))
    iteration_labels = ['Draft'] + [f'Refine {i}' for i in range(1, len(iterations))]
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(iterations, accuracy_per_iteration, 
             marker='o', linewidth=2.5, markersize=10,
             label='(i) Accuracy at iteration i', color='blue')
    
    plt.plot(iterations, best_accuracy_so_far, 
             marker='s', linewidth=2.5, markersize=10,
             label='(ii) Best accuracy so far', color='green', linestyle='--')
    
    plt.xlabel('Iteration', fontsize=13)
    plt.ylabel('Accuracy', fontsize=13)
    plt.title('Accuracy vs Iteration in Self-Refine', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    
    plt.xticks(iterations, iteration_labels, rotation=0)
    
    for i, (acc, best) in enumerate(zip(accuracy_per_iteration, best_accuracy_so_far)):
        plt.text(i, acc + 0.03, f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
        plt.text(i, best - 0.03, f'{best:.3f}', ha='center', va='top', fontsize=9, color='green')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
    plt.close()


def print_accuracy_analysis(
    accuracy_per_iteration: List[float],
    best_accuracy_so_far: List[float]
):
    print("\n" + "-"*80)
    print("SUMMARY:")
    print("-"*80)
    
    initial_acc = accuracy_per_iteration[0]
    final_acc = accuracy_per_iteration[-1]
    final_best = best_accuracy_so_far[-1]
    
    improvement = final_acc - initial_acc
    oracle_improvement = final_best - initial_acc
    gap = final_best - final_acc
    
    print(f"\nInitial accuracy (draft):        {initial_acc:.4f} ({initial_acc*100:.2f}%)")
    print(f"Final accuracy (last iteration): {final_acc:.4f} ({final_acc*100:.2f}%)")
    print(f"Best accuracy achieved:          {final_best:.4f} ({final_best*100:.2f}%)")
    print(f"\nImprovement (final - draft):     {improvement:+.4f} ({improvement*100:+.2f}pp)")
    print(f"Oracle improvement (best - draft): {oracle_improvement:+.4f} ({oracle_improvement*100:+.2f}pp)")
    print(f"Gap (best - final):              {gap:.4f} ({gap*100:.2f}pp)")


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
        default='accuracy_vs_iteration.png',
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
    
    print("Computing accuracy metrics...")
    accuracy_per_iteration, best_accuracy_so_far = compute_accuracy_metrics(
        results, 
        num_iterations=args.num_iterations
    )
    
    print_accuracy_analysis(accuracy_per_iteration, best_accuracy_so_far)
    
    plot_accuracy_metrics(
        accuracy_per_iteration,
        best_accuracy_so_far,
        output_path=args.output
    )


if __name__ == "__main__":
    main()