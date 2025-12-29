import json
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from pathlib import Path

def load_results(filepath="all_results_with_scores.json"):
    with open(filepath) as f:
        return json.load(f)

def compute_top1_score(method_scores, gold_scores):
    np.random.seed(42)
    correct = 0
    
    for m_scores, g_scores in zip(method_scores, gold_scores):
        max_gold = max(g_scores)
        gold_best_indices = [i for i, s in enumerate(g_scores) if s == max_gold]
        oracle_pick = np.random.choice(gold_best_indices)
        
        max_method = max(m_scores)
        method_best_indices = [i for i, s in enumerate(m_scores) if s == max_method]
        method_pick = np.random.choice(method_best_indices)
        
        if oracle_pick == method_pick:
            correct += 1
    
    return correct / len(method_scores)

def compute_avg_rank_of_best(method_scores, gold_scores):
    ranks = []
    
    for m_scores, g_scores in zip(method_scores, gold_scores):
        gold_best_idx = np.argmax(g_scores)
        
        sorted_indices = np.argsort(m_scores)[::-1]
        
        rank = np.where(sorted_indices == gold_best_idx)[0][0] + 1
        ranks.append(rank)
    
    return np.mean(ranks)

def compute_spearman_correlation(method_scores, gold_scores):
    all_method_scores = []
    all_gold_scores = []
    
    for m_scores, g_scores in zip(method_scores, gold_scores):
        all_method_scores.extend(m_scores)
        all_gold_scores.extend(g_scores)
    
    corr, _ = spearmanr(all_method_scores, all_gold_scores)
    return corr

def extract_scores_by_method(results):
    methods = {
        'Oracle': [],
        'Qwen3-4B log-probs': [],
        'Qwen3-14B log-probs': [],
        'Scalar reward': [],
        'Pairwise reward': [],
        'MBR with BLEU': [],
        'MBR with BERTScore': []
    }
    
    for result in results['results']:
        candidates = result['candidates']
        
        gold_scores = [c['scores']['infobench'] for c in candidates]
        methods['Oracle'].append(gold_scores)
        
        methods['Qwen3-4B log-probs'].append([c['scores']['qwen3_4b'] for c in candidates])
        methods['Qwen3-14B log-probs'].append([c['scores']['qwen3_14b'] for c in candidates])
        methods['Scalar reward'].append([c['scores']['r_scalar'] for c in candidates])
        methods['Pairwise reward'].append([c['scores']['r_pairwise'] for c in candidates])
        methods['MBR with BLEU'].append([c['scores']['mbr_bleu'] for c in candidates])
        methods['MBR with BERTScore'].append([c['scores']['mbr_bert'] for c in candidates])
    
    return methods

def compute_all_metrics(results):
    methods_scores = extract_scores_by_method(results)
    gold_scores = methods_scores['Oracle']
    
    results_table = []
    
    for method_name, method_scores in methods_scores.items():
        if method_name == 'Oracle':
            results_table.append({
                'Method': method_name,
                'Top-1 score': 1.0,
                'Avg. rank of best output': 1.0,
                'Spearman rank correlation': 1.0
            })
        else:
            top1 = compute_top1_score(method_scores, gold_scores)
            avg_rank = compute_avg_rank_of_best(method_scores, gold_scores)
            spearman = compute_spearman_correlation(method_scores, gold_scores)
            
            results_table.append({
                'Method': method_name,
                'Top-1 score': top1,
                'Avg. rank of best output': avg_rank,
                'Spearman rank correlation': spearman
            })
    
    return results_table

def create_scatter_plots(results, output_dir="plots"):
    Path(output_dir).mkdir(exist_ok=True)
    
    methods_scores = extract_scores_by_method(results)
    gold_scores = methods_scores['Oracle']
    
    all_gold = []
    for scores in gold_scores:
        all_gold.extend(scores)
    
    for method_name, method_scores in methods_scores.items():
        if method_name == 'Oracle':
            continue
        
        all_method = []
        for scores in method_scores:
            all_method.extend(scores)
        
        # Create scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(all_method, all_gold, alpha=0.5, s=20)
        plt.xlabel(f'{method_name} Score')
        plt.ylabel('Gold InfoBench Score')
        plt.title(f'{method_name} vs Gold Scores')
        plt.grid(True, alpha=0.3)
        
        corr, _ = spearmanr(all_method, all_gold)
        plt.text(0.05, 0.95, f'Spearman ρ = {corr:.4f}', 
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        filename = f"{output_dir}/{method_name.replace(' ', '_').replace('/', '_')}_scatter.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {filename}")

def compute_length_analysis(results):
    methods_scores = extract_scores_by_method(results)
    
    length_results = {}
    
    for method_name, method_scores in methods_scores.items():
        lengths = []
        
        for i, scores in enumerate(method_scores):
            top1_idx = np.argmax(scores)
            
            candidates = results['results'][i]['candidates']
            output_length = len(candidates[top1_idx]['generated_text'].split())
            lengths.append(output_length)
        
        length_results[method_name] = np.mean(lengths)
    
    return length_results

def main():
    # Load results with scores
    print("Loading results...")
    results = load_results("all_results_with_scores.json")
    
    results_table = compute_all_metrics(results)
    
    create_scatter_plots(results)
    
    length_results = compute_length_analysis(results)
    
    output = {
        'metrics': results_table,
        'length_analysis': length_results
    }
    
    with open('results.json', 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()