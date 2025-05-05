import os
import networkx as nx
import time
import matplotlib.pyplot as plt
import numpy as np
from graph_utils import GraphUtils
from s2l_algorithm import S2L
from saa_gs_algorithm import SAA_Gs
from kgs_algorithm import KGs
from evaluation import Evaluation

def test_algorithms_on_dataset(dataset_name, graph_path, target_k_ratios=[0.1, 0.2, 0.3, 0.4, 0.5]):
    """
    Test various graph summarization algorithms on a dataset.
    
    Args:
        dataset_name: Name of the dataset
        graph_path: Path to the graph file
        target_k_ratios: List of target supernode ratios
    """
    print(f"\n{'='*50}")
    print(f"Testing algorithms on {dataset_name} dataset")
    print(f"{'='*50}")
    
    # Load the graph
    G = GraphUtils.load_graph(graph_path)
    
    # Compute graph metrics
    metrics = GraphUtils.compute_graph_metrics(G)
    print("\nGraph metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Initialize algorithms
    s2l = S2L()
    saa_gs = SAA_Gs()
    kgs = KGs()
    
    # Calculate target k values
    n = G.number_of_nodes()
    target_k_values = [int(ratio * n) for ratio in target_k_ratios]
    
    # Initialize results
    results = {
        'algorithm': [],
        'k_ratio': [],
        'k': [],
        'reconstruction_error_l1': [],
        'reconstruction_error_l2': [],
        'size_in_bits': [],
        'runtime': [],
        'num_supernodes': [],
        'num_superedges': []
    }
    
    # Output directory for visualizations
    output_dir = f"./results/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test algorithms
    algorithms = [
        ("S2L (k-means)", lambda g, k: s2l.summarize(g, k, method='kmeans')),
        ("S2L (k-median)", lambda g, k: s2l.summarize(g, k, method='kmedian')),
        ("SAA-Gs (log n)", lambda g, k: saa_gs.summarize(g, k, log_n_sampling=True)),
        ("SAA-Gs (linear)", lambda g, k: saa_gs.summarize(g, k, log_n_sampling=False)),
        ("k-Gs (SamplePairs)", lambda g, k: kgs.sample_pairs(g, k, c=1.0))
    ]
    
    for algorithm_name, algorithm_func in algorithms:
        print(f"\nTesting {algorithm_name}...")
        
        for k_ratio, k in zip(target_k_ratios, target_k_values):
            print(f"\n  Running with k={k} ({k_ratio:.1%} of nodes)...")
            
            try:
                # Run the algorithm and measure time
                start_time = time.time()
                summary = algorithm_func(G, k)
                end_time = time.time()
                runtime = end_time - start_time
                
                # Evaluate the summary
                eval_metrics = Evaluation.evaluate_summary(summary)
                
                # Store results
                results['algorithm'].append(algorithm_name)
                results['k_ratio'].append(k_ratio)
                results['k'].append(k)
                results['reconstruction_error_l1'].append(eval_metrics['reconstruction_error'])
                
                # Calculate L2 error
                A_original = nx.to_numpy_array(G)
                A_expected = summary.get_expected_adjacency_matrix()
                l2_error = np.sqrt(np.sum((A_original - A_expected) ** 2)) / (G.number_of_nodes() * (G.number_of_nodes() - 1))
                results['reconstruction_error_l2'].append(l2_error)
                
                results['size_in_bits'].append(eval_metrics['size_in_bits'])
                results['runtime'].append(runtime)
                results['num_supernodes'].append(len(summary.supernodes))
                results['num_superedges'].append(len(summary.superedge_counts))
                
                print(f"    Completed: Error={eval_metrics['reconstruction_error']:.6f}, "
                      f"Size={eval_metrics['size_in_bits']} bits, "
                      f"Runtime={runtime:.2f}s")
                
                # Visualize the graph and its summary
                if k_ratio == 0.2:  # Only visualize for one k value to avoid too many plots
                    # Original graph visualization (small sample)
                    vis_G = G
                    if G.number_of_nodes() > 100:
                        # Sample a subset of nodes
                        sample_nodes = list(G.nodes())[:100]
                        vis_G = G.subgraph(sample_nodes).copy()
                    
                    GraphUtils.visualize_graph(
                        vis_G, 
                        title=f"Original {dataset_name} Graph", 
                        output_path=os.path.join(output_dir, f"{dataset_name}_original.png")
                    )
                    
                    # Summary graph visualization
                    summary_graph = summary.to_networkx_graph()
                    GraphUtils.visualize_graph(
                        summary_graph, 
                        title=f"{algorithm_name} Summary (k={k})", 
                        output_path=os.path.join(output_dir, f"{dataset_name}_{algorithm_name.replace(' ', '_')}_k{k}.png")
                    )
            
            except Exception as e:
                print(f"    Error: {e}")
                
                # Store failure in results
                results['algorithm'].append(algorithm_name)
                results['k_ratio'].append(k_ratio)
                results['k'].append(k)
                results['reconstruction_error_l1'].append(np.nan)
                results['reconstruction_error_l2'].append(np.nan)
                results['size_in_bits'].append(np.nan)
                results['runtime'].append(np.nan)
                results['num_supernodes'].append(np.nan)
                results['num_superedges'].append(np.nan)
    
    # Create comparison plots
    create_comparison_plots(results, dataset_name, output_dir)
    
    return results

def create_comparison_plots(results, dataset_name, output_dir):
    """
    Create comparison plots for the algorithms.
    
    Args:
        results: Dictionary of results
        dataset_name: Name of the dataset
        output_dir: Output directory for plots
    """
    # Convert results to a pandas DataFrame
    import pandas as pd
    df = pd.DataFrame(results)
    
    # Define algorithm colors
    algorithm_colors = {
        "S2L (k-means)": "blue",
        "S2L (k-median)": "purple",
        "SAA-Gs (log n)": "green",
        "SAA-Gs (linear)": "lime",
        "k-Gs (SamplePairs)": "red"
    }
    
    # 1. Error vs K Ratio
    plt.figure(figsize=(10, 6))
    for algorithm in df['algorithm'].unique():
        alg_data = df[df['algorithm'] == algorithm]
        plt.plot(alg_data['k_ratio'], alg_data['reconstruction_error_l1'], 
                 marker='o', label=algorithm, color=algorithm_colors.get(algorithm, None))
    
    plt.xlabel('Ratio of Supernodes (k/n)')
    plt.ylabel('L1 Reconstruction Error')
    plt.title(f'Reconstruction Error vs. Supernode Ratio - {dataset_name}')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_error_vs_k.png"), dpi=300, bbox_inches='tight')
    
    # 2. Runtime vs K Ratio
    plt.figure(figsize=(10, 6))
    for algorithm in df['algorithm'].unique():
        alg_data = df[df['algorithm'] == algorithm]
        plt.plot(alg_data['k_ratio'], alg_data['runtime'], 
                 marker='o', label=algorithm, color=algorithm_colors.get(algorithm, None))
    
    plt.xlabel('Ratio of Supernodes (k/n)')
    plt.ylabel('Runtime (seconds)')
    plt.title(f'Runtime vs. Supernode Ratio - {dataset_name}')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_runtime_vs_k.png"), dpi=300, bbox_inches='tight')
    
    # 3. Size vs K Ratio
    plt.figure(figsize=(10, 6))
    for algorithm in df['algorithm'].unique():
        alg_data = df[df['algorithm'] == algorithm]
        plt.plot(alg_data['k_ratio'], alg_data['size_in_bits'], 
                 marker='o', label=algorithm, color=algorithm_colors.get(algorithm, None))
    
    plt.xlabel('Ratio of Supernodes (k/n)')
    plt.ylabel('Size (bits)')
    plt.title(f'Summary Size vs. Supernode Ratio - {dataset_name}')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_size_vs_k.png"), dpi=300, bbox_inches='tight')
    
    # 4. Error vs Size
    plt.figure(figsize=(10, 6))
    for algorithm in df['algorithm'].unique():
        alg_data = df[df['algorithm'] == algorithm]
        plt.scatter(alg_data['size_in_bits'], alg_data['reconstruction_error_l1'], 
                   label=algorithm, color=algorithm_colors.get(algorithm, None), s=100, alpha=0.7)
        
        # Add k labels to points
        for i, k in enumerate(alg_data['k']):
            plt.annotate(f"k={k}", 
                        (alg_data['size_in_bits'].iloc[i], alg_data['reconstruction_error_l1'].iloc[i]),
                        textcoords="offset points", xytext=(0, 10), ha='center')
    
    plt.xlabel('Size (bits)')
    plt.ylabel('L1 Reconstruction Error')
    plt.title(f'Error vs. Size Trade-off - {dataset_name}')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_error_vs_size.png"), dpi=300, bbox_inches='tight')
    
    # 5. Create a CSV with the results
    df.to_csv(os.path.join(output_dir, f"{dataset_name}_results.csv"), index=False)
    
    print(f"Comparison plots saved to {output_dir}")

def main():
    """Run tests on sample datasets."""
    # Datasets to test
    datasets = [
        ("Amazon-0302", "./data/Amazon-0302.txt"),
        ("Ego-Facebook", "./data/DBLP.txt"),
        # Add more datasets as needed
    ]
    
    # Run tests
    all_results = {}
    for dataset_name, graph_path in datasets:
        #try:
        results = test_algorithms_on_dataset(dataset_name, graph_path)
        all_results[dataset_name] = results
        # except Exception as e:
        #     print(f"Error testing on {dataset_name}: {e}")
    
    print("\nTesting complete!")

if __name__ == "__main__":
    main()