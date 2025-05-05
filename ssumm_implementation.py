#!/usr/bin/env python3
"""
Demonstration of the SSumM (Sparse Summarization of Massive Graphs) algorithm implementation.
This script showcases the main features and compares performance with baseline methods.
"""

import os
import networkx as nx
import numpy as np
import argparse
import time
from typing import Dict, List, Tuple

# Import our implementations
from ssumm_algorithm import SSumM
from graph_summary import GraphSummary
from ssumm_cost_function import SsummCostFunction
from ssumm_evaluation import SummEvaluation
from graph_utils import GraphUtils
from kgs_algorithm import KGs
from s2l_algorithm import S2L
from saa_gs_algorithm import SAA_Gs

def setup_logging():
    """Set up logging configuration."""
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ssumm_demo.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('SSumMDemo')

def load_datasets() -> Dict[str, nx.Graph]:
    """Load and return sample datasets."""
    datasets = {}
    
    # Load processed datasets (you'll need these from the previous preprocessing step)
    dataset_names = ["DBLP", "Amazon-0302", "Email-Enron"]
    processed_dir = "./processed_data"
    
    for name in dataset_names:
        graph_path = os.path.join(processed_dir, name, f"{name}.gpickle")
        if os.path.exists(graph_path):
            try:
                with open(graph_path, 'rb') as f:
                    import pickle
                    G = pickle.load(f)
                datasets[name] = G
                print(f"Loaded {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            except Exception as e:
                print(f"Error loading {name}: {e}")
        else:
            print(f"Dataset {name} not found at {graph_path}")
    
    # Create a small test graph if no datasets loaded
    if not datasets:
        print("Creating small test graph...")
        datasets["test_graph"] = nx.karate_club_graph()
        print(f"Created test graph: {datasets['test_graph'].number_of_nodes()} nodes, {datasets['test_graph'].number_of_edges()} edges")
    
    return datasets

def create_algorithm_wrappers() -> Dict[str, callable]:
    """Create wrapper functions for different algorithms."""
    algorithms = {}
    
    # SSumM wrapper
    def ssumm_wrapper(G, target_size):
        ssumm = SSumM(max_iterations=20)
        return ssumm.summarize(G, target_size)
    
    algorithms["SSumM"] = ssumm_wrapper
    
    # k-Gs wrapper
    def kgs_wrapper(G, target_size):
        kgs = KGs()
        # Convert size to approximate number of supernodes
        n = G.number_of_nodes()
        target_k = int(n * 0.5)  # Use 50% of nodes as target
        return kgs.sample_pairs(G, target_k, c=1.0)
    
    algorithms["k-Gs"] = kgs_wrapper
    
    # S2L wrapper
    def s2l_wrapper(G, target_size):
        s2l = S2L()
        # Convert size to approximate number of supernodes
        n = G.number_of_nodes()
        target_k = int(n * 0.5)  # Use 50% of nodes as target
        return s2l.summarize(G, target_k, method='kmeans')
    
    algorithms["S2L"] = s2l_wrapper
    
    # SAA-Gs wrapper
    def saa_gs_wrapper(G, target_size):
        saa_gs = SAA_Gs()
        # Convert size to approximate number of supernodes
        n = G.number_of_nodes()
        target_k = int(n * 0.5)  # Use 50% of nodes as target
        return saa_gs.summarize(G, target_k, log_n_sampling=True)
    
    algorithms["SAA-Gs"] = saa_gs_wrapper
    
    return algorithms

def demonstrate_basic_functionality(G: nx.Graph, logger):
    """Demonstrate basic SSumM functionality."""
    logger.info("=== Demonstrating Basic SSumM Functionality ===")
    
    # Calculate original graph properties
    n = G.number_of_nodes()
    m = G.number_of_edges()
    original_size = 2 * m * np.ceil(np.log2(n))
    logger.info(f"Original graph: {n} nodes, {m} edges, {original_size} bits")
    
    # Create SSumM instance
    ssumm = SSumM(max_iterations=10)
    
    # Target 50% size reduction
    target_size = int(original_size * 0.5)
    logger.info(f"Target size: {target_size} bits")
    
    # Summarize graph
    start_time = time.time()
    summary = ssumm.summarize(G, target_size)
    end_time = time.time()
    
    # Display results
    summary_stats = summary.summary_stats()
    logger.info("\nSummary Results:")
    logger.info(f"  Number of supernodes: {summary_stats['num_supernodes']}")
    logger.info(f"  Number of superedges: {summary_stats['num_superedges']}")
    logger.info(f"  L1 reconstruction error: {summary_stats['reconstruction_error_l1']:.6f}")
    logger.info(f"  L2 reconstruction error: {summary_stats['reconstruction_error_l2']:.6f}")
    logger.info(f"  Summary size: {summary.size_in_bits()} bits")
    logger.info(f"  Size reduction: {summary_stats['size_reduction']:.2f}")
    logger.info(f"  Runtime: {end_time - start_time:.2f} seconds")
    
    return summary

def run_comparative_experiments(datasets: Dict[str, nx.Graph], algorithms: Dict[str, callable], logger):
    """Run comparative experiments across datasets and algorithms."""
    logger.info("=== Running Comparative Experiments ===")
    
    evaluator = SummEvaluation(output_dir="./ssumm_results")
    
    # Target size ratios to test
    target_size_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    
    # Run experiments on each dataset
    for dataset_name, graph in datasets.items():
        logger.info(f"\nEvaluating on dataset: {dataset_name}")
        
        # Run comparison
        results = evaluator.compare_algorithms(
            algorithms=algorithms,
            graph=graph,
            target_sizes=target_size_ratios,
            dataset_name=dataset_name,
            timeout=600  # 10 minute timeout per run
        )
        
        logger.info(f"Results saved for {dataset_name}")
        
        # Display best performance for each metric
        for algo in algorithms.keys():
            algo_results = results[results["algorithm"] == algo]
            if not algo_results.empty and "reconstruction_error_l1" in algo_results.columns:
                best_error = algo_results["reconstruction_error_l1"].min()
                logger.info(f"  {algo}: Best L1 error = {best_error:.6f}")

def run_parameter_sensitivity_analysis(datasets: Dict[str, nx.Graph], logger):
    """Run parameter sensitivity analysis for SSumM."""
    logger.info("=== Parameter Sensitivity Analysis ===")
    
    evaluator = SummEvaluation(output_dir="./ssumm_results")
    
    # Parameters to analyze
    parameters = {
        "max_iterations": [5, 10, 15, 20, 25, 30],
        "max_candidate_set_size": [100, 250, 500, 750, 1000],
        "max_recursion_depth": [5, 8, 10, 12, 15]
    }
    
    # Test on a sample dataset
    if "DBLP" in datasets:
        graph = datasets["DBLP"]
        dataset_name = "DBLP"
    else:
        graph = next(iter(datasets.values()))
        dataset_name = next(iter(datasets.keys()))
    
    logger.info(f"Analyzing parameters on {dataset_name} dataset")
    
    for param_name, param_values in parameters.items():
        logger.info(f"\nAnalyzing parameter: {param_name}")
        
        results = evaluator.analyze_parameter_sensitivity(
            graph=graph,
            parameter_name=param_name,
            parameter_values=param_values,
            dataset_name=dataset_name
        )
        
        logger.info(f"Sensitivity analysis complete for {param_name}")

def visualize_summary_structure(summary: GraphSummary, output_path: str):
    """Visualize the structure of a summary graph."""
    import matplotlib.pyplot as plt
    
    # Convert to NetworkX graph
    G_summary = summary.to_networkx_graph()
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Calculate node sizes based on supernode sizes
    node_sizes = [G_summary.nodes[n]['size'] * 100 for n in G_summary.nodes()]
    
    # Draw the graph
    pos = nx.spring_layout(G_summary, seed=42)
    
    # Draw nodes with size based on supernode size
    nx.draw_networkx_nodes(G_summary, pos, 
                          node_size=node_sizes,
                          node_color='skyblue',
                          alpha=0.7)
    
    # Draw edges with width based on weight
    edge_widths = [G_summary[u][v]['weight'] / 10 for u, v in G_summary.edges()]
    nx.draw_networkx_edges(G_summary, pos, 
                          width=edge_widths,
                          alpha=0.5)
    
    # Add labels for smaller summaries
    if G_summary.number_of_nodes() <= 20:
        nx.draw_networkx_labels(G_summary, pos, font_size=10)
    
    plt.title("Summary Graph Structure\n(Node size = # original nodes, Edge width = # connections)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="SSumM Algorithm Demonstration")
    parser.add_argument("--mode", choices=["demo", "compare", "params", "all"], default="all",
                        help="Run mode: demo, compare, params, or all")
    parser.add_argument("--dataset", type=str, help="Specific dataset to use (optional)")
    parser.add_argument("--output", type=str, default="./ssumm_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting SSumM demonstration")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # Load datasets
        datasets = load_datasets()
        
        if not datasets:
            logger.error("No datasets loaded. Exiting.")
            return
        
        # Filter to specific dataset if requested
        if args.dataset and args.dataset in datasets:
            datasets = {args.dataset: datasets[args.dataset]}
        
        # Run demonstrations
        if args.mode in ["demo", "all"]:
            # Use first available dataset for demo
            demo_dataset = next(iter(datasets.items()))
            demo_name, demo_graph = demo_dataset
            logger.info(f"Running basic demonstration on {demo_name}")
            
            summary = demonstrate_basic_functionality(demo_graph, logger)
            
            # Visualize the summary
            vis_path = os.path.join(args.output, f"{demo_name}_summary_visualization.png")
            visualize_summary_structure(summary, vis_path)
            logger.info(f"Summary visualization saved to {vis_path}")
        
        if args.mode in ["compare", "all"]:
            # Create algorithm wrappers
            algorithms = create_algorithm_wrappers()
            
            # Run comparative experiments
            run_comparative_experiments(datasets, algorithms, logger)
        
        if args.mode in ["params", "all"]:
            # Run parameter sensitivity analysis
            run_parameter_sensitivity_analysis(datasets, logger)
        
        logger.info(f"\nAll results saved to: {args.output}")
        logger.info("Demonstration complete!")
        
    except Exception as e:
        logger.error(f"Error in demonstration: {e}", exc_info=True)

if __name__ == "__main__":
    main()