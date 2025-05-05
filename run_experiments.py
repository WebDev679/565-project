#!/usr/bin/env python3
"""
Main script to run comprehensive experiments for the SSumM project.
This will compare SSumM with baseline algorithms (k-Gs, S2L, SAA-Gs)
on various datasets and report the results.
"""

import os
import argparse
import logging
import time
from experiment_runner import ExperimentRunner
import networkx as nx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiments.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('run_experiments')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run experiments for SSumM project')
    
    # Dataset options
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing datasets')
    parser.add_argument('--datasets', nargs='+', 
                        default=['DBLP', 'Amazon-0302', 'Email-Enron', 'Ego-Facebook'],
                        help='Names of datasets to use')
    parser.add_argument('--use_synthetic', action='store_true',
                        help='Include synthetic datasets in experiments')
    parser.add_argument('--sample_large', action='store_true',
                        help='Sample large datasets to make them manageable')
    parser.add_argument('--max_nodes', type=int, default=10000,
                        help='Maximum number of nodes when sampling large datasets')
    
    # Algorithm options
    parser.add_argument('--algorithms', nargs='+', 
                        default=['SSumM', 'k-Gs', 'S2L', 'SAA-Gs'],
                        help='Algorithms to include in experiments')
    parser.add_argument('--use_optimized', action='store_true',
                        help='Use optimized SSumM implementation')
    
    # Experiment options
    parser.add_argument('--results_dir', type=str, default='./experiment_results',
                        help='Directory to save results')
    parser.add_argument('--target_ratios', nargs='+', type=float,
                        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                        help='Target size ratios to test')
    parser.add_argument('--max_runtime', type=int, default=3600,
                        help='Maximum runtime in seconds per algorithm per dataset')
    parser.add_argument('--memory_limit', type=float, default=None,
                        help='Memory limit in GB')
    
    # Run mode
    parser.add_argument('--quick_test', action='store_true',
                        help='Run a quick test with smaller datasets and fewer iterations')
    
    return parser.parse_args()

def create_synthetic_datasets():
    """Create synthetic datasets for testing."""
    datasets = {}
    
    # Generate Barabasi-Albert graphs of different sizes
    for n, m in [(50, 2), (100, 3), (200, 4)]:
        name = f"ba_{n}_{m}"
        datasets[name] = nx.barabasi_albert_graph(n=n, m=m, seed=42)
    
    # Generate Erdos-Renyi graphs
    for n, p in [(50, 0.1), (100, 0.05), (200, 0.025)]:
        name = f"er_{n}_{int(p*100)}"
        datasets[name] = nx.erdos_renyi_graph(n=n, p=p, seed=42)
    
    # Generate Watts-Strogatz small-world graphs
    for n, k, p in [(50, 4, 0.1), (100, 6, 0.1), (200, 8, 0.1)]:
        name = f"ws_{n}_{k}_{int(p*100)}"
        datasets[name] = nx.watts_strogatz_graph(n=n, k=k, p=p, seed=42)
    
    # Generate LFR benchmark graphs (with community structure)
    try:
        import networkx.algorithms.community as nx_comm
        
        # Smaller LFR benchmark for testing
        G = nx_comm.LFR_benchmark_graph(
            n=100, tau1=3, tau2=1.5, mu=0.1, average_degree=6, min_community=10, seed=42)
        datasets["lfr_100"] = G
    except ImportError:
        logger.warning("Could not generate LFR benchmark graph: networkx community module not available")
    
    logger.info(f"Created {len(datasets)} synthetic datasets")
    return datasets

def run_quick_test():
    """Run a quick test with smaller datasets and fewer ratios."""
    logger.info("Running quick test mode")
    
    # Initialize experiment runner with shorter timeout
    runner = ExperimentRunner(
        data_dir="./data",
        results_dir="./quick_test_results",
        max_runtime=300,  # 5 minutes max
        use_optimized=True
    )
    
    # Create small synthetic datasets
    datasets = {
        "karate": nx.karate_club_graph(),
        "ba_50_2": nx.barabasi_albert_graph(n=50, m=2, seed=42)
    }
    
    # Run experiments with limited scope
    results = runner.run_experiments(
        datasets=datasets,
        algorithms=['SSumM', 'k-Gs'],  # Only test two algorithms
        target_size_ratios=[0.2, 0.4]  # Only two ratios
    )
    
    logger.info("Quick test completed successfully")
    return results

def main():
    """Main function to run experiments."""
    start_time = time.time()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Run quick test if requested
    if args.quick_test:
        run_quick_test()
        return
    
    # Initialize experiment runner
    runner = ExperimentRunner(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        max_runtime=args.max_runtime,
        memory_limit_gb=args.memory_limit,
        use_optimized=args.use_optimized
    )
    
    # Load datasets
    datasets = runner.load_datasets(
        dataset_names=args.datasets,
        sample_large=args.sample_large,
        max_nodes_sample=args.max_nodes
    )
    
    # Add synthetic datasets if requested
    if args.use_synthetic:
        synthetic_datasets = create_synthetic_datasets()
        datasets.update(synthetic_datasets)
    
    # Run experiments
    results = runner.run_experiments(
        datasets=datasets,
        algorithms=args.algorithms,
        target_size_ratios=args.target_ratios
    )
    
    # Calculate total runtime
    total_time = time.time() - start_time
    logger.info(f"All experiments completed in {total_time:.2f} seconds")
    
    # Report success
    logger.info(f"Results saved to {args.results_dir}")

if __name__ == "__main__":
    main()