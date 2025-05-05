#!/usr/bin/env python3
"""
Comprehensive experiment runner for graph summarization algorithms.

This script runs a complete set of experiments to compare the performance of different
graph summarization algorithms (SSumM, k-Gs, S2L, SAA-Gs) on various datasets.
It measures and reports the following metrics:
- Summary size (in bits)
- Reconstruction error (ℓ₁ and ℓ₂)
- Runtime performance
- Memory usage
- Quality metrics (distance from ideal quality)
"""

import os
import time
import argparse
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from memory_profiler import memory_usage
import psutil
import logging
from datetime import datetime
from tqdm import tqdm

# Import our algorithm implementations
from graph_utils import GraphUtils
from kgs_algorithm import KGs
from s2l_algorithm import S2L
from saa_gs_algorithm import SAA_Gs
from ssumm_algorithm import SSumM
from ssumm_optimized import OptimizedSSumM
from graph_visualizer import GraphVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiments.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ExperimentRunner')

class ExperimentRunner:
    """
    Runner for comprehensive experiments on graph summarization algorithms.
    """
    
    def __init__(self, 
                 data_dir="./data", 
                 results_dir="./experiment_results",
                 max_runtime=3600,  # 1 hour maximum runtime per algorithm per dataset
                 memory_limit_gb=None,  # Auto-detect memory limit
                 use_optimized=True):
        """
        Initialize the experiment runner.
        
        Args:
            data_dir: Directory containing datasets
            results_dir: Directory to save results
            max_runtime: Maximum runtime in seconds per algorithm per dataset
            memory_limit_gb: Memory limit in GB (None for auto-detection)
            use_optimized: Whether to use optimized SSumM implementation
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.max_runtime = max_runtime
        self.use_optimized = use_optimized
        
        # Set memory limit
        if memory_limit_gb is None:
            # Use 75% of available memory by default
            total_memory = psutil.virtual_memory().total
            self.memory_limit_bytes = int(0.75 * total_memory)
            self.memory_limit_gb = self.memory_limit_bytes / (1024**3)
        else:
            self.memory_limit_gb = memory_limit_gb
            self.memory_limit_bytes = int(memory_limit_gb * 1024**3)
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Algorithm configurations
        self.algorithms = {
            'SSumM': self._create_ssumm,
            'k-Gs': self._create_kgs,
            'S2L': self._create_s2l,
            'SAA-Gs': self._create_saags
        }
        
        # Initialize results dictionary
        self.results = {
            'summary': [],
            'detailed': {},
            'convergence': {}
        }
        
        logger.info(f"Experiment runner initialized with:")
        logger.info(f"  Data directory: {data_dir}")
        logger.info(f"  Results directory: {results_dir}")
        logger.info(f"  Max runtime: {max_runtime} seconds per run")
        logger.info(f"  Memory limit: {self.memory_limit_gb:.2f} GB")
        logger.info(f"  Using optimized SSumM: {use_optimized}")
    
    def _create_ssumm(self, iterations=20):
        """Create SSumM algorithm instance."""
        if self.use_optimized:
            return OptimizedSSumM(max_iterations=iterations)
        else:
            return SSumM(max_iterations=iterations)
    
    def _create_kgs(self):
        """Create k-Gs algorithm instance."""
        return KGs()
    
    def _create_s2l(self):
        """Create S2L algorithm instance."""
        return S2L()
    
    def _create_saags(self):
        """Create SAA-Gs algorithm instance."""
        return SAA_Gs()
    
    def load_datasets(self, dataset_names=None, sample_large=True, max_nodes_sample=10000):
        """
        Load datasets for experiments.
        
        Args:
            dataset_names: List of dataset names to load (None for all available)
            sample_large: Whether to sample large datasets to make them manageable
            max_nodes_sample: Maximum number of nodes for sampling
            
        Returns:
            dict: Dictionary of loaded graphs
        """
        datasets = {}
        
        # If no specific datasets provided, look for all datasets in the data directory
        if dataset_names is None:
            dataset_files = [f for f in os.listdir(self.data_dir) 
                          if os.path.isfile(os.path.join(self.data_dir, f)) and 
                          (f.endswith('.txt') or f.endswith('.gpickle') or f.endswith('.edges'))]
            dataset_names = [os.path.splitext(f)[0] for f in dataset_files]
        
        # Load each dataset
        for name in dataset_names:
            file_path = self._find_dataset_file(name)
            
            if file_path:
                try:
                    logger.info(f"Loading dataset: {name} from {file_path}")
                    
                    # Determine file format
                    if file_path.endswith('.gpickle'):
                        G = GraphUtils.load_graph(file_path, format="pickle")
                    elif file_path.endswith('.txt') or file_path.endswith('.edges'):
                        G = GraphUtils.load_graph(file_path, format="edge_list")
                    else:
                        logger.warning(f"Unsupported file format for {file_path}, skipping")
                        continue
                    
                    # Ensure graph is undirected and has no self-loops
                    if G.is_directed():
                        G = G.to_undirected()
                    G.remove_edges_from(nx.selfloop_edges(G))
                    
                    # Sample large graphs if needed
                    if sample_large and G.number_of_nodes() > max_nodes_sample:
                        logger.info(f"Sampling large graph {name} with {G.number_of_nodes()} nodes to {max_nodes_sample} nodes")
                        nodes = list(G.nodes())
                        sampled_nodes = np.random.choice(nodes, max_nodes_sample, replace=False)
                        G = G.subgraph(sampled_nodes).copy()
                        name = f"{name}_sampled_{max_nodes_sample}"
                    
                    # Store graph
                    datasets[name] = G
                    
                    # Log dataset info
                    logger.info(f"  Loaded {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
                
                except Exception as e:
                    logger.error(f"Error loading {name}: {e}", exc_info=True)
            else:
                logger.warning(f"Could not find dataset file for {name}")
        
        # Create at least one small test dataset if none were loaded
        if not datasets:
            logger.warning("No datasets loaded. Creating Zachary's Karate Club graph as fallback.")
            datasets["karate"] = nx.karate_club_graph()
        
        return datasets
    
    def _find_dataset_file(self, dataset_name):
        """Find the file for a dataset by name."""
        # Look for common file extensions
        for ext in ['.gpickle', '.txt', '.edges', '.edgelist']:
            path = os.path.join(self.data_dir, f"{dataset_name}{ext}")
            if os.path.exists(path):
                return path
        
        # Try without adding extension (if name already includes it)
        path = os.path.join(self.data_dir, dataset_name)
        if os.path.exists(path):
            return path
        
        return None
    
    def run_algorithm(self, algorithm_name, algorithm, graph, target_size_ratios=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 
                     track_convergence=True, timeout=None):
        """
        Run a graph summarization algorithm with different target size ratios.
        
        Args:
            algorithm_name: Name of the algorithm
            algorithm: Algorithm instance
            graph: Input graph
            target_size_ratios: List of target size ratios to test
            track_convergence: Whether to track convergence statistics
            timeout: Maximum runtime in seconds (None for default)
            
        Returns:
            dict: Results of the algorithm runs
        """
        if timeout is None:
            timeout = self.max_runtime
        
        algorithm_results = []
        convergence_data = {}
        
        # Calculate original graph size in bits
        original_size = 2 * graph.number_of_edges() * np.ceil(np.log2(graph.number_of_nodes()))
        
        # Run the algorithm for each target size ratio
        for ratio in target_size_ratios:
            target_size = int(ratio * original_size)
            
            logger.info(f"Running {algorithm_name} with target ratio {ratio:.2f} (target size: {target_size} bits)")
            
            # Handle algorithm-specific parameters and methods
            try:
                # Memory before run
                process = psutil.Process(os.getpid())
                memory_before = process.memory_info().rss
                
                # Start timing
                start_time = time.time()
                
                # Use the appropriate method based on algorithm type
                if algorithm_name == 'SSumM' or algorithm_name == 'OptimizedSSumM':
                    # SSumM takes target size in bits
                    if track_convergence:
                        summary, performance_data = algorithm.summarize(graph, target_size, track_convergence=True)
                        this_convergence = performance_data.get('convergence', {})
                    else:
                        summary = algorithm.summarize(graph, target_size, track_convergence=False)
                        this_convergence = {}
                
                elif algorithm_name == 'k-Gs':
                    # k-Gs takes target number of supernodes
                    k = max(2, int(ratio * graph.number_of_nodes()))
                    # Use the sample_pairs method for better efficiency
                    summary = algorithm.sample_pairs(graph, k, c=1.0)
                    this_convergence = {}
                
                elif algorithm_name == 'S2L':
                    # S2L takes target number of supernodes
                    k = max(2, int(ratio * graph.number_of_nodes()))
                    summary = algorithm.summarize(graph, k, method='kmeans')
                    this_convergence = {}
                
                elif algorithm_name == 'SAA-Gs':
                    # SAA-Gs takes target number of supernodes
                    k = max(2, int(ratio * graph.number_of_nodes()))
                    summary = algorithm.summarize(graph, k, log_n_sampling=True)
                    this_convergence = {}
                
                else:
                    raise ValueError(f"Unknown algorithm: {algorithm_name}")
                
                # End timing
                end_time = time.time()
                runtime = end_time - start_time
                
                # Memory after run
                memory_after = process.memory_info().rss
                memory_used = memory_after - memory_before
                
                # Calculate metrics
                size_bits = summary.size_in_bits()
                relative_size = size_bits / original_size
                l1_error = summary.compute_reconstruction_error(p=1)
                l2_error = summary.compute_reconstruction_error(p=2)
                
                # Calculate quality metric (distance from ideal - lower is better)
                # Normalize errors to [0,1] range
                normalized_l1_error = l1_error / 0.5  # Assume max error is 0.5 for normalization
                normalized_size = relative_size
                quality_distance = np.sqrt(normalized_l1_error**2 + normalized_size**2)
                
                # Store results
                result = {
                    'algorithm': algorithm_name,
                    'target_ratio': ratio,
                    'target_size': target_size,
                    'runtime_sec': runtime,
                    'memory_bytes': memory_used,
                    'size_bits': size_bits,
                    'relative_size': relative_size,
                    'l1_error': l1_error,
                    'l2_error': l2_error,
                    'quality_distance': quality_distance,
                    'num_supernodes': len(summary.supernodes),
                    'num_superedges': len(summary.superedge_counts),
                    'success': True,
                    'error': None
                }
                
                algorithm_results.append(result)
                
                # Store convergence data if available
                if this_convergence:
                    convergence_key = f"{algorithm_name}_{ratio:.2f}"
                    convergence_data[convergence_key] = this_convergence
                
                logger.info(f"  Completed in {runtime:.2f} seconds")
                logger.info(f"  Memory used: {memory_used / 1024**2:.2f} MB")
                logger.info(f"  Summary: {len(summary.supernodes)} supernodes, {len(summary.superedge_counts)} superedges")
                logger.info(f"  Size: {size_bits} bits ({relative_size:.2%} of original)")
                logger.info(f"  L1 Error: {l1_error:.6f}, L2 Error: {l2_error:.6f}")
                
            except Exception as e:
                logger.error(f"Error running {algorithm_name} with ratio {ratio}: {e}", exc_info=True)
                
                # Store error result
                result = {
                    'algorithm': algorithm_name,
                    'target_ratio': ratio,
                    'target_size': target_size,
                    'runtime_sec': timeout,  # Assume it ran until timeout
                    'memory_bytes': 0,
                    'size_bits': None,
                    'relative_size': None,
                    'l1_error': None,
                    'l2_error': None,
                    'quality_distance': None,
                    'num_supernodes': None,
                    'num_superedges': None,
                    'success': False,
                    'error': str(e)
                }
                
                algorithm_results.append(result)
        
        return {
            'results': algorithm_results,
            'convergence': convergence_data
        }
    
    def run_experiments(self, datasets, algorithms=None, target_size_ratios=None):
        """
        Run comprehensive experiments on multiple datasets with multiple algorithms.
        
        Args:
            datasets: Dictionary of datasets to test
            algorithms: List of algorithm names to test (None for all)
            target_size_ratios: List of target size ratios to test
            
        Returns:
            dict: Comprehensive results of all experiments
        """
        if algorithms is None:
            algorithms = list(self.algorithms.keys())
        
        if target_size_ratios is None:
            target_size_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        
        # Create experiment start timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"experiment_{timestamp}"
        
        # Create experiment directory
        experiment_dir = os.path.join(self.results_dir, experiment_id)
        os.makedirs(experiment_dir, exist_ok=True)
        
        logger.info(f"Starting experiments with ID: {experiment_id}")
        logger.info(f"Testing {len(datasets)} datasets with {len(algorithms)} algorithms")
        logger.info(f"Target size ratios: {target_size_ratios}")
        
        # Initialize results structure
        all_results = []
        detailed_results = {}
        all_convergence = {}
        
        # Run experiments for each dataset
        for dataset_name, graph in datasets.items():
            logger.info(f"\n{'='*50}\nProcessing dataset: {dataset_name}\n{'='*50}")
            
            # Create dataset directory
            dataset_dir = os.path.join(experiment_dir, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Get graph metrics
            graph_metrics = GraphUtils.compute_graph_metrics(graph)
            
            # Run each algorithm
            dataset_results = []
            dataset_convergence = {}
            
            for algorithm_name in algorithms:
                logger.info(f"\nRunning algorithm: {algorithm_name} on {dataset_name}")
                
                try:
                    # Create algorithm instance
                    if algorithm_name == 'SSumM':
                        algorithm = self._create_ssumm(iterations=20)
                    else:
                        algorithm = self.algorithms[algorithm_name]()
                    
                    # Run the algorithm
                    result = self.run_algorithm(
                        algorithm_name, 
                        algorithm, 
                        graph, 
                        target_size_ratios=target_size_ratios,
                        track_convergence=(algorithm_name == 'SSumM')
                    )
                    
                    # Add results to dataset results
                    dataset_results.extend(result['results'])
                    
                    # Store convergence data if available
                    if result['convergence']:
                        dataset_convergence.update(result['convergence'])
                
                except Exception as e:
                    logger.error(f"Error running {algorithm_name} on {dataset_name}: {e}", exc_info=True)
            
            # Add dataset info to results
            for result in dataset_results:
                result['dataset'] = dataset_name
                result['nodes'] = graph_metrics['nodes']
                result['edges'] = graph_metrics['edges']
                result['density'] = graph_metrics['density']
                all_results.append(result)
            
            # Store detailed results for this dataset
            detailed_results[dataset_name] = {
                'graph_metrics': graph_metrics,
                'results': dataset_results
            }
            
            # Store convergence data
            if dataset_convergence:
                all_convergence[dataset_name] = dataset_convergence
            
            # Save intermediate results for this dataset
            self._save_dataset_results(dataset_dir, dataset_name, dataset_results, dataset_convergence, graph_metrics)
        
        # Save overall results
        self.results = {
            'summary': all_results,
            'detailed': detailed_results,
            'convergence': all_convergence
        }
        
        # Save consolidated results
        self._save_consolidated_results(experiment_dir, self.results)
        
        # Create visualizations
        self._create_visualizations(experiment_dir, self.results)
        
        logger.info(f"\nExperiments completed. Results saved to {experiment_dir}")
        return self.results
    
    def _save_dataset_results(self, dataset_dir, dataset_name, results, convergence, graph_metrics):
        """Save results for a single dataset."""
        # Save results as CSV
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(dataset_dir, f"{dataset_name}_results.csv"), index=False)
        
        # Save graph metrics
        with open(os.path.join(dataset_dir, f"{dataset_name}_metrics.json"), 'w') as f:
            json.dump(graph_metrics, f, indent=2)
        
        # Save convergence data if available
        if convergence:
            with open(os.path.join(dataset_dir, f"{dataset_name}_convergence.pickle"), 'wb') as f:
                pickle.dump(convergence, f)
    
    def _save_consolidated_results(self, experiment_dir, results):
        """Save consolidated results from all experiments."""
        # Save summary results as CSV
        df = pd.DataFrame(results['summary'])
        df.to_csv(os.path.join(experiment_dir, "summary_results.csv"), index=False)
        
        # Save full results as pickle
        with open(os.path.join(experiment_dir, "full_results.pickle"), 'wb') as f:
            pickle.dump(results, f)
        
        # Save a readable summary
        self._create_summary_report(experiment_dir, results)
    
    def _create_summary_report(self, experiment_dir, results):
        """Create a human-readable summary report."""
        summary = pd.DataFrame(results['summary'])
        
        with open(os.path.join(experiment_dir, "summary_report.md"), 'w') as f:
            f.write("# Graph Summarization Experiment Results\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall statistics
            f.write("## Overall Statistics\n\n")
            
            # Dataset statistics
            datasets = summary['dataset'].unique()
            f.write(f"### Datasets ({len(datasets)})\n\n")
            
            for dataset in datasets:
                dataset_rows = summary[summary['dataset'] == dataset]
                nodes = dataset_rows['nodes'].iloc[0]
                edges = dataset_rows['edges'].iloc[0]
                density = dataset_rows['density'].iloc[0]
                
                f.write(f"- **{dataset}**: {nodes} nodes, {edges} edges, density: {density:.6f}\n")
            
            f.write("\n")
            
            # Algorithm statistics
            algorithms = summary['algorithm'].unique()
            f.write(f"### Algorithms ({len(algorithms)})\n\n")
            
            for algorithm in algorithms:
                algorithm_rows = summary[summary['algorithm'] == algorithm]
                success_count = algorithm_rows['success'].sum()
                total_count = len(algorithm_rows)
                
                f.write(f"- **{algorithm}**: {success_count}/{total_count} successful runs\n")
            
            f.write("\n")
            
            # Performance summary
            f.write("## Performance Summary\n\n")
            
            # Summarize by algorithm
            f.write("### Algorithm Performance\n\n")
            
            # Group by algorithm and compute average metrics
            alg_summary = summary.groupby('algorithm').agg({
                'runtime_sec': ['mean', 'std', 'min', 'max'],
                'memory_bytes': ['mean', 'max'],
                'l1_error': ['mean', 'min'],
                'l2_error': ['mean', 'min'],
                'relative_size': ['mean', 'min'],
                'quality_distance': ['mean', 'min']
            }).reset_index()
            
            # Convert to Markdown table
            f.write("| Algorithm | Avg Runtime (s) | Max Memory (MB) | Avg L1 Error | Avg Relative Size | Avg Quality |\n")
            f.write("|-----------|----------------|----------------|--------------|-------------------|-------------|\n")
            
            for _, row in alg_summary.iterrows():
                algo = row['algorithm']
                runtime = row[('runtime_sec', 'mean')]
                memory = row[('memory_bytes', 'mean')] / (1024**2)  # Convert to MB
                error = row[('l1_error', 'mean')]
                size = row[('relative_size', 'mean')]
                quality = row[('quality_distance', 'mean')]
                
                f.write(f"| {algo} | {runtime:.2f} | {memory:.2f} | {error:.6f} | {size:.2%} | {quality:.4f} |\n")
            
            f.write("\n")
            
            # Summarize by dataset
            f.write("### Dataset Complexity\n\n")
            
            # Group by dataset and compute average metrics across algorithms
            dataset_summary = summary.groupby('dataset').agg({
                'runtime_sec': ['mean', 'max'],
                'l1_error': ['mean', 'min'],
                'relative_size': ['mean', 'min'],
                'nodes': 'first',
                'edges': 'first',
                'density': 'first'
            }).reset_index()
            
            # Convert to Markdown table
            f.write("| Dataset | Nodes | Edges | Density | Avg Runtime (s) | Best L1 Error |\n")
            f.write("|---------|-------|-------|---------|----------------|---------------|\n")
            
            for _, row in dataset_summary.iterrows():
                dataset = row['dataset']
                nodes = row[('nodes', 'first')]
                edges = row[('edges', 'first')]
                density = row[('density', 'first')]
                runtime = row[('runtime_sec', 'mean')]
                error = row[('l1_error', 'min')]
                
                f.write(f"| {dataset} | {nodes} | {edges} | {density:.6f} | {runtime:.2f} | {error:.6f} |\n")
            
            f.write("\n")
            
            # Key Findings
            f.write("## Key Findings\n\n")
            
            # Best algorithm per metric
            f.write("### Best Algorithms by Metric\n\n")
            
            # L1 Error
            best_l1 = summary.loc[summary.groupby('algorithm')['l1_error'].idxmin()]
            best_l1_algo = best_l1.loc[best_l1['l1_error'].idxmin(), 'algorithm']
            best_l1_error = best_l1.loc[best_l1['l1_error'].idxmin(), 'l1_error']
            
            # Runtime
            best_runtime = summary.loc[summary.groupby('algorithm')['runtime_sec'].idxmin()]
            best_runtime_algo = best_runtime.loc[best_runtime['runtime_sec'].idxmin(), 'algorithm']
            best_runtime_time = best_runtime.loc[best_runtime['runtime_sec'].idxmin(), 'runtime_sec']
            
            # Size
            best_size = summary.loc[summary.groupby('algorithm')['relative_size'].idxmin()]
            best_size_algo = best_size.loc[best_size['relative_size'].idxmin(), 'algorithm']
            best_size_ratio = best_size.loc[best_size['relative_size'].idxmin(), 'relative_size']
            
            # Quality
            best_quality = summary.loc[summary.groupby('algorithm')['quality_distance'].idxmin()]
            best_quality_algo = best_quality.loc[best_quality['quality_distance'].idxmin(), 'algorithm']
            best_quality_value = best_quality.loc[best_quality['quality_distance'].idxmin(), 'quality_distance']
            
            f.write(f"- **Best for Accuracy (Lowest L1 Error)**: {best_l1_algo} ({best_l1_error:.6f})\n")
            f.write(f"- **Best for Runtime**: {best_runtime_algo} ({best_runtime_time:.2f} seconds)\n")
            f.write(f"- **Best for Compression (Smallest Size)**: {best_size_algo} ({best_size_ratio:.2%} of original)\n")
            f.write(f"- **Best Overall Quality**: {best_quality_algo} (Quality score: {best_quality_value:.4f})\n\n")
            
            # Trade-offs and recommendations
            f.write("### Trade-offs and Recommendations\n\n")
            
            f.write("Based on the experimental results, we can recommend the following:\n\n")
            
            # For small graphs
            small_datasets = [d for d in datasets if summary[summary['dataset'] == d]['nodes'].iloc[0] < 1000]
            if small_datasets:
                small_results = summary[summary['dataset'].isin(small_datasets)]
                best_small = small_results.groupby('algorithm')['quality_distance'].mean().idxmin()
                f.write(f"- **For small graphs (<1000 nodes)**: {best_small} provides the best balance of accuracy and compression\n")
            
            # For medium graphs
            medium_datasets = [d for d in datasets if 1000 <= summary[summary['dataset'] == d]['nodes'].iloc[0] < 10000]
            if medium_datasets:
                medium_results = summary[summary['dataset'].isin(medium_datasets)]
                best_medium = medium_results.groupby('algorithm')['quality_distance'].mean().idxmin()
                f.write(f"- **For medium graphs (1000-10000 nodes)**: {best_medium} provides the best balance of accuracy and compression\n")
            
            # For large graphs
            large_datasets = [d for d in datasets if summary[summary['dataset'] == d]['nodes'].iloc[0] >= 10000]
            if large_datasets:
                large_results = summary[summary['dataset'].isin(large_datasets)]
                best_large = large_results.groupby('algorithm')['quality_distance'].mean().idxmin()
                f.write(f"- **For large graphs (>10000 nodes)**: {best_large} provides the best balance of accuracy and compression\n")
            
            # For accuracy-critical applications
            f.write(f"- **For accuracy-critical applications**: {best_l1_algo} achieves the lowest reconstruction error\n")
            
            # For speed-critical applications
            f.write(f"- **For speed-critical applications**: {best_runtime_algo} has the fastest runtime\n")
            
            # For memory-constrained environments
            best_memory = summary.loc[summary.groupby('algorithm')['memory_bytes'].idxmin()]
            best_memory_algo = best_memory.loc[best_memory['memory_bytes'].idxmin(), 'algorithm']
            f.write(f"- **For memory-constrained environments**: {best_memory_algo} has the lowest memory footprint\n")
    
    def _create_visualizations(self, experiment_dir, results):
        """Create visualizations for experiment results."""
        # Create visualizations directory
        viz_dir = os.path.join(experiment_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        logger.info("Creating result visualizations...")
        
        # Convert results to DataFrame
        summary = pd.DataFrame(results['summary'])
        
        # Set visualization style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("colorblind")
        
        # 1. Error vs. Size scatter plot
        self._plot_error_vs_size(summary, viz_dir)
        
        # 2. Error by algorithm and dataset
        self._plot_error_by_algorithm(summary, viz_dir)
        
        # 3. Runtime by algorithm and dataset
        self._plot_runtime_by_algorithm(summary, viz_dir)
        
        # 4. Quality by algorithm and dataset
        self._plot_quality_by_algorithm(summary, viz_dir)
        
        # 5. Convergence plots for SSumM
        if results['convergence']:
            self._plot_convergence(results['convergence'], viz_dir)
        
        # 6. Algorithm comparison radar chart
        self._plot_algorithm_radar(summary, viz_dir)
        
        logger.info(f"Visualizations saved to {viz_dir}")
        
        # Create an HTML dashboard with all visualizations
        self._create_dashboard(viz_dir, experiment_dir)
    
    def _plot_error_vs_size(self, summary, viz_dir):
        """Plot error vs. size scatter plot."""
        plt.figure(figsize=(10, 8))
        
        # Filter out failed runs
        data = summary[summary['success']].copy()
        
        # Create scatter plot
        g = sns.scatterplot(
            data=data,
            x="relative_size", 
            y="l1_error", 
            hue="algorithm",
            style="algorithm",
            size="nodes",
            sizes=(50, 200),
            alpha=0.7
        )
        
        # Add annotations for datasets
        for _, row in data.iterrows():
            plt.annotate(
                row['dataset'],
                (row['relative_size'], row['l1_error']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.7
            )
        
        # Add labels and title
        plt.xlabel("Relative Size (summary / original)")
        plt.ylabel("L1 Reconstruction Error")
        plt.title("Error vs. Size Trade-off")
        
        # Add ideal point (0,0) marker
        plt.scatter([0], [0], marker='*', s=200, color='gold', label='Ideal Point', edgecolor='black', zorder=5)
        
        # Improve legend
        handles, labels = g.get_legend_handles_labels()
        plt.legend(handles, labels, title="Algorithm", loc="best")
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "error_vs_size.png"), dpi=300)
        plt.close()
    
    def _plot_error_by_algorithm(self, summary, viz_dir):
        """Plot error by algorithm for different datasets."""
        plt.figure(figsize=(12, 8))
        
        # Filter out failed runs
        data = summary[summary['success']].copy()
        
        # Sort by target ratio
        data = data.sort_values('target_ratio')
        
        # Create line plot
        sns.lineplot(
            data=data,
            x="target_ratio",
            y="l1_error",
            hue="algorithm",
            style="dataset",
            markers=True,
            dashes=False
        )
        
        # Add labels and title
        plt.xlabel("Target Size Ratio")
        plt.ylabel("L1 Reconstruction Error")
        plt.title("Reconstruction Error by Algorithm and Dataset")
        
        # Improve legend
        plt.legend(title="Algorithm", loc="best")
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "error_by_algorithm.png"), dpi=300)
        plt.close()
    
    def _plot_runtime_by_algorithm(self, summary, viz_dir):
        """Plot runtime by algorithm for different datasets."""
        plt.figure(figsize=(12, 8))
        
        # Filter out failed runs
        data = summary[summary['success']].copy()
        
        # Sort by target ratio
        data = data.sort_values('target_ratio')
        
        # Create line plot
        sns.lineplot(
            data=data,
            x="target_ratio",
            y="runtime_sec",
            hue="algorithm",
            style="dataset",
            markers=True,
            dashes=False
        )
        
        # Add labels and title
        plt.xlabel("Target Size Ratio")
        plt.ylabel("Runtime (seconds)")
        plt.title("Runtime by Algorithm and Dataset")
        
        # Use log scale for runtime if range is large
        if data['runtime_sec'].max() / data['runtime_sec'].min() > 100:
            plt.yscale('log')
        
        # Improve legend
        plt.legend(title="Algorithm", loc="best")
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "runtime_by_algorithm.png"), dpi=300)
        plt.close()
    
    def _plot_quality_by_algorithm(self, summary, viz_dir):
        """Plot quality by algorithm for different datasets."""
        plt.figure(figsize=(12, 8))
        
        # Filter out failed runs
        data = summary[summary['success']].copy()
        
        # Sort by target ratio
        data = data.sort_values('target_ratio')
        
        # Create line plot
        sns.lineplot(
            data=data,
            x="target_ratio",
            y="quality_distance",
            hue="algorithm",
            style="dataset",
            markers=True,
            dashes=False
        )
        
        # Add labels and title
        plt.xlabel("Target Size Ratio")
        plt.ylabel("Quality Distance (lower is better)")
        plt.title("Overall Quality by Algorithm and Dataset")
        
        # Improve legend
        plt.legend(title="Algorithm", loc="best")
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "quality_by_algorithm.png"), dpi=300)
        plt.close()
    
    def _plot_convergence(self, convergence_data, viz_dir):
        """Plot convergence data for SSumM algorithm."""
        # For each dataset
        for dataset, dataset_convergence in convergence_data.items():
            # For each SSumM run
            for run_key, convergence in dataset_convergence.items():
                if not convergence:
                    continue
                
                # Create figure with multiple subplots
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle(f"SSumM Convergence - {dataset} - {run_key}", fontsize=16)
                
                # 1. Error vs. Iteration
                ax = axes[0, 0]
                ax.plot(convergence['iteration'], convergence['reconstruction_error_l1'], 'o-', label='L1 Error')
                ax.set_xlabel('Iteration')
                ax.set_ylabel('L1 Reconstruction Error')
                ax.set_title('Error Convergence')
                ax.grid(alpha=0.3)
                
                # 2. Size vs. Iteration
                ax = axes[0, 1]
                ax.plot(convergence['iteration'], convergence['size_in_bits'], 'o-', color='green')
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Size (bits)')
                ax.set_title('Size Convergence')
                ax.grid(alpha=0.3)
                
                # 3. Supernodes & Superedges vs. Iteration
                ax = axes[1, 0]
                ax.plot(convergence['iteration'], convergence['num_supernodes'], 'o-', label='Supernodes')
                ax.plot(convergence['iteration'], convergence['num_superedges'], 's-', label='Superedges')
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Count')
                ax.set_title('Structure Convergence')
                ax.legend()
                ax.grid(alpha=0.3)
                
                # 4. Error vs. Size
                ax = axes[1, 1]
                sc = ax.scatter(
                    convergence['size_in_bits'], 
                    convergence['reconstruction_error_l1'],
                    c=convergence['iteration'],
                    cmap='viridis',
                    s=80,
                    alpha=0.8
                )
                
                # Add arrows to show progression
                for i in range(len(convergence['iteration']) - 1):
                    ax.annotate(
                        '',
                        xy=(convergence['size_in_bits'][i+1], convergence['reconstruction_error_l1'][i+1]),
                        xytext=(convergence['size_in_bits'][i], convergence['reconstruction_error_l1'][i]),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=1)
                    )
                
                ax.set_xlabel('Size (bits)')
                ax.set_ylabel('L1 Reconstruction Error')
                ax.set_title('Error-Size Trade-off')
                plt.colorbar(sc, ax=ax, label='Iteration')
                ax.grid(alpha=0.3)
                
                # Save figure
                plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
                plt.savefig(os.path.join(viz_dir, f"convergence_{dataset}_{run_key}.png"), dpi=300)
                plt.close()
    
    def _plot_algorithm_radar(self, summary, viz_dir):
        """Create radar chart to compare algorithms across multiple metrics."""
        # Filter out failed runs
        data = summary[summary['success']].copy()
        
        # Group by algorithm and compute average metrics
        metrics = ['l1_error', 'runtime_sec', 'memory_bytes', 'relative_size']
        radar_data = data.groupby('algorithm')[metrics].mean().reset_index()
        
        # Normalize metrics to [0,1] range (invert if lower is better)
        for metric in metrics:
            min_val = radar_data[metric].min()
            max_val = radar_data[metric].max()
            
            if max_val > min_val:
                # For metrics where lower is better, invert so 1 is best
                radar_data[f"{metric}_norm"] = 1 - ((radar_data[metric] - min_val) / (max_val - min_val))
            else:
                radar_data[f"{metric}_norm"] = 1.0
        
        # Create radar chart
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)
        
        # Number of metrics
        N = len(metrics)
        
        # Define friendly metric names
        metric_names = {
            'l1_error': 'Accuracy',
            'runtime_sec': 'Speed',
            'memory_bytes': 'Memory Efficiency',
            'relative_size': 'Compression'
        }
        
        # Angle of each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Plot each algorithm
        for i, row in radar_data.iterrows():
            algorithm = row['algorithm']
            values = [row[f"{metric}_norm"] for metric in metrics]
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, linestyle='-', label=algorithm)
            ax.fill(angles, values, alpha=0.1)
        
        # Set labels for each axis
        plt.xticks(angles[:-1], [metric_names[m] for m in metrics])
        
        # Draw y-labels (0-100%)
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75, 1.0], ["25%", "50%", "75%", "100%"], color="grey", size=8)
        plt.ylim(0, 1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title("Algorithm Comparison Across Metrics", size=15, y=1.1)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "algorithm_radar.png"), dpi=300)
        plt.close()
    
    def _create_dashboard(self, viz_dir, experiment_dir):
        """Create an HTML dashboard with all visualizations."""
        # Get all visualization files
        viz_files = [f for f in os.listdir(viz_dir) if f.endswith('.png')]
        
        # Create dashboard HTML
        dashboard_path = os.path.join(experiment_dir, "dashboard.html")
        
        with open(dashboard_path, 'w') as f:
            f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Graph Summarization Experiments Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1, h2 {
            color: #333;
        }
        .visualization {
            background-color: white;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .visualization img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
        }
        .visualization h3 {
            margin-top: 0;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Graph Summarization Experiments Dashboard</h1>
""")
            
            # Add visualizations
            f.write('<h2>Comparison Visualizations</h2>\n')
            
            # Add overview visualizations first
            overview_plots = ['error_vs_size.png', 'algorithm_radar.png', 
                             'error_by_algorithm.png', 'runtime_by_algorithm.png',
                             'quality_by_algorithm.png']
            
            for plot in overview_plots:
                if plot in viz_files:
                    viz_path = os.path.join('visualizations', plot)
                    title = plot.replace('.png', '').replace('_', ' ').title()
                    
                    f.write(f'''
        <div class="visualization">
            <h3>{title}</h3>
            <img src="{viz_path}" alt="{title}">
        </div>
''')
            
            # Add convergence plots
            convergence_plots = [p for p in viz_files if p.startswith('convergence_')]
            
            if convergence_plots:
                f.write('<h2>Convergence Analysis</h2>\n')
                
                for plot in convergence_plots:
                    viz_path = os.path.join('visualizations', plot)
                    parts = plot.replace('.png', '').split('_')
                    if len(parts) >= 3:
                        title = f"Convergence: {parts[1]} - {parts[2]}"
                    else:
                        title = plot.replace('.png', '').replace('_', ' ').title()
                    
                    f.write(f'''
        <div class="visualization">
            <h3>{title}</h3>
            <img src="{viz_path}" alt="{title}">
        </div>
''')
            
            f.write('''
    </div>
</body>
</html>
''')
        
        logger.info(f"Dashboard created at {dashboard_path}")
        return dashboard_path