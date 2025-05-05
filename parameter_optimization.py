#!/usr/bin/env python3
"""
Parameter optimization script for SSumM algorithm.
This script tests different parameter configurations of the SSumM algorithm
to identify optimal settings for different types of graphs.
"""

import os
import argparse
import logging
import time
import pickle
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from itertools import product
from tqdm import tqdm

# Import our algorithm implementations
from graph_utils import GraphUtils
from ssumm_optimized import OptimizedSSumM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parameter_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ParameterOptimization')

class ParameterOptimizer:
    """
    Class for optimizing SSumM algorithm parameters on different graph types.
    """
    
    def __init__(self, data_dir="./data", results_dir="./parameter_optimization_results"):
        """
        Initialize the parameter optimizer.
        
        Args:
            data_dir: Directory containing datasets
            results_dir: Directory to save results
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize results dictionary
        self.results = []
    
    def load_datasets(self, dataset_names=None, synthetic=True):
        """
        Load datasets for parameter optimization.
        
        Args:
            dataset_names: List of dataset names to load
            synthetic: Whether to include synthetic datasets
            
        Returns:
            dict: Dictionary of loaded graphs
        """
        datasets = {}
        
        # Load real datasets if names provided
        if dataset_names:
            for name in dataset_names:
                try:
                    # Try to find the dataset file
                    file_path = self._find_dataset_file(name)
                    
                    if file_path:
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
                        
                        # Sample large graphs
                        if G.number_of_nodes() > 1000:
                            logger.info(f"Sampling large graph {name} to 1000 nodes for parameter optimization")
                            nodes = list(G.nodes())
                            sampled_nodes = np.random.choice(nodes, 1000, replace=False)
                            G = G.subgraph(sampled_nodes).copy()
                            name = f"{name}_sampled_1000"
                        
                        # Store graph
                        datasets[name] = G
                        logger.info(f"  Loaded {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
                    else:
                        logger.warning(f"Could not find dataset file for {name}")
                
                except Exception as e:
                    logger.error(f"Error loading {name}: {e}", exc_info=True)
        
        # Add synthetic datasets if requested
        if synthetic:
            logger.info("Creating synthetic datasets for different graph structures")
            
            # Barabasi-Albert (scale-free network)
            G_ba = nx.barabasi_albert_graph(n=500, m=3, seed=42)
            datasets["ba_500_3"] = G_ba
            logger.info(f"  Created BA graph: {G_ba.number_of_nodes()} nodes, {G_ba.number_of_edges()} edges")
            
            # Erdos-Renyi (random graph)
            G_er = nx.erdos_renyi_graph(n=500, p=0.01, seed=42)
            datasets["er_500_0.01"] = G_er
            logger.info(f"  Created ER graph: {G_er.number_of_nodes()} nodes, {G_er.number_of_edges()} edges")
            
            # Watts-Strogatz (small-world network)
            G_ws = nx.watts_strogatz_graph(n=500, k=10, p=0.1, seed=42)
            datasets["ws_500_10_0.1"] = G_ws
            logger.info(f"  Created WS graph: {G_ws.number_of_nodes()} nodes, {G_ws.number_of_edges()} edges")
            
            # Regular graph
            G_reg = nx.random_regular_graph(d=6, n=500, seed=42)
            datasets["reg_500_6"] = G_reg
            logger.info(f"  Created regular graph: {G_reg.number_of_nodes()} nodes, {G_reg.number_of_edges()} edges")
        
        # Add at least one small test graph
        if not datasets:
            logger.warning("No datasets loaded. Adding Zachary's Karate Club graph as fallback.")
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
    
    def generate_parameter_grid(self, detailed=False):
        """
        Generate a grid of parameter combinations to test.
        
        Args:
            detailed: Whether to generate a detailed grid (more combinations)
            
        Returns:
            list: List of parameter dictionaries
        """
        if detailed:
            # More detailed grid for thorough optimization
            max_iterations = [10, 20, 30, 40]
            max_candidate_set_size = [100, 500, 1000]
            max_recursion_depth = [5, 10, 15]
        else:
            # Simpler grid for quicker testing
            max_iterations = [10, 20]
            max_candidate_set_size = [200, 500]
            max_recursion_depth = [5, 10]
        
        # Generate all combinations
        parameter_grid = []
        for iters, cand_size, rec_depth in product(max_iterations, max_candidate_set_size, max_recursion_depth):
            parameter_grid.append({
                'max_iterations': iters,
                'max_candidate_set_size': cand_size,
                'max_recursion_depth': rec_depth
            })
        
        logger.info(f"Generated parameter grid with {len(parameter_grid)} combinations")
        return parameter_grid
    
    def optimize_parameters(self, datasets, parameter_grid, target_ratio=0.3, timeout=600):
        """
        Test different parameter configurations on datasets.
        
        Args:
            datasets: Dictionary of datasets
            parameter_grid: List of parameter dictionaries to test
            target_ratio: Target size ratio for summaries
            timeout: Maximum runtime in seconds per configuration
            
        Returns:
            pd.DataFrame: Results of all parameter configurations
        """
        results = []
        
        # Test each dataset with each parameter configuration
        for dataset_name, graph in datasets.items():
            logger.info(f"\nTesting parameters on dataset: {dataset_name}")
            
            # Calculate original graph size in bits
            original_size = 2 * graph.number_of_edges() * np.ceil(np.log2(graph.number_of_nodes()))
            target_size = int(target_ratio * original_size)
            
            logger.info(f"Original size: {original_size} bits, Target size: {target_size} bits")
            
            # Test each parameter configuration
            for params in tqdm(parameter_grid, desc=f"Testing parameters on {dataset_name}"):
                try:
                    # Create algorithm instance with these parameters
                    algorithm = OptimizedSSumM(
                        max_iterations=params['max_iterations'],
                        max_candidate_set_size=params['max_candidate_set_size'],
                        max_recursion_depth=params['max_recursion_depth']
                    )
                    
                    # Start timing
                    start_time = time.time()
                    
                    # Run the algorithm with this configuration
                    summary, performance_data = algorithm.summarize(graph, target_size, track_convergence=True)
                    
                    # End timing
                    runtime = time.time() - start_time
                    
                    # Calculate metrics
                    size_bits = summary.size_in_bits()
                    relative_size = size_bits / original_size
                    l1_error = summary.compute_reconstruction_error(p=1)
                    l2_error = summary.compute_reconstruction_error(p=2)
                    
                    # Calculate quality metric (lower is better)
                    normalized_l1_error = l1_error / 0.5  # Normalize to approximate [0,1] range
                    normalized_size = relative_size
                    quality_distance = np.sqrt(normalized_l1_error**2 + normalized_size**2)
                    
                    # Store convergence statistics
                    convergence = performance_data.get('convergence', {})
                    num_iterations = len(convergence.get('iteration', [])) if convergence else 0
                    
                    # Store results
                    result = {
                        'dataset': dataset_name,
                        'max_iterations': params['max_iterations'],
                        'max_candidate_set_size': params['max_candidate_set_size'],
                        'max_recursion_depth': params['max_recursion_depth'],
                        'runtime_sec': runtime,
                        'size_bits': size_bits,
                        'relative_size': relative_size,
                        'l1_error': l1_error,
                        'l2_error': l2_error,
                        'quality_distance': quality_distance,
                        'num_supernodes': len(summary.supernodes),
                        'num_superedges': len(summary.superedge_counts),
                        'actual_iterations': num_iterations,
                        'graph_nodes': graph.number_of_nodes(),
                        'graph_edges': graph.number_of_edges(),
                        'graph_density': nx.density(graph),
                        'success': True
                    }
                    
                    results.append(result)
                    
                    logger.debug(f"Configuration complete: quality={quality_distance:.4f}, runtime={runtime:.2f}s")
                
                except Exception as e:
                    logger.error(f"Error with parameters {params} on {dataset_name}: {e}")
                    
                    # Store error result
                    result = {
                        'dataset': dataset_name,
                        'max_iterations': params['max_iterations'],
                        'max_candidate_set_size': params['max_candidate_set_size'],
                        'max_recursion_depth': params['max_recursion_depth'],
                        'runtime_sec': timeout,
                        'size_bits': None,
                        'relative_size': None,
                        'l1_error': None,
                        'l2_error': None,
                        'quality_distance': None,
                        'num_supernodes': None,
                        'num_superedges': None,
                        'actual_iterations': None,
                        'graph_nodes': graph.number_of_nodes(),
                        'graph_edges': graph.number_of_edges(),
                        'graph_density': nx.density(graph),
                        'success': False
                    }
                    
                    results.append(result)
        
        # Convert to DataFrame
        self.results = results
        return pd.DataFrame(results)
    
    def analyze_results(self, results_df=None):
        """
        Analyze parameter optimization results.
        
        Args:
            results_df: DataFrame with results (None to use self.results)
            
        Returns:
            dict: Analysis results with optimal parameters
        """
        if results_df is None:
            if not self.results:
                logger.error("No results available for analysis")
                return {}
            results_df = pd.DataFrame(self.results)
        
        # Filter out failed runs
        results_df = results_df[results_df['success']].copy()
        
        if len(results_df) == 0:
            logger.error("No successful runs to analyze")
            return {}
        
        logger.info("\nAnalyzing parameter optimization results")
        
        # Overall best parameters (based on quality)
        best_quality_idx = results_df['quality_distance'].idxmin()
        best_quality_params = results_df.loc[best_quality_idx]
        
        logger.info("\nOverall best parameters (based on quality):")
        logger.info(f"  max_iterations: {best_quality_params['max_iterations']}")
        logger.info(f"  max_candidate_set_size: {best_quality_params['max_candidate_set_size']}")
        logger.info(f"  max_recursion_depth: {best_quality_params['max_recursion_depth']}")
        logger.info(f"  quality: {best_quality_params['quality_distance']:.4f}")
        logger.info(f"  runtime: {best_quality_params['runtime_sec']:.2f}s")
        
        # Best parameters by graph type
        dataset_groups = {}
        
        # Group datasets by type
        for dataset in results_df['dataset'].unique():
            if 'ba_' in dataset:
                group = 'scale_free'
            elif 'er_' in dataset:
                group = 'random'
            elif 'ws_' in dataset:
                group = 'small_world'
            elif 'reg_' in dataset:
                group = 'regular'
            else:
                group = 'real'
            
            if group not in dataset_groups:
                dataset_groups[group] = []
            dataset_groups[group].append(dataset)
        
        # Find best parameters for each graph type
        best_params_by_type = {}
        
        for graph_type, datasets in dataset_groups.items():
            type_results = results_df[results_df['dataset'].isin(datasets)]
            
            if len(type_results) > 0:
                # Group by parameter configuration and calculate average quality
                param_cols = ['max_iterations', 'max_candidate_set_size', 'max_recursion_depth']
                grouped = type_results.groupby(param_cols).agg({
                    'quality_distance': 'mean',
                    'runtime_sec': 'mean',
                    'l1_error': 'mean',
                    'relative_size': 'mean'
                }).reset_index()
                
                # Find best configuration
                best_idx = grouped['quality_distance'].idxmin()
                best_params = grouped.loc[best_idx]
                
                best_params_by_type[graph_type] = {
                    'max_iterations': int(best_params['max_iterations']),
                    'max_candidate_set_size': int(best_params['max_candidate_set_size']),
                    'max_recursion_depth': int(best_params['max_recursion_depth']),
                    'quality_distance': best_params['quality_distance'],
                    'runtime_sec': best_params['runtime_sec'],
                    'l1_error': best_params['l1_error'],
                    'relative_size': best_params['relative_size']
                }
                
                logger.info(f"\nBest parameters for {graph_type} graphs:")
                logger.info(f"  max_iterations: {best_params_by_type[graph_type]['max_iterations']}")
                logger.info(f"  max_candidate_set_size: {best_params_by_type[graph_type]['max_candidate_set_size']}")
                logger.info(f"  max_recursion_depth: {best_params_by_type[graph_type]['max_recursion_depth']}")
                logger.info(f"  quality: {best_params_by_type[graph_type]['quality_distance']:.4f}")
                logger.info(f"  runtime: {best_params_by_type[graph_type]['runtime_sec']:.2f}s")
        
        # Best parameters by graph size
        # Categorize graphs by size
        results_df['size_category'] = pd.cut(
            results_df['graph_nodes'],
            bins=[0, 100, 500, 1000, float('inf')],
            labels=['small', 'medium', 'large', 'very_large']
        )
        
        best_params_by_size = {}
        
        for size_category in results_df['size_category'].unique():
            size_results = results_df[results_df['size_category'] == size_category]
            
            if len(size_results) > 0:
                # Group by parameter configuration and calculate average quality
                param_cols = ['max_iterations', 'max_candidate_set_size', 'max_recursion_depth']
                grouped = size_results.groupby(param_cols).agg({
                    'quality_distance': 'mean',
                    'runtime_sec': 'mean',
                    'l1_error': 'mean',
                    'relative_size': 'mean'
                }).reset_index()
                
                # Find best configuration
                best_idx = grouped['quality_distance'].idxmin()
                best_params = grouped.loc[best_idx]
                
                best_params_by_size[size_category] = {
                    'max_iterations': int(best_params['max_iterations']),
                    'max_candidate_set_size': int(best_params['max_candidate_set_size']),
                    'max_recursion_depth': int(best_params['max_recursion_depth']),
                    'quality_distance': best_params['quality_distance'],
                    'runtime_sec': best_params['runtime_sec'],
                    'l1_error': best_params['l1_error'],
                    'relative_size': best_params['relative_size']
                }
                
                logger.info(f"\nBest parameters for {size_category} graphs:")
                logger.info(f"  max_iterations: {best_params_by_size[size_category]['max_iterations']}")
                logger.info(f"  max_candidate_set_size: {best_params_by_size[size_category]['max_candidate_set_size']}")
                logger.info(f"  max_recursion_depth: {best_params_by_size[size_category]['max_recursion_depth']}")
                logger.info(f"  quality: {best_params_by_size[size_category]['quality_distance']:.4f}")
                logger.info(f"  runtime: {best_params_by_size[size_category]['runtime_sec']:.2f}s")
        
        # Create analysis results object
        analysis = {
            'overall_best': {
                'max_iterations': int(best_quality_params['max_iterations']),
                'max_candidate_set_size': int(best_quality_params['max_candidate_set_size']),
                'max_recursion_depth': int(best_quality_params['max_recursion_depth']),
                'quality_distance': float(best_quality_params['quality_distance']),
                'runtime_sec': float(best_quality_params['runtime_sec']),
                'l1_error': float(best_quality_params['l1_error']),
                'relative_size': float(best_quality_params['relative_size'])
            },
            'by_graph_type': best_params_by_type,
            'by_graph_size': best_params_by_size
        }
        
        return analysis
    
    def visualize_results(self, results_df=None):
        """
        Create visualizations of parameter optimization results.
        
        Args:
            results_df: DataFrame with results (None to use self.results)
            
        Returns:
            list: Paths to generated visualizations
        """
        if results_df is None:
            if not self.results:
                logger.error("No results available for visualization")
                return []
            results_df = pd.DataFrame(self.results)
        
        # Filter out failed runs
        results_df = results_df[results_df['success']].copy()
        
        if len(results_df) == 0:
            logger.error("No successful runs to visualize")
            return []
        
        # Create visualizations directory
        viz_dir = os.path.join(self.results_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Set visualization style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("colorblind")
        
        visualizations = []
        
        # 1. Parameter impact on quality
        visualizations.append(self._plot_parameter_impact(results_df, 'quality_distance', 'Quality Distance (lower is better)', viz_dir))
        
        # 2. Parameter impact on runtime
        visualizations.append(self._plot_parameter_impact(results_df, 'runtime_sec', 'Runtime (seconds)', viz_dir))
        
        # 3. Parameter impact on L1 error
        visualizations.append(self._plot_parameter_impact(results_df, 'l1_error', 'L1 Reconstruction Error', viz_dir))
        
        # 4. Quality vs. runtime scatter plot
        visualizations.append(self._plot_quality_vs_runtime(results_df, viz_dir))
        
        # 5. Parameter heatmaps
        visualizations.append(self._plot_parameter_heatmaps(results_df, viz_dir))
        
        # 6. Parameter impact by graph type
        visualizations.append(self._plot_impact_by_graph_type(results_df, viz_dir))
        
        return visualizations
    
    def _plot_parameter_impact(self, results_df, metric, metric_label, viz_dir):
        """Plot impact of each parameter on a given metric."""
        plt.figure(figsize=(15, 5))
        
        # Plot each parameter's impact
        for i, param in enumerate(['max_iterations', 'max_candidate_set_size', 'max_recursion_depth']):
            plt.subplot(1, 3, i+1)
            
            sns.boxplot(x=param, y=metric, data=results_df)
            
            plt.xlabel(param.replace('_', ' ').title())
            plt.ylabel(metric_label)
            plt.title(f'Impact of {param.replace("_", " ").title()} on {metric_label.split("(")[0].strip()}')
            
            if metric == 'runtime_sec' and results_df['runtime_sec'].max() / results_df['runtime_sec'].min() > 10:
                plt.yscale('log')
        
        plt.tight_layout()
        output_path = os.path.join(viz_dir, f"parameter_impact_{metric}.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
    
    def _plot_quality_vs_runtime(self, results_df, viz_dir):
        """Plot quality vs. runtime scatter plot."""
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot with parameter combinations
        scatter = plt.scatter(
            results_df['runtime_sec'],
            results_df['quality_distance'],
            c=results_df['max_iterations'],
            s=results_df['max_candidate_set_size'] / 10,
            alpha=0.7,
            cmap='viridis'
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Max Iterations')
        
        # Add legend for point sizes
        sizes = results_df['max_candidate_set_size'].unique()
        handles, labels = [], []
        for size in sorted(sizes):
            handles.append(plt.scatter([], [], s=size/10, color='gray', alpha=0.7))
            labels.append(f'Candidate Set Size: {size}')
        
        plt.legend(handles, labels, loc='upper right', title='Marker Size')
        
        # Add labels and title
        plt.xlabel('Runtime (seconds)')
        plt.ylabel('Quality Distance (lower is better)')
        plt.title('Quality vs. Runtime Trade-off for Different Parameter Combinations')
        
        # Log scale for runtime if range is large
        if results_df['runtime_sec'].max() / results_df['runtime_sec'].min() > 10:
            plt.xscale('log')
        
        # Highlight best configuration
        best_idx = results_df['quality_distance'].idxmin()
        best_config = results_df.loc[best_idx]
        
        plt.scatter(
            best_config['runtime_sec'],
            best_config['quality_distance'],
            s=200,
            color='red',
            marker='*',
            edgecolor='black',
            zorder=10,
            label='Best Configuration'
        )
        
        plt.annotate(
            f"Best: iterations={int(best_config['max_iterations'])}, "
            f"candidate_size={int(best_config['max_candidate_set_size'])}, "
            f"depth={int(best_config['max_recursion_depth'])}",
            (best_config['runtime_sec'], best_config['quality_distance']),
            xytext=(20, -20),
            textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color='red'),
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7)
        )
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = os.path.join(viz_dir, "quality_vs_runtime.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
    
    def _plot_parameter_heatmaps(self, results_df, viz_dir):
        """Plot heatmaps of parameter interactions."""
        # Get unique parameter values
        iterations = sorted(results_df['max_iterations'].unique())
        candidate_sizes = sorted(results_df['max_candidate_set_size'].unique())
        recursion_depths = sorted(results_df['max_recursion_depth'].unique())
        
        # Create figure
        fig, axes = plt.subplots(len(recursion_depths), 1, figsize=(10, 4 * len(recursion_depths)))
        if len(recursion_depths) == 1:
            axes = [axes]
        
        # Create a heatmap for each recursion depth
        for i, depth in enumerate(recursion_depths):
            # Filter data for this depth
            depth_data = results_df[results_df['max_recursion_depth'] == depth].copy()
            
            # Reshape data for heatmap
            heatmap_data = depth_data.pivot_table(
                index='max_iterations',
                columns='max_candidate_set_size',
                values='quality_distance',
                aggfunc='mean'
            )
            
            # Plot heatmap
            ax = axes[i]
            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt=".4f",
                cmap="YlGnBu_r",  # Reversed colormap so darker is better (lower quality distance)
                ax=ax
            )
            
            ax.set_title(f'Quality by Max Iterations and Candidate Set Size (Recursion Depth = {depth})')
            ax.set_xlabel('Max Candidate Set Size')
            ax.set_ylabel('Max Iterations')
        
        plt.tight_layout()
        output_path = os.path.join(viz_dir, "parameter_heatmaps.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
    
    def _plot_impact_by_graph_type(self, results_df, viz_dir):
        """Plot parameter impact by graph type."""
        # Add graph type column
        results_df['graph_type'] = results_df['dataset'].apply(lambda x: 
            'scale_free' if 'ba_' in x else
            'random' if 'er_' in x else
            'small_world' if 'ws_' in x else
            'regular' if 'reg_' in x else
            'real'
        )
        
        # Check if we have multiple graph types
        graph_types = results_df['graph_type'].unique()
        if len(graph_types) <= 1:
            logger.warning("Not enough graph types for comparison")
            return None
        
        plt.figure(figsize=(15, 10))
        
        # Create subplots for each parameter
        for i, param in enumerate(['max_iterations', 'max_candidate_set_size', 'max_recursion_depth']):
            plt.subplot(3, 1, i+1)
            
            # Plot parameter impact on quality by graph type
            sns.lineplot(
                data=results_df,
                x=param,
                y='quality_distance',
                hue='graph_type',
                marker='o',
                err_style='band'
            )
            
            plt.xlabel(param.replace('_', ' ').title())
            plt.ylabel('Quality Distance (lower is better)')
            plt.title(f'Impact of {param.replace("_", " ").title()} on Quality by Graph Type')
            plt.grid(True, alpha=0.3)
            plt.legend(title="Graph Type")
        
        plt.tight_layout()
        output_path = os.path.join(viz_dir, "parameter_impact_by_graph_type.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
    
    def run_optimization(self, dataset_names=None, detailed=False, target_ratio=0.3):
        """
        Run the full parameter optimization process.
        
        Args:
            dataset_names: List of dataset names to use
            detailed: Whether to use a more detailed parameter grid
            target_ratio: Target size ratio for summaries
            
        Returns:
            dict: Optimization results with best parameters
        """
        # Load datasets
        datasets = self.load_datasets(dataset_names)
        
        # Generate parameter grid
        parameter_grid = self.generate_parameter_grid(detailed)
        
        # Run parameter optimization
        results_df = self.optimize_parameters(datasets, parameter_grid, target_ratio)
        
        # Save raw results
        results_df.to_csv(os.path.join(self.results_dir, "parameter_optimization_results.csv"), index=False)
        
        # Analyze results
        analysis = self.analyze_results(results_df)
        
        # Save analysis
        with open(os.path.join(self.results_dir, "parameter_optimization_analysis.json"), 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Create visualizations
        self.visualize_results(results_df)
        
        # Create recommendations and report
        self._create_recommendations_report(analysis, results_df)
        
        return analysis
    
    def _create_recommendations_report(self, analysis, results_df):
        """Create a recommendations report based on parameter optimization."""
        report_path = os.path.join(self.results_dir, "parameter_recommendations.md")
        
        with open(report_path, 'w') as f:
            f.write("# SSumM Parameter Optimization Results\n\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall recommendations
            f.write("## Recommended Parameter Settings\n\n")
            
            f.write("### Overall Best Parameters\n\n")
            f.write("These parameters provide the best balance of quality and performance across all tested graphs:\n\n")
            f.write("```python\n")
            f.write(f"max_iterations = {analysis['overall_best']['max_iterations']}\n")
            f.write(f"max_candidate_set_size = {analysis['overall_best']['max_candidate_set_size']}\n")
            f.write(f"max_recursion_depth = {analysis['overall_best']['max_recursion_depth']}\n")
            f.write("```\n\n")
            
            # Graph type specific recommendations
            f.write("### Parameters by Graph Type\n\n")
            
            for graph_type, params in analysis['by_graph_type'].items():
                f.write(f"#### {graph_type.replace('_', ' ').title()} Graphs\n\n")
                f.write("```python\n")
                f.write(f"max_iterations = {params['max_iterations']}\n")
                f.write(f"max_candidate_set_size = {params['max_candidate_set_size']}\n")
                f.write(f"max_recursion_depth = {params['max_recursion_depth']}\n")
                f.write("```\n\n")
                f.write(f"Quality: {params['quality_distance']:.4f}, Runtime: {params['runtime_sec']:.2f}s\n\n")
            
            # Graph size specific recommendations
            f.write("### Parameters by Graph Size\n\n")
            
            for size_category, params in analysis['by_graph_size'].items():
                f.write(f"#### {size_category.replace('_', ' ').title()} Graphs\n\n")
                f.write("```python\n")
                f.write(f"max_iterations = {params['max_iterations']}\n")
                f.write(f"max_candidate_set_size = {params['max_candidate_set_size']}\n")
                f.write(f"max_recursion_depth = {params['max_recursion_depth']}\n")
                f.write("```\n\n")
                f.write(f"Quality: {params['quality_distance']:.4f}, Runtime: {params['runtime_sec']:.2f}s\n\n")
            
            # Parameter sensitivity analysis
            f.write("## Parameter Sensitivity Analysis\n\n")
            
            # Analyze impact of each parameter
            for param in ['max_iterations', 'max_candidate_set_size', 'max_recursion_depth']:
                f.write(f"### Impact of {param.replace('_', ' ').title()}\n\n")
                
                # Calculate average quality for each parameter value
                param_impact = results_df.groupby(param)['quality_distance'].mean().reset_index()
                param_impact = param_impact.sort_values(param)
                
                f.write("| Value | Average Quality |\n")
                f.write("|-------|----------------|\n")
                
                for _, row in param_impact.iterrows():
                    f.write(f"| {int(row[param])} | {row['quality_distance']:.4f} |\n")
                
                f.write("\n")
                
                # Write observations
                best_value = param_impact.loc[param_impact['quality_distance'].idxmin(), param]
                worst_value = param_impact.loc[param_impact['quality_distance'].idxmax(), param]
                
                f.write(f"- Best value: {int(best_value)}\n")
                f.write(f"- Worst value: {int(worst_value)}\n")
                
                # Trend analysis
                first_value = param_impact.iloc[0][param]
                last_value = param_impact.iloc[-1][param]
                first_quality = param_impact.iloc[0]['quality_distance']
                last_quality = param_impact.iloc[-1]['quality_distance']
                
                if first_quality > last_quality:
                    trend = "increasing this parameter generally improves quality"
                elif first_quality < last_quality:
                    trend = "increasing this parameter generally worsens quality"
                else:
                    trend = "changing this parameter has no clear trend on quality"
                
                f.write(f"- Trend: {trend}\n\n")
            
            # Conclusions and recommendations
            f.write("## Conclusions and Recommendations\n\n")
            
            # Compare overall best with type-specific bests
            f.write("### When to Use Specialized Parameters\n\n")
            
            # Scale-free graphs
            if 'scale_free' in analysis['by_graph_type']:
                scale_free_params = analysis['by_graph_type']['scale_free']
                overall_params = analysis['overall_best']
                
                quality_diff = scale_free_params['quality_distance'] - overall_params['quality_distance']
                runtime_diff = scale_free_params['runtime_sec'] - overall_params['runtime_sec']
                
                f.write("**Scale-Free Networks (e.g., social networks, web graphs):**\n\n")
                
                if quality_diff < -0.01:  # More than 1% improvement
                    f.write(f"Using specialized parameters improves quality by {-quality_diff:.1%} ")
                    f.write(f"with a runtime {runtime_diff/overall_params['runtime_sec']:.1%} difference.\n\n")
                    f.write("**Recommendation**: Use specialized parameters for scale-free networks.\n\n")
                else:
                    f.write("Using specialized parameters offers minimal improvement over the overall best parameters.\n\n")
                    f.write("**Recommendation**: Use the overall best parameters for simplicity.\n\n")
            
            # Random graphs
            if 'random' in analysis['by_graph_type']:
                random_params = analysis['by_graph_type']['random']
                overall_params = analysis['overall_best']
                
                quality_diff = random_params['quality_distance'] - overall_params['quality_distance']
                runtime_diff = random_params['runtime_sec'] - overall_params['runtime_sec']
                
                f.write("**Random Networks (e.g., Erdos-Renyi graphs):**\n\n")
                
                if quality_diff < -0.01:  # More than 1% improvement
                    f.write(f"Using specialized parameters improves quality by {-quality_diff:.1%} ")
                    f.write(f"with a runtime {runtime_diff/overall_params['runtime_sec']:.1%} difference.\n\n")
                    f.write("**Recommendation**: Use specialized parameters for random networks.\n\n")
                else:
                    f.write("Using specialized parameters offers minimal improvement over the overall best parameters.\n\n")
                    f.write("**Recommendation**: Use the overall best parameters for simplicity.\n\n")
            
            # Final summary
            f.write("### Final Parameter Recommendations\n\n")
            f.write("For most use cases, we recommend using the overall best parameters:\n\n")
            f.write("```python\n")
            f.write(f"max_iterations = {analysis['overall_best']['max_iterations']}\n")
            f.write(f"max_candidate_set_size = {analysis['overall_best']['max_candidate_set_size']}\n")
            f.write(f"max_recursion_depth = {analysis['overall_best']['max_recursion_depth']}\n")
            f.write("```\n\n")
            
            # Trade-off advice
            f.write("### Performance Trade-offs\n\n")
            
            f.write("If you need to prioritize runtime over quality:\n\n")
            
            # Find fastest parameter combination with reasonable quality
            fastest_combo = results_df.sort_values('runtime_sec').iloc[0]
            fastest_quality = fastest_combo['quality_distance']
            fastest_runtime = fastest_combo['runtime_sec']
            
            f.write("```python\n")
            f.write(f"max_iterations = {int(fastest_combo['max_iterations'])}\n")
            f.write(f"max_candidate_set_size = {int(fastest_combo['max_candidate_set_size'])}\n")
            f.write(f"max_recursion_depth = {int(fastest_combo['max_recursion_depth'])}\n")
            f.write("```\n\n")
            
            quality_diff = fastest_quality - analysis['overall_best']['quality_distance']
            runtime_diff = analysis['overall_best']['runtime_sec'] - fastest_runtime
            
            f.write(f"This reduces runtime by {runtime_diff:.2f}s ({runtime_diff/analysis['overall_best']['runtime_sec']:.1%}) ")
            f.write(f"with a quality penalty of {quality_diff:.1%}.\n\n")
            
            # Memory efficiency recommendation
            f.write("If memory efficiency is your primary concern:\n\n")
            
            f.write("```python\n")
            f.write(f"max_iterations = {min(results_df['max_iterations'].unique())}\n")
            f.write(f"max_candidate_set_size = {min(results_df['max_candidate_set_size'].unique())}\n")
            f.write(f"max_recursion_depth = {min(results_df['max_recursion_depth'].unique())}\n")
            f.write("```\n\n")
            
            f.write("This configuration minimizes memory usage but may result in lower quality summaries.\n\n")
            
            f.write("---\n\n")
            f.write("*This report was generated automatically based on parameter optimization experiments.*\n")
        
        logger.info(f"Recommendations report saved to {report_path}")
        return report_path


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Optimize SSumM algorithm parameters')
    
    # Dataset options
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing datasets')
    parser.add_argument('--datasets', nargs='+', 
                        default=None,
                        help='Names of datasets to use (None for synthetic only)')
    
    # Optimization options
    parser.add_argument('--results_dir', type=str, default='./parameter_optimization_results',
                        help='Directory to save results')
    parser.add_argument('--detailed', action='store_true',
                        help='Use detailed parameter grid (more combinations)')
    parser.add_argument('--target_ratio', type=float, default=0.3,
                        help='Target size ratio for summaries')
    
    # Quick test mode
    parser.add_argument('--quick_test', action='store_true',
                        help='Run a quick test with limited parameters')
    
    return parser.parse_args()

def run_quick_test():
    """Run a quick parameter optimization test."""
    logger.info("Running quick test mode")
    
    optimizer = ParameterOptimizer(
        data_dir="./data",
        results_dir="./quick_test_results"
    )
    
    # Create simple parameter grid
    simple_grid = [
        {'max_iterations': 10, 'max_candidate_set_size': 200, 'max_recursion_depth': 5},
        {'max_iterations': 20, 'max_candidate_set_size': 200, 'max_recursion_depth': 5},
        {'max_iterations': 10, 'max_candidate_set_size': 500, 'max_recursion_depth': 5},
        {'max_iterations': 20, 'max_candidate_set_size': 500, 'max_recursion_depth': 5}
    ]
    
    # Use only small synthetic datasets
    datasets = {
        "karate": nx.karate_club_graph(),
        "ba_50_2": nx.barabasi_albert_graph(n=50, m=2, seed=42)
    }
    
    # Run optimization with limited scope
    results_df = optimizer.optimize_parameters(datasets, simple_grid, target_ratio=0.3)
    
    # Analyze and visualize
    analysis = optimizer.analyze_results(results_df)
    optimizer.visualize_results(results_df)
    optimizer._create_recommendations_report(analysis, results_df)
    
    logger.info("Quick test completed successfully")
    return analysis

def main():
    """Main function to run parameter optimization."""
    args = parse_arguments()
    
    if args.quick_test:
        run_quick_test()
        return
    
    optimizer = ParameterOptimizer(
        data_dir=args.data_dir,
        results_dir=args.results_dir
    )
    
    # Run full optimization
    optimizer.run_optimization(
        dataset_names=args.datasets,
        detailed=args.detailed,
        target_ratio=args.target_ratio
    )
    
    logger.info("Parameter optimization completed successfully")

if __name__ == "__main__":
    main()