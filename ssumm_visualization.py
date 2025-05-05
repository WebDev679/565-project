#!/usr/bin/env python3
"""
Advanced visualization tools for SSumM results and comparisons.
Generates detailed plots, diagrams, and visual analyses.
"""

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

# Import our implementations
from graph_summary import GraphSummary
from ssumm_algorithm import SSumM

class SSumMVisualizer:
    """Advanced visualization tools for SSumM algorithm results."""
    
    def __init__(self, output_dir: str = "./ssumm_visualizations"):
        """Initialize the visualizer with output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Color scheme
        self.colors = {
            'original': '#1f77b4',    # blue
            'summary': '#ff7f0e',     # orange
            'merged': '#2ca02c',      # green
            'superedge': '#d62728',   # red
            'sparsified': '#9467bd',  # purple
            'reconstructed': '#8c564b' # brown
        }
    
    def visualize_algorithm_convergence(self, convergence_data: Dict[str, List], 
                                      dataset_name: str,
                                      output_path: Optional[str] = None):
        """
        Visualize SSumM algorithm convergence across iterations.
        
        Args:
            convergence_data: Dictionary containing iteration data
            dataset_name: Name of the dataset
            output_path: Custom output path (optional)
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"{dataset_name}_convergence.png")
        
        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Ensure iteration data exists
        iterations = convergence_data.get('iteration', list(range(len(convergence_data.get('reconstruction_error_l1', [])))))
        
        # Plot 1: Reconstruction Error convergence
        ax = axes[0, 0]
        if 'reconstruction_error_l1' in convergence_data:
            ax.plot(iterations, convergence_data['reconstruction_error_l1'], 
                   marker='o', linestyle='-', linewidth=2, markersize=8,
                   color=self.colors['original'], label='L1 Error')
        if 'reconstruction_error_l2' in convergence_data:
            ax.plot(iterations, convergence_data['reconstruction_error_l2'], 
                   marker='s', linestyle='-', linewidth=2, markersize=8,
                   color=self.colors['summary'], label='L2 Error')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Reconstruction Error')
        ax.set_title('Error Convergence')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 2: Size reduction over iterations
        ax = axes[0, 1]
        if 'size_in_bits' in convergence_data:
            ax.plot(iterations, convergence_data['size_in_bits'], 
                   marker='D', linestyle='-', linewidth=2, markersize=8,
                   color=self.colors['merged'], label='Summary Size')
        
        if 'target_size' in convergence_data and convergence_data['target_size']:
            ax.axhline(y=convergence_data['target_size'][0], color='red', 
                      linestyle='--', linewidth=2, label='Target Size')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Size (bits)')
        ax.set_title('Size Reduction Progress')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 3: Number of supernodes and superedges
        ax = axes[1, 0]
        if 'num_supernodes' in convergence_data:
            ax.plot(iterations, convergence_data['num_supernodes'], 
                   marker='^', linestyle='-', linewidth=2, markersize=8,
                   color=self.colors['sparsified'], label='Supernodes')
        if 'num_superedges' in convergence_data:
            ax.plot(iterations, convergence_data['num_superedges'], 
                   marker='v', linestyle='-', linewidth=2, markersize=8,
                   color=self.colors['superedge'], label='Superedges')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Count')
        ax.set_title('Supernode/Superedge Evolution')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 4: Quality metric (error/size ratio)
        ax = axes[1, 1]
        if 'reconstruction_error_l1' in convergence_data and 'size_in_bits' in convergence_data:
            quality_ratio = np.array(convergence_data['reconstruction_error_l1']) / np.array(convergence_data['size_in_bits'])
            ax.plot(iterations, quality_ratio, 
                   marker='o', linestyle='-', linewidth=2, markersize=8,
                   color=self.colors['reconstructed'], label='Error/Size Ratio')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Quality Metric (Error/Size)')
        ax.set_title('Quality Evolution')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.suptitle(f'SSumM Algorithm Convergence - {dataset_name}', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_summary_structure(self, summary: GraphSummary, 
                                  dataset_name: str,
                                  output_path: Optional[str] = None):
        """
        Create detailed visualization of summary graph structure.
        
        Args:
            summary: GraphSummary object
            dataset_name: Name of the dataset
            output_path: Custom output path (optional)
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"{dataset_name}_summary_structure.png")
        
        # Convert to NetworkX graph
        G_summary = summary.to_networkx_graph()
        
        # Create large figure with multiple panels
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.5], width_ratios=[1, 1])
        
        # Main summary graph visualization
        ax1 = fig.add_subplot(gs[0:2, :])
        
        # Calculate node sizes and positions
        node_sizes = []
        node_colors = []
        for node in G_summary.nodes():
            size = G_summary.nodes[node]['size']
            node_sizes.append(size * 500)  # Scale for visibility
            # Color nodes by cluster size
            node_colors.append(size)
        
        # Use spring layout for main visualization
        pos = nx.spring_layout(G_summary, seed=42, k=1/np.sqrt(G_summary.number_of_nodes()), iterations=50)
        
        # Create colormap for node coloring
        cmap = plt.cm.viridis
        
        # Draw the network
        nodes = nx.draw_networkx_nodes(G_summary, pos, 
                                     node_size=node_sizes,
                                     node_color=node_colors,
                                     cmap=cmap,
                                     alpha=0.8,
                                     ax=ax1)
        
        # Draw edges with varying widths
        edge_widths = []
        edge_colors = []
        for u, v in G_summary.edges():
            weight = G_summary[u][v]['weight']
            edge_widths.append(weight / 5)
            edge_colors.append(weight)
        
        edges = nx.draw_networkx_edges(G_summary, pos,
                                     width=edge_widths,
                                     edge_color=edge_colors,
                                     edge_cmap=plt.cm.plasma,
                                     alpha=0.6,
                                     ax=ax1)
        
        # Add colorbar for nodes
        cbar = plt.colorbar(nodes, ax=ax1, orientation='horizontal', pad=0.05)
        cbar.set_label('Supernode Size (# original nodes)', fontsize=12)
        
        ax1.set_title(f'Summary Graph Structure - {dataset_name}', fontsize=16)
        ax1.set_aspect('equal')
        
        # Degree distribution histogram
        ax2 = fig.add_subplot(gs[2, 0])
        degrees = [d for n, d in G_summary.degree()]
        ax2.hist(degrees, bins=20, color=self.colors['sparsified'], alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Supernode Degree')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Supernode Degree Distribution')
        ax2.grid(alpha=0.3)
        
        # Edge weight distribution
        ax3 = fig.add_subplot(gs[2, 1])
        edge_weights = [G_summary[u][v]['weight'] for u, v in G_summary.edges()]
        ax3.hist(edge_weights, bins=20, color=self.colors['superedge'], alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Edge Weight')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Superedge Weight Distribution')
        ax3.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_error_distribution(self, original_adj: np.ndarray, 
                                   reconstructed_adj: np.ndarray,
                                   dataset_name: str,
                                   output_path: Optional[str] = None):
        """
        Visualize the distribution of reconstruction errors.
        
        Args:
            original_adj: Original adjacency matrix
            reconstructed_adj: Reconstructed adjacency matrix
            dataset_name: Name of the dataset
            output_path: Custom output path (optional)
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"{dataset_name}_error_distribution.png")
        
        # Calculate error matrix
        error_matrix = np.abs(original_adj - reconstructed_adj)
        
        # Create figure with multiple visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Error histogram
        ax = axes[0, 0]
        error_values = error_matrix[error_matrix > 0]
        ax.hist(error_values, bins=50, color=self.colors['reconstructed'], alpha=0.7, edgecolor='black')
        ax.set_xlabel('Error Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Error Distribution')
        ax.grid(alpha=0.3)
        
        # Error heatmap (sample)
        ax = axes[0, 1]
        n = min(100, error_matrix.shape[0])  # Limit for visibility
        error_sample = error_matrix[:n, :n]
        im = ax.imshow(error_sample, cmap='YlOrRd', aspect='auto')
        ax.set_title(f'Error Heatmap (first {n}x{n})')
        plt.colorbar(im, ax=ax)
        
        # Cumulative error distribution
        ax = axes[0, 2]
        error_sorted = np.sort(error_matrix.flatten())
        cumsum = np.cumsum(error_sorted)
        ax.plot(error_sorted, cumsum / cumsum[-1], linewidth=2, color=self.colors['original'])
        ax.set_xlabel('Error Value')
        ax.set_ylabel('Cumulative Proportion')
        ax.set_title('Cumulative Error Distribution')
        ax.grid(alpha=0.3)
        
        # Error by node degree
        ax = axes[1, 0]
        # Assume original graph structure
        original_degrees = np.sum(original_adj > 0, axis=1)
        node_errors = np.sum(error_matrix, axis=1)
        ax.scatter(original_degrees, node_errors, alpha=0.5, color=self.colors['sparsified'])
        ax.set_xlabel('Original Node Degree')
        ax.set_ylabel('Total Error for Node')
        ax.set_title('Error vs Node Degree')
        ax.grid(alpha=0.3)
        
        # Error by predicted probability
        ax = axes[1, 1]
        # Show relationship between predicted values and error
        predicted_values = reconstructed_adj[reconstructed_adj > 0]
        actual_values = original_adj[reconstructed_adj > 0]
        errors_by_prediction = np.abs(predicted_values - actual_values)
        
        ax.scatter(predicted_values, errors_by_prediction, alpha=0.5, color=self.colors['merged'])
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Absolute Error')
        ax.set_title('Error vs Predicted Probability')
        ax.grid(alpha=0.3)
        
        # Error statistics panel
        ax = axes[1, 2]
        ax.axis('off')
        
        # Calculate error statistics
        total_error = np.sum(error_matrix)
        mean_error = np.mean(error_matrix)
        max_error = np.max(error_matrix)
        nonzero_errors = error_matrix[error_matrix > 0]
        
        stats_text = f"Error Statistics:\n\n"
        stats_text += f"Total Error: {total_error:.6f}\n"
        stats_text += f"Mean Error: {mean_error:.6f}\n"
        stats_text += f"Max Error: {max_error:.6f}\n"
        stats_text += f"# Non-zero Errors: {len(nonzero_errors)}\n"
        stats_text += f"% Elements with Error: {len(nonzero_errors) / error_matrix.size * 100:.2f}%\n"
        
        ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, 
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle(f'Reconstruction Error Analysis - {dataset_name}', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_algorithm_comparison_dashboard(self, results_df: pd.DataFrame,
                                             dataset_name: str,
                                             output_path: Optional[str] = None):
        """
        Create a comprehensive dashboard comparing algorithm performance.
        
        Args:
            results_df: DataFrame with algorithm comparison results
            dataset_name: Name of the dataset
            output_path: Custom output path (optional)
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"{dataset_name}_algorithm_comparison_dashboard.png")
        
        # Create figure with multiple panels
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
        
        # Define algorithm colors
        algo_colors = {
            'SSumM': self.colors['original'],
            'k-Gs': self.colors['summary'],
            'S2L': self.colors['merged'],
            'SAA-Gs': self.colors['superedge'],
            'SAA-Gs-linear': self.colors['sparsified']
        }
        
        # Plot 1: Error vs Size Trade-off
        ax1 = fig.add_subplot(gs[0, :])
        for algo in results_df['algorithm'].unique():
            algo_data = results_df[results_df['algorithm'] == algo]
            if not algo_data.empty and 'relative_size' in algo_data.columns and 'reconstruction_error_l1' in algo_data.columns:
                ax1.scatter(algo_data['relative_size'], algo_data['reconstruction_error_l1'],
                          label=algo, color=algo_colors.get(algo, 'gray'), s=100, alpha=0.7)
                ax1.plot(algo_data['relative_size'], algo_data['reconstruction_error_l1'],
                        color=algo_colors.get(algo, 'gray'), alpha=0.5)
        
        ax1.set_xlabel('Relative Size (Summary/Original)', fontsize=12)
        ax1.set_ylabel('L1 Reconstruction Error', fontsize=12)
        ax1.set_title('Error vs Size Trade-off', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)
        
        # Plot 2: Runtime Comparison
        ax2 = fig.add_subplot(gs[1, 0])
        runtime_data = []
        for algo in results_df['algorithm'].unique():
            algo_data = results_df[results_df['algorithm'] == algo]
            if not algo_data.empty and 'runtime' in algo_data.columns:
                runtime_data.append({
                    'algorithm': algo,
                    'avg_runtime': algo_data['runtime'].mean(),
                    'max_runtime': algo_data['runtime'].max()
                })
        
        if runtime_data:
            runtime_df = pd.DataFrame(runtime_data)
            bars = ax2.bar(runtime_df['algorithm'], runtime_df['avg_runtime'], 
                          color=[algo_colors.get(algo, 'gray') for algo in runtime_df['algorithm']],
                          alpha=0.7)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}s', ha='center', va='bottom', fontsize=10)
            
            ax2.set_xlabel('Algorithm', fontsize=12)
            ax2.set_ylabel('Average Runtime (seconds)', fontsize=12)
            ax2.set_title('Runtime Comparison', fontsize=14)
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Quality Box Plot
        ax3 = fig.add_subplot(gs[1, 1])
        quality_data = []
        for algo in results_df['algorithm'].unique():
            algo_data = results_df[results_df['algorithm'] == algo]
            if not algo_data.empty and 'reconstruction_error_l1' in algo_data.columns:
                quality_data.extend([{'algorithm': algo, 'error': err} 
                                   for err in algo_data['reconstruction_error_l1'] if not pd.isna(err)])
        
        if quality_data:
            quality_df = pd.DataFrame(quality_data)
            box_colors = [algo_colors.get(algo, 'gray') for algo in quality_df['algorithm'].unique()]
            sns.boxplot(data=quality_df, x='algorithm', y='error', ax=ax3, palette=box_colors)
            ax3.set_xlabel('Algorithm', fontsize=12)
            ax3.set_ylabel('L1 Reconstruction Error', fontsize=12)
            ax3.set_title('Error Distribution by Algorithm', fontsize=14)
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Efficiency Metric (Error/Runtime)
        ax4 = fig.add_subplot(gs[1, 2])
        efficiency_data = []
        for algo in results_df['algorithm'].unique():
            algo_data = results_df[results_df['algorithm'] == algo]
            if not algo_data.empty and 'reconstruction_error_l1' in algo_data.columns and 'runtime' in algo_data.columns:
                # Calculate efficiency metric: lower is better
                algo_data['efficiency'] = algo_data['reconstruction_error_l1'] * algo_data['runtime']
                efficiency_data.append({
                    'algorithm': algo,
                    'efficiency': algo_data['efficiency'].mean()
                })
        
        if efficiency_data:
            efficiency_df = pd.DataFrame(efficiency_data)
            # Sort by efficiency (lower is better)
            efficiency_df = efficiency_df.sort_values('efficiency', ascending=False)
            
            bars = ax4.barh(efficiency_df['algorithm'], efficiency_df['efficiency'],
                          color=[algo_colors.get(algo, 'gray') for algo in efficiency_df['algorithm']],
                          alpha=0.7)
            
            # Add value labels on bars
            for bar in bars:
                width = bar.get_width()
                ax4.text(width, bar.get_y() + bar.get_height()/2.,
                        f'{width:.6f}', ha='left', va='center', fontsize=10)
            
            ax4.set_xlabel('Efficiency Metric (Error × Runtime)', fontsize=12)
            ax4.set_ylabel('Algorithm', fontsize=12)
            ax4.set_title('Algorithm Efficiency', fontsize=14)
            ax4.grid(axis='x', alpha=0.3)
        
        # Plot 5: Scalability Analysis
        ax5 = fig.add_subplot(gs[2, 0])
        if 'target_size_ratio' in results_df.columns and 'num_supernodes' in results_df.columns:
            for algo in results_df['algorithm'].unique():
                algo_data = results_df[results_df['algorithm'] == algo]
                if not algo_data.empty:
                    ax5.scatter(algo_data['target_size_ratio'], algo_data['num_supernodes'],
                              label=algo, color=algo_colors.get(algo, 'gray'), s=80, alpha=0.7)
                    ax5.plot(algo_data['target_size_ratio'], algo_data['num_supernodes'],
                            color=algo_colors.get(algo, 'gray'), alpha=0.5)
            
            ax5.set_xlabel('Target Size Ratio', fontsize=12)
            ax5.set_ylabel('Number of Supernodes', fontsize=12)
            ax5.set_title('Supernode Count vs Target Size', fontsize=14)
            ax5.legend(fontsize=10)
            ax5.grid(alpha=0.3)
        
        # Plot 6: Size Reduction Factor
        ax6 = fig.add_subplot(gs[2, 1])
        if 'relative_size' in results_df.columns:
            reduction_data = []
            for algo in results_df['algorithm'].unique():
                algo_data = results_df[results_df['algorithm'] == algo]
                if not algo_data.empty:
                    # Calculate average size reduction
                    avg_reduction = (1 - algo_data['relative_size'].mean()) * 100
                    reduction_data.append({
                        'algorithm': algo,
                        'reduction': avg_reduction
                    })
            
            if reduction_data:
                reduction_df = pd.DataFrame(reduction_data)
                bars = ax6.bar(reduction_df['algorithm'], reduction_df['reduction'],
                              color=[algo_colors.get(algo, 'gray') for algo in reduction_df['algorithm']],
                              alpha=0.7)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax6.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
                
                ax6.set_xlabel('Algorithm', fontsize=12)
                ax6.set_ylabel('Average Size Reduction (%)', fontsize=12)
                ax6.set_title('Size Reduction Performance', fontsize=14)
                ax6.tick_params(axis='x', rotation=45)
                ax6.grid(axis='y', alpha=0.3)
        
        # Plot 7: Summary Statistics
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        # Compile summary statistics
        summary_text = f"Algorithm Comparison Summary - {dataset_name}\n\n"
        
        for algo in results_df['algorithm'].unique():
            algo_data = results_df[results_df['algorithm'] == algo]
            if not algo_data.empty:
                avg_error = algo_data['reconstruction_error_l1'].mean() if 'reconstruction_error_l1' in algo_data.columns else np.nan
                avg_runtime = algo_data['runtime'].mean() if 'runtime' in algo_data.columns else np.nan
                avg_size_reduction = (1 - algo_data['relative_size'].mean()) * 100 if 'relative_size' in algo_data.columns else np.nan
                
                summary_text += f"{algo}:\n"
                summary_text += f"  Avg Error: {avg_error:.6f}\n" if not np.isnan(avg_error) else "  Avg Error: N/A\n"
                summary_text += f"  Avg Runtime: {avg_runtime:.2f}s\n" if not np.isnan(avg_runtime) else "  Avg Runtime: N/A\n"
                summary_text += f"  Avg Reduction: {avg_size_reduction:.1f}%\n\n" if not np.isnan(avg_size_reduction) else "  Avg Reduction: N/A\n\n"
        
        ax7.text(0.1, 0.9, summary_text, transform=ax7.transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle(f'Algorithm Comparison Dashboard - {dataset_name}', fontsize=18, y=0.98)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_algorithm_strength_weakness_map(self, results_df: pd.DataFrame,
                                             dataset_name: str,
                                             output_path: Optional[str] = None):
        """
        Create a visual map of algorithm strengths and weaknesses.
        
        Args:
            results_df: DataFrame with algorithm comparison results
            dataset_name: Name of the dataset
            output_path: Custom output path (optional)
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"{dataset_name}_strength_weakness_map.png")
        
        # Metrics to evaluate
        metrics = {
            'compactness': 'relative_size',
            'accuracy': 'reconstruction_error_l1',
            'speed': 'runtime',
            'efficiency': 'efficiency'
        }
        
        # Prepare data for radar chart
        algorithms = results_df['algorithm'].unique()
        radar_data = {}
        
        for algo in algorithms:
            algo_data = results_df[results_df['algorithm'] == algo]
            scores = {}
            
            # Normalize each metric to 0-1 scale (1 is best)
            for metric_name, column_name in metrics.items():
                if column_name in algo_data.columns:
                    if metric_name in ['compactness', 'speed']:
                        # Lower is better, invert
                        value = 1 - (algo_data[column_name].mean() / results_df[column_name].max())
                    else:
                        # Lower is better
                        value = 1 - (algo_data[column_name].mean() / results_df[column_name].max())
                else:
                    value = 0
                scores[metric_name] = value
            
            radar_data[algo] = scores
        
        # Create radar chart
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Number of variables
        categories = list(metrics.keys())
        num_vars = len(categories)
        
        # Compute angle for each axis
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]  # Close the loop
        
        # Plot data for each algorithm
        for algo, scores in radar_data.items():
            values = [scores.get(cat, 0) for cat in categories]
            values += values[:1]  # Close the loop
            
            color = {
                'SSumM': self.colors['original'],
                'k-Gs': self.colors['summary'],
                'S2L': self.colors['merged'],
                'SAA-Gs': self.colors['superedge']
            }.get(algo, 'gray')
            
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=algo, color=color)
            ax.fill(angles, values, alpha=0.1, color=color)
        
        # Customize the chart
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.title(f'Algorithm Performance Profile - {dataset_name}', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """
    Example usage of the SSumM visualization tools.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="SSumM Visualization Tools")
    parser.add_argument("--results", type=str, default="./ssumm_results",
                        help="Directory containing SSumM results")
    parser.add_argument("--output", type=str, default="./ssumm_visualizations",
                        help="Output directory for visualizations")
    parser.add_argument("--dataset", type=str, help="Specific dataset to visualize")
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = SSumMVisualizer(output_dir=args.output)
    
    # Check for comparison results
    results_files = []
    if os.path.exists(args.results):
        for file in os.listdir(args.results):
            if file.endswith("_comparison.csv"):
                results_files.append(file)
    
    # Process each dataset comparison
    for file in results_files:
        dataset_name = file.replace("_comparison.csv", "")
        
        # Skip if specific dataset requested and this isn't it
        if args.dataset and dataset_name != args.dataset:
            continue
        
        print(f"Processing visualization for {dataset_name}")
        
        # Load results
        results_df = pd.read_csv(os.path.join(args.results, file))
        
        # Create dashboard
        visualizer.create_algorithm_comparison_dashboard(
            results_df, 
            dataset_name
        )
        
        # Create strength/weakness map
        visualizer.create_algorithm_strength_weakness_map(
            results_df,
            dataset_name
        )
        
        print(f"Visualizations created for {dataset_name}")
    
    print("All visualizations complete!")

if __name__ == "__main__":
    main()