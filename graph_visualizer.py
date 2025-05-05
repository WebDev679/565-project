import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pickle
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

class GraphVisualizer:
    """
    A class for visualizing graph structures, summaries, and comparison metrics.
    """
    
    def __init__(self, output_dir="./visualizations"):
        """
        Initialize the visualizer with output directory.
        
        Args:
            output_dir: Directory where visualizations will be saved
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created visualization directory: {output_dir}")
        
        # Set consistent styling for all plots
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Define custom color palettes
        self.node_colors = {
            'original': '#1f77b4',    # blue
            'summary': '#ff7f0e',     # orange
            'merged': '#2ca02c',      # green
            'superedge': '#d62728',   # red
            'sparsified': '#9467bd',  # purple
            'reconstructed': '#8c564b' # brown
        }
        
        # Enable LaTeX rendering if available
        try:
            plt.rcParams['text.usetex'] = True
            plt.rcParams['font.family'] = 'serif'
        except:
            plt.rcParams['font.family'] = 'serif'
    
    def visualize_original_graph(self, G, dataset_name, layout='spring', 
                                max_nodes=1000, plot_labels=False, 
                                node_size=None, fig_size=(12, 10)):
        """
        Visualize the structure of the original graph.
        
        Args:
            G: NetworkX graph to visualize
            dataset_name: Name of the dataset
            layout: Graph layout algorithm ('spring', 'kamada_kawai', 'circular', 'spectral')
            max_nodes: Maximum number of nodes to visualize (for large graphs)
            plot_labels: Whether to plot node labels
            node_size: Size of nodes in visualization (if None, will be computed based on degree)
            fig_size: Size of the figure
            
        Returns:
            The path to the saved visualization
        """
        # Skip if graph is too large
        if G.number_of_nodes() > max_nodes:
            print(f"Graph too large to visualize ({G.number_of_nodes()} nodes). "
                  f"Sampling {max_nodes} nodes...")
            nodes = list(G.nodes())
            sampled_nodes = np.random.choice(nodes, max_nodes, replace=False)
            G = G.subgraph(sampled_nodes).copy()
        
        print(f"Visualizing original graph for {dataset_name} with {G.number_of_nodes()} nodes "
              f"and {G.number_of_edges()} edges...")
        
        plt.figure(figsize=fig_size)
        
        # Compute layout
        if layout == 'spring':
            pos = nx.spring_layout(G, seed=42)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'spectral':
            pos = nx.spectral_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42)
        
        # Compute node sizes based on degree if not provided
        if node_size is None:
            degrees = dict(G.degree())
            min_degree, max_degree = min(degrees.values()), max(degrees.values())
            node_size = {n: 20 + 100 * (degrees[n] - min_degree) / (max_degree - min_degree + 1) 
                        for n in G.nodes()}
            node_sizes = [node_size[n] for n in G.nodes()]
        else:
            node_sizes = node_size
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, 
                               node_color=self.node_colors['original'],
                               node_size=node_sizes,
                               alpha=0.7)
        
        nx.draw_networkx_edges(G, pos, 
                               edge_color='gray',
                               width=0.5,
                               alpha=0.5)
        
        if plot_labels and G.number_of_nodes() <= 100:
            nx.draw_networkx_labels(G, pos, font_size=10)
        
        # Add title and remove axes
        plt.title(f"Original Graph Structure - {dataset_name}\n"
                 f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        plt.axis('off')
        
        # Save and show
        output_path = os.path.join(self.output_dir, f"{dataset_name}_original_graph.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization to {output_path}")
        return output_path
    
    def visualize_summary_graph(self, G_summary, dataset_name, 
                               original_graph=None, supernode_info=None,
                               layout='spring', fig_size=(14, 10),
                               show_supernode_size=True):
        """
        Visualize the summary graph structure.
        
        Args:
            G_summary: NetworkX graph representing the summary graph
            dataset_name: Name of the dataset
            original_graph: Original graph (if available)
            supernode_info: Dictionary mapping supernode ID to list of original nodes
            layout: Graph layout algorithm
            fig_size: Size of the figure
            show_supernode_size: Whether to scale nodes by the number of original nodes they contain
            
        Returns:
            The path to the saved visualization
        """
        print(f"Visualizing summary graph for {dataset_name} with {G_summary.number_of_nodes()} supernodes "
              f"and {G_summary.number_of_edges()} superedges...")
        
        plt.figure(figsize=fig_size)
        
        # Compute layout
        if layout == 'spring':
            pos = nx.spring_layout(G_summary, seed=42)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G_summary)
        elif layout == 'circular':
            pos = nx.circular_layout(G_summary)
        elif layout == 'spectral':
            pos = nx.spectral_layout(G_summary)
        else:
            pos = nx.spring_layout(G_summary, seed=42)
        
        # Compute node sizes based on the number of original nodes in each supernode
        if supernode_info and show_supernode_size:
            node_sizes = [100 + 10 * len(supernode_info[n]) for n in G_summary.nodes()]
        else:
            node_sizes = 300
        
        # Compute edge weights if available
        if nx.get_edge_attributes(G_summary, 'weight'):
            edge_weights = [G_summary[u][v]['weight'] * 2 for u, v in G_summary.edges()]
        else:
            edge_weights = 2.0
        
        # Draw the summary graph
        nx.draw_networkx_nodes(G_summary, pos, 
                               node_color=self.node_colors['summary'],
                               node_size=node_sizes,
                               alpha=0.8)
        
        nx.draw_networkx_edges(G_summary, pos, 
                               edge_color=self.node_colors['superedge'],
                               width=edge_weights,
                               alpha=0.7)
        
        if G_summary.number_of_nodes() <= 50:
            nx.draw_networkx_labels(G_summary, pos, font_size=10)
            
            # Add edge labels for weights if available and if few enough edges
            if G_summary.number_of_edges() <= 30 and nx.get_edge_attributes(G_summary, 'weight'):
                edge_labels = {(u, v): f"{G_summary[u][v]['weight']:.1f}" 
                              for u, v in G_summary.edges()}
                nx.draw_networkx_edge_labels(G_summary, pos, edge_labels=edge_labels, font_size=8)
        
        # Add compression ratio information
        if original_graph:
            compression_ratio = G_summary.number_of_nodes() / original_graph.number_of_nodes()
            edge_ratio = G_summary.number_of_edges() / original_graph.number_of_edges()
            title = (f"Summary Graph - {dataset_name}\n"
                    f"{G_summary.number_of_nodes()} supernodes ({compression_ratio:.2%} of original), "
                    f"{G_summary.number_of_edges()} superedges ({edge_ratio:.2%} of original)")
        else:
            title = (f"Summary Graph - {dataset_name}\n"
                    f"{G_summary.number_of_nodes()} supernodes, {G_summary.number_of_edges()} superedges")
        
        plt.title(title)
        plt.axis('off')
        
        # Add a legend for supernode sizes if applicable
        if supernode_info and show_supernode_size:
            sizes = [len(nodes) for nodes in supernode_info.values()]
            min_size, max_size = min(sizes), max(sizes)
            
            # Create legend with sample sizes
            size_markers = []
            size_labels = []
            
            # Create 3 sample sizes
            sample_sizes = [min_size, (min_size + max_size) // 2, max_size]
            for size in sample_sizes:
                marker_size = 100 + 10 * size
                size_markers.append(Line2D([0], [0], marker='o', color=self.node_colors['summary'],
                                         markersize=np.sqrt(marker_size/100), linestyle=''))
                size_labels.append(f"{size} nodes")
            
            plt.legend(size_markers, size_labels, title="Supernode Content", 
                      loc="upper right", frameon=True)
        
        # Save and show
        output_path = os.path.join(self.output_dir, f"{dataset_name}_summary_graph.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization to {output_path}")
        return output_path
    
    def visualize_graph_comparison(self, G_original, G_summary, G_reconstructed, 
                                  dataset_name, supernode_info=None,
                                  algorithm_name="SSumM", layout='spring', 
                                  fig_size=(20, 10)):
        """
        Create a side-by-side comparison of original, summary, and reconstructed graphs.
        
        Args:
            G_original: Original NetworkX graph
            G_summary: Summary graph
            G_reconstructed: Reconstructed graph
            dataset_name: Name of the dataset
            supernode_info: Dictionary mapping supernode ID to list of original nodes
            algorithm_name: Name of the summarization algorithm
            layout: Graph layout algorithm
            fig_size: Size of the figure
            
        Returns:
            The path to the saved visualization
        """
        print(f"Creating comparison visualization for {dataset_name} using {algorithm_name}...")
        
        if G_original.number_of_nodes() > 1000:
            print("Original graph too large for visualization. Sampling 1000 nodes...")
            nodes = list(G_original.nodes())
            sampled_nodes = np.random.choice(nodes, 1000, replace=False)
            G_original = G_original.subgraph(sampled_nodes).copy()
            
            # Adjust reconstructed graph to match
            G_reconstructed = G_reconstructed.subgraph(sampled_nodes).copy()
        
        fig, axes = plt.subplots(1, 3, figsize=fig_size)
        
        # Determine a common layout based on the original graph
        if layout == 'spring':
            pos_original = nx.spring_layout(G_original, seed=42)
        elif layout == 'kamada_kawai':
            pos_original = nx.kamada_kawai_layout(G_original)
        elif layout == 'circular':
            pos_original = nx.circular_layout(G_original)
        elif layout == 'spectral':
            pos_original = nx.spectral_layout(G_original)
        else:
            pos_original = nx.spring_layout(G_original, seed=42)
        
        # Calculate positions for summary graph
        if layout == 'spring':
            pos_summary = nx.spring_layout(G_summary, seed=42)
        else:
            pos_summary = nx.spring_layout(G_summary, seed=42)  # Fallback for summary graph
        
        # 1. Original Graph
        ax = axes[0]
        nx.draw_networkx_nodes(G_original, pos_original, 
                              node_color=self.node_colors['original'],
                              node_size=30,
                              alpha=0.7,
                              ax=ax)
        
        nx.draw_networkx_edges(G_original, pos_original, 
                              edge_color='gray',
                              width=0.5,
                              alpha=0.4,
                              ax=ax)
        
        ax.set_title(f"Original Graph\n{G_original.number_of_nodes()} nodes, {G_original.number_of_edges()} edges")
        ax.axis('off')
        
        # 2. Summary Graph
        ax = axes[1]
        
        # Compute node sizes based on the number of original nodes in each supernode
        if supernode_info:
            node_sizes = [100 + 5 * len(supernode_info[n]) for n in G_summary.nodes()]
        else:
            node_sizes = 200
        
        nx.draw_networkx_nodes(G_summary, pos_summary, 
                              node_color=self.node_colors['summary'],
                              node_size=node_sizes,
                              alpha=0.8,
                              ax=ax)
        
        # Compute edge weights if available
        if nx.get_edge_attributes(G_summary, 'weight'):
            edge_weights = [G_summary[u][v]['weight'] * 1.5 for u, v in G_summary.edges()]
        else:
            edge_weights = 1.5
        
        nx.draw_networkx_edges(G_summary, pos_summary, 
                              edge_color=self.node_colors['superedge'],
                              width=edge_weights,
                              alpha=0.7,
                              ax=ax)
        
        if G_summary.number_of_nodes() <= 50:
            nx.draw_networkx_labels(G_summary, pos_summary, font_size=8, ax=ax)
        
        compression_ratio = G_summary.number_of_nodes() / G_original.number_of_nodes()
        edge_ratio = G_summary.number_of_edges() / G_original.number_of_edges()
        
        ax.set_title(f"Summary Graph ({algorithm_name})\n"
                   f"{G_summary.number_of_nodes()} supernodes ({compression_ratio:.2%}), "
                   f"{G_summary.number_of_edges()} superedges ({edge_ratio:.2%})")
        ax.axis('off')
        
        # 3. Reconstructed Graph
        ax = axes[2]
        
        # Calculate error for edges
        original_adj = nx.to_numpy_array(G_original)
        reconstructed_adj = nx.to_numpy_array(G_reconstructed)
        edge_errors = np.abs(original_adj - reconstructed_adj)
        
        # Determine edge colors based on error
        error_cmap = plt.cm.YlOrRd
        max_error = np.max(edge_errors)
        
        # Draw reconstructed graph edges with color based on error
        for u, v in G_reconstructed.edges():
            error_val = edge_errors[u, v]
            error_color = error_cmap(error_val / max_error if max_error > 0 else 0)
            nx.draw_networkx_edges(G_reconstructed, pos_original, 
                                  edgelist=[(u, v)],
                                  edge_color=[error_color],
                                  width=1.0,
                                  alpha=0.7,
                                  ax=ax)
        
        # Draw nodes
        nx.draw_networkx_nodes(G_reconstructed, pos_original, 
                              node_color=self.node_colors['reconstructed'],
                              node_size=30,
                              alpha=0.7,
                              ax=ax)
        
        # Create colorbar for edge errors
        sm = plt.cm.ScalarMappable(cmap=error_cmap, norm=plt.Normalize(0, max_error))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Reconstruction Error')
        
        # Calculate reconstruction error
        l1_error = np.sum(edge_errors) / (G_original.number_of_nodes() * (G_original.number_of_nodes() - 1))
        l2_error = np.sqrt(np.sum(edge_errors ** 2)) / (G_original.number_of_nodes() * (G_original.number_of_nodes() - 1))
        
        ax.set_title(f"Reconstructed Graph\n"
                   f"L1 Error: {l1_error:.6f}, L2 Error: {l2_error:.6f}")
        ax.axis('off')
        
        # Overall title
        plt.suptitle(f"Graph Summarization Comparison - {dataset_name} ({algorithm_name})", 
                    fontsize=16, y=0.98)
        
        # Save and show
        output_path = os.path.join(self.output_dir, f"{dataset_name}_{algorithm_name}_comparison.png")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved comparison visualization to {output_path}")
        return output_path
    
    def visualize_error_distribution(self, original_adj, reconstructed_adj, 
                                    dataset_name, algorithm_name="SSumM"):
        """
        Visualize the distribution of reconstruction errors.
        
        Args:
            original_adj: Adjacency matrix of the original graph
            reconstructed_adj: Adjacency matrix of the reconstructed graph
            dataset_name: Name of the dataset
            algorithm_name: Name of the summarization algorithm
            
        Returns:
            The path to the saved visualization
        """
        print(f"Visualizing error distribution for {dataset_name} using {algorithm_name}...")
        
        # Calculate absolute errors
        error_matrix = np.abs(original_adj - reconstructed_adj)
        
        # Convert to 1D array for histogram (ignoring diagonal)
        n = original_adj.shape[0]
        indices = ~np.eye(n, dtype=bool)
        errors = error_matrix[indices]
        
        # Create figure with multiple panels
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Reconstruction Error Analysis - {dataset_name} ({algorithm_name})", 
                    fontsize=16, y=0.98)
        
        # 1. Histogram of errors
        ax = axes[0, 0]
        ax.hist(errors, bins=50, alpha=0.7, color=self.node_colors['reconstructed'])
        ax.set_xlabel('Absolute Error')
        ax.set_ylabel('Frequency')
        ax.set_title('Histogram of Reconstruction Errors')
        
        # Add error statistics as text
        stats_text = (
            f"Mean Error: {np.mean(errors):.6f}\n"
            f"Median Error: {np.median(errors):.6f}\n"
            f"Max Error: {np.max(errors):.6f}\n"
            f"L1 Error: {np.mean(errors):.6f}\n"
            f"L2 Error: {np.sqrt(np.mean(errors**2)):.6f}"
        )
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Error density plot
        ax = axes[0, 1]
        
        # Use kernel density estimation
        if len(errors) > 100:  # Only if enough data points
            try:
                density = gaussian_kde(errors)
                x_range = np.linspace(0, min(1.0, np.max(errors) * 1.2), 1000)
                ax.plot(x_range, density(x_range), color=self.node_colors['reconstructed'], linewidth=2)
                ax.fill_between(x_range, density(x_range), alpha=0.3, color=self.node_colors['reconstructed'])
                ax.set_xlabel('Absolute Error')
                ax.set_ylabel('Density')
                ax.set_title('Error Density Distribution')
            except:
                # Fallback if KDE fails
                ax.hist(errors, bins=50, density=True, alpha=0.7, color=self.node_colors['reconstructed'])
                ax.set_xlabel('Absolute Error')
                ax.set_ylabel('Density')
                ax.set_title('Error Distribution (Histogram)')
        else:
            # Fallback for small datasets
            ax.hist(errors, bins=20, density=True, alpha=0.7, color=self.node_colors['reconstructed'])
            ax.set_xlabel('Absolute Error')
            ax.set_ylabel('Density')
            ax.set_title('Error Distribution (Histogram)')
        
        # 3. Error heatmap (sample if too large)
        ax = axes[1, 0]
        max_heatmap_size = 100
        
        if n > max_heatmap_size:
            # Sample a subset of the matrix
            indices = np.random.choice(n, max_heatmap_size, replace=False)
            indices = np.sort(indices)  # Sort for better visualization
            error_sample = error_matrix[np.ix_(indices, indices)]
            im = ax.imshow(error_sample, cmap='YlOrRd', aspect='auto')
            ax.set_title(f'Error Heatmap (Sample of {max_heatmap_size}x{max_heatmap_size})')
        else:
            im = ax.imshow(error_matrix, cmap='YlOrRd', aspect='auto')
            ax.set_title('Error Heatmap')
        
        ax.set_xlabel('Node Index')
        ax.set_ylabel('Node Index')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # 4. Error vs. Degree correlation
        ax = axes[1, 1]
        
        # Calculate node degrees
        degrees = np.sum(original_adj, axis=1)
        
        # Calculate average error per node
        node_errors = np.sum(error_matrix, axis=1) / (n - 1)  # Excluding self-loops
        
        # Scatter plot with hexbin for large datasets
        if n > 1000:
            hb = ax.hexbin(degrees, node_errors, gridsize=30, cmap='Blues', mincnt=1)
            plt.colorbar(hb, ax=ax, label='Count')
        else:
            ax.scatter(degrees, node_errors, alpha=0.5, color=self.node_colors['original'], s=10)
            
            # Add trend line
            if n > 5:  # Only if enough data points
                try:
                    z = np.polyfit(degrees, node_errors, 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(min(degrees), max(degrees), 100)
                    ax.plot(x_range, p(x_range), color='red', linestyle='--')
                    
                    # Add correlation coefficient
                    corr = np.corrcoef(degrees, node_errors)[0, 1]
                    ax.text(0.05, 0.95, f"Correlation: {corr:.4f}", transform=ax.transAxes,
                           verticalalignment='top', horizontalalignment='left',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                except:
                    pass
        
        ax.set_xlabel('Node Degree')
        ax.set_ylabel('Average Reconstruction Error')
        ax.set_title('Error vs. Node Degree')
        
        # Save and show
        output_path = os.path.join(self.output_dir, f"{dataset_name}_{algorithm_name}_error_distribution.png")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved error distribution visualization to {output_path}")
        return output_path
    
    def visualize_algorithm_comparison(self, results_data, dataset_name, 
                                      algorithms=None, metrics=None):
        """
        Create visualizations comparing multiple algorithms on the same dataset.
        
        Args:
            results_data: Dictionary or DataFrame containing algorithm comparison results
            dataset_name: Name of the dataset
            algorithms: List of algorithm names to include (if None, use all)
            metrics: List of metrics to visualize (if None, use all available)
            
        Returns:
            List of paths to the saved visualizations
        """
        print(f"Creating algorithm comparison visualizations for {dataset_name}...")
        
        # Convert dictionary to DataFrame if needed
        if isinstance(results_data, dict):
            results_df = pd.DataFrame(results_data)
        else:
            results_df = results_data
        
        # Filter algorithms if specified
        if algorithms:
            results_df = results_df[results_df['algorithm'].isin(algorithms)]
        
        # Get available metrics if not specified
        if not metrics:
            # Identify likely metric columns (exclude metadata)
            exclude_cols = ['algorithm', 'dataset', 'timestamp', 'parameters', 'size_ratio']
            metrics = [col for col in results_df.columns if col not in exclude_cols]
        
        output_paths = []
        
        # 1. Size vs. Reconstruction Error plot
        if 'size_in_bits' in results_df.columns and 'reconstruction_error_l1' in results_df.columns:
            plt.figure(figsize=(10, 8))
            
            # Create scatter plot with different markers for each algorithm
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
            algorithms_in_data = results_df['algorithm'].unique()
            
            for i, algo in enumerate(algorithms_in_data):
                algo_data = results_df[results_df['algorithm'] == algo]
                
                # Use relative size for better comparison
                if 'relative_size' in algo_data.columns:
                    size_metric = 'relative_size'
                else:
                    original_size = results_df['original_size'].iloc[0] if 'original_size' in results_df.columns else 1
                    algo_data['relative_size'] = algo_data['size_in_bits'] / original_size
                    size_metric = 'relative_size'
                
                plt.scatter(algo_data[size_metric], 
                           algo_data['reconstruction_error_l1'],
                           label=algo,
                           marker=markers[i % len(markers)],
                           s=100,
                           alpha=0.7)
                
                # Add algorithm name as annotation
                for _, row in algo_data.iterrows():
                    plt.annotate(algo, 
                               (row[size_metric], row['reconstruction_error_l1']),
                               textcoords="offset points", 
                               xytext=(0, 7), 
                               ha='center',
                               fontsize=8)
            
            plt.xlabel('Relative Size (compared to original)')
            plt.ylabel('L1 Reconstruction Error')
            plt.title(f'Size vs. Accuracy Trade-off - {dataset_name}')
            plt.grid(True, alpha=0.3)
            plt.legend(title='Algorithm')
            
            # Save the plot
            output_path = os.path.join(self.output_dir, f"{dataset_name}_size_vs_error.png")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            output_paths.append(output_path)
            print(f"Saved size vs. error comparison to {output_path}")
        
        # 2. Metric comparison bar charts
        for metric in metrics:
            if metric in results_df.columns and metric not in ['algorithm', 'dataset', 'timestamp', 'parameters']:
                plt.figure(figsize=(12, 6))
                
                # Sort algorithms by metric value for better visualization
                plot_data = results_df.sort_values(by=metric)
                
                # Create bar chart
                bars = plt.bar(plot_data['algorithm'], plot_data[metric], alpha=0.7)
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.001 * max(plot_data[metric]),
                           f'{height:.6f}', ha='center', va='bottom', fontsize=9, rotation=0)
                
                plt.xlabel('Algorithm')
                plt.ylabel(metric.replace('_', ' ').title())
                plt.title(f'{metric.replace("_", " ").title()} Comparison - {dataset_name}')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3, axis='y')
                
                # Save the plot
                output_path = os.path.join(self.output_dir, f"{dataset_name}_{metric}_comparison.png")
                plt.tight_layout()
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                output_paths.append(output_path)
                print(f"Saved {metric} comparison to {output_path}")
        
        # 3. Runtime comparison
        if 'runtime' in results_df.columns:
            plt.figure(figsize=(12, 6))
            
            # Sort algorithms by runtime
            plot_data = results_df.sort_values(by='runtime')
            
            # Create bar chart with log scale for wide runtime ranges
            bars = plt.bar(plot_data['algorithm'], plot_data['runtime'], alpha=0.7)
            plt.yscale('log')
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                       f'{height:.2f}s', ha='center', va='bottom', fontsize=9, rotation=0)
            
            plt.xlabel('Algorithm')
            plt.ylabel('Runtime (seconds, log scale)')
            plt.title(f'Runtime Comparison - {dataset_name}')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            
            # Save the plot
            output_path = os.path.join(self.output_dir, f"{dataset_name}_runtime_comparison.png")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            output_paths.append(output_path)
            print(f"Saved runtime comparison to {output_path}")
        
        # 4. Radar chart for multi-metric comparison
        # Select metrics that are comparable (lower is better or higher is better)
        if len(metrics) >= 3:
            comparable_metrics = [m for m in metrics if m in results_df.columns and 
                                 m not in ['algorithm', 'dataset', 'timestamp', 'parameters', 'runtime']]
            
            if len(comparable_metrics) >= 3:
                # Create radar chart
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, polar=True)
                
                # Number of metrics
                N = len(comparable_metrics)
                
                # Angle of each axis
                angles = [n / N * 2 * np.pi for n in range(N)]
                angles += angles[:1]  # Close the loop
                
                # Normalize data for radar chart (0 to 1 scale)
                normalized_data = {}
                for metric in comparable_metrics:
                    # Determine if lower is better (errors, size) or higher is better
                    if 'error' in metric or 'size' in metric:
                        min_val = results_df[metric].min()
                        max_val = results_df[metric].max()
                        if min_val == max_val:
                            normalized_data[metric] = [0.5] * len(results_df)
                        else:
                            # Invert so that lower values are better (outer on radar)
                            normalized_data[metric] = 1 - (results_df[metric] - min_val) / (max_val - min_val)
                    else:
                        min_val = results_df[metric].min()
                        max_val = results_df[metric].max()
                        if min_val == max_val:
                            normalized_data[metric] = [0.5] * len(results_df)
                        else:
                            normalized_data[metric] = (results_df[metric] - min_val) / (max_val - min_val)
                
                # Plot each algorithm
                for i, algo in enumerate(results_df['algorithm']):
                    values = [normalized_data[metric][i] for metric in comparable_metrics]
                    values += values[:1]  # Close the loop
                    
                    ax.plot(angles, values, linewidth=2, label=algo)
                    ax.fill(angles, values, alpha=0.1)
                
                # Set labels
                plt.xticks(angles[:-1], comparable_metrics, size=10)
                
                # Add legend
                plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                
                plt.title(f'Multi-Metric Comparison - {dataset_name}', size=15, y=1.1)
                
                # Save the plot
                output_path = os.path.join(self.output_dir, f"{dataset_name}_radar_comparison.png")
                plt.tight_layout()
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                output_paths.append(output_path)
                print(f"Saved radar chart comparison to {output_path}")
        
        return output_paths
    
    def visualize_parameter_sensitivity(self, sensitivity_data, dataset_name, 
                                       algorithm_name="SSumM", parameter_name=None):
        """
        Visualize how parameter changes affect algorithm performance.
        
        Args:
            sensitivity_data: DataFrame or dictionary with parameter sensitivity results
            dataset_name: Name of the dataset
            algorithm_name: Name of the algorithm
            parameter_name: If specified, only visualize sensitivity for this parameter
            
        Returns:
            List of paths to the saved visualizations
        """
        print(f"Creating parameter sensitivity visualizations for {algorithm_name} on {dataset_name}...")
        
        # Convert dictionary to DataFrame if needed
        if isinstance(sensitivity_data, dict):
            sensitivity_df = pd.DataFrame(sensitivity_data)
        else:
            sensitivity_df = sensitivity_data
        
        # Identify parameter columns (those not metrics or metadata)
        exclude_cols = ['algorithm', 'dataset', 'timestamp', 'reconstruction_error_l1', 
                       'reconstruction_error_l2', 'size_in_bits', 'runtime']
        
        param_cols = [col for col in sensitivity_df.columns if col not in exclude_cols]
        
        # Filter to specific parameter if requested
        if parameter_name and parameter_name in param_cols:
            param_cols = [parameter_name]
        
        output_paths = []
        
        # For each parameter, create sensitivity plots
        for param in param_cols:
            # Check if parameter has multiple values
            unique_values = sensitivity_df[param].unique()
            if len(unique_values) <= 1:
                continue
            
            # Create figure with subplots for different metrics
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Parameter Sensitivity: {param} - {algorithm_name} on {dataset_name}', 
                        fontsize=16, y=0.98)
            
            # Sort by parameter value
            plot_data = sensitivity_df.sort_values(by=param)
            
            # 1. Effect on L1 Reconstruction Error
            ax = axes[0, 0]
            if 'reconstruction_error_l1' in plot_data.columns:
                ax.plot(plot_data[param], plot_data['reconstruction_error_l1'], 
                       marker='o', linestyle='-', linewidth=2, markersize=8)
                
                # Add value labels
                for x, y in zip(plot_data[param], plot_data['reconstruction_error_l1']):
                    ax.annotate(f'{y:.6f}', (x, y), textcoords="offset points", 
                              xytext=(0, 5), ha='center')
                
                ax.set_xlabel(param)
                ax.set_ylabel('L1 Reconstruction Error')
                ax.set_title('Effect on L1 Reconstruction Error')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No L1 error data available', 
                      ha='center', va='center', transform=ax.transAxes)
            
            # 2. Effect on Size
            ax = axes[0, 1]
            if 'size_in_bits' in plot_data.columns:
                ax.plot(plot_data[param], plot_data['size_in_bits'], 
                       marker='s', linestyle='-', linewidth=2, markersize=8, 
                       color=self.node_colors['summary'])
                
                # Add value labels
                for x, y in zip(plot_data[param], plot_data['size_in_bits']):
                    ax.annotate(f'{y:.0f}', (x, y), textcoords="offset points", 
                              xytext=(0, 5), ha='center')
                
                ax.set_xlabel(param)
                ax.set_ylabel('Size (bits)')
                ax.set_title('Effect on Summary Size')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No size data available', 
                      ha='center', va='center', transform=ax.transAxes)
            
            # 3. Effect on Runtime
            ax = axes[1, 0]
            if 'runtime' in plot_data.columns:
                ax.plot(plot_data[param], plot_data['runtime'], 
                       marker='^', linestyle='-', linewidth=2, markersize=8, 
                       color=self.node_colors['sparsified'])
                
                # Add value labels
                for x, y in zip(plot_data[param], plot_data['runtime']):
                    ax.annotate(f'{y:.2f}s', (x, y), textcoords="offset points", 
                              xytext=(0, 5), ha='center')
                
                ax.set_xlabel(param)
                ax.set_ylabel('Runtime (seconds)')
                ax.set_title('Effect on Runtime')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No runtime data available', 
                      ha='center', va='center', transform=ax.transAxes)
            
            # 4. Size-Accuracy trade-off
            ax = axes[1, 1]
            if 'size_in_bits' in plot_data.columns and 'reconstruction_error_l1' in plot_data.columns:
                # Convert to relative size if original size available
                if 'original_size' in plot_data.columns:
                    x_values = plot_data['size_in_bits'] / plot_data['original_size'].iloc[0]
                    x_label = 'Relative Size'
                else:
                    x_values = plot_data['size_in_bits']
                    x_label = 'Size (bits)'
                
                sc = ax.scatter(x_values, plot_data['reconstruction_error_l1'], 
                              c=plot_data[param], cmap='viridis', 
                              s=100, alpha=0.8)
                
                # Add parameter value as annotation
                for i, txt in enumerate(plot_data[param]):
                    ax.annotate(f'{txt}', (x_values.iloc[i], plot_data['reconstruction_error_l1'].iloc[i]), 
                              textcoords="offset points", xytext=(0, 7), ha='center')
                
                plt.colorbar(sc, ax=ax, label=param)
                ax.set_xlabel(x_label)
                ax.set_ylabel('L1 Reconstruction Error')
                ax.set_title('Size-Accuracy Trade-off')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Insufficient data for trade-off analysis', 
                      ha='center', va='center', transform=ax.transAxes)
            
            # Save the plot
            output_path = os.path.join(self.output_dir, f"{dataset_name}_{algorithm_name}_{param}_sensitivity.png")
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            output_paths.append(output_path)
            print(f"Saved {param} sensitivity visualization to {output_path}")
        
        return output_paths
    
    def visualize_supernode_content(self, G_original, G_summary, supernode_info,
                                  dataset_name, algorithm_name="SSumM",
                                  max_supernodes=5, projection_method='PCA'):
        """
        Visualize the content of supernodes to understand grouping patterns.
        
        Args:
            G_original: Original graph
            G_summary: Summary graph
            supernode_info: Dictionary mapping supernode ID to list of original nodes
            dataset_name: Name of the dataset
            algorithm_name: Name of the algorithm
            max_supernodes: Maximum number of supernodes to visualize
            projection_method: Method for dimensionality reduction ('PCA' or 'TSNE')
            
        Returns:
            The path to the saved visualization
        """
        print(f"Visualizing supernode content for {algorithm_name} on {dataset_name}...")
        
        # Sample supernodes if there are too many
        supernode_ids = list(supernode_info.keys())
        if len(supernode_ids) > max_supernodes:
            # Select largest supernodes for better visualization
            supernode_sizes = {sid: len(nodes) for sid, nodes in supernode_info.items()}
            supernode_ids = sorted(supernode_ids, key=lambda x: supernode_sizes[x], reverse=True)[:max_supernodes]
        
        # Create adjacency matrix of original graph
        adj_matrix = nx.to_numpy_array(G_original)
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Supernode Analysis - {algorithm_name} on {dataset_name}', 
                    fontsize=16, y=0.98)
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        # 1. Distribution of supernode sizes
        ax = axes[0]
        supernode_sizes = [len(nodes) for nodes in supernode_info.values()]
        ax.hist(supernode_sizes, bins=min(50, len(supernode_sizes)), alpha=0.7, color=self.node_colors['summary'])
        ax.set_xlabel('Number of Nodes in Supernode')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Supernode Sizes')
        
        # Add statistics as text
        stats_text = (
            f"Total Supernodes: {len(supernode_info)}\n"
            f"Mean Size: {np.mean(supernode_sizes):.2f}\n"
            f"Median Size: {np.median(supernode_sizes):.2f}\n"
            f"Max Size: {max(supernode_sizes)}\n"
            f"Min Size: {min(supernode_sizes)}"
        )
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Intra-supernode vs. Inter-supernode edge density
        ax = axes[1]
        
        # Calculate intra and inter densities
        intra_densities = []
        inter_densities = []
        
        for supernode_id, nodes in supernode_info.items():
            if len(nodes) > 1:  # Only for supernodes with multiple nodes
                # Intra-supernode edge density
                subgraph = G_original.subgraph(nodes)
                actual_edges = subgraph.number_of_edges()
                possible_edges = (len(nodes) * (len(nodes) - 1)) / 2
                intra_density = actual_edges / possible_edges if possible_edges > 0 else 0
                intra_densities.append(intra_density)
                
                # Inter-supernode edge density
                external_nodes = [n for n in G_original.nodes() if n not in nodes]
                cross_edges = sum(1 for u in nodes for v in external_nodes if G_original.has_edge(u, v))
                possible_cross_edges = len(nodes) * len(external_nodes)
                inter_density = cross_edges / possible_cross_edges if possible_cross_edges > 0 else 0
                inter_densities.append(inter_density)
        
        # Create box plots
        ax.boxplot([intra_densities, inter_densities], labels=['Intra-Supernode', 'Inter-Supernode'])
        ax.set_ylabel('Edge Density')
        ax.set_title('Edge Density Comparison')
        ax.grid(True, alpha=0.3)
        
        # Add mean values as text
        ax.text(1, np.mean(intra_densities), f' Mean: {np.mean(intra_densities):.4f}', 
               verticalalignment='center', horizontalalignment='left')
        ax.text(2, np.mean(inter_densities), f' Mean: {np.mean(inter_densities):.4f}', 
               verticalalignment='center', horizontalalignment='left')
        
        # 3. Selected supernode visualization in feature space
        ax = axes[2]
        
        # Use node feature vectors (rows of adjacency matrix) for visualization
        node_features = adj_matrix
        
        if projection_method == 'TSNE':
            try:
                # Use t-SNE for dimensionality reduction
                tsne = TSNE(n_components=2, random_state=42)
                node_embeddings = tsne.fit_transform(node_features)
            except:
                # Fallback to PCA if t-SNE fails
                pca = PCA(n_components=2, random_state=42)
                node_embeddings = pca.fit_transform(node_features)
                projection_method = 'PCA'  # Update method name
        else:
            # Use PCA for dimensionality reduction
            pca = PCA(n_components=2, random_state=42)
            node_embeddings = pca.fit_transform(node_features)
        
        # Create a colormap for supernodes
        cmap = plt.cm.get_cmap('tab10', len(supernode_ids))
        
        # Plot each supernode's nodes in the embedding space
        for i, supernode_id in enumerate(supernode_ids):
            nodes = supernode_info[supernode_id]
            node_indices = [list(G_original.nodes()).index(n) for n in nodes]
            embeddings = node_embeddings[node_indices]
            
            ax.scatter(embeddings[:, 0], embeddings[:, 1], 
                      label=f'Supernode {supernode_id}',
                      color=cmap(i),
                      alpha=0.7,
                      s=30)
        
        ax.set_xlabel(f'Component 1')
        ax.set_ylabel(f'Component 2')
        ax.set_title(f'Node Embedding ({projection_method})')
        ax.legend(title="Supernodes", loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 4. Visualize a sample supernode's subgraph structure
        ax = axes[3]
        
        if supernode_ids:
            # Select the largest supernode
            largest_supernode_id = max([(sid, len(nodes)) for sid, nodes in supernode_info.items() 
                                      if sid in supernode_ids], key=lambda x: x[1])[0]
            
            nodes = supernode_info[largest_supernode_id]
            
            # Create subgraph for visualization
            subgraph = G_original.subgraph(nodes).copy()
            
            # Only visualize if not too large
            if subgraph.number_of_nodes() <= 100:
                pos = nx.spring_layout(subgraph, seed=42)
                
                nx.draw_networkx_nodes(subgraph, pos, 
                                      node_color=self.node_colors['merged'],
                                      node_size=50,
                                      alpha=0.8,
                                      ax=ax)
                
                nx.draw_networkx_edges(subgraph, pos, 
                                      edge_color='gray',
                                      width=0.7,
                                      alpha=0.6,
                                      ax=ax)
                
                if subgraph.number_of_nodes() <= 30:
                    nx.draw_networkx_labels(subgraph, pos, font_size=8, ax=ax)
                
                ax.set_title(f'Supernode {largest_supernode_id} Structure ({len(nodes)} nodes)')
                ax.axis('off')
            else:
                ax.text(0.5, 0.5, f'Supernode too large to visualize ({len(nodes)} nodes)', 
                      ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No supernode data available', 
                  ha='center', va='center', transform=ax.transAxes)
        
        # 5. Supernode connectivity in summary graph
        ax = axes[4]
        
        if nx.number_of_nodes(G_summary) <= 50:  # Only if not too large
            pos = nx.spring_layout(G_summary, seed=42)
            
            # Draw highlighting selected supernodes
            selected_nodes = [n for n in G_summary.nodes() if n in supernode_ids]
            
            # Draw all nodes
            nx.draw_networkx_nodes(G_summary, pos, 
                                  node_color='lightgray',
                                  node_size=200,
                                  alpha=0.5,
                                  ax=ax)
            
            # Highlight selected nodes
            nx.draw_networkx_nodes(G_summary, pos, 
                                  nodelist=selected_nodes,
                                  node_color=self.node_colors['summary'],
                                  node_size=300,
                                  alpha=0.9,
                                  ax=ax)
            
            # Draw edges
            nx.draw_networkx_edges(G_summary, pos, 
                                  edge_color='gray',
                                  width=1.0,
                                  alpha=0.6,
                                  ax=ax)
            
            # Labels only for selected nodes
            if selected_nodes:
                node_labels = {n: str(n) for n in selected_nodes}
                nx.draw_networkx_labels(G_summary, pos, 
                                      labels=node_labels,
                                      font_size=10, 
                                      ax=ax)
            
            ax.set_title('Supernode Connectivity in Summary Graph')
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'Summary graph too large to visualize', 
                  ha='center', va='center', transform=ax.transAxes)
        
        # 6. Specific algorithm information or additional statistics
        ax = axes[5]
        
        # Calculate additional statistics for SSumM
        if algorithm_name == "SSumM":
            # Analyze superedge density
            superedges = G_summary.number_of_edges()
            possible_superedges = (G_summary.number_of_nodes() * (G_summary.number_of_nodes() - 1)) / 2
            superedge_density = superedges / possible_superedges if possible_superedges > 0 else 0
            
            # Calculate sparsification statistics: ratio of actual superedges to all possible
            sparsification_ratio = superedges / possible_superedges if possible_superedges > 0 else 0
            
            # Compression statistics
            node_compression = G_summary.number_of_nodes() / G_original.number_of_nodes()
            edge_compression = G_summary.number_of_edges() / G_original.number_of_edges()
            
            # Create pie chart for compression
            labels = ['Supernodes', 'Merged Nodes']
            sizes = [G_summary.number_of_nodes(), G_original.number_of_nodes() - G_summary.number_of_nodes()]
            explode = (0, 0.1)
            
            ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                  shadow=True, startangle=90, colors=[self.node_colors['summary'], self.node_colors['merged']])
            
            ax.axis('equal')
            ax.set_title('Node Compression')
            
            # Add statistics as text
            stats_text = (
                f"Compression Statistics:\n"
                f"Node Ratio: {node_compression:.2%}\n"
                f"Edge Ratio: {edge_compression:.2%}\n"
                f"Superedge Density: {superedge_density:.4f}\n"
                f"Sparsification Ratio: {sparsification_ratio:.4f}"
            )
            
            ax.text(0, -1.2, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            # Generic statistics for other algorithms
            node_compression = G_summary.number_of_nodes() / G_original.number_of_nodes()
            edge_compression = G_summary.number_of_edges() / G_original.number_of_edges()
            
            # Create bar chart for compression ratios
            labels = ['Nodes', 'Edges']
            values = [node_compression, edge_compression]
            
            ax.bar(labels, values, color=[self.node_colors['summary'], self.node_colors['superedge']])
            ax.set_ylim(0, 1)
            ax.set_ylabel('Ratio (Summary / Original)')
            ax.set_title('Compression Ratios')
            
            # Add value labels on top of bars
            for i, v in enumerate(values):
                ax.text(i, v + 0.02, f'{v:.2%}', ha='center')
            
            ax.grid(True, alpha=0.3, axis='y')
        
        # Save and show
        output_path = os.path.join(self.output_dir, f"{dataset_name}_{algorithm_name}_supernode_analysis.png")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved supernode analysis visualization to {output_path}")
        return output_path
    
    def visualize_convergence(self, convergence_data, dataset_name, algorithm_name="SSumM"):
        """
        Visualize the convergence of iterative graph summarization (e.g., SSumM iterations).
        
        Args:
            convergence_data: DataFrame or dictionary with convergence information
            dataset_name: Name of the dataset
            algorithm_name: Name of the algorithm
            
        Returns:
            The path to the saved visualization
        """
        print(f"Visualizing convergence for {algorithm_name} on {dataset_name}...")
        
        # Convert dictionary to DataFrame if needed
        if isinstance(convergence_data, dict):
            convergence_df = pd.DataFrame(convergence_data)
        else:
            convergence_df = convergence_data
        
        # Create figure with multiple metrics
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Convergence Analysis - {algorithm_name} on {dataset_name}', 
                    fontsize=16, y=0.98)
        
        # Ensure we have iteration column
        if 'iteration' not in convergence_df.columns:
            # Assume data is ordered by iteration
            convergence_df['iteration'] = range(len(convergence_df))
        
        # 1. Reconstruction Error vs. Iteration
        ax = axes[0, 0]
        if 'reconstruction_error_l1' in convergence_df.columns:
            ax.plot(convergence_df['iteration'], convergence_df['reconstruction_error_l1'], 
                   marker='o', linestyle='-', linewidth=2, color=self.node_colors['original'])
            
            # Add endpoint annotation
            last_iter = convergence_df['iteration'].iloc[-1]
            last_error = convergence_df['reconstruction_error_l1'].iloc[-1]
            ax.annotate(f'Final: {last_error:.6f}', (last_iter, last_error), 
                       textcoords="offset points", xytext=(10, 0))
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel('L1 Reconstruction Error')
            ax.set_title('Reconstruction Error Convergence')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No L1 error convergence data available', 
                  ha='center', va='center', transform=ax.transAxes)
        
        # 2. Summary Size vs. Iteration
        ax = axes[0, 1]
        
        size_metrics = [col for col in convergence_df.columns 
                       if 'size' in col.lower() or 'bits' in col.lower()]
        
        if size_metrics:
            size_metric = size_metrics[0]  # Use first available size metric
            
            ax.plot(convergence_df['iteration'], convergence_df[size_metric], 
                   marker='s', linestyle='-', linewidth=2, color=self.node_colors['summary'])
            
            # Add endpoint annotation
            last_iter = convergence_df['iteration'].iloc[-1]
            last_size = convergence_df[size_metric].iloc[-1]
            ax.annotate(f'Final: {last_size:.0f}', (last_iter, last_size), 
                       textcoords="offset points", xytext=(10, 0))
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel(size_metric)
            ax.set_title('Summary Size Convergence')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No size convergence data available', 
                  ha='center', va='center', transform=ax.transAxes)
        
        # 3. Number of Supernodes vs. Iteration
        ax = axes[1, 0]
        
        node_metrics = [col for col in convergence_df.columns 
                      if 'node' in col.lower() or 'supernode' in col.lower()]
        
        if node_metrics:
            node_metric = node_metrics[0]  # Use first available node count metric
            
            ax.plot(convergence_df['iteration'], convergence_df[node_metric], 
                   marker='^', linestyle='-', linewidth=2, color=self.node_colors['merged'])
            
            # Add endpoint annotation
            last_iter = convergence_df['iteration'].iloc[-1]
            last_nodes = convergence_df[node_metric].iloc[-1]
            ax.annotate(f'Final: {last_nodes:.0f}', (last_iter, last_nodes), 
                       textcoords="offset points", xytext=(10, 0))
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel(node_metric)
            ax.set_title('Supernode Count Convergence')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No node count convergence data available', 
                  ha='center', va='center', transform=ax.transAxes)
        
        # 4. Error vs. Size trade-off during convergence
        ax = axes[1, 1]
        
        if size_metrics and 'reconstruction_error_l1' in convergence_df.columns:
            size_metric = size_metrics[0]
            
            # Create scatter plot with colormap based on iteration
            sc = ax.scatter(convergence_df[size_metric], 
                          convergence_df['reconstruction_error_l1'],
                          c=convergence_df['iteration'],
                          cmap='viridis',
                          s=80, alpha=0.8)
            
            # Add colorbar
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('Iteration')
            
            # Add arrows to show direction of progression
            for i in range(len(convergence_df) - 1):
                ax.annotate('', 
                          xy=(convergence_df[size_metric].iloc[i+1], 
                             convergence_df['reconstruction_error_l1'].iloc[i+1]),
                          xytext=(convergence_df[size_metric].iloc[i], 
                                 convergence_df['reconstruction_error_l1'].iloc[i]),
                          arrowprops=dict(arrowstyle='->', color='gray', lw=1))
            
            ax.set_xlabel(size_metric)
            ax.set_ylabel('L1 Reconstruction Error')
            ax.set_title('Error-Size Trade-off During Convergence')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient data for trade-off analysis', 
                  ha='center', va='center', transform=ax.transAxes)
        
        # Save and show
        output_path = os.path.join(self.output_dir, f"{dataset_name}_{algorithm_name}_convergence.png")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved convergence visualization to {output_path}")
        return output_path
    
    def create_dashboard(self, dataset_name, all_images, output_filename=None):
        """
        Create an HTML dashboard with all visualizations for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            all_images: List of paths to visualization images
            output_filename: Name of the output HTML file (if None, use dataset name)
            
        Returns:
            The path to the saved HTML dashboard
        """
        if output_filename is None:
            output_filename = f"{dataset_name}_dashboard.html"
        
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Graph Summarization Dashboard - {dataset_name}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                }}
                header {{
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    text-align: center;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .visualization {{
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    margin-bottom: 30px;
                    padding: 20px;
                }}
                .visualization img {{
                    max-width: 100%;
                    height: auto;
                    display: block;
                    margin: 0 auto;
                }}
                h2 {{
                    color: #2c3e50;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 10px;
                }}
                footer {{
                    background-color: #2c3e50;
                    color: white;
                    text-align: center;
                    padding: 10px;
                    margin-top: 30px;
                }}
            </style>
        </head>
        <body>
            <header>
                <h1>Graph Summarization Dashboard - {dataset_name}</h1>
            </header>
            
            <div class="container">
        """
        
        # Categorize images
        categories = {
            'overview': [],
            'comparison': [],
            'error': [],
            'parameter': [],
            'convergence': [],
            'supernode': []
        }
        
        for img_path in all_images:
            img_filename = os.path.basename(img_path)
            
            if 'comparison' in img_filename:
                categories['comparison'].append(img_path)
            elif 'error' in img_filename:
                categories['error'].append(img_path)
            elif 'parameter' in img_filename or 'sensitivity' in img_filename:
                categories['parameter'].append(img_path)
            elif 'convergence' in img_filename:
                categories['convergence'].append(img_path)
            elif 'supernode' in img_filename:
                categories['supernode'].append(img_path)
            else:
                categories['overview'].append(img_path)
        
        # Add images by category
        for category, images in categories.items():
            if images:
                category_title = category.replace('_', ' ').title()
                html_content += f"""
                <section>
                    <h2>{category_title} Visualizations</h2>
                """
                
                for img_path in images:
                    img_filename = os.path.basename(img_path)
                    img_title = img_filename.replace('.png', '').replace('_', ' ').title()
                    
                    # Create relative path
                    rel_path = os.path.relpath(img_path, self.output_dir)
                    
                    html_content += f"""
                    <div class="visualization">
                        <h3>{img_title}</h3>
                        <img src="{rel_path}" alt="{img_title}">
                    </div>
                    """
                
                html_content += """
                </section>
                """
        
        # Close HTML
        html_content += """
            </div>
            
            <footer>
                <p>Generated using SSumM Graph Summarization Analysis</p>
            </footer>
        </body>
        </html>
        """
        
        # Write HTML file
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"Created visualization dashboard at {output_path}")
        return output_path