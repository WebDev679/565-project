import os
import networkx as nx
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from graph_visualizer import GraphVisualizer
import argparse
import time

def load_graph(graph_path):
    """
    Load a graph from file based on file extension.
    """
    if graph_path.endswith('.gpickle'):
        try:
            return nx.read_gpickle(graph_path)
        except:
            # If networkx.read_gpickle doesn't work, try direct pickle loading
            with open(graph_path, 'rb') as f:
                return pickle.load(f)
    elif graph_path.endswith('.edgelist') or graph_path.endswith('.edges') or graph_path.endswith('.txt'):
        return nx.read_edgelist(graph_path, nodetype=int)
    elif graph_path.endswith('.adjlist'):
        return nx.read_adjlist(graph_path, nodetype=int)
    else:
        raise ValueError(f"Unsupported graph file format: {graph_path}")

def create_visualization_suite(original_graph_path, dataset_name, output_dir, 
                              summary_graph_path=None, reconstructed_graph_path=None,
                              supernode_info_path=None, algorithm_results_path=None,
                              convergence_data_path=None, max_nodes=1000):
    """
    Create a comprehensive suite of visualizations for graph summarization.
    
    Args:
        original_graph_path: Path to the original graph file
        dataset_name: Name of the dataset
        output_dir: Directory to save visualizations
        summary_graph_path: Path to summary graph file (if available)
        reconstructed_graph_path: Path to reconstructed graph file (if available)
        supernode_info_path: Path to supernode mapping file (if available)
        algorithm_results_path: Path to algorithm comparison results (if available)
        convergence_data_path: Path to convergence data (if available)
        max_nodes: Maximum number of nodes for graph visualization
    
    Returns:
        List of paths to all generated visualizations
    """
    # Initialize visualizer
    visualizer = GraphVisualizer(output_dir=output_dir)
    
    # Track all generated visualizations
    all_visualizations = []
    
    print(f"Creating visualization suite for {dataset_name}...")
    start_time = time.time()
    
    # 1. Load original graph
    print(f"Loading original graph from {original_graph_path}...")
    G_original = load_graph(original_graph_path)
    print(f"Loaded original graph with {G_original.number_of_nodes()} nodes and {G_original.number_of_edges()} edges")
    
    # 2. Visualize original graph structure
    viz_path = visualizer.visualize_original_graph(G_original, dataset_name, max_nodes=max_nodes)
    all_visualizations.append(viz_path)
    
    # 3. Load and visualize summary graph if available
    if summary_graph_path and os.path.exists(summary_graph_path):
        print(f"Loading summary graph from {summary_graph_path}...")
        G_summary = load_graph(summary_graph_path)
        print(f"Loaded summary graph with {G_summary.number_of_nodes()} supernodes and {G_summary.number_of_edges()} superedges")
        
        # Load supernode info if available
        supernode_info = None
        if supernode_info_path and os.path.exists(supernode_info_path):
            print(f"Loading supernode mapping from {supernode_info_path}...")
            with open(supernode_info_path, 'rb') as f:
                supernode_info = pickle.load(f)
        
        # Visualize summary graph
        viz_path = visualizer.visualize_summary_graph(G_summary, dataset_name, 
                                                    original_graph=G_original,
                                                    supernode_info=supernode_info)
        all_visualizations.append(viz_path)
        
        # 4. Load and visualize reconstructed graph if available
        if reconstructed_graph_path and os.path.exists(reconstructed_graph_path):
            print(f"Loading reconstructed graph from {reconstructed_graph_path}...")
            G_reconstructed = load_graph(reconstructed_graph_path)
            print(f"Loaded reconstructed graph with {G_reconstructed.number_of_nodes()} nodes and {G_reconstructed.number_of_edges()} edges")
            
            # Create comparison visualization
            viz_path = visualizer.visualize_graph_comparison(
                G_original, G_summary, G_reconstructed, 
                dataset_name, supernode_info=supernode_info
            )
            all_visualizations.append(viz_path)
            
            # Create error distribution visualization
            # Convert graphs to adjacency matrices for error calculation
            original_adj = nx.to_numpy_array(G_original)
            reconstructed_adj = nx.to_numpy_array(G_reconstructed)
            
            viz_path = visualizer.visualize_error_distribution(
                original_adj, reconstructed_adj, dataset_name
            )
            all_visualizations.append(viz_path)
        
        # 5. Visualize supernode content if info available
        if supernode_info:
            viz_path = visualizer.visualize_supernode_content(
                G_original, G_summary, supernode_info, dataset_name
            )
            all_visualizations.append(viz_path)
    
    # 6. Visualize algorithm comparison if results available
    if algorithm_results_path and os.path.exists(algorithm_results_path):
        print(f"Loading algorithm comparison results from {algorithm_results_path}...")
        
        # Load results based on file format
        if algorithm_results_path.endswith('.csv'):
            results_data = pd.read_csv(algorithm_results_path)
        elif algorithm_results_path.endswith('.pkl') or algorithm_results_path.endswith('.pickle'):
            with open(algorithm_results_path, 'rb') as f:
                results_data = pickle.load(f)
        elif algorithm_results_path.endswith('.json'):
            import json
            with open(algorithm_results_path, 'r') as f:
                results_data = json.load(f)
        else:
            print(f"Unsupported results file format: {algorithm_results_path}")
            results_data = None
        
        if results_data is not None:
            viz_paths = visualizer.visualize_algorithm_comparison(results_data, dataset_name)
            all_visualizations.extend(viz_paths)
    
    # 7. Visualize convergence if data available
    if convergence_data_path and os.path.exists(convergence_data_path):
        print(f"Loading convergence data from {convergence_data_path}...")
        
        # Load convergence data based on file format
        if convergence_data_path.endswith('.csv'):
            convergence_data = pd.read_csv(convergence_data_path)
        elif convergence_data_path.endswith('.pkl') or convergence_data_path.endswith('.pickle'):
            with open(convergence_data_path, 'rb') as f:
                convergence_data = pickle.load(f)
        elif convergence_data_path.endswith('.json'):
            import json
            with open(convergence_data_path, 'r') as f:
                convergence_data = json.load(f)
        else:
            print(f"Unsupported convergence data format: {convergence_data_path}")
            convergence_data = None
        
        if convergence_data is not None:
            viz_path = visualizer.visualize_convergence(convergence_data, dataset_name)
            all_visualizations.append(viz_path)
    
    # 8. Create dashboard with all visualizations
    dashboard_path = visualizer.create_dashboard(dataset_name, all_visualizations)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Visualization suite created in {elapsed_time:.2f} seconds")
    print(f"Generated {len(all_visualizations)} visualizations and dashboard at {dashboard_path}")
    
    return all_visualizations, dashboard_path

def main():
    parser = argparse.ArgumentParser(description='Create graph summarization visualizations')
    parser.add_argument('--original', required=True, help='Path to original graph file')
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--output_dir', default='./visualizations', help='Output directory for visualizations')
    parser.add_argument('--summary', default=None, help='Path to summary graph file')
    parser.add_argument('--reconstructed', default=None, help='Path to reconstructed graph file')
    parser.add_argument('--supernode_info', default=None, help='Path to supernode mapping file')
    parser.add_argument('--algorithm_results', default=None, help='Path to algorithm comparison results')
    parser.add_argument('--convergence_data', default=None, help='Path to convergence data')
    parser.add_argument('--max_nodes', type=int, default=1000, help='Maximum number of nodes for graph visualization')
    
    args = parser.parse_args()
    
    create_visualization_suite(
        args.original,
        args.dataset,
        args.output_dir,
        summary_graph_path=args.summary,
        reconstructed_graph_path=args.reconstructed,
        supernode_info_path=args.supernode_info,
        algorithm_results_path=args.algorithm_results,
        convergence_data_path=args.convergence_data,
        max_nodes=args.max_nodes
    )

if __name__ == "__main__":
    main()