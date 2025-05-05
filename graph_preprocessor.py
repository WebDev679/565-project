import networkx as nx
import pandas as pd
import numpy as np
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import time

class GraphPreprocessor:
    """
    A class for preprocessing graph datasets for the SSumM algorithm and baselines.
    """
    
    def __init__(self, data_dir="./data", processed_dir="./processed_data"):
        """
        Initialize the preprocessor with directories for raw and processed data.
        
        Args:
            data_dir: Directory containing the raw datasets
            processed_dir: Directory where processed data will be stored
        """
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        
        # Create processed data directory if it doesn't exist
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
            print(f"Created directory: {processed_dir}")
            
        # Dictionary to store the loaded graphs
        self.graphs = {}
        
    def load_dataset(self, dataset_name, file_format="edge_list", delimiter=None, 
                     has_header=False, comment="#"):
        """
        Load a graph dataset from file.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'DBLP', 'Amazon-0302')
            file_format: Format of the dataset file ('edge_list', 'adjacency_list', etc.)
            delimiter: Delimiter used in the file
            has_header: Whether the file has a header
            comment: Character used for comments in the file
            
        Returns:
            nx.Graph: The loaded graph
        """
        file_path = os.path.join(self.data_dir, f"{dataset_name}.txt")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        print(f"Loading dataset: {dataset_name}")
        
        if file_format == "edge_list":
            # Load as edge list
            G = nx.read_edgelist(
                file_path, 
                delimiter=delimiter,
                comments=comment,
                nodetype=int,
                data=False  # We don't need edge attributes for this project
            )
        elif file_format == "adjacency_list":
            # Load as adjacency list
            G = nx.read_adjlist(
                file_path,
                delimiter=delimiter,
                comments=comment,
                nodetype=int
            )
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        print(f"Successfully loaded {dataset_name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        self.graphs[dataset_name] = G
        return G
    
    def preprocess_graph(self, dataset_name, make_undirected=True, remove_self_loops=True, 
                         largest_cc_only=False, relabel_nodes=True):
        """
        Preprocess a loaded graph.
        
        Args:
            dataset_name: Name of the dataset to preprocess
            make_undirected: Convert directed graphs to undirected
            remove_self_loops: Remove self-loops from the graph
            largest_cc_only: Extract only the largest connected component
            relabel_nodes: Relabel nodes to be consecutive integers starting from 0
            
        Returns:
            nx.Graph: The preprocessed graph
        """
        if dataset_name not in self.graphs:
            raise ValueError(f"Dataset {dataset_name} not loaded. Call load_dataset first.")
        
        G = self.graphs[dataset_name]
        original_n = G.number_of_nodes()
        original_m = G.number_of_edges()
        
        print(f"\nPreprocessing {dataset_name}...")
        print(f"Original graph: {original_n} nodes, {original_m} edges")
        
        # Convert to undirected if requested
        if make_undirected and G.is_directed():
            print("Converting to undirected graph...")
            G = G.to_undirected()
            print(f"After conversion: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Remove self-loops if requested
        if remove_self_loops:
            print("Removing self-loops...")
            G.remove_edges_from(nx.selfloop_edges(G))
            print(f"After removing self-loops: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Extract largest connected component if requested
        if largest_cc_only:
            print("Extracting largest connected component...")
            if not nx.is_connected(G):
                largest_cc = max(nx.connected_components(G), key=len)
                G = G.subgraph(largest_cc).copy()
                print(f"Largest connected component: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            else:
                print("Graph is already connected.")
        
        # Relabel nodes to be consecutive integers if requested
        if relabel_nodes:
            print("Relabeling nodes to consecutive integers...")
            G = nx.convert_node_labels_to_integers(G, first_label=0)
            print(f"After relabeling: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Update the graph in the dictionary
        self.graphs[dataset_name] = G
        
        # Calculate some basic statistics
        density = nx.density(G)
        avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
        
        print(f"\nPreprocessing complete for {dataset_name}:")
        print(f"Nodes: {G.number_of_nodes()} (reduced by {original_n - G.number_of_nodes()})")
        print(f"Edges: {G.number_of_edges()} (reduced by {original_m - G.number_of_edges()})")
        print(f"Density: {density:.6f}")
        print(f"Average degree: {avg_degree:.2f}")
        
        return G
    
    def create_subsamples(self, dataset_name, sizes=[0.1, 0.2, 0.5], 
                          methods=["random_nodes", "random_edges", "snowball"],
                          seed=42):
        """
        Create subsamples of a graph for development testing.
        
        Args:
            dataset_name: Name of the dataset to subsample
            sizes: List of subsample sizes (as fractions of the original)
            methods: List of subsampling methods to use
            seed: Random seed for reproducibility
            
        Returns:
            dict: Dictionary of created subsamples
        """
        if dataset_name not in self.graphs:
            raise ValueError(f"Dataset {dataset_name} not loaded. Call load_dataset first.")
        
        G = self.graphs[dataset_name]
        subsamples = {}
        
        random.seed(seed)
        np.random.seed(seed)
        
        print(f"\nCreating subsamples for {dataset_name}...")
        
        for method in methods:
            for size in sizes:
                subsample_name = f"{dataset_name}_{method}_{int(size*100)}pct"
                print(f"Creating {subsample_name}...")
                
                if method == "random_nodes":
                    # Randomly sample nodes
                    num_nodes = int(size * G.number_of_nodes())
                    sampled_nodes = random.sample(list(G.nodes()), num_nodes)
                    sub_G = G.subgraph(sampled_nodes).copy()
                    
                elif method == "random_edges":
                    # Randomly sample edges
                    sub_G = nx.Graph()
                    sub_G.add_nodes_from(G.nodes())
                    
                    num_edges = int(size * G.number_of_edges())
                    sampled_edges = random.sample(list(G.edges()), num_edges)
                    sub_G.add_edges_from(sampled_edges)
                    
                    # Keep only non-isolated nodes
                    sub_G.remove_nodes_from(list(nx.isolates(sub_G)))
                    
                elif method == "snowball":
                    # Snowball sampling (BFS-based)
                    sub_G = nx.Graph()
                    start_node = random.choice(list(G.nodes()))
                    nodes_to_explore = {start_node}
                    explored_nodes = set()
                    
                    target_nodes = int(size * G.number_of_nodes())
                    
                    while len(explored_nodes) < target_nodes and nodes_to_explore:
                        current = nodes_to_explore.pop()
                        if current not in explored_nodes:
                            explored_nodes.add(current)
                            
                            # Add neighbors to exploration queue
                            neighbors = set(G.neighbors(current)) - explored_nodes
                            nodes_to_explore.update(neighbors)
                            
                            if len(explored_nodes) >= target_nodes:
                                break
                    
                    sub_G = G.subgraph(explored_nodes).copy()
                    
                else:
                    raise ValueError(f"Unknown subsampling method: {method}")
                
                # Store the subsample
                subsamples[subsample_name] = sub_G
                self.graphs[subsample_name] = sub_G
                
                print(f"Created {subsample_name}: {sub_G.number_of_nodes()} nodes, {sub_G.number_of_edges()} edges")
                
        return subsamples
    
    def convert_to_common_format(self, dataset_name, format_type="edge_list"):
        """
        Convert a graph to a common format for all algorithms.
        
        Args:
            dataset_name: Name of the dataset to convert
            format_type: Type of format to convert to
            
        Returns:
            The graph data in the requested format
        """
        if dataset_name not in self.graphs:
            raise ValueError(f"Dataset {dataset_name} not loaded. Call load_dataset first.")
        
        G = self.graphs[dataset_name]
        
        if format_type == "edge_list":
            # Return a list of edges
            return list(G.edges())
        
        elif format_type == "adjacency_list":
            # Return a dictionary of adjacency lists
            return {n: list(G.neighbors(n)) for n in G.nodes()}
        
        elif format_type == "adjacency_matrix":
            # Return a dense adjacency matrix
            return nx.to_numpy_array(G)
        
        elif format_type == "csr_matrix":
            # Return a sparse CSR matrix
            return nx.to_scipy_sparse_array(G, format="csr")
        
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def save_processed_graph(self, dataset_name, formats=["edge_list", "adjacency_list", "csr_matrix"]):
        """
        Save a processed graph in multiple formats for efficient access.
        
        Args:
            dataset_name: Name of the dataset to save
            formats: List of formats to save the graph in
        """
        if dataset_name not in self.graphs:
            raise ValueError(f"Dataset {dataset_name} not loaded. Call load_dataset first.")
        
        # Create directory for this dataset if it doesn't exist
        dataset_dir = os.path.join(self.processed_dir, dataset_name)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            
        G = self.graphs[dataset_name]
        
        # Save graph metadata
        metadata = {
            "name": dataset_name,
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "density": nx.density(G),
            "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
            "is_directed": G.is_directed(),
            "is_connected": nx.is_connected(G),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(dataset_dir, "metadata.json"), 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
        
        # Save the graph in requested formats
        print(f"\nSaving {dataset_name} in multiple formats...")
        
        # # Always save the NetworkX graph object
        # nx.write_gpickle(G, os.path.join(dataset_dir, f"{dataset_name}.gpickle"))
        # print(f"Saved as NetworkX pickle: {dataset_name}.gpickle")

        import pickle
        with open(os.path.join(dataset_dir, f"{dataset_name}.gpickle"), 'wb') as f:
            pickle.dump(G, f)
        print(f"Saved as NetworkX pickle: {dataset_name}.gpickle")
        
        for fmt in formats:
            if fmt == "edge_list":
                # Save as edge list
                edge_list_file = os.path.join(dataset_dir, f"{dataset_name}_edges.txt")
                nx.write_edgelist(G, edge_list_file, data=False)
                print(f"Saved as edge list: {dataset_name}_edges.txt")
                
            elif fmt == "adjacency_list":
                # Save as adjacency list
                adj_list_file = os.path.join(dataset_dir, f"{dataset_name}_adj.txt")
                nx.write_adjlist(G, adj_list_file)
                print(f"Saved as adjacency list: {dataset_name}_adj.txt")
                
            elif fmt == "csr_matrix":
                # Save as sparse CSR matrix
                csr_file = os.path.join(dataset_dir, f"{dataset_name}_csr.npz")
                sparse_matrix = nx.to_scipy_sparse_array(G, format="csr")
                import scipy.sparse
                scipy.sparse.save_npz(csr_file, sparse_matrix)
                print(f"Saved as CSR matrix: {dataset_name}_csr.npz")
                
            elif fmt == "adjacency_matrix":
                # Save as dense adjacency matrix
                adj_matrix_file = os.path.join(dataset_dir, f"{dataset_name}_adj_matrix.npy")
                adj_matrix = nx.to_numpy_array(G)
                np.save(adj_matrix_file, adj_matrix)
                print(f"Saved as adjacency matrix: {dataset_name}_adj_matrix.npy")
        
        print(f"Successfully saved {dataset_name} in all requested formats.")
    
    def analyze_graph(self, dataset_name, plot=True):
        """
        Perform basic analysis of a graph and generate visualizations.
        
        Args:
            dataset_name: Name of the dataset to analyze
            plot: Whether to generate and save plots
            
        Returns:
            dict: Dictionary of analysis results
        """
        if dataset_name not in self.graphs:
            raise ValueError(f"Dataset {dataset_name} not loaded. Call load_dataset first.")
        
        G = self.graphs[dataset_name]
        
        print(f"\nAnalyzing {dataset_name}...")
        
        # Create analysis directory
        analysis_dir = os.path.join(self.processed_dir, dataset_name, "analysis")
        if not os.path.exists(analysis_dir):
            os.makedirs(analysis_dir)
        
        # Collect basic graph properties
        results = {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "density": nx.density(G),
            "is_directed": G.is_directed(),
            "is_connected": nx.is_connected(G)
        }
        
        # Compute degree statistics
        degrees = [d for _, d in G.degree()]
        results["min_degree"] = min(degrees)
        results["max_degree"] = max(degrees)
        results["avg_degree"] = sum(degrees) / len(degrees)
        results["median_degree"] = np.median(degrees)
        
        print(f"Basic properties:")
        print(f"  Nodes: {results['num_nodes']}")
        print(f"  Edges: {results['num_edges']}")
        print(f"  Density: {results['density']:.6f}")
        print(f"  Average degree: {results['avg_degree']:.2f}")
        print(f"  Degree range: {results['min_degree']} - {results['max_degree']}")
        
        # Connected components analysis
        if not results["is_connected"]:
            components = list(nx.connected_components(G))
            results["num_components"] = len(components)
            component_sizes = [len(c) for c in components]
            results["largest_component_size"] = max(component_sizes)
            results["largest_component_ratio"] = results["largest_component_size"] / results["num_nodes"]
            
            print(f"Connected components:")
            print(f"  Number of components: {results['num_components']}")
            print(f"  Largest component: {results['largest_component_size']} nodes ({results['largest_component_ratio']:.2%})")
        
        # Generate visualizations if requested
        if plot:
            print("Generating visualizations...")
            
            # Degree distribution
            plt.figure(figsize=(10, 6))
            plt.hist(degrees, bins=min(50, results["max_degree"]), alpha=0.7)
            plt.xlabel('Degree')
            plt.ylabel('Number of Nodes')
            plt.title(f'Degree Distribution - {dataset_name}')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, f"{dataset_name}_degree_dist.png"), dpi=300)
            plt.close()
            
            # Log-log degree distribution
            degree_counts = pd.Series(degrees).value_counts().sort_index()
            plt.figure(figsize=(10, 6))
            plt.loglog(degree_counts.index, degree_counts.values, 'o-', alpha=0.7)
            plt.xlabel('Degree (log scale)')
            plt.ylabel('Number of Nodes (log scale)')
            plt.title(f'Log-Log Degree Distribution - {dataset_name}')
            plt.grid(alpha=0.3, which='both')
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, f"{dataset_name}_loglog_degree_dist.png"), dpi=300)
            plt.close()
            
            # Only try to visualize the graph if it's small enough
            if G.number_of_nodes() <= 1000:
                try:
                    plt.figure(figsize=(12, 10))
                    pos = nx.spring_layout(G, seed=42)
                    nx.draw(G, pos, node_size=20, node_color='blue', alpha=0.7, 
                            with_labels=False, width=0.5)
                    plt.title(f'Graph Visualization - {dataset_name}')
                    plt.savefig(os.path.join(analysis_dir, f"{dataset_name}_graph_viz.png"), dpi=300)
                    plt.close()
                except Exception as e:
                    print(f"Could not generate graph visualization: {e}")
        
        # Save analysis results
        with open(os.path.join(analysis_dir, f"{dataset_name}_analysis.json"), 'w') as f:
            import json
            json.dump(results, f, indent=2)
        
        return results
    
    def process_all_datasets(self):
        """
        Process all loaded datasets with standard preprocessing steps.
        """
        for dataset_name in self.graphs.keys():
            self.preprocess_graph(dataset_name)
            self.analyze_graph(dataset_name)
            self.save_processed_graph(dataset_name)

# Usage example
def main():
    # Initialize the preprocessor
    processor = GraphPreprocessor(data_dir="./data", processed_dir="./processed_data")
    
    # Load datasets with appropriate parameters
    # DBLP
    processor.load_dataset("DBLP", file_format="edge_list", delimiter=None, comment="#")
    
    # Amazon-0302
    processor.load_dataset("Amazon-0302", file_format="edge_list", delimiter=None, comment="#")
    
    # Email-Enron
    processor.load_dataset("Email-Enron", file_format="edge_list", delimiter=None, comment="#")
    
    # Preprocess all datasets with standard steps
    for dataset_name in processor.graphs.keys():
        # Standard preprocessing
        processor.preprocess_graph(
            dataset_name,
            make_undirected=True,
            remove_self_loops=True,
            largest_cc_only=True,  # Focus on largest connected component
            relabel_nodes=True     # Ensure nodes are labeled as consecutive integers
        )
        
        # Create subsamples for development testing
        processor.create_subsamples(
            dataset_name,
            sizes=[0.01, 0.05, 0.1, 0.25],  # Different subsample sizes
            methods=["random_nodes", "snowball"]  # Different sampling methods
        )
        
        # Analyze and save processed data
        processor.analyze_graph(dataset_name)
        processor.save_processed_graph(dataset_name)
    
    print("\nAll datasets processed successfully!")

if __name__ == "__main__":
    main()