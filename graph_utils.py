import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import random
import pickle
import os

class GraphUtils:
    """
    Utility class for graph operations, including data structures,
    metrics calculation, and I/O operations.
    """
    
    @staticmethod
    def load_graph(file_path, format="edge_list"):
        """
        Load a graph from file based on specified format.
        
        Args:
            file_path (str): Path to the graph file
            format (str): Format of the graph file ('edge_list', 'adjacency_list', 'pickle')
            
        Returns:
            nx.Graph: Loaded graph
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Graph file not found: {file_path}")
        
        if format == "edge_list":
            G = nx.read_edgelist(file_path, nodetype=int, data=False)
        elif format == "adjacency_list":
            G = nx.read_adjlist(file_path, nodetype=int)
        elif format == "pickle":
            with open(file_path, 'rb') as f:
                G = pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Make sure graph is undirected and simple (as required by the paper)
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        
        print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    @staticmethod
    def save_graph(G, file_path, format="pickle"):
        """
        Save a graph to file based on specified format.
        
        Args:
            G (nx.Graph): Graph to save
            file_path (str): Path for saving the graph
            format (str): Format for saving ('edge_list', 'adjacency_list', 'pickle')
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if format == "edge_list":
            nx.write_edgelist(G, file_path, data=False)
        elif format == "adjacency_list":
            nx.write_adjlist(G, file_path)
        elif format == "pickle":
            with open(file_path, 'wb') as f:
                pickle.dump(G, f)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Saved graph to {file_path}")
    
    @staticmethod
    def compute_graph_metrics(G):
        """
        Compute various metrics for the input graph.
        
        Args:
            G (nx.Graph): Input graph
            
        Returns:
            dict: Dictionary containing graph metrics
        """
        metrics = {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "density": nx.density(G),
            "is_connected": nx.is_connected(G),
            "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
            "max_degree": max(dict(G.degree()).values()) if G.number_of_nodes() > 0 else 0,
            "min_degree": min(dict(G.degree()).values()) if G.number_of_nodes() > 0 else 0,
            "size_in_bits": 2 * G.number_of_edges() * np.ceil(np.log2(G.number_of_nodes())) if G.number_of_nodes() > 0 else 0,
        }
        
        return metrics
    
    @staticmethod
    def visualize_graph(G, title="Graph Visualization", node_size=None, layout='spring', 
                    max_nodes=100, output_path=None):
        """
        Visualize a graph using matplotlib.
        
        Args:
            G (nx.Graph): Graph to visualize
            title (str): Title of the visualization
            node_size (dict): Dictionary mapping nodes to sizes
            layout (str): Layout algorithm ('spring', 'kamada_kawai', 'circular', 'spectral')
            max_nodes (int): Maximum number of nodes to visualize
            output_path (str): Path to save the visualization
            
        Returns:
            fig, ax: Figure and axes objects
        """
        if G.number_of_nodes() > max_nodes:
            print(f"Graph too large to visualize. Sampling {max_nodes} nodes...")
            nodes = list(G.nodes())
            sampled_nodes = random.sample(nodes, min(max_nodes, len(nodes)))
            G = G.subgraph(sampled_nodes).copy()
        
        # Choose layout
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
        
        # Determine node sizes
        if node_size is None:
            # Try to use 'size' attribute if available
            has_size_attr = all('size' in G.nodes[n] for n in G.nodes())
            
            if has_size_attr:
                node_size = {n: 50 + 100 * G.nodes[n]['size'] for n in G.nodes()}
            else:
                # Fall back to degree-based sizing
                degrees = dict(G.degree())
                min_degree = min(degrees.values()) if degrees else 0
                max_degree = max(degrees.values()) if degrees else 1
                node_size = {n: 50 + 200 * (degrees[n] - min_degree) / (max_degree - min_degree + 1) 
                        for n in G.nodes()}
        
        fig, ax = plt.subplots(figsize=(10, 8))
        nx.draw_networkx_nodes(G, pos, node_size=[node_size[n] for n in G.nodes()], 
                            node_color='skyblue', alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
        
        if G.number_of_nodes() <= 50:  # Only show labels for small graphs
            nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
        
        plt.title(title)
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
        
        return fig, ax