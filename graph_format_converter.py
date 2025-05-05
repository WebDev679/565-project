import os
import networkx as nx
import numpy as np
import scipy.sparse as sp
import pickle
import json
from tqdm import tqdm

class GraphFormatConverter:
    """
    Converts processed graphs into formats required by SSumM and baseline algorithms.
    """
    
    def __init__(self, processed_dir="./processed_data", output_dir="./algorithm_data"):
        """
        Initialize the converter.
        
        Args:
            processed_dir: Directory containing processed data
            output_dir: Directory where algorithm-specific data will be stored
        """
        self.processed_dir = processed_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
            
        # Create subdirectories for each algorithm
        for algo in ["SSumM", "k-Gs", "S2L", "SAA-Gs"]:
            algo_dir = os.path.join(output_dir, algo)
            if not os.path.exists(algo_dir):
                os.makedirs(algo_dir)
                print(f"Created directory: {algo_dir}")
    
    def load_processed_graph(self, dataset_name):
        """
        Load a processed graph from the processed data directory.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            nx.Graph: The loaded graph
        """
        graph_path = os.path.join(self.processed_dir, dataset_name, f"{dataset_name}.gpickle")
        
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Processed graph not found: {graph_path}")
        
        print(f"Loading processed graph: {dataset_name}")
        #G = nx.read_gpickle(graph_path)

        with open(graph_path, 'rb') as f:
            G = pickle.load(f)
        
        print(f"Loaded {dataset_name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def convert_for_SSumM(self, dataset_name, G=None):
        """
        Convert a graph to the format required by SSumM.
        SSumM needs:
        - Edge list (for input graph)
        - Size calculation in bits
        
        Args:
            dataset_name: Name of the dataset to convert
            G: Preloaded graph (if None, will be loaded)
            
        Returns:
            dict: SSumM-compatible data
        """
        if G is None:
            G = self.load_processed_graph(dataset_name)
        
        print(f"Converting {dataset_name} for SSumM...")
        
        # Create directory for SSumM data
        ssumm_dir = os.path.join(self.output_dir, "SSumM", dataset_name)
        if not os.path.exists(ssumm_dir):
            os.makedirs(ssumm_dir)
        
        # Prepare data in SSumM format
        ssumm_data = {
            "nodes": list(G.nodes()),
            "edges": list(G.edges()),
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "size_in_bits": 2 * G.number_of_edges() * np.ceil(np.log2(G.number_of_nodes()))
        }
        
        # Save data
        with open(os.path.join(ssumm_dir, f"{dataset_name}_ssumm.pickle"), 'wb') as f:
            pickle.dump(ssumm_data, f)
        
        # Also save as text files for easier inspection
        with open(os.path.join(ssumm_dir, f"{dataset_name}_ssumm_edges.txt"), 'w') as f:
            for u, v in G.edges():
                f.write(f"{u} {v}\n")
        
        print(f"Saved SSumM data for {dataset_name}")
        return ssumm_data
    
    def convert_for_k_Gs(self, dataset_name, G=None):
        """
        Convert a graph to the format required by k-Gs.
        k-Gs needs:
        - Adjacency matrix representation
        - Target number of supernodes (k)
        
        Args:
            dataset_name: Name of the dataset to convert
            G: Preloaded graph (if None, will be loaded)
            
        Returns:
            dict: k-Gs-compatible data
        """
        if G is None:
            G = self.load_processed_graph(dataset_name)
        
        print(f"Converting {dataset_name} for k-Gs...")
        
        # Create directory for k-Gs data
        kgs_dir = os.path.join(self.output_dir, "k-Gs", dataset_name)
        if not os.path.exists(kgs_dir):
            os.makedirs(kgs_dir)
        
        # Calculate target k values (from 10% to 60% of nodes)
        k_values = [int(G.number_of_nodes() * pct) for pct in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]
        
        # Prepare data in k-Gs format
        kgs_data = {
            "adjacency_matrix": nx.to_numpy_array(G),
            "nodes": list(G.nodes()),
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "k_values": k_values
        }
        
        # Save data
        with open(os.path.join(kgs_dir, f"{dataset_name}_kgs.pickle"), 'wb') as f:
            pickle.dump(kgs_data, f)
        
        # Also save target k values for reference
        with open(os.path.join(kgs_dir, f"{dataset_name}_kgs_k_values.txt"), 'w') as f:
            for k in k_values:
                f.write(f"{k}\n")
        
        print(f"Saved k-Gs data for {dataset_name}")
        return kgs_data
    
    def convert_for_S2L(self, dataset_name, G=None):
        """
        Convert a graph to the format required by S2L.
        S2L needs:
        - Adjacency matrix or CSR matrix
        - Target number of supernodes (k)
        
        Args:
            dataset_name: Name of the dataset to convert
            G: Preloaded graph (if None, will be loaded)
            
        Returns:
            dict: S2L-compatible data
        """
        if G is None:
            G = self.load_processed_graph(dataset_name)
        
        print(f"Converting {dataset_name} for S2L...")
        
        # Create directory for S2L data
        s2l_dir = os.path.join(self.output_dir, "S2L", dataset_name)
        if not os.path.exists(s2l_dir):
            os.makedirs(s2l_dir)
        
        # Calculate target k values (from 10% to 60% of nodes)
        k_values = [int(G.number_of_nodes() * pct) for pct in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]
        
        # Prepare data in S2L format (using sparse CSR matrix for efficiency)
        s2l_data = {
            "csr_matrix": nx.to_scipy_sparse_matrix(G, format="csr"),
            "nodes": list(G.nodes()),
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "k_values": k_values
        }
        
        # Save data
        with open(os.path.join(s2l_dir, f"{dataset_name}_s2l.pickle"), 'wb') as f:
            pickle.dump(s2l_data, f)
        
        # Also save CSR matrix in a format that C++ implementation can read
        # (since S2L implementation might be in C++)
        sp.save_npz(os.path.join(s2l_dir, f"{dataset_name}_s2l_matrix.npz"), s2l_data["csr_matrix"])
        
        # Save edge list for maximum compatibility
        with open(os.path.join(s2l_dir, f"{dataset_name}_s2l_edges.txt"), 'w') as f:
            for u, v in G.edges():
                f.write(f"{u} {v}\n")
        
        print(f"Saved S2L data for {dataset_name}")
        return s2l_data
    
    def convert_for_SAA_Gs(self, dataset_name, G=None):
        """
        Convert a graph to the format required by SAA-Gs.
        SAA-Gs needs:
        - Edge list or adjacency list
        - Target number of supernodes (k)
        
        Args:
            dataset_name: Name of the dataset to convert
            G: Preloaded graph (if None, will be loaded)
            
        Returns:
            dict: SAA-Gs-compatible data
        """
        if G is None:
            G = self.load_processed_graph(dataset_name)
        
        print(f"Converting {dataset_name} for SAA-Gs...")
        
        # Create directory for SAA-Gs data
        saags_dir = os.path.join(self.output_dir, "SAA-Gs", dataset_name)
        if not os.path.exists(saags_dir):
            os.makedirs(saags_dir)
        
        # Calculate target k values (from 10% to 60% of nodes)
        k_values = [int(G.number_of_nodes() * pct) for pct in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]
        
        # Prepare data in SAA-Gs format
        saags_data = {
            "adjacency_list": {n: list(G.neighbors(n)) for n in G.nodes()},
            "edges": list(G.edges()),
            "nodes": list(G.nodes()),
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "k_values": k_values
        }
        
        # Save data
        with open(os.path.join(saags_dir, f"{dataset_name}_saags.pickle"), 'wb') as f:
            pickle.dump(saags_data, f)
        
        # Save edge list for Java implementation
        with open(os.path.join(saags_dir, f"{dataset_name}_saags_edges.txt"), 'w') as f:
            for u, v in G.edges():
                f.write(f"{u} {v}\n")
        
        # Save adjacency list for Java implementation
        with open(os.path.join(saags_dir, f"{dataset_name}_saags_adjlist.txt"), 'w') as f:
            for n in sorted(G.nodes()):
                neighbors = list(G.neighbors(n))
                f.write(f"{n} {' '.join(map(str, neighbors))}\n")
        
        print(f"Saved SAA-Gs data for {dataset_name}")
        return saags_data
    
    def convert_all_datasets(self, dataset_names=None):
        """
        Convert all specified datasets for all algorithms.
        
        Args:
            dataset_names: List of dataset names to convert (if None, all directories in processed_dir)
        """
        if dataset_names is None:
            # Get all subdirectories in the processed directory
            dataset_names = [d for d in os.listdir(self.processed_dir) 
                             if os.path.isdir(os.path.join(self.processed_dir, d))]
        
        print(f"Converting {len(dataset_names)} datasets for all algorithms...")
        
        for dataset_name in dataset_names:
            try:
                G = self.load_processed_graph(dataset_name)
                
                # Convert for all algorithms
                self.convert_for_SSumM(dataset_name, G)
                self.convert_for_k_Gs(dataset_name, G)
                self.convert_for_S2L(dataset_name, G)
                self.convert_for_SAA_Gs(dataset_name, G)
                
                print(f"Successfully converted {dataset_name} for all algorithms")
            except Exception as e:
                print(f"Error converting {dataset_name}: {e}")
    
    def generate_experiment_config(self, dataset_names=None):
        """
        Generate a configuration file for experiments.
        
        Args:
            dataset_names: List of dataset names to include (if None, all directories in processed_dir)
        """
        if dataset_names is None:
            # Get all subdirectories in the processed directory
            dataset_names = [d for d in os.listdir(self.processed_dir) 
                             if os.path.isdir(os.path.join(self.processed_dir, d))]
        
        print(f"Generating experiment configuration for {len(dataset_names)} datasets...")
        
        # Collect metadata for each dataset
        datasets_metadata = {}
        for dataset_name in dataset_names:
            try:
                # Load metadata
                metadata_path = os.path.join(self.processed_dir, dataset_name, "metadata.json")
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                datasets_metadata[dataset_name] = metadata
            except Exception as e:
                print(f"Error loading metadata for {dataset_name}: {e}")
        
        # Create experiment configuration
        config = {
            "algorithms": ["SSumM", "k-Gs", "S2L", "SAA-Gs"],
            "datasets": datasets_metadata,
            "parameters": {
                "SSumM": {
                    "num_iterations": 20,
                    "target_sizes": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
                },
                "k-Gs": {
                    "sampling_method": "SamplePairs",
                    "c": 1.0
                },
                "S2L": {},
                "SAA-Gs": {
                    "sample_pairs_log_n": True,
                    "count_min_sketch_w": 50,
                    "count_min_sketch_d": 2
                }
            },
            "metrics": ["reconstruction_error_l1", "reconstruction_error_l2", "size_in_bits", "runtime", "memory_usage"],
            "output_dir": "./results"
        }
        
        # Save configuration
        config_path = os.path.join(self.output_dir, "experiment_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Saved experiment configuration to {config_path}")
        return config

# Usage example
def main():
    # Initialize the converter
    converter = GraphFormatConverter(
        processed_dir="./processed_data", 
        output_dir="./algorithm_data"
    )
    
    # Process main datasets
    main_datasets = ["DBLP", "Amazon-0302", "Email-Enron"]
    
    # Convert all datasets for all algorithms
    converter.convert_all_datasets(main_datasets)
    
    # Generate experiment configuration
    converter.generate_experiment_config(main_datasets)
    
    print("\nAll datasets converted successfully for all algorithms!")

if __name__ == "__main__":
    main()