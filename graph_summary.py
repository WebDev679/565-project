import networkx as nx
import numpy as np

class GraphSummary:
    """
    Representation of a graph summary as defined in the LeFevre and Terzi paper.
    A summary consists of:
    1. A partition of nodes into supernodes
    2. Edge counts within supernodes
    3. Edge counts between supernodes
    """
    
    def __init__(self, G=None):
        """
        Initialize a graph summary, optionally from an input graph G.
        
        Args:
            G (nx.Graph, optional): Input graph to initialize the summary
        """
        self.supernodes = {}  # Dictionary mapping supernode ID to set of original nodes
        self.supernode_edge_counts = {}  # Dictionary mapping supernode ID to internal edge count
        self.superedge_counts = {}  # Dictionary mapping (supernode1, supernode2) to edge count
        self.original_graph = G
        
        # Initialize with trivial summary if G is provided
        if G is not None:
            self.initialize_trivial_summary(G)
    
    def initialize_trivial_summary(self, G):
        """
        Initialize with a trivial summary where each node is in its own supernode.
        
        Args:
            G (nx.Graph): Original graph
        """
        self.original_graph = G
        
        # Create one supernode per original node
        for node in G.nodes():
            self.supernodes[node] = {node}
            self.supernode_edge_counts[node] = 0  # No self-loops
        
        # Count edges between supernodes
        for u, v in G.edges():
            if u != v:  # Skip self-loops
                self.add_to_superedge_count(u, v, 1)
    
    def add_to_superedge_count(self, supernode1, supernode2, count=1):
        """
        Add count to the superedge between two supernodes.
        
        Args:
            supernode1: ID of first supernode
            supernode2: ID of second supernode
            count (int): Count to add
        """
        # Ensure supernodes are ordered (since graph is undirected)
        if supernode1 > supernode2:
            supernode1, supernode2 = supernode2, supernode1
        
        key = (supernode1, supernode2)
        self.superedge_counts[key] = self.superedge_counts.get(key, 0) + count
    
    def merge_supernodes(self, supernode1, supernode2):
        """
        Merge two supernodes and update edge counts.
        
        Args:
            supernode1: ID of first supernode
            supernode2: ID of second supernode
            
        Returns:
            new_id: ID of the newly created supernode
        """
        if supernode1 not in self.supernodes or supernode2 not in self.supernodes:
            raise ValueError("One or both supernodes not found in the summary")
        
        # Create new supernode with a unique ID
        new_id = min(supernode1, supernode2)
        
        # Merge nodes
        self.supernodes[new_id] = self.supernodes[supernode1].union(self.supernodes[supernode2])
        
        # Calculate internal edges for the new supernode
        # 1. Internal edges from both source supernodes
        new_internal_edges = (
            self.supernode_edge_counts.get(supernode1, 0) + 
            self.supernode_edge_counts.get(supernode2, 0)
        )
        
        # 2. Edges between the two supernodes being merged
        key = (min(supernode1, supernode2), max(supernode1, supernode2))
        between_edges = self.superedge_counts.get(key, 0)
        new_internal_edges += between_edges
        
        # Set internal edge count for new supernode
        self.supernode_edge_counts[new_id] = new_internal_edges
        
        # Update edges to other supernodes
        for supernode in list(self.supernodes.keys()):
            if supernode in (supernode1, supernode2) or supernode == new_id:
                continue
            
            # Get edge counts from both source supernodes to the current supernode
            count1 = self.get_superedge_count(supernode1, supernode)
            count2 = self.get_superedge_count(supernode2, supernode)
            
            # Set the new edge count
            if count1 > 0 or count2 > 0:
                self.add_to_superedge_count(new_id, supernode, count1 + count2)
        
        # Remove old supernodes and their connections
        if supernode1 != new_id:
            del self.supernodes[supernode1]
            del self.supernode_edge_counts[supernode1]
        
        if supernode2 != new_id:
            del self.supernodes[supernode2]
            del self.supernode_edge_counts[supernode2]
        
        # Remove old superedges
        for key in list(self.superedge_counts.keys()):
            sn1, sn2 = key
            if sn1 in (supernode1, supernode2) or sn2 in (supernode1, supernode2):
                if sn1 == new_id or sn2 == new_id:
                    continue
                del self.superedge_counts[key]
        
        return new_id
    
    def get_superedge_count(self, supernode1, supernode2):
        """
        Get the number of edges between two supernodes.
        
        Args:
            supernode1: ID of first supernode
            supernode2: ID of second supernode
            
        Returns:
            int: Number of edges between the supernodes
        """
        key = (min(supernode1, supernode2), max(supernode1, supernode2))
        return self.superedge_counts.get(key, 0)
    
    def get_supernode_size(self, supernode):
        """
        Get the number of original nodes in a supernode.
        
        Args:
            supernode: ID of the supernode
            
        Returns:
            int: Number of original nodes
        """
        return len(self.supernodes.get(supernode, set()))
    
    def get_expected_adjacency_matrix(self):
        """
        Compute the expected adjacency matrix for the summary.
        
        Returns:
            numpy.ndarray: Expected adjacency matrix
        """
        if self.original_graph is None:
            raise ValueError("Original graph not available")
        
        n = self.original_graph.number_of_nodes()
        nodes = list(self.original_graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        # Initialize expected adjacency matrix
        A_expected = np.zeros((n, n))
        
        # Map each node to its supernode
        node_to_supernode = {}
        for supernode, nodes_set in self.supernodes.items():
            for node in nodes_set:
                node_to_supernode[node] = supernode
        
        # Fill in the expected adjacency matrix
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if i == j:  # No self-loops
                    continue
                    
                supernode_u = node_to_supernode[u]
                supernode_v = node_to_supernode[v]
                
                if supernode_u == supernode_v:
                    # Nodes in the same supernode
                    supernode_size = self.get_supernode_size(supernode_u)
                    internal_edges = self.supernode_edge_counts[supernode_u]
                    # Using Equation 3.1 from the paper
                    if supernode_size > 1:
                        A_expected[i, j] = 2 * internal_edges / (supernode_size * (supernode_size - 1))
                else:
                    # Nodes in different supernodes
                    cross_edges = self.get_superedge_count(supernode_u, supernode_v)
                    # Using Equation 3.2 from the paper
                    size_u = self.get_supernode_size(supernode_u)
                    size_v = self.get_supernode_size(supernode_v)
                    A_expected[i, j] = cross_edges / (size_u * size_v)
        
        return A_expected
    
    def compute_reconstruction_error(self):
        """
        Compute the reconstruction error as defined in the paper.
        
        Returns:
            float: Reconstruction error
        """
        if self.original_graph is None:
            raise ValueError("Original graph not available")
            
        # Get adjacency matrices
        A_original = nx.to_numpy_array(self.original_graph)
        A_expected = self.get_expected_adjacency_matrix()
        
        # Calculate L1 error according to Definition 5 in the paper
        error = np.abs(A_original - A_expected).sum() / (len(A_original) ** 2)
        return error
    
    def size_in_bits(self):
        """
        Calculate the size in bits of the summary as defined in the paper.
        
        Returns:
            float: Size in bits
        """
        if self.original_graph is None:
            raise ValueError("Original graph not available")
            
        # Get numbers
        n = self.original_graph.number_of_nodes()
        k = len(self.supernodes)
        
        # Calculate maximum edge weight
        w_max = max([max(self.supernode_edge_counts.values(), default=0)] + 
                   [max(self.superedge_counts.values(), default=0)])
        
        if w_max == 0:
            w_max = 1  # Avoid log(0)
        
        # Using Equation 4 from the paper
        size_bits = (
            len(self.superedge_counts) * (2 * np.ceil(np.log2(k)) + np.ceil(np.log2(w_max))) +
            n * np.ceil(np.log2(k))
        )
        
        return size_bits
    
    def to_networkx_graph(self):
        """
        Convert the summary to a NetworkX graph for visualization.
        
        Returns:
            nx.Graph: Graph representation of the summary
        """
        G = nx.Graph()
        
        # Add nodes with size attributes
        for supernode, nodes in self.supernodes.items():
            G.add_node(supernode, size=len(nodes), original_nodes=list(nodes))
        
        # Add edges with weight attributes
        for (sn1, sn2), count in self.superedge_counts.items():
            G.add_edge(sn1, sn2, weight=count)
        
        return G
    
    def summary_stats(self):
        """
        Print summary statistics.
        
        Returns:
            dict: Dictionary of summary statistics
        """
        stats = {
            "num_supernodes": len(self.supernodes),
            "num_superedges": len(self.superedge_counts),
            "avg_supernode_size": sum(len(nodes) for nodes in self.supernodes.values()) / len(self.supernodes) if self.supernodes else 0,
            "max_supernode_size": max(len(nodes) for nodes in self.supernodes.values()) if self.supernodes else 0,
            "min_supernode_size": min(len(nodes) for nodes in self.supernodes.values()) if self.supernodes else 0,
        }
        
        if self.original_graph is not None:
            stats["reconstruction_error"] = self.compute_reconstruction_error()
            stats["size_reduction"] = self.size_in_bits() / (2 * self.original_graph.number_of_edges() * 
                                                           np.ceil(np.log2(self.original_graph.number_of_nodes())))
        
        return stats