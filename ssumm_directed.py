import networkx as nx
import numpy as np
import random
import time
from tqdm import tqdm

class GraphSummaryDirected:
    """
    Representation of a directed graph summary based on the SSumM paper.
    Extends the original implementation to handle directed graphs.
    """
    
    def __init__(self, G=None):
        """
        Initialize a directed graph summary.
        
        Args:
            G (nx.DiGraph, optional): Input directed graph to initialize the summary
        """
        self.supernodes = {}  # Dictionary mapping supernode ID to set of original nodes
        self.supernode_edge_counts_out = {}  # Outgoing edges within supernodes
        self.supernode_edge_counts_in = {}   # Incoming edges within supernodes
        self.supernode_edge_counts_self = {} # Self-loop edges within supernodes
        self.superedge_counts = {}  # Directed edges between supernodes: (source, dest)
        self.original_graph = G
        
        # Initialize with trivial summary if G is provided
        if G is not None:
            self.initialize_trivial_summary(G)
    
    def initialize_trivial_summary(self, G):
        """
        Initialize with a trivial summary where each node is in its own supernode.
        
        Args:
            G (nx.DiGraph): Original directed graph
        """
        self.original_graph = G
        
        # Create one supernode per original node
        for node in G.nodes():
            self.supernodes[node] = {node}
            # No internal edges in singleton supernodes
            self.supernode_edge_counts_out[node] = 0
            self.supernode_edge_counts_in[node] = 0
            self.supernode_edge_counts_self[node] = 0
        
        # Count directed edges between supernodes
        for u, v in G.edges():
            if u == v:  # Self-loop
                self.add_to_superedge_count(u, u, 1)
            else:
                self.add_to_superedge_count(u, v, 1)
    
    def add_to_superedge_count(self, source_supernode, dest_supernode, count=1):
        """
        Add count to the directed superedge from source to destination.
        
        Args:
            source_supernode: ID of source supernode
            dest_supernode: ID of destination supernode
            count (int): Count to add
        """
        key = (source_supernode, dest_supernode)
        self.superedge_counts[key] = self.superedge_counts.get(key, 0) + count
    
    def merge_supernodes(self, supernode1, supernode2):
        """
        Merge two supernodes and update edge counts for directed graphs.
        
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
        new_out_edges = (
            self.supernode_edge_counts_out.get(supernode1, 0) + 
            self.supernode_edge_counts_out.get(supernode2, 0)
        )
        new_in_edges = (
            self.supernode_edge_counts_in.get(supernode1, 0) + 
            self.supernode_edge_counts_in.get(supernode2, 0)
        )
        new_self_edges = (
            self.supernode_edge_counts_self.get(supernode1, 0) + 
            self.supernode_edge_counts_self.get(supernode2, 0)
        )
        
        # 2. Edges between the two supernodes being merged
        # These become internal edges
        edge_key_12 = (supernode1, supernode2)
        edge_key_21 = (supernode2, supernode1)
        
        if edge_key_12 in self.superedge_counts:
            edges_12 = self.superedge_counts[edge_key_12]
            new_out_edges += edges_12
            new_in_edges += edges_12
        
        if edge_key_21 in self.superedge_counts:
            edges_21 = self.superedge_counts[edge_key_21]
            new_out_edges += edges_21
            new_in_edges += edges_21
        
        # Set internal edge counts for new supernode
        self.supernode_edge_counts_out[new_id] = new_out_edges
        self.supernode_edge_counts_in[new_id] = new_in_edges
        self.supernode_edge_counts_self[new_id] = new_self_edges
        
        # Update edges to/from other supernodes
        for supernode in list(self.supernodes.keys()):
            if supernode in (supernode1, supernode2) or supernode == new_id:
                continue
            
            # Outgoing edges from new_id to supernode
            count_out1 = self.get_superedge_count(supernode1, supernode)
            count_out2 = self.get_superedge_count(supernode2, supernode)
            if count_out1 > 0 or count_out2 > 0:
                self.add_to_superedge_count(new_id, supernode, count_out1 + count_out2)
            
            # Incoming edges to new_id from supernode
            count_in1 = self.get_superedge_count(supernode, supernode1)
            count_in2 = self.get_superedge_count(supernode, supernode2)
            if count_in1 > 0 or count_in2 > 0:
                self.add_to_superedge_count(supernode, new_id, count_in1 + count_in2)
        
        # Remove old supernodes and their connections
        if supernode1 != new_id:
            del self.supernodes[supernode1]
            del self.supernode_edge_counts_out[supernode1]
            del self.supernode_edge_counts_in[supernode1]
            del self.supernode_edge_counts_self[supernode1]
        
        if supernode2 != new_id:
            del self.supernodes[supernode2]
            del self.supernode_edge_counts_out[supernode2]
            del self.supernode_edge_counts_in[supernode2]
            del self.supernode_edge_counts_self[supernode2]
        
        # Remove old superedges
        keys_to_remove = []
        for key in self.superedge_counts.keys():
            source, dest = key
            if source in (supernode1, supernode2) or dest in (supernode1, supernode2):
                if source == new_id or dest == new_id:
                    continue
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.superedge_counts[key]
        
        return new_id
    
    def get_superedge_count(self, source_supernode, dest_supernode):
        """
        Get the number of directed edges from source to destination.
        
        Args:
            source_supernode: ID of source supernode
            dest_supernode: ID of destination supernode
            
        Returns:
            int: Number of directed edges from source to destination
        """
        key = (source_supernode, dest_supernode)
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
        Compute the expected adjacency matrix for the directed summary.
        
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
                supernode_u = node_to_supernode[u]
                supernode_v = node_to_supernode[v]
                
                if supernode_u == supernode_v:
                    # Nodes in the same supernode
                    supernode_size = self.get_supernode_size(supernode_u)
                    
                    if u == v:  # Self-loop
                        internal_edges = self.supernode_edge_counts_self[supernode_u]
                        A_expected[i, j] = internal_edges / supernode_size if supernode_size > 0 else 0
                    else:
                        # Different nodes in same supernode
                        # For directed graphs, use outgoing internal edges for probability
                        internal_edges = self.supernode_edge_counts_out[supernode_u]
                        possible_internal_edges = supernode_size * (supernode_size - 1)
                        A_expected[i, j] = internal_edges / possible_internal_edges if possible_internal_edges > 0 else 0
                else:
                    # Nodes in different supernodes
                    cross_edges = self.get_superedge_count(supernode_u, supernode_v)
                    size_u = self.get_supernode_size(supernode_u)
                    size_v = self.get_supernode_size(supernode_v)
                    possible_cross_edges = size_u * size_v
                    A_expected[i, j] = cross_edges / possible_cross_edges if possible_cross_edges > 0 else 0
        
        return A_expected
    
    def compute_reconstruction_error(self):
        """
        Compute the L1 reconstruction error for directed graphs.
        
        Returns:
            float: L1 reconstruction error
        """
        if self.original_graph is None:
            raise ValueError("Original graph not available")
            
        # Get adjacency matrices
        A_original = nx.to_numpy_array(self.original_graph, nodelist=sorted(self.original_graph.nodes()))
        A_expected = self.get_expected_adjacency_matrix()
        
        # Calculate L1 error
        error = np.abs(A_original - A_expected).sum() / (len(A_original) ** 2)
        return error
    
    def size_in_bits(self):
        """
        Calculate the size in bits of the directed summary.
        
        Returns:
            float: Size in bits
        """
        if self.original_graph is None:
            raise ValueError("Original graph not available")
            
        # Get numbers
        n = self.original_graph.number_of_nodes()
        k = len(self.supernodes)
        
        # Calculate maximum edge weight
        w_max = max([max(self.supernode_edge_counts_out.values(), default=0)] + 
                   [max(self.supernode_edge_counts_in.values(), default=0)] +
                   [max(self.supernode_edge_counts_self.values(), default=0)] + 
                   [max(self.superedge_counts.values(), default=0)])
        
        if w_max == 0:
            w_max = 1  # Avoid log(0)
        
        # For directed graphs, encode direction explicitly
        # Each superedge requires: 2 * log2(k) + log2(w_max)
        size_bits = (
            len(self.superedge_counts) * (2 * np.ceil(np.log2(k)) + np.ceil(np.log2(w_max))) +
            n * np.ceil(np.log2(k))
        )
        
        return size_bits
    
    def to_networkx_graph(self):
        """
        Convert the summary to a NetworkX directed graph for visualization.
        
        Returns:
            nx.DiGraph: Directed graph representation of the summary
        """
        G = nx.DiGraph()
        
        # Add nodes with size attributes
        for supernode, nodes in self.supernodes.items():
            G.add_node(supernode, size=len(nodes), original_nodes=list(nodes))
        
        # Add directed edges with weight attributes
        for (source, dest), count in self.superedge_counts.items():
            G.add_edge(source, dest, weight=count)
        
        return G
    
    def summary_stats(self):
        """
        Print summary statistics for directed graph.
        
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
            stats["size_reduction"] = self.size_in_bits() / (len(self.original_graph.edges()) * 
                                                           2 * np.ceil(np.log2(self.original_graph.number_of_nodes())))
        
        return stats


class SSuMMDirected:
    """
    Sparse Summarization of Massive Directed Graphs (SSumM).
    Extends the original SSumM algorithm to handle directed graphs.
    """
    
    def __init__(self, num_iterations=20):
        """
        Initialize SSumM for directed graphs.
        
        Args:
            num_iterations (int): Number of iterations for the algorithm
        """
        self.num_iterations = num_iterations
    
    def summarize(self, G, k_bits, verbose=False):
        """
        Summarize a directed graph to fit within k bits.
        
        Args:
            G (nx.DiGraph): Input directed graph
            k_bits (int): Target size in bits
            verbose (bool): Print progress information
            
        Returns:
            GraphSummaryDirected: Summary graph
        """
        if not G.is_directed():
            raise ValueError("Input graph must be directed. Use nx.DiGraph.")
        
        # Initialize with trivial summary
        summary = GraphSummaryDirected(G)
        
        # Compute initial cost
        current_cost = self._compute_cost(summary)
        
        if verbose:
            print(f"Initial cost: {current_cost:.2f} bits")
            print(f"Target: {k_bits} bits")
        
        for iteration in range(self.num_iterations):
            if verbose:
                print(f"\nIteration {iteration + 1}/{self.num_iterations}")
            
            # Check if we've reached the target size
            current_size = summary.size_in_bits()
            if current_size <= k_bits:
                if verbose:
                    print(f"Target size reached. Current size: {current_size:.2f} bits")
                break
            
            # Generate candidate sets
            candidate_sets = self._generate_candidate_sets(summary)
            
            best_reduction = 0
            best_pair = None
            best_summary = None
            
            # Process each candidate set
            for candidate_set in candidate_sets:
                if len(candidate_set) < 2:
                    continue
                
                # Sample pairs from the candidate set
                pairs_to_check = self._sample_pairs(candidate_set, iteration)
                
                for pair in pairs_to_check:
                    if verbose:
                        print(f"  Checking pair: {pair}")
                    
                    # Create temporary summary to test this merge
                    temp_summary = self._clone_summary(summary)
                    new_supernode = temp_summary.merge_supernodes(pair[0], pair[1])
                    
                    # Sparsify the summary
                    temp_summary = self._sparsify_summary(temp_summary, k_bits)
                    
                    # Calculate cost reduction
                    new_cost = self._compute_cost(temp_summary)
                    reduction = current_cost - new_cost
                    
                    if reduction > best_reduction:
                        best_reduction = reduction
                        best_pair = pair
                        best_summary = temp_summary
            
            # Apply best merge if found
            if best_pair is not None:
                summary = best_summary
                current_cost = self._compute_cost(summary)
                
                if verbose:
                    print(f"  Merged {best_pair[0]} and {best_pair[1]}")
                    print(f"  Cost reduction: {best_reduction:.2f}")
                    print(f"  Current cost: {current_cost:.2f}")
                    print(f"  Current size: {summary.size_in_bits():.2f} bits")
            else:
                if verbose:
                    print("  No beneficial merge found")
        
        # Final sparsification to ensure size constraint
        summary = self._sparsify_summary(summary, k_bits, aggressive=True)
        
        return summary
    
    def _generate_candidate_sets(self, summary):
        """
        Generate candidate sets for merging using shingles (as in original SSumM).
        
        Args:
            summary (GraphSummaryDirected): Current summary
            
        Returns:
            list: List of candidate sets
        """
        supernodes = list(summary.supernodes.keys())
        
        # Create random hash function for shingles
        random_hash = {node: random.randint(0, 2**32 - 1) for node in range(max(supernodes) + 1) if node in supernodes}
        
        # Group supernodes by shingles
        shingle_groups = {}
        
        for supernode in supernodes:
            # Calculate shingle for this supernode
            neighbors = []
            
            # Check outgoing edges
            for dest in supernodes:
                if summary.get_superedge_count(supernode, dest) > 0:
                    neighbors.append(dest)
            
            # Check incoming edges
            for source in supernodes:
                if summary.get_superedge_count(source, supernode) > 0:
                    neighbors.append(source)
            
            # Calculate shingle
            if neighbors:
                shingle = min(random_hash[supernode], *[random_hash[n] for n in neighbors])
            else:
                shingle = random_hash[supernode]
            
            if shingle not in shingle_groups:
                shingle_groups[shingle] = []
            shingle_groups[shingle].append(supernode)
        
        # Return groups as candidate sets
        return list(shingle_groups.values())
    
    def _sample_pairs(self, candidate_set, iteration):
        """
        Sample pairs from a candidate set for merging.
        
        Args:
            candidate_set (list): Candidate supernodes
            iteration (int): Current iteration number
            
        Returns:
            list: List of pairs to check
        """
        # Adaptive sampling: more pairs in early iterations
        num_pairs = min(len(candidate_set) ** 2, max(10, len(candidate_set) // (iteration + 1)))
        
        pairs = []
        for i in range(len(candidate_set)):
            for j in range(i + 1, len(candidate_set)):
                pairs.append((candidate_set[i], candidate_set[j]))
        
        if len(pairs) > num_pairs:
            pairs = random.sample(pairs, num_pairs)
        
        return pairs
    
    def _compute_cost(self, summary):
        """
        Compute the MDL cost for a directed graph summary.
        
        Args:
            summary (GraphSummaryDirected): Summary to evaluate
            
        Returns:
            float: Total cost in bits
        """
        # Model cost (encoding the structure)
        model_cost = summary.size_in_bits()
        
        # Data cost (reconstruction error)
        if summary.original_graph is not None:
            data_cost = summary.compute_reconstruction_error() * (summary.original_graph.number_of_nodes() ** 2)
        else:
            data_cost = 0
        
        return model_cost + data_cost
    
    def _sparsify_summary(self, summary, k_bits, aggressive=False):
        """
        Sparsify the summary by removing less important edges.
        
        Args:
            summary (GraphSummaryDirected): Summary to sparsify
            k_bits (int): Target size in bits
            aggressive (bool): Whether to use aggressive sparsification
            
        Returns:
            GraphSummaryDirected: Sparsified summary
        """
        current_size = summary.size_in_bits()
        if current_size <= k_bits:
            return summary
        
        # Calculate error increase when removing each edge
        edge_candidates = []
        
        for (source, dest), count in summary.superedge_counts.items():
            # Estimate error increase if this edge is removed
            source_size = summary.get_supernode_size(source)
            dest_size = summary.get_supernode_size(dest)
            
            if source_size == 0 or dest_size == 0:
                continue
            
            edge_density = count / (source_size * dest_size)
            error_increase = (1 - edge_density) * count
            
            edge_candidates.append(((source, dest), error_increase))
        
        # Sort by error increase (ascending)
        edge_candidates.sort(key=lambda x: x[1])
        
        # Remove edges until we reach target size
        edges_to_remove = []
        reduction_per_edge = 2 * np.ceil(np.log2(len(summary.supernodes))) + np.ceil(np.log2(max(summary.superedge_counts.values(), default=1)))
        
        while current_size > k_bits and edge_candidates:
            edge, _ = edge_candidates.pop(0)
            edges_to_remove.append(edge)
            current_size -= reduction_per_edge
        
        # Apply edge removals
        for edge in edges_to_remove:
            if edge in summary.superedge_counts:
                del summary.superedge_counts[edge]
        
        return summary
    
    def _clone_summary(self, summary):
        """
        Create a deep copy of a graph summary.
        
        Args:
            summary (GraphSummaryDirected): Summary to clone
            
        Returns:
            GraphSummaryDirected: Cloned summary
        """
        new_summary = GraphSummaryDirected(summary.original_graph)
        new_summary.supernodes = {k: set(v) for k, v in summary.supernodes.items()}
        new_summary.supernode_edge_counts_out = dict(summary.supernode_edge_counts_out)
        new_summary.supernode_edge_counts_in = dict(summary.supernode_edge_counts_in)
        new_summary.supernode_edge_counts_self = dict(summary.supernode_edge_counts_self)
        new_summary.superedge_counts = dict(summary.superedge_counts)
        return new_summary