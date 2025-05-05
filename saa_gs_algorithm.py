import networkx as nx
import numpy as np
from graph_summary import GraphSummary
from tqdm import tqdm
import time
import random
import math

class SAA_Gs:
    """
    Implementation of the SAA-Gs (Scalable Approximation Algorithm for Graph Summarization)
    from Beg et al. paper. This method uses weighted sampling to efficiently merge supernodes.
    """
    
    def __init__(self):
        """Initialize the SAA-Gs algorithm."""
        self.count_min_sketch = None
    
    def summarize(self, G, k, num_pairs=None, log_n_sampling=True,
                 count_min_sketch_w=50, count_min_sketch_d=2):
        """
        Summarize the graph using the SAA-Gs algorithm.
        
        Args:
            G (nx.Graph): Input graph
            k (int): Target number of supernodes
            num_pairs (int): Number of pairs to sample in each iteration (default: log n or n)
            log_n_sampling (bool): Whether to sample log(n) pairs (True) or n pairs (False)
            count_min_sketch_w (int): Width of the count-min sketch
            count_min_sketch_d (int): Depth of the count-min sketch
            
        Returns:
            GraphSummary: Graph summary with k supernodes
        """
        print(f"Running SAA-Gs algorithm with target k={k}...")
        start_time = time.time()
        
        # Initialize summary with trivial summary
        summary = GraphSummary(G)
        n = G.number_of_nodes()
        
        # Determine number of pairs to sample in each iteration
        if num_pairs is None:
            if log_n_sampling:
                num_pairs = int(math.log2(n))
                print(f"Using log(n) sampling: {num_pairs} pairs per iteration")
            else:
                num_pairs = n
                print(f"Using linear sampling: {num_pairs} pairs per iteration")
        
        # Initialize count-min sketch for approximating reconstruction errors
        self.initialize_count_min_sketch(count_min_sketch_w, count_min_sketch_d)
        
        # Initialize the weight tree with all node pairs
        weight_tree = self.initialize_weight_tree(summary)
        
        # Continue merging until we have at most k supernodes
        with tqdm(total=n-k) as pbar:
            while len(summary.supernodes) > k:
                # Sample node pairs according to weights
                sampled_pairs = self.sample_node_pairs(summary, weight_tree, num_pairs)
                
                # Find best pair to merge
                best_pair = None
                min_error_increase = float('inf')
                
                for pair in sampled_pairs:
                    # Create a temporary summary to test this merge
                    error_change = self.estimate_error_change(summary, pair)
                    
                    # Update best pair if this is better
                    if error_change < min_error_increase:
                        min_error_increase = error_change
                        best_pair = pair
                
                # Merge the best pair
                if best_pair:
                    sn1, sn2 = best_pair
                    new_supernode_id = summary.merge_supernodes(sn1, sn2)
                    
                    # Update weight tree after merge
                    self.update_weight_tree(weight_tree, summary, sn1, sn2, new_supernode_id)
                    
                    pbar.update(1)
                    pbar.set_postfix({"error": f"{summary.compute_reconstruction_error():.6f}"})
                else:
                    break  # No valid pairs found
        
        end_time = time.time()
        print(f"SAA-Gs completed in {end_time - start_time:.2f} seconds.")
        print(f"Final summary has {len(summary.supernodes)} supernodes "
              f"and {len(summary.superedge_counts)} superedges "
              f"with reconstruction error {summary.compute_reconstruction_error():.6f}")
        
        return summary
    
    def initialize_count_min_sketch(self, w, d):
        """
        Initialize the count-min sketch for error approximation.
        
        Args:
            w (int): Width of the sketch
            d (int): Depth of the sketch
        """
        self.cms_width = w
        self.cms_depth = d
        self.count_min_sketch = np.zeros((d, w), dtype=np.float64)
        
        # Generate hash functions (using simple linear hash for demonstration)
        self.hash_a = np.random.randint(1, 1000000, size=d)
        self.hash_b = np.random.randint(0, 1000000, size=d)
        
        print(f"Initialized Count-Min Sketch with dimensions {d}x{w}")
    
    def cms_hash(self, x, i):
        """
        Hash function for count-min sketch.
        
        Args:
            x (int): Value to hash
            i (int): Hash function index
            
        Returns:
            int: Hash value
        """
        return ((self.hash_a[i] * x + self.hash_b[i]) % 1000000) % self.cms_width
    
    def cms_update(self, key, value):
        """
        Update count-min sketch for a key with a value.
        
        Args:
            key (int): Key to update
            value (float): Value to add
        """
        for i in range(self.cms_depth):
            idx = self.cms_hash(key, i)
            self.count_min_sketch[i, idx] += value
    
    def cms_query(self, key):
        """
        Query count-min sketch for a key.
        
        Args:
            key (int): Key to query
            
        Returns:
            float: Estimated value
        """
        return min(self.count_min_sketch[i, self.cms_hash(key, i)] 
                  for i in range(self.cms_depth))
    
    def initialize_weight_tree(self, summary):
        """
        Initialize the weight tree for efficient sampling.
        
        The weight tree is a binary indexed tree (Fenwick tree) that allows
        efficient weighted sampling of node pairs.
        
        Args:
            summary (GraphSummary): Current graph summary
            
        Returns:
            dict: Weight tree structure containing weights and cumulative sums
        """
        print("Initializing weight tree...")
        
        # Initialize weights for all supernode pairs
        weights = {}
        supernode_list = list(summary.supernodes.keys())
        
        for i, sn1 in enumerate(supernode_list):
            for j in range(i+1, len(supernode_list)):
                sn2 = supernode_list[j]
                
                # Calculate initial weight based on potential error reduction
                weight = self.calculate_pair_weight(summary, sn1, sn2)
                pair_key = self.get_pair_key(sn1, sn2)
                weights[pair_key] = weight
        
        # Calculate cumulative weights for binary search during sampling
        cumulative_weights = [0]
        pair_mapping = []  # Maps index to pair
        
        total_weight = 0
        for pair, weight in weights.items():
            total_weight += weight
            cumulative_weights.append(total_weight)
            pair_mapping.append(pair)
        
        # Create weight tree structure
        weight_tree = {
            'weights': weights,
            'cumulative_weights': cumulative_weights,
            'pair_mapping': pair_mapping,
            'total_weight': total_weight
        }
        
        return weight_tree
    
    def calculate_pair_weight(self, summary, sn1, sn2):
        """
        Calculate the weight of a supernode pair for sampling.
        
        The weight is designed to prioritize promising pairs that are likely
        to reduce the reconstruction error when merged.
        
        Args:
            summary (GraphSummary): Current graph summary
            sn1 (int): First supernode ID
            sn2 (int): Second supernode ID
            
        Returns:
            float: Weight value
        """
        # Calculate size of supernodes
        size1 = len(summary.supernodes[sn1])
        size2 = len(summary.supernodes[sn2])
        
        # Check edge density between the supernodes
        key = (min(sn1, sn2), max(sn1, sn2))
        edge_count = summary.superedge_counts.get(key, 0)
        max_edges = size1 * size2
        edge_density = edge_count / max_edges if max_edges > 0 else 0
        
        # Weight is higher for dense connections or similar sized nodes
        size_similarity = 1 - abs(size1 - size2) / (size1 + size2) if (size1 + size2) > 0 else 0
        
        # Combine factors - high density and size similarity get higher weights
        weight = (0.5 * edge_density + 0.5 * size_similarity) + 0.1  # Add small constant to avoid zero weights
        
        return weight
    
    def get_pair_key(self, sn1, sn2):
        """
        Get a unique key for a pair of supernodes.
        
        Args:
            sn1 (int): First supernode ID
            sn2 (int): Second supernode ID
            
        Returns:
            tuple: Ordered pair key
        """
        return (min(sn1, sn2), max(sn1, sn2))
    
    def sample_node_pairs(self, summary, weight_tree, num_pairs):
        """
        Sample supernode pairs according to their weights.
        
        Args:
            summary (GraphSummary): Current graph summary
            weight_tree (dict): Weight tree structure
            num_pairs (int): Number of pairs to sample
            
        Returns:
            list: List of sampled supernode pairs
        """
        cumulative_weights = weight_tree['cumulative_weights']
        pair_mapping = weight_tree['pair_mapping']
        total_weight = weight_tree['total_weight']
        
        if not pair_mapping:  # No pairs available
            return []
        
        sampled_pairs = []
        for _ in range(min(num_pairs, len(pair_mapping))):
            # Binary search to find the pair
            r = random.random() * total_weight
            idx = np.searchsorted(cumulative_weights, r) - 1
            if idx >= len(pair_mapping):
                idx = len(pair_mapping) - 1
            elif idx < 0:
                idx = 0
            
            pair_key = pair_mapping[idx]
            sn1, sn2 = pair_key
            sampled_pairs.append((sn1, sn2))
        
        return sampled_pairs
    
    def estimate_error_change(self, summary, pair):
        """
        Estimate the change in reconstruction error after merging a pair.
        
        This uses the count-min sketch for a fast approximation.
        
        Args:
            summary (GraphSummary): Current graph summary
            pair (tuple): Pair of supernode IDs to merge
            
        Returns:
            float: Estimated change in reconstruction error
        """
        sn1, sn2 = pair
        pair_key = hash(self.get_pair_key(sn1, sn2))
        
        # Check if we have this value cached in the count-min sketch
        cached_value = self.cms_query(pair_key)
        
        if cached_value > 0:
            # Use cached estimate
            return cached_value
        
        # Otherwise, compute a direct estimate
        size1 = len(summary.supernodes[sn1])
        size2 = len(summary.supernodes[sn2])
        
        # Get the edge count between supernodes
        key = (min(sn1, sn2), max(sn1, sn2))
        edge_count = summary.superedge_counts.get(key, 0)
        
        # Calculate error increase due to merging
        # More sophisticated error calculation can be implemented here
        max_edges = size1 * size2
        current_density = edge_count / max_edges if max_edges > 0 else 0
        
        # Error increases when merging dissimilar supernodes
        # Estimate based on how far the density is from 0 or 1
        error_increase = min(current_density, 1 - current_density) * max_edges
        
        # Add internal errors of both supernodes
        internal_edges1 = summary.supernode_edge_counts.get(sn1, 0)
        max_internal1 = (size1 * (size1 - 1)) / 2
        internal_density1 = internal_edges1 / max_internal1 if max_internal1 > 0 else 0
        
        internal_edges2 = summary.supernode_edge_counts.get(sn2, 0)
        max_internal2 = (size2 * (size2 - 1)) / 2
        internal_density2 = internal_edges2 / max_internal2 if max_internal2 > 0 else 0
        
        # Error increases due to internal structure differences
        internal_diff = abs(internal_density1 - internal_density2) * (max_internal1 + max_internal2)
        
        # Combine error components
        error_estimate = error_increase + internal_diff
        
        # Cache the result in the count-min sketch
        self.cms_update(pair_key, error_estimate)
        
        return error_estimate
    
    def update_weight_tree(self, weight_tree, summary, sn1, sn2, new_supernode_id):
        """
        Update the weight tree after merging two supernodes.
        
        Args:
            weight_tree (dict): Weight tree structure
            summary (GraphSummary): Current graph summary
            sn1 (int): First merged supernode ID
            sn2 (int): Second merged supernode ID
            new_supernode_id (int): ID of the newly created supernode
        """
        weights = weight_tree['weights']
        pair_mapping = weight_tree['pair_mapping']
        
        # Remove all pairs involving sn1 or sn2
        pairs_to_remove = []
        for pair in pair_mapping:
            if sn1 in pair or sn2 in pair:
                pairs_to_remove.append(pair)
                weights.pop(pair, None)
        
        for pair in pairs_to_remove:
            if pair in pair_mapping:
                pair_mapping.remove(pair)
        
        # Add new pairs with the new supernode
        for sn in summary.supernodes:
            if sn != new_supernode_id:
                pair = self.get_pair_key(new_supernode_id, sn)
                weight = self.calculate_pair_weight(summary, new_supernode_id, sn)
                weights[pair] = weight
                pair_mapping.append(pair)
        
        # Recalculate cumulative weights
        cumulative_weights = [0]
        total_weight = 0
        
        for pair in pair_mapping:
            weight = weights[pair]
            total_weight += weight
            cumulative_weights.append(total_weight)
        
        # Update the weight tree
        weight_tree['cumulative_weights'] = cumulative_weights
        weight_tree['total_weight'] = total_weight