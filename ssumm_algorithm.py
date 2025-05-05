import networkx as nx
import numpy as np
import random
import time
import math
from tqdm import tqdm
import logging
import copy
from collections import defaultdict

# Import our implementations
from graph_summary import GraphSummary
from ssumm_cost_function import SsummCostFunction

class SSumM:
    """
    Implementation of the SSumM (Sparse Summarization of Massive Graphs) algorithm.
    
    SSumM creates a concise summary of a graph by:
    1. Merging nodes into supernodes
    2. Selectively creating superedges to sparsify the graph
    3. Balancing reconstruction error and output size using MDL principle
    """
    
    def __init__(self, max_iterations=20, max_candidate_set_size=500, max_recursion_depth=10):
        """
        Initialize the SSumM algorithm.
        
        Args:
            max_iterations (int): Maximum number of iterations for the algorithm
            max_candidate_set_size (int): Maximum number of nodes in a candidate set
            max_recursion_depth (int): Maximum recursion depth for shingle-based partitioning
        """
        self.max_iterations = max_iterations
        self.max_candidate_set_size = max_candidate_set_size
        self.max_recursion_depth = max_recursion_depth
        self.logger = logging.getLogger("SSumM")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def summarize(self, G, target_size_bits):
        """
        Summarize the input graph to a smaller summary graph.
        
        Args:
            G (nx.Graph): Input graph to be summarized
            target_size_bits (int): Target size of the summary graph in bits
            
        Returns:
            GraphSummary: The summarized graph
        """
        self.logger.info(f"Starting SSumM summarization for graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        self.logger.info(f"Target summary size: {target_size_bits} bits")
        
        # Start with the trivial summary (each node in its own supernode)
        summary = GraphSummary(G)
        
        # Initialize the cost function
        cost_function = SsummCostFunction(G)
        
        # Iterate until we reach the target size or max iterations
        current_iteration = 1
        
        while current_iteration <= self.max_iterations and summary.size_in_bits() > target_size_bits:
            self.logger.info(f"Iteration {current_iteration}/{self.max_iterations}")
            self.logger.info(f"Current summary size: {summary.size_in_bits()} bits")
            
            # Calculate adaptive threshold for this iteration
            threshold = self._calculate_threshold(current_iteration)
            self.logger.info(f"Current threshold: {threshold}")
            
            # Generate candidate sets for this iteration
            candidate_sets = self._generate_candidate_sets(summary)
            self.logger.info(f"Generated {len(candidate_sets)} candidate sets")
            
            # Process each candidate set
            for candidate_set in tqdm(candidate_sets, desc="Processing candidate sets"):
                self._process_candidate_set(summary, candidate_set, threshold, cost_function)
            
            current_iteration += 1
        
        # Perform final sparsification if needed
        if summary.size_in_bits() > target_size_bits:
            self.logger.info("Performing final sparsification to meet target size")
            self._further_sparsify(summary, target_size_bits)
        
        self.logger.info(f"Summarization complete. Final summary: {len(summary.supernodes)} supernodes, {len(summary.superedge_counts)} superedges")
        self.logger.info(f"Final summary size: {summary.size_in_bits()} bits")
        
        return summary
    
    def _calculate_threshold(self, iteration):
        """
        Calculate the adaptive threshold for the current iteration.
        The threshold decreases over time to allow merging less optimal pairs in later iterations.
        
        Args:
            iteration (int): Current iteration number
            
        Returns:
            float: Threshold value for this iteration
        """
        if iteration < self.max_iterations:
            return (1 + iteration) ** (-1)
        else:
            return 0  # In the final iteration, accept any merge that reduces cost
    
    def _generate_candidate_sets(self, summary):
        """
        Generate candidate sets of supernodes for potential merging.
        Uses shingle-based partitioning to group closely-connected supernodes.
        
        Args:
            summary (GraphSummary): Current summary graph
            
        Returns:
            list: List of candidate sets (each a set of supernode IDs)
        """
        # Create a temporary graph for node connectivity
        temp_graph = nx.Graph()
        temp_graph.add_nodes_from(summary.supernodes.keys())
        
        # Add edges between supernodes that have connections in the original graph
        for (sn1, sn2), count in summary.superedge_counts.items():
            temp_graph.add_edge(sn1, sn2)
        
        # Use shingle-based partitioning
        candidate_sets = self._shingle_partition(temp_graph)
        
        # Ensure sets are not too large
        final_sets = []
        for candidate_set in candidate_sets:
            # If the set is too large, split it randomly
            if len(candidate_set) > self.max_candidate_set_size:
                nodes = list(candidate_set)
                random.shuffle(nodes)
                for i in range(0, len(nodes), self.max_candidate_set_size):
                    subset = set(nodes[i:i+self.max_candidate_set_size])
                    final_sets.append(subset)
            else:
                final_sets.append(candidate_set)
        
        return final_sets
    
    def _shingle_partition(self, graph, depth=0):
        """
        Partition nodes using shingles to group closely-connected nodes.
        
        Args:
            graph (nx.Graph): Graph to partition
            depth (int): Current recursion depth
            
        Returns:
            list: List of node sets
        """
        # Base cases
        if graph.number_of_nodes() == 0:
            return []
        
        if graph.number_of_nodes() <= self.max_candidate_set_size or depth >= self.max_recursion_depth:
            return [set(graph.nodes())]
        
        # Create a random hash function for shingles
        node_list = list(graph.nodes())
        hash_values = {node: random.randint(1, 1000000) for node in node_list}
        
        # Calculate shingles for each node (min hash of node + neighbors)
        shingles = {}
        for node in node_list:
            # Get minimum hash among node and its neighbors
            node_hash = hash_values[node]
            neighbor_hashes = [hash_values.get(neighbor, float('inf')) 
                              for neighbor in graph.neighbors(node)]
            
            min_hash = min([node_hash] + neighbor_hashes)
            shingles[node] = min_hash
        
        # Group nodes by shingle
        partitions = defaultdict(set)
        for node, shingle in shingles.items():
            partitions[shingle].add(node)
        
        # If no meaningful partitioning happened, try random partitioning
        if len(partitions) <= 1:
            # Random partitioning as fallback
            nodes = list(graph.nodes())
            random.shuffle(nodes)
            mid = len(nodes) // 2
            partitions = {
                0: set(nodes[:mid]),
                1: set(nodes[mid:])
            }
        
        # Recursively partition large sets
        result = []
        for partition_nodes in partitions.values():
            if len(partition_nodes) > self.max_candidate_set_size:
                subgraph = graph.subgraph(partition_nodes).copy()
                sub_partitions = self._shingle_partition(subgraph, depth + 1)
                result.extend(sub_partitions)
            else:
                result.append(partition_nodes)
        
        return result
    
    def _process_candidate_set(self, summary, candidate_set, threshold, cost_function):
        """
        Process a candidate set by greedily merging supernodes to minimize the cost function.
        
        Args:
            summary (GraphSummary): Current summary graph
            candidate_set (set): Set of supernode IDs to consider for merging
            threshold (float): Current threshold for accepting merges
            cost_function (SsummCostFunction): Cost function for evaluating merges
        """
        # Keep track of consecutive skips
        consecutive_skips = 0
        max_skips = max(int(math.log2(len(candidate_set))), 1)
        
        while consecutive_skips < max_skips and len(candidate_set) > 1:
            # Randomly sample log2(n) pairs to find the best merge
            best_pair = None
            best_reduction = -float('inf')
            
            # Number of pairs to sample
            num_pairs = min(int(math.log2(len(candidate_set)) + 1), len(candidate_set) * (len(candidate_set) - 1) // 2)
            
            # Sample pairs
            candidate_list = list(candidate_set)
            all_pairs = [(i, j) for i in range(len(candidate_list)) 
                        for j in range(i+1, len(candidate_list))]
            
            if len(all_pairs) <= num_pairs:
                sampled_pairs = all_pairs
            else:
                sampled_pairs = random.sample(all_pairs, num_pairs)
            
            # Evaluate each sampled pair
            for i, j in sampled_pairs:
                supernode1 = candidate_list[i]
                supernode2 = candidate_list[j]
                
                # Calculate relative reduction in cost
                relative_reduction = cost_function.calculate_relative_reduction(
                    summary, supernode1, supernode2
                )
                
                # Update best pair if this is better
                if relative_reduction > best_reduction:
                    best_reduction = relative_reduction
                    best_pair = (supernode1, supernode2)
            
            # Check if the best reduction exceeds the threshold
            if best_pair and best_reduction > threshold:
                supernode1, supernode2 = best_pair
                
                # Merge the supernodes
                new_supernode = summary.merge_supernodes(supernode1, supernode2)
                
                # Find optimal superedges for the merged supernode
                self._optimize_superedges(summary, new_supernode, cost_function)
                
                # Update candidate set
                candidate_set.discard(supernode1)
                candidate_set.discard(supernode2)
                candidate_set.add(new_supernode)
                
                # Reset skip counter
                consecutive_skips = 0
            else:
                consecutive_skips += 1
    
    def _optimize_superedges(self, summary, supernode, cost_function):
        """
        Find the optimal set of superedges for a supernode to minimize the cost function.
        
        Args:
            summary (GraphSummary): Current summary graph
            supernode: The supernode to optimize superedges for
            cost_function (SsummCostFunction): Cost function for evaluating superedges
        """
        # Remove all existing superedges connected to this supernode
        for key in list(summary.superedge_counts.keys()):
            sn1, sn2 = key
            if sn1 == supernode or sn2 == supernode:
                del summary.superedge_counts[key]
        
        # For each potential superedge, determine if it should be created
        for other_supernode in summary.supernodes:
            if other_supernode == supernode:
                # Handle self-loop
                A, B = supernode, supernode
            else:
                A, B = min(supernode, other_supernode), max(supernode, other_supernode)
            
            # Get subedges between these supernodes
            E_AB = cost_function.get_subedges_between_supernodes(summary, A, B)
            
            # Only consider creating superedge if there are actual connections
            if len(E_AB) > 0:
                # Get all possible subedges
                Pi_AB = cost_function.get_possible_subedges_between_supernodes(summary, A, B)
                
                # Calculate cost with and without the superedge
                cost_with_superedge = cost_function.calculate_cost_method1(summary, A, B, E_AB, Pi_AB)
                cost_without_superedge = cost_function.calculate_cost_method2(E_AB)
                
                # Create superedge if it reduces cost
                if cost_with_superedge <= cost_without_superedge:
                    key = (A, B)
                    summary.superedge_counts[key] = len(E_AB)
    
    def _further_sparsify(self, summary, target_size_bits):
        """
        Further sparsify the summary graph to meet the target size constraint.
        This is done by removing superedges that increase the reconstruction error the least.
        
        Args:
            summary (GraphSummary): Current summary graph
            target_size_bits (int): Target size in bits
        """
        # Calculate how many superedges to remove
        current_size = summary.size_in_bits()
        k = len(summary.supernodes)
        
        # Get maximum weight for calculating superedge size
        w_max = max([max(summary.supernode_edge_counts.values(), default=0)] + 
                   [max(summary.superedge_counts.values(), default=0)])
        
        # Size of each superedge in bits
        superedge_size = 2 * np.ceil(np.log2(k)) + np.ceil(np.log2(w_max))
        
        # Number of superedges to remove
        num_to_remove = math.ceil((current_size - target_size_bits) / superedge_size)
        
        if num_to_remove <= 0:
            return
        
        # Calculate increase in reconstruction error for removing each superedge
        error_increases = {}
        
        for key, count in summary.superedge_counts.items():
            sn1, sn2 = key
            
            # Create a copy of the summary without this superedge
            temp_summary = copy.deepcopy(summary)
            del temp_summary.superedge_counts[key]
            
            # Calculate increase in L1 reconstruction error
            original_error = summary.compute_reconstruction_error(p=1)
            new_error = temp_summary.compute_reconstruction_error(p=1)
            error_increases[key] = new_error - original_error
        
        # Sort superedges by error increase (remove those with smallest impact first)
        sorted_superedges = sorted(error_increases.items(), key=lambda x: x[1])
        
        # Remove superedges
        for (key, _), _ in zip(sorted_superedges, range(min(num_to_remove, len(sorted_superedges)))):
            if key in summary.superedge_counts:
                del summary.superedge_counts[key]