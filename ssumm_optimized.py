import networkx as nx
import numpy as np
import random
import time
import math
import logging
import copy
from collections import defaultdict, Counter
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import psutil
import os
from scipy.sparse import csr_matrix, lil_matrix

# Import our implementations
from graph_summary import GraphSummary
from ssumm_cost_function import SsummCostFunction

class OptimizedSSumM:
    """
    Optimized implementation of the SSumM (Sparse Summarization of Massive Graphs) algorithm.
    
    This implementation includes:
    1. Parallelized processing of candidate sets
    2. Memory-efficient data structures
    3. Optimized cost calculations
    4. Progress tracking and performance monitoring
    """
    
    def __init__(self, max_iterations=20, max_candidate_set_size=500, max_recursion_depth=10,
                 n_jobs=-1, memory_limit_gb=None, use_sparse_matrices=True):
        """
        Initialize the optimized SSumM algorithm.
        
        Args:
            max_iterations (int): Maximum number of iterations for the algorithm
            max_candidate_set_size (int): Maximum number of nodes in a candidate set
            max_recursion_depth (int): Maximum recursion depth for shingle-based partitioning
            n_jobs (int): Number of parallel jobs to run (-1 means use all available cores)
            memory_limit_gb (float): Memory limit in GB (None means no limit)
            use_sparse_matrices (bool): Whether to use sparse matrices for large graphs
        """
        self.max_iterations = max_iterations
        self.max_candidate_set_size = max_candidate_set_size
        self.max_recursion_depth = max_recursion_depth
        self.use_sparse_matrices = use_sparse_matrices
        
        # Set number of parallel jobs
        if n_jobs == -1:
            self.n_jobs = mp.cpu_count()
        else:
            self.n_jobs = min(n_jobs, mp.cpu_count())
        
        # Set memory limit
        if memory_limit_gb is None:
            # Use 75% of available memory by default
            total_memory = psutil.virtual_memory().total
            self.memory_limit_bytes = int(0.75 * total_memory)
        else:
            self.memory_limit_bytes = int(memory_limit_gb * 1024**3)
        
        # Initialize logger
        self.logger = logging.getLogger("OptimizedSSumM")
        self.logger.setLevel(logging.INFO)
        
        # Add console handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Performance metrics tracking
        self.performance_metrics = {
            'iteration_times': [],
            'memory_usage': [],
            'candidate_set_counts': [],
            'merge_counts': [],
            'sparsification_counts': []
        }
    
    def summarize(self, G, target_size_bits, track_convergence=True):
        """
        Summarize the input graph to a smaller summary graph with optimized performance.
        
        Args:
            G (nx.Graph): Input graph to be summarized
            target_size_bits (int): Target size of the summary graph in bits
            track_convergence (bool): Whether to track convergence statistics
            
        Returns:
            GraphSummary: The summarized graph
            dict: Performance metrics and convergence data
        """
        self.logger.info(f"Starting OptimizedSSumM summarization for graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        self.logger.info(f"Target summary size: {target_size_bits} bits")
        self.logger.info(f"Using {self.n_jobs} parallel processes")
        
        # Check if input graph is too large for in-memory processing
        estimated_memory = self._estimate_memory_requirements(G)
        if estimated_memory > self.memory_limit_bytes:
            self.logger.warning(f"Input graph may require {estimated_memory / 1024**3:.2f} GB of memory, "
                              f"which exceeds the limit of {self.memory_limit_bytes / 1024**3:.2f} GB")
            self.logger.info("Switching to memory-efficient mode")
            use_memory_efficient = True
        else:
            use_memory_efficient = False
        
        # Prepare for convergence tracking
        if track_convergence:
            convergence_data = {
                'iteration': [],
                'num_supernodes': [],
                'num_superedges': [],
                'reconstruction_error_l1': [],
                'reconstruction_error_l2': [],
                'size_in_bits': [],
                'time_elapsed': []
            }
        
        # Start with the trivial summary (each node in its own supernode)
        start_time = time.time()
        summary = GraphSummary(G)
        
        # Initialize the cost function with optimization
        cost_function = self._create_optimized_cost_function(G, use_memory_efficient)
        
        # Iterate until we reach the target size or max iterations
        current_iteration = 1
        total_merges = 0
        total_sparsifications = 0
        
        while current_iteration <= self.max_iterations and summary.size_in_bits() > target_size_bits:
            iteration_start_time = time.time()
            self.logger.info(f"Iteration {current_iteration}/{self.max_iterations}")
            self.logger.info(f"Current summary size: {summary.size_in_bits()} bits")
            
            # Calculate adaptive threshold for this iteration
            threshold = self._calculate_threshold(current_iteration)
            self.logger.info(f"Current threshold: {threshold}")
            
            # Monitor memory usage
            current_memory = psutil.Process(os.getpid()).memory_info().rss
            self.performance_metrics['memory_usage'].append(current_memory)
            self.logger.info(f"Current memory usage: {current_memory / 1024**3:.2f} GB")
            
            # Generate candidate sets for this iteration
            candidate_sets = self._generate_candidate_sets(summary)
            num_candidate_sets = len(candidate_sets)
            self.performance_metrics['candidate_set_counts'].append(num_candidate_sets)
            self.logger.info(f"Generated {num_candidate_sets} candidate sets")
            
            # Process candidate sets in parallel if there are enough of them and they're not too small
            if num_candidate_sets >= self.n_jobs and self.n_jobs > 1:
                # Filter out very small candidate sets
                valid_candidate_sets = [cs for cs in candidate_sets if len(cs) > 1]
                
                if len(valid_candidate_sets) > 0:
                    self.logger.info(f"Processing {len(valid_candidate_sets)} candidate sets in parallel")
                    
                    # Prepare shared summary object (need to be thread-safe)
                    # For real parallelism, we'd need a more complex approach with a shared-memory structure
                    # or a distributed processing framework. This is a simplified version.
                    
                    # Create copies for each process
                    summaries = [copy.deepcopy(summary) for _ in range(len(valid_candidate_sets))]
                    iteration_merges = 0
                    
                    # Process each candidate set with a separate summary copy
                    results = []
                    with mp.Pool(processes=min(self.n_jobs, len(valid_candidate_sets))) as pool:
                        process_func = partial(self._process_candidate_set_wrapper, 
                                              threshold=threshold, 
                                              cost_function=cost_function)
                        
                        # Map function to all candidate sets
                        results = pool.starmap(process_func, 
                                              zip(summaries, valid_candidate_sets))
                    
                    # Merge results back into a single summary
                    for result_summary, merges, sparsifications in results:
                        iteration_merges += merges
                        total_sparsifications += sparsifications
                        
                        # Integrate the merged supernodes
                        # This is a simplified approach; in practice, we'd need a more
                        # sophisticated method to combine the results from different processes
                        
                        # For now, we'll take the summary with the most merges
                        if merges > 0 and result_summary.size_in_bits() < summary.size_in_bits():
                            summary = result_summary
                            break
                
                else:
                    # Process sequentially if no valid candidate sets for parallelization
                    iteration_merges = 0
                    for candidate_set in tqdm(candidate_sets, desc="Processing candidate sets"):
                        merges, sparsifications = self._process_candidate_set(
                            summary, candidate_set, threshold, cost_function)
                        iteration_merges += merges
                        total_sparsifications += sparsifications
            
            else:
                # Process sequentially
                iteration_merges = 0
                for candidate_set in tqdm(candidate_sets, desc="Processing candidate sets"):
                    merges, sparsifications = self._process_candidate_set(
                        summary, candidate_set, threshold, cost_function)
                    iteration_merges += merges
                    total_sparsifications += sparsifications
            
            total_merges += iteration_merges
            self.performance_metrics['merge_counts'].append(iteration_merges)
            
            # Track iteration time
            iteration_end_time = time.time()
            iteration_time = iteration_end_time - iteration_start_time
            self.performance_metrics['iteration_times'].append(iteration_time)
            self.logger.info(f"Iteration {current_iteration} completed in {iteration_time:.2f} seconds")
            self.logger.info(f"Performed {iteration_merges} merges in this iteration")
            
            # Track convergence data
            if track_convergence:
                current_time = time.time() - start_time
                convergence_data['iteration'].append(current_iteration)
                convergence_data['num_supernodes'].append(len(summary.supernodes))
                convergence_data['num_superedges'].append(len(summary.superedge_counts))
                
                # Compute errors (this can be expensive, so we use a quick approximation for large graphs)
                if G.number_of_nodes() > 10000:
                    # Use sampled error calculation for large graphs
                    l1_error = self._estimate_reconstruction_error(summary, p=1, sample_size=1000)
                    l2_error = self._estimate_reconstruction_error(summary, p=2, sample_size=1000)
                else:
                    l1_error = summary.compute_reconstruction_error(p=1)
                    l2_error = summary.compute_reconstruction_error(p=2)
                
                convergence_data['reconstruction_error_l1'].append(l1_error)
                convergence_data['reconstruction_error_l2'].append(l2_error)
                convergence_data['size_in_bits'].append(summary.size_in_bits())
                convergence_data['time_elapsed'].append(current_time)
            
            # Stop early if no merges were performed
            if iteration_merges == 0:
                self.logger.info("No merges performed in this iteration. Early stopping.")
                break
            
            current_iteration += 1
        
        # Perform final sparsification if needed
        if summary.size_in_bits() > target_size_bits:
            self.logger.info("Performing final sparsification to meet target size")
            sparsification_count = self._further_sparsify(summary, target_size_bits)
            total_sparsifications += sparsification_count
            self.performance_metrics['sparsification_counts'].append(sparsification_count)
        
        # Calculate final statistics
        final_time = time.time() - start_time
        self.logger.info(f"Summarization complete in {final_time:.2f} seconds")
        self.logger.info(f"Final summary: {len(summary.supernodes)} supernodes, {len(summary.superedge_counts)} superedges")
        self.logger.info(f"Final summary size: {summary.size_in_bits()} bits")
        self.logger.info(f"Total merges: {total_merges}, Total sparsifications: {total_sparsifications}")
        
        # Return both the summary and performance data
        if track_convergence:
            return summary, {'performance': self.performance_metrics, 'convergence': convergence_data}
        else:
            return summary, {'performance': self.performance_metrics}
    
    def _estimate_memory_requirements(self, G):
        """
        Estimate memory requirements for processing a graph.
        
        Args:
            G (nx.Graph): Input graph
            
        Returns:
            int: Estimated memory requirement in bytes
        """
        n = G.number_of_nodes()
        m = G.number_of_edges()
        
        # Base graph memory estimate (nodes + edges)
        graph_memory = (n * 16) + (m * 24)  # Rough estimate for Python objects
        
        # Memory for adjacency matrix
        adj_matrix_memory = n * n * 8  # 8 bytes per float64 entry
        
        # Memory for internal data structures
        internal_memory = n * 100  # Approximate for dictionaries, sets, etc.
        
        # Total estimated memory
        if self.use_sparse_matrices and n > 10000:
            # Sparse matrix is more efficient for large sparse graphs
            sparsity = m / (n * (n - 1) / 2)
            sparse_matrix_memory = m * 20  # Rough estimate for CSR format
            return graph_memory + sparse_matrix_memory + internal_memory
        else:
            return graph_memory + adj_matrix_memory + internal_memory
    
    def _create_optimized_cost_function(self, G, use_memory_efficient=False):
        """
        Create an optimized cost function based on graph properties.
        
        Args:
            G (nx.Graph): Input graph
            use_memory_efficient (bool): Whether to use memory-efficient implementation
            
        Returns:
            SsummCostFunction: Optimized cost function instance
        """
        # For now, we're using the standard cost function
        # In a real implementation, we'd customize it based on graph properties
        return SsummCostFunction(G)
    
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
        Generate candidate sets of supernodes for potential merging using an optimized approach.
        
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
        
        # Optimize for different graph sizes
        if len(summary.supernodes) < 1000:
            # For small graphs, use community detection for better partitioning
            candidate_sets = self._community_based_partition(temp_graph)
        else:
            # For larger graphs, use shingle-based partitioning
            candidate_sets = self._shingle_partition(temp_graph)
        
        # Ensure sets are not too large
        final_sets = []
        for candidate_set in candidate_sets:
            # If the set is too large, split it using a more intelligent approach
            if len(candidate_set) > self.max_candidate_set_size:
                # Try to split based on connectivity
                subsets = self._intelligent_split(temp_graph, candidate_set)
                final_sets.extend(subsets)
            else:
                final_sets.append(candidate_set)
        
        return final_sets
    
    def _community_based_partition(self, graph):
        """
        Partition nodes based on community detection for better merging decisions.
        
        Args:
            graph (nx.Graph): Graph to partition
            
        Returns:
            list: List of node sets
        """
        try:
            # Try to use the Louvain community detection algorithm
            import community as community_louvain
            partition = community_louvain.best_partition(graph)
            
            # Group nodes by community
            communities = defaultdict(set)
            for node, community_id in partition.items():
                communities[community_id].add(node)
            
            return list(communities.values())
        
        except ImportError:
            # Fall back to connected components if community detection is not available
            self.logger.warning("Community detection module not available. Falling back to connected components.")
            return [set(c) for c in nx.connected_components(graph)]
    
    def _intelligent_split(self, graph, node_set):
        """
        Intelligently split a large set of nodes based on graph structure.
        
        Args:
            graph (nx.Graph): Original graph
            node_set (set): Set of nodes to split
            
        Returns:
            list: List of smaller node sets
        """
        # Create subgraph of the nodes
        subgraph = graph.subgraph(node_set).copy()
        
        # Try spectral clustering first
        try:
            # Get the normalized Laplacian
            laplacian = nx.normalized_laplacian_matrix(subgraph)
            
            if laplacian.shape[0] > 2:  # Only meaningful for 3+ nodes
                from scipy.sparse.linalg import eigsh
                
                # Compute the second smallest eigenvalue and eigenvector
                eigenvalues, eigenvectors = eigsh(laplacian.astype(float), k=2, which='SM')
                
                # Use the eigenvector corresponding to the second smallest eigenvalue (Fiedler vector)
                fiedler_vector = eigenvectors[:, 1]
                
                # Split nodes based on the sign of the Fiedler vector
                nodes = list(subgraph.nodes())
                partition1 = {nodes[i] for i, val in enumerate(fiedler_vector) if val >= 0}
                partition2 = {nodes[i] for i, val in enumerate(fiedler_vector) if val < 0}
                
                # If the split is too imbalanced, revert to a more balanced approach
                if len(partition1) < 0.1 * len(node_set) or len(partition2) < 0.1 * len(node_set):
                    raise ValueError("Spectral clustering produced imbalanced partitions")
                
                return [partition1, partition2]
        
        except (ValueError, ImportError) as e:
            self.logger.debug(f"Spectral clustering failed: {e}. Using fallback approach.")
        
        # Fallback: Split based on degree
        if subgraph.number_of_nodes() > 0:
            # Sort nodes by degree
            nodes_by_degree = sorted(subgraph.degree(), key=lambda x: x[1], reverse=True)
            nodes_sorted = [n for n, d in nodes_by_degree]
            
            # Split into roughly equal parts
            mid = len(nodes_sorted) // 2
            return [set(nodes_sorted[:mid]), set(nodes_sorted[mid:])]
        
        # Final fallback: random split
        nodes = list(node_set)
        random.shuffle(nodes)
        mid = len(nodes) // 2
        return [set(nodes[:mid]), set(nodes[mid:])]
    
    def _shingle_partition(self, graph, depth=0):
        """
        Partition nodes using optimized shingles to group closely-connected nodes.
        
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
        
        # Optimize shingle calculation by pre-computing neighborhood information
        neighbors = {node: set(graph.neighbors(node)) for node in node_list}
        
        # Calculate shingles more efficiently
        shingles = {}
        for node in node_list:
            # Get minimum hash among node and its neighbors
            node_hash = hash_values[node]
            neighbor_hashes = [hash_values.get(neigh, float('inf')) for neigh in neighbors[node]]
            
            min_hash = min([node_hash] + neighbor_hashes)
            shingles[node] = min_hash
        
        # Group nodes by shingle using Counter for efficiency
        partitions = defaultdict(set)
        for node, shingle in shingles.items():
            partitions[shingle].add(node)
        
        # Better handling of the case when most nodes have the same shingle
        if len(partitions) <= 1 or max(len(nodes) for nodes in partitions.values()) > 0.9 * graph.number_of_nodes():
            # Try a different partitioning approach based on edge betweenness
            try:
                # Use edge betweenness for more meaningful partitioning
                if graph.number_of_edges() < 1000:  # Only use for reasonably small graphs
                    communities = list(nx.algorithms.community.girvan_newman(graph))
                    # Take the partitioning with a reasonable number of communities
                    for comm in communities:
                        if len(comm) >= 2 and len(comm) <= 10:
                            return [set(c) for c in comm]
            except:
                self.logger.debug("Edge betweenness partitioning failed, using fallback")
            
            # Fallback to more balanced random partitioning
            nodes = list(graph.nodes())
            random.shuffle(nodes)
            
            # Create more than 2 partitions for large graphs
            if len(nodes) > 2 * self.max_candidate_set_size:
                num_partitions = max(2, min(10, len(nodes) // self.max_candidate_set_size))
                partition_size = len(nodes) // num_partitions
                partitions = {}
                
                for i in range(num_partitions):
                    start_idx = i * partition_size
                    end_idx = start_idx + partition_size if i < num_partitions - 1 else len(nodes)
                    partitions[i] = set(nodes[start_idx:end_idx])
            else:
                # Simple binary split for smaller graphs
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
    
    def _process_candidate_set_wrapper(self, summary, candidate_set, threshold, cost_function):
        """
        Wrapper for parallel processing of candidate sets.
        
        Args:
            summary (GraphSummary): Current summary graph (copy for this process)
            candidate_set (set): Set of supernode IDs to consider for merging
            threshold (float): Current threshold for accepting merges
            cost_function (SsummCostFunction): Cost function for evaluating merges
            
        Returns:
            tuple: (updated_summary, merge_count, sparsification_count)
        """
        merges, sparsifications = self._process_candidate_set(
            summary, candidate_set, threshold, cost_function)
        return summary, merges, sparsifications
    
    def _process_candidate_set(self, summary, candidate_set, threshold, cost_function):
        """
        Process a candidate set by greedily merging supernodes to minimize the cost function.
        This is an optimized version of the algorithm.
        
        Args:
            summary (GraphSummary): Current summary graph
            candidate_set (set): Set of supernode IDs to consider for merging
            threshold (float): Current threshold for accepting merges
            cost_function (SsummCostFunction): Cost function for evaluating merges
            
        Returns:
            tuple: (merge_count, sparsification_count)
        """
        # Keep track of consecutive skips
        consecutive_skips = 0
        max_skips = max(int(math.log2(len(candidate_set))), 1)
        merge_count = 0
        sparsification_count = 0
        
        # Pre-compute connectivity information for faster lookups
        node_pairs = {}
        
        while consecutive_skips < max_skips and len(candidate_set) > 1:
            # Determine number of pairs to sample
            candidate_list = list(candidate_set)
            num_pairs = min(int(math.log2(len(candidate_set)) + 1), 
                           len(candidate_set) * (len(candidate_set) - 1) // 2)
            
            # Generate all possible pairs if number of pairs is small
            if len(candidate_set) * (len(candidate_set) - 1) // 2 <= num_pairs * 2:
                # Generate all pairs
                all_pairs = [(i, j) for i in range(len(candidate_list)) 
                          for j in range(i+1, len(candidate_list))]
                sampled_pairs = all_pairs
            else:
                # Sample pairs efficiently without generating all pairs
                sampled_pairs = set()
                max_attempts = num_pairs * 2  # Limit attempts to avoid infinite loop
                attempts = 0
                
                while len(sampled_pairs) < num_pairs and attempts < max_attempts:
                    i = random.randint(0, len(candidate_list) - 2)
                    j = random.randint(i + 1, len(candidate_list) - 1)
                    sampled_pairs.add((i, j))
                    attempts += 1
                
                sampled_pairs = list(sampled_pairs)
            
            # Find best pair to merge
            best_pair = None
            best_reduction = -float('inf')
            
            # Evaluate sampled pairs
            for i, j in sampled_pairs:
                supernode1 = candidate_list[i]
                supernode2 = candidate_list[j]
                
                # Skip if already computed
                pair_key = (min(supernode1, supernode2), max(supernode1, supernode2))
                if pair_key in node_pairs:
                    relative_reduction = node_pairs[pair_key]
                else:
                    # Calculate relative reduction in cost
                    relative_reduction = cost_function.calculate_relative_reduction(
                        summary, supernode1, supernode2
                    )
                    # Cache result
                    node_pairs[pair_key] = relative_reduction
                
                # Update best pair if this is better
                if relative_reduction > best_reduction:
                    best_reduction = relative_reduction
                    best_pair = (supernode1, supernode2)
            
            # Check if the best reduction exceeds the threshold
            if best_pair and best_reduction > threshold:
                supernode1, supernode2 = best_pair
                
                # Merge the supernodes
                new_supernode = summary.merge_supernodes(supernode1, supernode2)
                merge_count += 1
                
                # Find optimal superedges for the merged supernode
                sparsification_count += self._optimize_superedges(
                    summary, new_supernode, cost_function)
                
                # Update candidate set and clear cached entries for merged nodes
                candidate_set.discard(supernode1)
                candidate_set.discard(supernode2)
                candidate_set.add(new_supernode)
                
                # Remove cached entries that involve the merged nodes
                for key in list(node_pairs.keys()):
                    if supernode1 in key or supernode2 in key:
                        del node_pairs[key]
                
                # Reset skip counter
                consecutive_skips = 0
            else:
                consecutive_skips += 1
        
        return merge_count, sparsification_count
    
    def _optimize_superedges(self, summary, supernode, cost_function):
        """
        Find the optimal set of superedges for a supernode to minimize the cost function.
        Optimized version that returns the number of sparsifications.
        
        Args:
            summary (GraphSummary): Current summary graph
            supernode: The supernode to optimize superedges for
            cost_function (SsummCostFunction): Cost function for evaluating superedges
            
        Returns:
            int: Number of sparsifications performed
        """
        # Count existing superedges
        existing_count = 0
        for key in list(summary.superedge_counts.keys()):
            sn1, sn2 = key
            if sn1 == supernode or sn2 == supernode:
                existing_count += 1
                del summary.superedge_counts[key]
        
        # Count new superedges created
        new_count = 0
        
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
                    new_count += 1
        
        # Return number of sparsifications (difference between previous and new superedges)
        return max(0, existing_count - new_count)
    
    def _further_sparsify(self, summary, target_size_bits):
        """
        Further sparsify the summary graph to meet the target size constraint.
        This optimized version selects superedges to remove based on their impact on reconstruction error.
        
        Args:
            summary (GraphSummary): Current summary graph
            target_size_bits (int): Target size in bits
            
        Returns:
            int: Number of superedges removed
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
            return 0
        
        self.logger.info(f"Need to remove {num_to_remove} superedges to meet target size")
        
        # Optimization: For large graphs, use a more efficient approach
        if len(summary.superedge_counts) > 1000:
            return self._efficient_sparsification(summary, num_to_remove)
        
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
        removed_count = 0
        for (key, _), _ in zip(sorted_superedges, range(min(num_to_remove, len(sorted_superedges)))):
            if key in summary.superedge_counts:
                del summary.superedge_counts[key]
                removed_count += 1
        
        return removed_count
    
    def _efficient_sparsification(self, summary, num_to_remove):
        """
        A more efficient approach to sparsification for large graphs.
        Instead of calculating exact error for each superedge, use heuristics.
        
        Args:
            summary (GraphSummary): Current summary graph
            num_to_remove (int): Number of superedges to remove
            
        Returns:
            int: Number of superedges removed
        """
        # Heuristic: Prioritize removing superedges with:
        # 1. Low edge count relative to potential edges
        # 2. Connecting large supernodes (higher impact on size, lower on quality)
        
        # Calculate scores for each superedge
        scores = {}
        for key, count in summary.superedge_counts.items():
            sn1, sn2 = key
            
            # Get sizes of supernodes
            size1 = len(summary.supernodes[sn1])
            size2 = len(summary.supernodes[sn2])
            
            # Calculate maximum possible edges between these supernodes
            if sn1 == sn2:
                max_edges = size1 * (size1 - 1) // 2
            else:
                max_edges = size1 * size2
            
            # Calculate density (proportion of possible edges that exist)
            density = count / max_edges if max_edges > 0 else 0
            
            # Score is based on density and supernode sizes
            # Lower score = better candidate for removal
            scores[key] = density * (1 / (size1 + size2))
        
        # Sort superedges by score (ascending)
        sorted_superedges = sorted(scores.items(), key=lambda x: x[1])
        
        # Remove superedges
        removed_count = 0
        for key, _ in sorted_superedges[:num_to_remove]:
            if key in summary.superedge_counts:
                del summary.superedge_counts[key]
                removed_count += 1
        
        return removed_count
    
    def _estimate_reconstruction_error(self, summary, p=1, sample_size=1000):
        """
        Estimate reconstruction error by sampling node pairs instead of computing full matrix.
        This is much more efficient for large graphs.
        
        Args:
            summary (GraphSummary): Summary graph
            p (int): Norm to use (1 for L1, 2 for L2)
            sample_size (int): Number of node pairs to sample
            
        Returns:
            float: Estimated reconstruction error
        """
        if summary.original_graph is None:
            raise ValueError("Original graph not available")
        
        G = summary.original_graph
        nodes = list(G.nodes())
        n = len(nodes)
        
        # Get adjacency information
        adj_dict = {u: set(G.neighbors(u)) for u in nodes}
        
        # Create expected adjacency function
        def get_expected_adj(u, v):
            if u == v:  # No self-loops
                return 0
            
            # Get supernodes
            sn_u = None
            sn_v = None
            
            for sn, members in summary.supernodes.items():
                if u in members:
                    sn_u = sn
                if v in members:
                    sn_v = sn
                if sn_u is not None and sn_v is not None:
                    break
            
            if sn_u == sn_v:
                # Nodes in the same supernode
                supernode_size = len(summary.supernodes[sn_u])
                internal_edges = summary.supernode_edge_counts.get(sn_u, 0)
                
                # Using the formula from the paper
                if supernode_size > 1:
                    return 2 * internal_edges / (supernode_size * (supernode_size - 1))
                else:
                    return 0
            else:
                # Nodes in different supernodes
                key = (min(sn_u, sn_v), max(sn_u, sn_v))
                cross_edges = summary.superedge_counts.get(key, 0)
                
                # Using the formula from the paper
                size_u = len(summary.supernodes[sn_u])
                size_v = len(summary.supernodes[sn_v])
                return cross_edges / (size_u * size_v) if size_u * size_v > 0 else 0
        
        # Sample random node pairs
        error_sum = 0
        for _ in range(sample_size):
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i != j:
                u, v = nodes[i], nodes[j]
                
                # Actual edge
                actual = 1 if v in adj_dict[u] else 0
                
                # Expected value
                expected = get_expected_adj(u, v)
                
                # Add to error
                if p == 1:
                    error_sum += abs(actual - expected)
                elif p == 2:
                    error_sum += (actual - expected) ** 2
        
        # Scale up to estimate full error
        total_pairs = n * (n - 1)
        scaling_factor = total_pairs / sample_size
        
        if p == 2:
            # Return RMSE for p=2
            return (error_sum * scaling_factor) ** 0.5 / total_pairs
        else:
            # Return MAE for p=1
            return error_sum * scaling_factor / total_pairs