import numpy as np
from graph_summary import GraphSummary
from tqdm import tqdm
import random


class KGs:
    """
    Implementation of the k-Graph Summarization (k-Gs) algorithm from 
    the LeFevre and Terzi paper.
    """
    
    def __init__(self):
        """Initialize the k-Gs algorithm."""
        pass
    
    def greedy(self, G, k):
        """
        Greedy algorithm for k-Gs (full enumeration of node pairs).
        
        Args:
            G (nx.Graph): Input graph
            k (int): Target number of supernodes
            
        Returns:
            GraphSummary: Graph summary with k supernodes
        """
        print(f"Running Greedy k-Gs algorithm with target k={k}...")
        
        # Initialize with trivial summary
        summary = GraphSummary(G)
        n = G.number_of_nodes()
        
        # Continue merging until we have at most k supernodes
        with tqdm(total=n-k) as pbar:
            while len(summary.supernodes) > k:
                # Find best pair to merge
                best_pair = None
                min_error_increase = float('inf')
                
                # Enumerate all pairs of supernodes
                supernode_ids = list(summary.supernodes.keys())
                for i in range(len(supernode_ids)):
                    for j in range(i+1, len(supernode_ids)):
                        sn1, sn2 = supernode_ids[i], supernode_ids[j]
                        
                        # Create a temporary summary to test this merge
                        temp_summary = self._clone_summary(summary)
                        temp_summary.merge_supernodes(sn1, sn2)
                        
                        # Calculate reconstruction error after merge
                        new_error = temp_summary.compute_reconstruction_error()
                        old_error = summary.compute_reconstruction_error()
                        error_change = new_error - old_error
                        
                        # Update best pair if this is better
                        if error_change < min_error_increase:
                            min_error_increase = error_change
                            best_pair = (sn1, sn2)
                
                # Merge the best pair
                if best_pair:
                    summary.merge_supernodes(best_pair[0], best_pair[1])
                    pbar.update(1)
                    pbar.set_postfix({"error": f"{summary.compute_reconstruction_error():.6f}"})
                else:
                    break  # No valid pairs found
        
        print(f"Greedy k-Gs completed. Final summary has {len(summary.supernodes)} supernodes "
              f"with reconstruction error {summary.compute_reconstruction_error():.6f}")
        
        return summary
    
    def sample_pairs(self, G, k, c=1.0):
        """
        SamplePairs algorithm for k-Gs as described in the paper.
        Instead of enumerating all pairs, sample a subset of pairs in each iteration.
        
        Args:
            G (nx.Graph): Input graph
            k (int): Target number of supernodes
            c (float): Sampling parameter (controls how many pairs to sample)
            
        Returns:
            GraphSummary: Graph summary with k supernodes
        """
        print(f"Running SamplePairs k-Gs algorithm with target k={k}, c={c}...")
        
        # Initialize with trivial summary
        summary = GraphSummary(G)
        n = G.number_of_nodes()
        
        # Continue merging until we have at most k supernodes
        with tqdm(total=n-k) as pbar:
            while len(summary.supernodes) > k:
                # Determine number of pairs to sample
                num_supernodes = len(summary.supernodes)
                num_samples = max(1, int(c * np.log2(num_supernodes)))
                
                # Sample pairs
                supernode_ids = list(summary.supernodes.keys())
                if len(supernode_ids) < 2:
                    break
                
                all_pairs = [(i, j) for i in range(len(supernode_ids)) 
                           for j in range(i+1, len(supernode_ids))]
                
                # If fewer pairs than samples, use all pairs
                if len(all_pairs) <= num_samples:
                    sampled_pairs = all_pairs
                else:
                    sampled_pairs = random.sample(all_pairs, num_samples)
                
                # Find best pair among samples
                best_pair = None
                min_error_increase = float('inf')
                
                for i, j in sampled_pairs:
                    sn1, sn2 = supernode_ids[i], supernode_ids[j]
                    
                    # Create a temporary summary to test this merge
                    temp_summary = self._clone_summary(summary)
                    temp_summary.merge_supernodes(sn1, sn2)
                    
                    # Calculate reconstruction error after merge
                    new_error = temp_summary.compute_reconstruction_error()
                    old_error = summary.compute_reconstruction_error()
                    error_change = new_error - old_error
                    
                    # Update best pair if this is better
                    if error_change < min_error_increase:
                        min_error_increase = error_change
                        best_pair = (sn1, sn2)
                
                # Merge the best pair
                if best_pair:
                    summary.merge_supernodes(best_pair[0], best_pair[1])
                    pbar.update(1)
                    pbar.set_postfix({"error": f"{summary.compute_reconstruction_error():.6f}"})
                else:
                    break  # No valid pairs found
        
        print(f"SamplePairs k-Gs completed. Final summary has {len(summary.supernodes)} supernodes "
              f"with reconstruction error {summary.compute_reconstruction_error():.6f}")
        
        return summary
    
    def linear_check(self, G, k):
        """
        LinearCheck algorithm for k-Gs as described in the paper.
        In each iteration, randomly select one supernode and find the best
        supernode to merge it with.
        
        Args:
            G (nx.Graph): Input graph
            k (int): Target number of supernodes
            
        Returns:
            GraphSummary: Graph summary with k supernodes
        """
        print(f"Running LinearCheck k-Gs algorithm with target k={k}...")
        
        # Initialize with trivial summary
        summary = GraphSummary(G)
        n = G.number_of_nodes()
        
        # Continue merging until we have at most k supernodes
        with tqdm(total=n-k) as pbar:
            while len(summary.supernodes) > k:
                # Randomly select a supernode
                supernode_ids = list(summary.supernodes.keys())
                if len(supernode_ids) < 2:
                    break
                
                selected = random.choice(supernode_ids)
                
                # Find best pair involving the selected supernode
                best_pair = None
                min_error_increase = float('inf')
                
                for other in supernode_ids:
                    if other == selected:
                        continue
                    
                    # Create a temporary summary to test this merge
                    temp_summary = self._clone_summary(summary)
                    temp_summary.merge_supernodes(selected, other)
                    
                    # Calculate reconstruction error after merge
                    new_error = temp_summary.compute_reconstruction_error()
                    old_error = summary.compute_reconstruction_error()
                    error_change = new_error - old_error
                    
                    # Update best pair if this is better
                    if error_change < min_error_increase:
                        min_error_increase = error_change
                        best_pair = (selected, other)
                
                # Merge the best pair
                if best_pair:
                    summary.merge_supernodes(best_pair[0], best_pair[1])
                    pbar.update(1)
                    pbar.set_postfix({"error": f"{summary.compute_reconstruction_error():.6f}"})
                else:
                    break  # No valid pairs found
        
        print(f"LinearCheck k-Gs completed. Final summary has {len(summary.supernodes)} supernodes "
              f"with reconstruction error {summary.compute_reconstruction_error():.6f}")
        
        return summary
    
    def random_merge(self, G, k):
        """
        Random merging algorithm as a baseline.
        
        Args:
            G (nx.Graph): Input graph
            k (int): Target number of supernodes
            
        Returns:
            GraphSummary: Graph summary with k supernodes
        """
        print(f"Running Random k-Gs algorithm with target k={k}...")
        
        # Initialize with trivial summary
        summary = GraphSummary(G)
        n = G.number_of_nodes()
        
        # Continue merging until we have at most k supernodes
        with tqdm(total=n-k) as pbar:
            while len(summary.supernodes) > k:
                # Randomly select a pair of supernodes to merge
                supernode_ids = list(summary.supernodes.keys())
                if len(supernode_ids) < 2:
                    break
                
                i, j = random.sample(range(len(supernode_ids)), 2)
                sn1, sn2 = supernode_ids[i], supernode_ids[j]
                
                # Merge the random pair
                summary.merge_supernodes(sn1, sn2)
                pbar.update(1)
                pbar.set_postfix({"error": f"{summary.compute_reconstruction_error():.6f}"})
        
        print(f"Random k-Gs completed. Final summary has {len(summary.supernodes)} supernodes "
              f"with reconstruction error {summary.compute_reconstruction_error():.6f}")
        
        return summary
    
    def _clone_summary(self, summary):
        """
        Create a deep copy of a graph summary.
        
        Args:
            summary (GraphSummary): Summary to clone
            
        Returns:
            GraphSummary: Cloned summary
        """
        new_summary = GraphSummary(summary.original_graph)
        new_summary.supernodes = {k: set(v) for k, v in summary.supernodes.items()}
        new_summary.supernode_edge_counts = dict(summary.supernode_edge_counts)
        new_summary.superedge_counts = dict(summary.superedge_counts)
        return new_summary