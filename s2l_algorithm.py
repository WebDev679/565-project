import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import time
from graph_summary import GraphSummary
from tqdm import tqdm
import random

class S2L:
    """
    Implementation of S2L (Summarization via Structural Locality) algorithm from
    Riondato et al. paper. This method uses geometric clustering for graph summarization.
    """
    
    def __init__(self):
        """Initialize the S2L algorithm."""
        pass
    
    def summarize(self, G, k, method='kmeans', dim_reduction=True, 
                  reduction_dim=None, sample_nodes=False, sample_ratio=0.1):
        """
        Summarize the graph using the S2L algorithm.
        
        Args:
            G (nx.Graph): Input graph
            k (int): Target number of supernodes
            method (str): Clustering method ('kmeans' or 'kmedian')
            dim_reduction (bool): Whether to apply dimensionality reduction
            reduction_dim (int): Dimension to reduce to (if None, use min(n/3, 100))
            sample_nodes (bool): Whether to sample nodes for large graphs
            sample_ratio (float): Ratio of nodes to sample if sampling is used
            
        Returns:
            GraphSummary: Graph summary with k supernodes
        """
        print(f"Running S2L algorithm with target k={k} using {method}...")
        start_time = time.time()
        
        n = G.number_of_nodes()
        nodes = list(G.nodes())
        
        # Create adjacency matrix 
        A = nx.to_numpy_array(G, nodelist=nodes)
        
        # Sample nodes if the graph is large and sampling is enabled
        sampled_indices = None
        if sample_nodes and n > 10000:
            sample_size = int(n * sample_ratio)
            print(f"Sampling {sample_size} nodes out of {n} for initial clustering...")
            sampled_indices = np.random.choice(n, sample_size, replace=False)
            A_sampled = A[sampled_indices, :]
        else:
            A_sampled = A
        
        # Apply dimensionality reduction if enabled
        if dim_reduction:
            if reduction_dim is None:
                reduction_dim = min(int(n/3), 100)  # Heuristic for dimension choice
            
            print(f"Applying dimensionality reduction to {reduction_dim} dimensions...")
            pca = PCA(n_components=reduction_dim, random_state=42)
            A_reduced = pca.fit_transform(A_sampled)
        else:
            A_reduced = A_sampled
        
        # Apply clustering
        print(f"Applying {method} clustering to find {k} clusters...")
        if method == 'kmeans':
            clustering = KMeans(n_clusters=k, random_state=42, n_init=10)
            if sampled_indices is not None:
                # Fit clustering on sampled data
                cluster_labels_sampled = clustering.fit_predict(A_reduced)
                
                # Now we need to assign the rest of the nodes
                # We'll use nearest centroid assignment
                centroids = clustering.cluster_centers_
                
                # For non-sampled nodes, compute distances to centroids
                non_sampled_mask = np.ones(n, dtype=bool)
                non_sampled_mask[sampled_indices] = False
                non_sampled_indices = np.where(non_sampled_mask)[0]
                
                # Project non-sampled nodes to the same space
                if dim_reduction:
                    non_sampled_reduced = pca.transform(A[non_sampled_indices, :])
                else:
                    non_sampled_reduced = A[non_sampled_indices, :]
                
                # Assign each non-sampled node to nearest centroid
                cluster_labels_nonsample = np.argmin(
                    np.sqrt(((non_sampled_reduced[:, np.newaxis, :] - 
                             centroids[np.newaxis, :, :]) ** 2).sum(axis=2)), 
                    axis=1
                )
                
                # Combine sampled and non-sampled cluster assignments
                cluster_labels = np.zeros(n, dtype=int)
                cluster_labels[sampled_indices] = cluster_labels_sampled
                cluster_labels[non_sampled_indices] = cluster_labels_nonsample
            else:
                cluster_labels = clustering.fit_predict(A_reduced)
        elif method == 'kmedian':
            # Implement k-median clustering (using k-means as approximation with Manhattan distance)
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import pairwise_distances
            
            # Scale the data for better clustering
            scaler = StandardScaler()
            A_scaled = scaler.fit_transform(A_reduced)
            
            # Use k-means with Manhattan distance-based initialization
            distances = pairwise_distances(A_scaled, metric='manhattan')
            
            # Choose initial centroids using k-means++ like strategy but with Manhattan distance
            centroids_indices = [random.randint(0, len(A_scaled) - 1)]
            
            for _ in range(k - 1):
                dist_sq = np.array([min([distances[i, j] for j in centroids_indices]) 
                                  for i in range(len(A_scaled))])
                probs = dist_sq / dist_sq.sum()
                cumprobs = probs.cumsum()
                r = random.random()
                ind = np.searchsorted(cumprobs, r)
                centroids_indices.append(ind)
            
            # Run k-median iterations
            max_iterations = 100
            prev_medians = None
            cluster_assignments = np.zeros(len(A_scaled), dtype=int)
            
            for iteration in range(max_iterations):
                # Assign points to clusters based on closest medians
                for i in range(len(A_scaled)):
                    cluster_assignments[i] = np.argmin([distances[i, j] for j in centroids_indices])
                
                # Update medians
                new_medians = []
                for c in range(k):
                    points_in_cluster = [i for i, cluster in enumerate(cluster_assignments) if cluster == c]
                    if not points_in_cluster:  # If cluster is empty, keep old centroid
                        new_medians.append(centroids_indices[c])
                        continue
                        
                    # For each cluster, find the point that minimizes the sum of distances to other points
                    if len(points_in_cluster) == 1:
                        median_idx = points_in_cluster[0]
                    else:
                        within_distances = distances[points_in_cluster][:, points_in_cluster]
                        median_idx = points_in_cluster[np.argmin(within_distances.sum(axis=1))]
                    
                    new_medians.append(median_idx)
                
                # Check for convergence
                if prev_medians == new_medians:
                    break
                
                centroids_indices = new_medians
                prev_medians = new_medians
            
            # Assign the final cluster labels
            if sampled_indices is not None:
                # Similar process as for k-means with sampled data
                cluster_labels_sampled = cluster_assignments
                
                # For non-sampled nodes, compute distances to medians
                non_sampled_mask = np.ones(n, dtype=bool)
                non_sampled_mask[sampled_indices] = False
                non_sampled_indices = np.where(non_sampled_mask)[0]
                
                # Project non-sampled nodes to the same space
                if dim_reduction:
                    non_sampled_reduced = pca.transform(A[non_sampled_indices, :])
                    non_sampled_scaled = scaler.transform(non_sampled_reduced)
                else:
                    non_sampled_scaled = scaler.transform(A[non_sampled_indices, :])
                
                # Get median points
                median_points = A_scaled[centroids_indices]
                
                # Compute distances from non-sampled points to medians
                non_sampled_distances = pairwise_distances(
                    non_sampled_scaled, median_points, metric='manhattan'
                )
                
                # Assign to closest median
                cluster_labels_nonsample = np.argmin(non_sampled_distances, axis=1)
                
                # Combine sampled and non-sampled cluster assignments
                cluster_labels = np.zeros(n, dtype=int)
                cluster_labels[sampled_indices] = cluster_labels_sampled
                cluster_labels[non_sampled_indices] = cluster_labels_nonsample
            else:
                cluster_labels = cluster_assignments
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Create supernodes based on clustering
        summary = GraphSummary(G)
        
        # Initialize with an empty set of supernodes
        summary.supernodes = {}
        summary.supernode_edge_counts = {}
        summary.superedge_counts = {}
        
        # Create supernodes from clusters
        for cluster_id in range(k):
            cluster_nodes = [nodes[i] for i in range(n) if cluster_labels[i] == cluster_id]
            
            if cluster_nodes:  # Only create supernode if the cluster has nodes
                supernode_id = cluster_id
                summary.supernodes[supernode_id] = set(cluster_nodes)
                
                # Count internal edges
                internal_edges = 0
                for u in cluster_nodes:
                    for v in cluster_nodes:
                        if u < v and G.has_edge(u, v):
                            internal_edges += 1
                
                summary.supernode_edge_counts[supernode_id] = internal_edges
        
        # Create superedges
        for supernode_i in summary.supernodes:
            for supernode_j in summary.supernodes:
                if supernode_i <= supernode_j:  # Avoid duplicates
                    if supernode_i == supernode_j:
                        continue  # Skip self-loops (they're handled already)
                    
                    # Count edges between supernodes
                    cross_edges = 0
                    for u in summary.supernodes[supernode_i]:
                        for v in summary.supernodes[supernode_j]:
                            if G.has_edge(u, v):
                                cross_edges += 1
                    
                    if cross_edges > 0:
                        # Add superedge
                        summary.superedge_counts[(supernode_i, supernode_j)] = cross_edges
        
        end_time = time.time()
        print(f"S2L completed in {end_time - start_time:.2f} seconds.")
        print(f"Final summary has {len(summary.supernodes)} supernodes "
              f"and {len(summary.superedge_counts)} superedges "
              f"with reconstruction error {summary.compute_reconstruction_error():.6f}")
        
        # Calculate theoretical error guarantees
        self.calculate_error_guarantees(A, cluster_labels, k)
        
        return summary
    
    def calculate_error_guarantees(self, A, cluster_labels, k):
        """
        Calculate error guarantees for the S2L algorithm based on the paper.
        
        Args:
            A (numpy.ndarray): Adjacency matrix
            cluster_labels (numpy.ndarray): Cluster assignments
            k (int): Number of clusters
        """
        n = A.shape[0]
        
        # Calculate the total squared error of clustering
        cluster_centers = np.zeros((k, A.shape[1]))
        cluster_sizes = np.zeros(k)
        
        for i in range(n):
            cluster_id = cluster_labels[i]
            cluster_centers[cluster_id] += A[i]
            cluster_sizes[cluster_id] += 1
        
        # Calculate centroids
        for i in range(k):
            if cluster_sizes[i] > 0:
                cluster_centers[i] /= cluster_sizes[i]
        
        # Calculate sum of squared errors
        sse = 0
        for i in range(n):
            cluster_id = cluster_labels[i]
            sse += np.sum((A[i] - cluster_centers[cluster_id]) ** 2)
        
        # Theoretical error bounds (from Riondato et al.)
        theoretical_l2_error = np.sqrt(sse / (n * n))
        theoretical_l1_error = theoretical_l2_error * n
        
        print(f"Theoretical error guarantees:")
        print(f"  L2 Reconstruction Error Bound: {theoretical_l2_error:.6f}")
        print(f"  L1 Reconstruction Error Bound: {theoretical_l1_error:.6f}")