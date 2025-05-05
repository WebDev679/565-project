import os
import networkx as nx
import pickle
import matplotlib.pyplot as plt
import time

class Evaluation:
    """
    Utilities for evaluating graph summaries.
    """
    
    @staticmethod
    def evaluate_summary(summary, original_graph=None):
        """
        Evaluate a graph summary.
        
        Args:
            summary (GraphSummary): Graph summary to evaluate
            original_graph (nx.Graph, optional): Original graph
            
        Returns:
            dict: Evaluation metrics
        """
        if original_graph is not None:
            summary.original_graph = original_graph
        
        metrics = summary.summary_stats()
        metrics["size_in_bits"] = summary.size_in_bits()
        
        if summary.original_graph:
            # Calculate degree error
            degree_error = Evaluation.compute_degree_error(summary)
            metrics["avg_degree_error"] = degree_error
            
            # Calculate centrality error
            centrality_error = Evaluation.compute_centrality_error(summary)
            metrics["avg_centrality_error"] = centrality_error
        
        return metrics
    
    @staticmethod
    def compute_degree_error(summary):
        """
        Compute the average absolute degree error.
        
        Args:
            summary (GraphSummary): Graph summary
            
        Returns:
            float: Average absolute degree error
        """
        if summary.original_graph is None:
            raise ValueError("Original graph not available")
            
        # Get original degrees
        original_degrees = dict(summary.original_graph.degree())
        nodes = list(summary.original_graph.nodes())
        
        # Get expected adjacency matrix
        A_expected = summary.get_expected_adjacency_matrix()
        
        # Calculate expected degrees
        expected_degrees = {}
        for i, node in enumerate(nodes):
            # Sum the row in the expected adjacency matrix
            expected_degrees[node] = A_expected[i, :].sum()
        
        # Calculate average absolute error
        error_sum = sum(abs(original_degrees[node] - expected_degrees[node]) for node in nodes)
        avg_error = error_sum / len(nodes)
        
        return avg_error
    
    @staticmethod
    def compute_centrality_error(summary):
        """
        Compute the average absolute eigenvector centrality error.
        
        Args:
            summary (GraphSummary): Graph summary
            
        Returns:
            float: Average absolute centrality error
        """
        if summary.original_graph is None:
            raise ValueError("Original graph not available")
            
        # Get original graph
        G_original = summary.original_graph
        nodes = list(G_original.nodes())
        
        # Compute original centrality scores
        try:
            original_centrality = nx.eigenvector_centrality(G_original, max_iter=100)
        except:
            # If it doesn't converge, use degree centrality as fallback
            print("Warning: Eigenvector centrality did not converge. Using degree centrality instead.")
            degrees = dict(G_original.degree())
            total_edges = G_original.number_of_edges()
            original_centrality = {node: degrees[node] / (2 * total_edges) for node in nodes}
        
        # Get expected adjacency matrix
        A_expected = summary.get_expected_adjacency_matrix()
        
        # Create a graph from the expected adjacency matrix
        G_expected = nx.Graph()
        G_expected.add_nodes_from(nodes)
        
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if i < j and A_expected[i, j] > 0:
                    G_expected.add_edge(u, v, weight=A_expected[i, j])
        
        # Compute expected centrality scores
        try:
            expected_centrality = nx.eigenvector_centrality_numpy(G_expected, weight='weight')
        except:
            # If it doesn't converge, use degree centrality as fallback
            print("Warning: Expected eigenvector centrality did not converge. Using Theorem 3.3 instead.")
            # Using Theorem 3.3 from the paper
            degrees = {node: sum(A_expected[nodes.index(node), :]) for node in nodes}
            total_expected_edges = sum(degrees.values()) / 2
            expected_centrality = {node: degrees[node] / (2 * total_expected_edges) for node in nodes}
        
        # Calculate average absolute error
        error_sum = sum(abs(original_centrality[node] - expected_centrality[node]) for node in nodes)
        avg_error = error_sum / len(nodes)
        
        return avg_error
    
    @staticmethod
    def compute_pagerank_error(summary, alpha=0.85):
        """
        Compute the average absolute PageRank error.
        
        Args:
            summary (GraphSummary): Graph summary
            alpha (float): Damping factor for PageRank
            
        Returns:
            float: Average absolute PageRank error
        """
        if summary.original_graph is None:
            raise ValueError("Original graph not available")
            
        # Get original graph
        G_original = summary.original_graph
        nodes = list(G_original.nodes())
        
        # Compute original PageRank scores
        original_pagerank = nx.pagerank(G_original, alpha=alpha)
        
        # Get expected adjacency matrix
        A_expected = summary.get_expected_adjacency_matrix()
        
        # Create a graph from the expected adjacency matrix
        G_expected = nx.Graph()
        G_expected.add_nodes_from(nodes)
        
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if i < j and A_expected[i, j] > 0:
                    G_expected.add_edge(u, v, weight=A_expected[i, j])
        
        # Compute expected PageRank scores
        expected_pagerank = nx.pagerank(G_expected, alpha=alpha, weight='weight')
        
        # Calculate average absolute error
        error_sum = sum(abs(original_pagerank[node] - expected_pagerank[node]) for node in nodes)
        avg_error = error_sum / len(nodes)
        
        return avg_error
    
    @staticmethod
    def compare_algorithms(graph, algorithms, k_values, output_dir=None):
        """
        Run multiple algorithms and compare their performance.
        
        Args:
            graph (nx.Graph): Input graph
            algorithms (dict): Dictionary mapping algorithm names to functions
            k_values (list): List of k values to evaluate
            output_dir (str, optional): Directory to save results
            
        Returns:
            dict: Dictionary of results
        """
        results = {}
        
        for alg_name, alg_func in algorithms.items():
            alg_results = []
            
            for k in k_values:
                start_time = time.time()
                summary = alg_func(graph, k)
                end_time = time.time()
                
                metrics = Evaluation.evaluate_summary(summary)
                metrics["time"] = end_time - start_time
                metrics["k"] = k
                
                alg_results.append(metrics)
                
                print(f"{alg_name} with k={k}: "
                      f"Error={metrics['reconstruction_error']:.6f}, "
                      f"Time={metrics['time']:.2f}s")
            
            results[alg_name] = alg_results
        
        # Save results if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save numerical results
            with open(os.path.join(output_dir, "comparison_results.pickle"), 'wb') as f:
                pickle.dump(results, f)
            
            # Create plots
            Evaluation.plot_algorithm_comparison(results, output_dir)
        
        return results
    
    @staticmethod
    def plot_algorithm_comparison(results, output_dir):
        """
        Plot comparison of algorithms.
        
        Args:
            results (dict): Results from compare_algorithms
            output_dir (str): Directory to save plots
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract algorithms and k values
        algorithms = list(results.keys())
        k_values = sorted(set(result["k"] for alg_results in results.values() for result in alg_results))
        
        # Plot reconstruction error vs. k
        plt.figure(figsize=(10, 6))
        for alg in algorithms:
            errors = [next(r["reconstruction_error"] for r in results[alg] if r["k"] == k) for k in k_values]
            plt.plot(k_values, errors, marker='o', linestyle='-', label=alg)
        
        plt.xlabel('Number of Supernodes (k)')
        plt.ylabel('Reconstruction Error')
        plt.title('Reconstruction Error vs. Number of Supernodes')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, "error_vs_k.png"), dpi=300, bbox_inches='tight')
        
        # Plot runtime vs. k
        plt.figure(figsize=(10, 6))
        for alg in algorithms:
            times = [next(r["time"] for r in results[alg] if r["k"] == k) for k in k_values]
            plt.plot(k_values, times, marker='o', linestyle='-', label=alg)
        
        plt.xlabel('Number of Supernodes (k)')
        plt.ylabel('Runtime (seconds)')
        plt.title('Runtime vs. Number of Supernodes')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, "time_vs_k.png"), dpi=300, bbox_inches='tight')
        
        # Plot degree error vs. k (if available)
        if "avg_degree_error" in results[algorithms[0]][0]:
            plt.figure(figsize=(10, 6))
            for alg in algorithms:
                errors = [next(r["avg_degree_error"] for r in results[alg] if r["k"] == k) for k in k_values]
                plt.plot(k_values, errors, marker='o', linestyle='-', label=alg)
            
            plt.xlabel('Number of Supernodes (k)')
            plt.ylabel('Average Degree Error')
            plt.title('Average Degree Error vs. Number of Supernodes')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(output_dir, "degree_error_vs_k.png"), dpi=300, bbox_inches='tight')
        
        # Plot centrality error vs. k (if available)
        if "avg_centrality_error" in results[algorithms[0]][0]:
            plt.figure(figsize=(10, 6))
            for alg in algorithms:
                errors = [next(r["avg_centrality_error"] for r in results[alg] if r["k"] == k) for k in k_values]
                plt.plot(k_values, errors, marker='o', linestyle='-', label=alg)
            
            plt.xlabel('Number of Supernodes (k)')
            plt.ylabel('Average Centrality Error')
            plt.title('Average Centrality Error vs. Number of Supernodes')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(output_dir, "centrality_error_vs_k.png"), dpi=300, bbox_inches='tight')
        
        # Plot trade-off: reconstruction error vs. time
        plt.figure(figsize=(10, 6))
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
        
        for i, alg in enumerate(algorithms):
            errors = [r["reconstruction_error"] for r in results[alg]]
            times = [r["time"] for r in results[alg]]
            k_vals = [r["k"] for r in results[alg]]
            
            # Sort by k value for consistent coloring
            sorted_data = sorted(zip(k_vals, errors, times))
            k_vals, errors, times = zip(*sorted_data)
            
            # Create scatter plot with k as color
            sc = plt.scatter(times, errors, c=k_vals, cmap='viridis', 
                           marker=markers[i % len(markers)], label=alg, s=100, alpha=0.7)
            
            # Add connecting lines
            plt.plot(times, errors, linestyle='-', alpha=0.3)
        
        plt.xlabel('Runtime (seconds)')
        plt.ylabel('Reconstruction Error')
        plt.title('Trade-off: Reconstruction Error vs. Runtime')
        plt.colorbar(sc, label='Number of Supernodes (k)')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, "error_vs_time.png"), dpi=300, bbox_inches='tight')