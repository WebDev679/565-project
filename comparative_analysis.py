import os
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import gc
import pickle
from memory_profiler import profile, memory_usage
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

# Import our algorithm implementations
from graph_utils import GraphUtils
from graph_summary import GraphSummary
from kgs_algorithm import KGs
from s2l_algorithm import S2L
from saa_gs_algorithm import SAA_Gs

class ComparativeAnalysis:
    """
    Class for conducting comparative analysis of different graph summarization algorithms.
    """
    
    def __init__(self, output_dir="./analysis_results"):
        """
        Initialize the comparative analysis.
        
        Args:
            output_dir: Directory to save analysis results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Dictionary to store results
        self.results = {}
        
        # Define algorithms to compare
        self.algorithms = {
            "k-Gs": KGs(),
            "S2L (k-means)": S2L(),
            "SAA-Gs (log n)": SAA_Gs()
        }
        
        # Algorithm parameters
        self.algorithm_params = {
            "k-Gs": {"method": "sample_pairs", "c": 1.0},
            "S2L (k-means)": {"method": "kmeans"},
            "SAA-Gs (log n)": {"log_n_sampling": True}
        }
        
        # Set color palette for consistent visualization
        self.colors = {
            "k-Gs": "#1f77b4",  # blue
            "S2L (k-means)": "#ff7f0e",  # orange
            "SAA-Gs (log n)": "#2ca02c",  # green
        }
        
        print(f"Comparative analysis initialized. Results will be saved to {output_dir}")
    
    def load_datasets(self, dataset_names=None):
        """
        Load test datasets.
        
        Args:
            dataset_names: List of dataset names to load (if None, use defaults)
            
        Returns:
            dict: Dictionary of loaded graphs
        """
        if dataset_names is None:
            dataset_names = ["Amazon-0302", "DBLP", "Email-Enron"]
        
        datasets = {}
        
        # Define dataset paths
        data_dir = "./data"
        dataset_paths = {
            "Amazon-0302": os.path.join(data_dir, "Amazon-0302.txt"),
            "DBLP": os.path.join(data_dir, "DBLP.txt"),
            "Email-Enron": os.path.join(data_dir, "Email-Enron.txt"),
        }
        
        # Load each dataset
        for name in dataset_names:
            if name in dataset_paths:
                try:
                    path = dataset_paths[name]
                    print(f"Loading dataset: {name} from {path}")
                    
                    if os.path.exists(path):
                        G = GraphUtils.load_graph(path)
                        
                        # Ensure dataset is undirected and has no self-loops
                        if G.is_directed():
                            G = G.to_undirected()
                        G.remove_edges_from(nx.selfloop_edges(G))
                        
                        # Store graph
                        datasets[name] = G
                        
                        # Print dataset info
                        print(f"  Loaded {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
                    else:
                        print(f"  Dataset file not found: {path}")
                except Exception as e:
                    print(f"  Error loading {name}: {e}")
            else:
                print(f"  Unknown dataset: {name}")
        
        if not datasets:
            # Create at least one simple graph if no datasets were loaded
            print("No datasets loaded. Creating Zachary's Karate Club graph as fallback.")
            datasets["karate"] = nx.karate_club_graph()
        
        return datasets

    def generate_synthetic_datasets(self, types=None, sizes=None):
        """
        Generate synthetic datasets for testing.
        
        Args:
            types: List of graph types to generate
            sizes: List of graph sizes (node counts)
            
        Returns:
            dict: Dictionary of generated graphs
        """
        if types is None:
            types = ["erdos_renyi", "barabasi_albert", "watts_strogatz"]
        
        if sizes is None:
            sizes = [50, 100]
        
        datasets = {}
        
        for graph_type in types:
            for size in sizes:
                name = f"{graph_type}_{size}"
                print(f"Generating synthetic dataset: {name}")
                
                try:
                    if graph_type == "erdos_renyi":
                        # Random graph with probability p for edge creation
                        p = 0.1  # Edge probability
                        G = nx.erdos_renyi_graph(size, p, seed=42)
                    elif graph_type == "barabasi_albert":
                        # Preferential attachment graph
                        m = 3  # Number of edges to attach from a new node
                        G = nx.barabasi_albert_graph(size, m, seed=42)
                    elif graph_type == "watts_strogatz":
                        # Small-world graph
                        k = 4  # Each node is connected to k nearest neighbors
                        p = 0.1  # Probability of rewiring each edge
                        G = nx.watts_strogatz_graph(size, k, p, seed=42)
                    else:
                        print(f"  Unknown graph type: {graph_type}")
                        continue
                    
                    # Store graph
                    datasets[name] = G
                    
                    # Print dataset info
                    print(f"  Generated {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
                    
                except Exception as e:
                    print(f"  Error generating {name}: {e}")
        
        return datasets
    
    def compute_baseline_statistics(self, datasets):
        """
        Compute baseline statistics for each dataset.
        
        Args:
            datasets: Dictionary of graphs to analyze
            
        Returns:
            pd.DataFrame: DataFrame with baseline statistics
        """
        print("\nComputing baseline statistics for datasets:")
        
        stats = []
        
        for name, G in datasets.items():
            print(f"  Analyzing {name}...")
            
            # Compute basic metrics
            n = G.number_of_nodes()
            m = G.number_of_edges()
            density = nx.density(G)
            avg_degree = 2 * m / n if n > 0 else 0
            
            # Additional metrics
            try:
                avg_clustering = nx.average_clustering(G)
            except:
                avg_clustering = np.nan
                
            try:
                diameter = nx.diameter(G) if nx.is_connected(G) else np.nan
            except:
                diameter = np.nan
            
            # Size in bits (as defined in the SSumM paper)
            size_bits = 2 * m * np.ceil(np.log2(n)) if n > 0 else 0
            
            # Store statistics
            stats.append({
                "Dataset": name,
                "Nodes": n,
                "Edges": m,
                "Density": density,
                "Avg. Degree": avg_degree,
                "Avg. Clustering": avg_clustering,
                "Diameter": diameter,
                "Size (bits)": size_bits
            })
        
        # Create DataFrame
        stats_df = pd.DataFrame(stats)
        
        # Print statistics table
        print("\nDataset Statistics:")
        print(tabulate(stats_df, headers='keys', tablefmt='pretty', showindex=False))
        
        # Save statistics
        stats_df.to_csv(os.path.join(self.output_dir, "dataset_statistics.csv"), index=False)
        
        return stats_df

    def run_comparative_analysis(self, datasets, k_ratios=None, max_runtime=600):
        """
        Run comparative analysis of graph summarization algorithms.
        
        Args:
            datasets: Dictionary of graphs to analyze
            k_ratios: List of k/n ratios to test
            max_runtime: Maximum runtime in seconds for each algorithm run
            
        Returns:
            pd.DataFrame: DataFrame with comparison results
        """
        if k_ratios is None:
            k_ratios = [0.1, 0.2, 0.3]
        
        print("\nRunning comparative analysis with the following parameters:")
        print(f"  k ratios: {k_ratios}")
        print(f"  max runtime: {max_runtime} seconds per run")
        
        results = []
        
        for dataset_name, G in datasets.items():
            n = G.number_of_nodes()
            print(f"\nAnalyzing dataset: {dataset_name} ({n} nodes, {G.number_of_edges()} edges)")
            
            # Calculate target k values
            k_values = [max(int(ratio * n), 2) for ratio in k_ratios]  # Ensure k >= 2
            
            for algorithm_name, algorithm in self.algorithms.items():
                print(f"\n  Testing {algorithm_name}:")
                
                for k_ratio, k in zip(k_ratios, k_values):
                    print(f"    Running with k={k} ({k_ratio:.1%} of nodes)...")
                    
                    try:
                        # Force garbage collection before running
                        gc.collect()
                        
                        # Prepare parameters based on algorithm
                        if algorithm_name == "k-Gs":
                            run_func = getattr(algorithm, self.algorithm_params[algorithm_name]["method"])
                            run_args = {"c": self.algorithm_params[algorithm_name]["c"]}
                        elif algorithm_name == "S2L (k-means)":
                            run_func = algorithm.summarize
                            run_args = {"method": self.algorithm_params[algorithm_name]["method"]}
                        elif algorithm_name == "SAA-Gs (log n)":
                            run_func = algorithm.summarize
                            run_args = {"log_n_sampling": self.algorithm_params[algorithm_name]["log_n_sampling"]}
                        
                        # Measure memory usage and runtime
                        start_time = time.time()
                        
                        # Define a function to measure memory for
                        def run_algorithm():
                            return run_func(G, k, **run_args)
                        
                        # Run with memory profiling and timeout
                        mem_usage = []
                        if max_runtime > 0:
                            # Set timeout - not completely reliable but helps
                            import signal
                            
                            def timeout_handler(signum, frame):
                                raise TimeoutError(f"Algorithm {algorithm_name} exceeded {max_runtime} seconds")
                            
                            # Set the timeout
                            signal.signal(signal.SIGALRM, timeout_handler)
                            signal.alarm(max_runtime)
                            
                            try:
                                # Use memory_usage to track peak memory
                                mem_usage, summary = memory_usage(
                                    (run_algorithm, [], {}),
                                    retval=True,
                                    interval=0.1,
                                    timeout=max_runtime
                                )
                                
                                # Disable the alarm
                                signal.alarm(0)
                            except TimeoutError as e:
                                print(f"      {e}")
                                signal.alarm(0)  # Disable the alarm
                                raise
                            except Exception as e:
                                signal.alarm(0)  # Disable the alarm
                                raise
                        else:
                            # No timeout, just run and measure memory
                            mem_usage, summary = memory_usage(
                                (run_algorithm, [], {}),
                                retval=True,
                                interval=0.1
                            )
                        
                        end_time = time.time()
                        runtime = end_time - start_time
                        peak_memory = max(mem_usage) - min(mem_usage) if mem_usage else 0
                        
                        # Evaluate the summary
                        l1_error = summary.compute_reconstruction_error()
                        
                        # Calculate L2 error
                        A_original = nx.to_numpy_array(G)
                        A_expected = summary.get_expected_adjacency_matrix()
                        l2_error = np.sqrt(np.sum((A_original - A_expected) ** 2)) / (n * (n - 1))
                        
                        # Calculate relative size
                        original_size = 2 * G.number_of_edges() * np.ceil(np.log2(n))
                        summary_size = summary.size_in_bits()
                        relative_size = summary_size / original_size if original_size > 0 else np.nan
                        
                        # Store results
                        results.append({
                            "Dataset": dataset_name,
                            "Algorithm": algorithm_name,
                            "k_ratio": k_ratio,
                            "k": k,
                            "L1_Error": l1_error,
                            "L2_Error": l2_error,
                            "Size_bits": summary_size,
                            "Relative_Size": relative_size,
                            "Runtime_sec": runtime,
                            "Peak_Memory_MB": peak_memory,
                            "Num_Supernodes": len(summary.supernodes),
                            "Num_Superedges": len(summary.superedge_counts)
                        })
                        
                        print(f"      Completed: L1 Error={l1_error:.6f}, Size={summary_size} bits ({relative_size:.2%}), "
                              f"Runtime={runtime:.2f}s, Peak Memory={peak_memory:.2f} MB")
                        
                        # Clean up
                        del summary
                        gc.collect()
                        
                    except TimeoutError:
                        print(f"      Timeout: Runtime exceeded {max_runtime} seconds")
                        
                        # Store timeout in results
                        results.append({
                            "Dataset": dataset_name,
                            "Algorithm": algorithm_name,
                            "k_ratio": k_ratio,
                            "k": k,
                            "L1_Error": np.nan,
                            "L2_Error": np.nan,
                            "Size_bits": np.nan,
                            "Relative_Size": np.nan,
                            "Runtime_sec": max_runtime,  # Set to maximum
                            "Peak_Memory_MB": np.nan,
                            "Num_Supernodes": np.nan,
                            "Num_Superedges": np.nan
                        })
                        
                    except Exception as e:
                        print(f"      Error: {e}")
                        
                        # Store error in results
                        results.append({
                            "Dataset": dataset_name,
                            "Algorithm": algorithm_name,
                            "k_ratio": k_ratio,
                            "k": k,
                            "L1_Error": np.nan,
                            "L2_Error": np.nan,
                            "Size_bits": np.nan,
                            "Relative_Size": np.nan,
                            "Runtime_sec": np.nan,
                            "Peak_Memory_MB": np.nan,
                            "Num_Supernodes": np.nan,
                            "Num_Superedges": np.nan
                        })
                        
                        # Clean up after error
                        gc.collect()
        
        # Create DataFrame and save results
        results_df = pd.DataFrame(results)
        
        # Save results
        results_df.to_csv(os.path.join(self.output_dir, "algorithm_comparison_results.csv"), index=False)
        
        # Print summary
        print("\nComparative analysis complete. Summary of results:")
        summary = results_df.groupby(['Algorithm', 'Dataset']).agg({
            'L1_Error': ['mean', 'min'],
            'Runtime_sec': ['mean', 'max'],
            'Peak_Memory_MB': ['mean', 'max']
        }).reset_index()
        
        print(tabulate(summary, headers='keys', tablefmt='pretty', showindex=False))
        
        self.results = results_df
        return results_df
    
    def create_visualizations(self):
        """
        Create visualizations from the comparative analysis results.
        
        Returns:
            list: List of paths to generated visualizations
        """
        if self.results.empty:
            print("No results to visualize. Run comparative analysis first.")
            return []
        
        print("\nCreating visualizations from comparative analysis results...")
        
        # Convert results to DataFrame if it's a dictionary
        results_df = pd.DataFrame(self.results) if isinstance(self.results, dict) else self.results
        
        visualizations = []
        
        # Create subdirectory for visualizations
        viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Set plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 1. Error vs. K Ratio by Algorithm and Dataset
        print("  Creating Error vs. K Ratio visualization...")
        plt.figure(figsize=(12, 8))
        
        for dataset in results_df['Dataset'].unique():
            plt.subplot(2, 2, list(results_df['Dataset'].unique()).index(dataset) + 1)
            
            dataset_results = results_df[results_df['Dataset'] == dataset]
            
            for algorithm in dataset_results['Algorithm'].unique():
                alg_data = dataset_results[dataset_results['Algorithm'] == algorithm]
                plt.plot(alg_data['k_ratio'], alg_data['L1_Error'], 
                         marker='o', label=algorithm, color=self.colors.get(algorithm, None))
            
            plt.xlabel('Ratio of Supernodes (k/n)')
            plt.ylabel('L1 Reconstruction Error')
            plt.title(f'Dataset: {dataset}')
            plt.grid(alpha=0.3)
            if dataset == list(results_df['Dataset'].unique())[0]:  # Only show legend on first plot
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
        plt.tight_layout()
        error_vs_k_path = os.path.join(viz_dir, "error_vs_k_ratio.png")
        plt.savefig(error_vs_k_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualizations.append(error_vs_k_path)
        
        # 2. Runtime vs. K Ratio by Algorithm and Dataset
        print("  Creating Runtime vs. K Ratio visualization...")
        plt.figure(figsize=(12, 8))
        
        for dataset in results_df['Dataset'].unique():
            plt.subplot(2, 2, list(results_df['Dataset'].unique()).index(dataset) + 1)
            
            dataset_results = results_df[results_df['Dataset'] == dataset]
            
            for algorithm in dataset_results['Algorithm'].unique():
                alg_data = dataset_results[dataset_results['Algorithm'] == algorithm]
                plt.plot(alg_data['k_ratio'], alg_data['Runtime_sec'], 
                         marker='o', label=algorithm, color=self.colors.get(algorithm, None))
            
            plt.xlabel('Ratio of Supernodes (k/n)')
            plt.ylabel('Runtime (seconds)')
            plt.title(f'Dataset: {dataset}')
            plt.grid(alpha=0.3)
            if dataset == list(results_df['Dataset'].unique())[0]:  # Only show legend on first plot
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
        plt.tight_layout()
        runtime_vs_k_path = os.path.join(viz_dir, "runtime_vs_k_ratio.png")
        plt.savefig(runtime_vs_k_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualizations.append(runtime_vs_k_path)
        
        # 3. Memory Usage vs. K Ratio by Algorithm and Dataset
        print("  Creating Memory Usage vs. K Ratio visualization...")
        plt.figure(figsize=(12, 8))
        
        for dataset in results_df['Dataset'].unique():
            plt.subplot(2, 2, list(results_df['Dataset'].unique()).index(dataset) + 1)
            
            dataset_results = results_df[results_df['Dataset'] == dataset]
            
            for algorithm in dataset_results['Algorithm'].unique():
                alg_data = dataset_results[dataset_results['Algorithm'] == algorithm]
                plt.plot(alg_data['k_ratio'], alg_data['Peak_Memory_MB'], 
                         marker='o', label=algorithm, color=self.colors.get(algorithm, None))
            
            plt.xlabel('Ratio of Supernodes (k/n)')
            plt.ylabel('Peak Memory Usage (MB)')
            plt.title(f'Dataset: {dataset}')
            plt.grid(alpha=0.3)
            if dataset == list(results_df['Dataset'].unique())[0]:  # Only show legend on first plot
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
        plt.tight_layout()
        memory_vs_k_path = os.path.join(viz_dir, "memory_vs_k_ratio.png")
        plt.savefig(memory_vs_k_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualizations.append(memory_vs_k_path)
        
        # 4. Size vs. K Ratio by Algorithm and Dataset
        print("  Creating Size vs. K Ratio visualization...")
        plt.figure(figsize=(12, 8))
        
        for dataset in results_df['Dataset'].unique():
            plt.subplot(2, 2, list(results_df['Dataset'].unique()).index(dataset) + 1)
            
            dataset_results = results_df[results_df['Dataset'] == dataset]
            
            for algorithm in dataset_results['Algorithm'].unique():
                alg_data = dataset_results[dataset_results['Algorithm'] == algorithm]
                plt.plot(alg_data['k_ratio'], alg_data['Relative_Size'], 
                         marker='o', label=algorithm, color=self.colors.get(algorithm, None))
            
            plt.xlabel('Ratio of Supernodes (k/n)')
            plt.ylabel('Relative Size (summary/original)')
            plt.title(f'Dataset: {dataset}')
            plt.grid(alpha=0.3)
            if dataset == list(results_df['Dataset'].unique())[0]:  # Only show legend on first plot
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
        plt.tight_layout()
        size_vs_k_path = os.path.join(viz_dir, "size_vs_k_ratio.png")
        plt.savefig(size_vs_k_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualizations.append(size_vs_k_path)
        
        # 5. Error vs Size scatter plot
        print("  Creating Error vs. Size scatter plot...")
        plt.figure(figsize=(10, 8))
        
        for algorithm in results_df['Algorithm'].unique():
            alg_data = results_df[results_df['Algorithm'] == algorithm]
            plt.scatter(alg_data['Relative_Size'], alg_data['L1_Error'], 
                       label=algorithm, color=self.colors.get(algorithm, None), s=80, alpha=0.7)
        
        plt.xlabel('Relative Size (summary/original)')
        plt.ylabel('L1 Reconstruction Error')
        plt.title('Error vs. Size Trade-off Across All Datasets')
        plt.grid(alpha=0.3)
        plt.legend()
        
        error_vs_size_path = os.path.join(viz_dir, "error_vs_size.png")
        plt.savefig(error_vs_size_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualizations.append(error_vs_size_path)
        
        # 6. Algorithm Performance Radar Chart
        print("  Creating Algorithm Performance Radar Chart...")
        
        # Prepare data for radar chart - normalize metrics
        radar_metrics = ['L1_Error', 'Runtime_sec', 'Peak_Memory_MB', 'Relative_Size']
        
        # Aggregate by algorithm
        radar_data = results_df.groupby('Algorithm')[radar_metrics].mean().reset_index()
        
        # Normalize metrics (invert error and size so higher is better for all metrics)
        for metric in radar_metrics:
            if metric in ['L1_Error', 'Runtime_sec', 'Peak_Memory_MB', 'Relative_Size']:
                # Invert so lower values become higher scores (better)
                min_val = radar_data[metric].min()
                max_val = radar_data[metric].max()
                if max_val > min_val:
                    radar_data[f"{metric}_normalized"] = 1 - (radar_data[metric] - min_val) / (max_val - min_val)
                else:
                    radar_data[f"{metric}_normalized"] = 1.0  # All values are equal
        
        # Create radar chart
        normalized_metrics = [f"{m}_normalized" for m in radar_metrics]
        
        # Set up radar chart
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Number of variables
        N = len(normalized_metrics)
        
        # Angle of each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Plot each algorithm
        for i, algorithm in enumerate(radar_data['Algorithm']):
            values = radar_data.loc[radar_data['Algorithm'] == algorithm, normalized_metrics].values.flatten().tolist()
            values += values[:1]  # Close the loop
            
            # Plot values
            ax.plot(angles, values, linewidth=2, linestyle='-', label=algorithm, color=self.colors.get(algorithm, None))
            ax.fill(angles, values, alpha=0.1, color=self.colors.get(algorithm, None))
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        plt.xticks(angles[:-1], ['Error', 'Runtime', 'Memory', 'Size'])
        
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=8)
        plt.ylim(0, 1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        radar_chart_path = os.path.join(viz_dir, "algorithm_radar_chart.png")
        plt.savefig(radar_chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualizations.append(radar_chart_path)
        
        print(f"Visualizations saved to {viz_dir}")
        return visualizations
    
    def identify_implementation_challenges(self):
        """
        Identify implementation challenges for larger datasets based on analysis results.
        
        Returns:
            dict: Dictionary of identified challenges by algorithm
        """
        if self.results.empty:
            print("No results to analyze. Run comparative analysis first.")
            return {}
        
        print("\nIdentifying implementation challenges for larger datasets...")
        
        # Convert results to DataFrame if it's a dictionary
        results_df = pd.DataFrame(self.results) if isinstance(self.results, dict) else self.results
        
        challenges = {}
        
        # Analyze scaling behavior for each algorithm
        for algorithm in results_df['Algorithm'].unique():
            alg_data = results_df[results_df['Algorithm'] == algorithm]
            
            # Check for timeout occurrences
            timeouts = alg_data['Runtime_sec'].isna().sum()
            timeout_datasets = alg_data[alg_data['Runtime_sec'].isna()]['Dataset'].unique()
            
            # Check scaling of runtime vs. dataset size
            runtime_scaling = {}
            memory_scaling = {}
            
            for dataset in alg_data['Dataset'].unique():
                dataset_info = results_df[results_df['Dataset'] == dataset].iloc[0]
                dataset_size = dataset_info.get('Nodes', 0)
                
                dataset_runs = alg_data[alg_data['Dataset'] == dataset]
                avg_runtime = dataset_runs['Runtime_sec'].mean()
                avg_memory = dataset_runs['Peak_Memory_MB'].mean()
                
                if not np.isnan(avg_runtime) and not np.isnan(avg_memory):
                    runtime_scaling[dataset] = (dataset_size, avg_runtime)
                    memory_scaling[dataset] = (dataset_size, avg_memory)
            
            # Calculate growth rates if possible
            runtime_growth = "Unknown"
            memory_growth = "Unknown"
            
            if len(runtime_scaling) >= 2:
                # Sort by dataset size
                runtime_points = sorted(runtime_scaling.values())
                
                # Calculate approximate growth rate
                if len(runtime_points) >= 2 and runtime_points[0][0] > 0 and runtime_points[-1][0] > 0:
                    # Use the smallest and largest datasets
                    small_size, small_runtime = runtime_points[0]
                    large_size, large_runtime = runtime_points[-1]
                    
                    size_ratio = large_size / small_size
                    runtime_ratio = large_runtime / small_runtime
                    
                    # Estimate growth factor
                    if runtime_ratio > 0 and size_ratio > 1:
                        exponent = np.log(runtime_ratio) / np.log(size_ratio)
                        if exponent < 1.2:
                            runtime_growth = "Approximately linear"
                        elif exponent < 2.2:
                            runtime_growth = "Approximately quadratic"
                        else:
                            runtime_growth = f"Polynomial with exponent ~{exponent:.1f}"
            
            if len(memory_scaling) >= 2:
                # Sort by dataset size
                memory_points = sorted(memory_scaling.values())
                
                # Calculate approximate growth rate
                if len(memory_points) >= 2 and memory_points[0][0] > 0 and memory_points[-1][0] > 0:
                    # Use the smallest and largest datasets
                    small_size, small_memory = memory_points[0]
                    large_size, large_memory = memory_points[-1]
                    
                    size_ratio = large_size / small_size
                    memory_ratio = large_memory / small_memory
                    
                    # Estimate growth factor
                    if memory_ratio > 0 and size_ratio > 1:
                        exponent = np.log(memory_ratio) / np.log(size_ratio)
                        if exponent < 1.2:
                            memory_growth = "Approximately linear"
                        elif exponent < 2.2:
                            memory_growth = "Approximately quadratic"
                        else:
                            memory_growth = f"Polynomial with exponent ~{exponent:.1f}"
            
            # Check for other challenges
            error_inconsistency = alg_data['L1_Error'].std() / alg_data['L1_Error'].mean() if not alg_data['L1_Error'].isna().all() and alg_data['L1_Error'].mean() > 0 else np.nan
            
            # Collect challenges
            alg_challenges = []
            
            # Runtime challenges
            if timeouts > 0:
                alg_challenges.append(f"Timeout occurred on {timeouts} runs with datasets: {', '.join(timeout_datasets)}")
            
            if runtime_growth != "Unknown":
                if "quadratic" in runtime_growth or "Polynomial" in runtime_growth:
                    alg_challenges.append(f"Runtime scaling is {runtime_growth}, which may limit application to larger graphs")
            
            # Memory challenges
            if "quadratic" in memory_growth or "Polynomial" in memory_growth:
                alg_challenges.append(f"Memory usage scaling is {memory_growth}, which may cause out-of-memory errors on larger graphs")
            
            # Quality challenges
            if not np.isnan(error_inconsistency) and error_inconsistency > 0.5:
                alg_challenges.append(f"High variability in reconstruction error (CV={error_inconsistency:.2f}), suggesting inconsistent quality")
            
            # Algorithm-specific challenges
            if algorithm == "k-Gs":
                alg_challenges.append("No built-in sparsification strategy may lead to large summary sizes")
            elif algorithm == "S2L (k-means)":
                alg_challenges.append("High memory usage for adjacency matrix representation limits scalability")
                alg_challenges.append("Clustering quality highly depends on initial centroids")
            elif algorithm == "SAA-Gs (log n)":
                alg_challenges.append("Count-min sketch may have hash collisions affecting error estimation accuracy")
            
            challenges[algorithm] = alg_challenges
        
        # Write challenges to file
        with open(os.path.join(self.output_dir, "implementation_challenges.txt"), 'w') as f:
            f.write("Implementation Challenges for Larger Datasets\n")
            f.write("===========================================\n\n")
            
            for algorithm, alg_challenges in challenges.items():
                f.write(f"{algorithm}:\n")
                for i, challenge in enumerate(alg_challenges):
                    f.write(f"  {i+1}. {challenge}\n")
                f.write("\n")
            
            # Add general recommendations
            f.write("\nGeneral Recommendations for Scaling to Larger Datasets:\n")
            f.write("--------------------------------------------------------\n")
            f.write("1. Implement sparse matrix operations where possible\n")
            f.write("2. Use sampling strategies for large graphs\n")
            f.write("3. Consider parallel or distributed implementations\n")
            f.write("4. Optimize memory usage with more efficient data structures\n")
            f.write("5. Implement progressive processing for very large graphs\n")
        
        print(f"Implementation challenges identified and saved to {os.path.join(self.output_dir, 'implementation_challenges.txt')}")
        return challenges
    
    def generate_report(self):
        """
        Generate a comprehensive report of the comparative analysis.
        
        Returns:
            str: Path to the generated report
        """
        if self.results.empty:
            print("No results to report. Run comparative analysis first.")
            return None
        
        print("\nGenerating comprehensive report...")
        
        # Convert results to DataFrame if it's a dictionary
        results_df = pd.DataFrame(self.results) if isinstance(self.results, dict) else self.results
        
        # Create report file
        report_path = os.path.join(self.output_dir, "comparative_analysis_report.md")
        
        with open(report_path, 'w') as f:
            # Title and introduction
            f.write("# Comparative Analysis of Graph Summarization Algorithms\n\n")
            f.write("## Introduction\n\n")
            f.write("This report presents a comparative analysis of different graph summarization algorithms: k-Gs, S2L, and SAA-Gs. ")
            f.write("The analysis evaluates these algorithms in terms of reconstruction error, runtime efficiency, memory usage, and output size ")
            f.write("on various datasets.\n\n")
            
            # Datasets summary
            f.write("## Datasets\n\n")
            
            datasets = results_df['Dataset'].unique()
            f.write(f"The analysis was performed on {len(datasets)} datasets:\n\n")
            
            for dataset in datasets:
                dataset_info = results_df[results_df['Dataset'] == dataset].iloc[0]
                f.write(f"- **{dataset}**: ")
                if 'Nodes' in dataset_info and not np.isnan(dataset_info['Nodes']):
                    f.write(f"{int(dataset_info['Nodes'])} nodes, ")
                if 'Edges' in dataset_info and not np.isnan(dataset_info['Edges']):
                    f.write(f"{int(dataset_info['Edges'])} edges")
                f.write("\n")
            
            f.write("\n")
            
            # Algorithm descriptions
            f.write("## Algorithm Descriptions\n\n")
            
            f.write("### k-Gs (k-Graph Summarization)\n")
            f.write("k-Gs is a greedy algorithm that iteratively merges pairs of supernodes to minimize reconstruction error. ")
            f.write("The implementation uses the SamplePairs optimization which samples node pairs to reduce computational complexity.\n\n")
            
            f.write("### S2L (Summarization via Structural Locality)\n")
            f.write("S2L uses a geometric clustering approach, treating rows of the adjacency matrix as points in high-dimensional space. ")
            f.write("The implementation uses k-means clustering with dimensionality reduction preprocessing.\n\n")
            
            f.write("### SAA-Gs (Scalable Approximation Algorithm for Graph Summarization)\n")
            f.write("SAA-Gs employs a weighted sampling approach with a tree structure for efficient candidate pair selection. ")
            f.write("The implementation incorporates a count-min sketch for fast approximation of error estimates.\n\n")
            
            # Performance comparison
            f.write("## Performance Comparison\n\n")
            
            # Reconstruction error
            f.write("### Reconstruction Error\n\n")
            
            f.write("The L1 reconstruction error measures the average absolute difference between the original and reconstructed adjacency matrices. ")
            f.write("Lower values indicate better preservation of the original graph structure.\n\n")
            
            error_summary = results_df.groupby(['Algorithm', 'Dataset'])['L1_Error'].mean().reset_index()
            error_summary = error_summary.pivot(index='Dataset', columns='Algorithm', values='L1_Error')
            
            f.write("Average L1 Reconstruction Error by Algorithm and Dataset:\n\n")
            f.write("```\n")
            f.write(tabulate(error_summary, headers="keys", tablefmt="pipe"))
            f.write("\n```\n\n")
            
            # Runtime
            f.write("### Runtime Performance\n\n")
            
            f.write("Runtime measures the execution time in seconds for each algorithm. ")
            f.write("This metric is crucial for understanding the computational efficiency of each approach.\n\n")
            
            runtime_summary = results_df.groupby(['Algorithm', 'Dataset'])['Runtime_sec'].mean().reset_index()
            runtime_summary = runtime_summary.pivot(index='Dataset', columns='Algorithm', values='Runtime_sec')
            
            f.write("Average Runtime (seconds) by Algorithm and Dataset:\n\n")
            f.write("```\n")
            f.write(tabulate(runtime_summary, headers="keys", tablefmt="pipe"))
            f.write("\n```\n\n")
            
            # Memory usage
            f.write("### Memory Usage\n\n")
            
            f.write("Peak memory usage (in MB) provides insight into the memory efficiency of each algorithm, ")
            f.write("which is a critical factor for processing large graphs.\n\n")
            
            memory_summary = results_df.groupby(['Algorithm', 'Dataset'])['Peak_Memory_MB'].mean().reset_index()
            memory_summary = memory_summary.pivot(index='Dataset', columns='Algorithm', values='Peak_Memory_MB')
            
            f.write("Average Peak Memory Usage (MB) by Algorithm and Dataset:\n\n")
            f.write("```\n")
            f.write(tabulate(memory_summary, headers="keys", tablefmt="pipe"))
            f.write("\n```\n\n")
            
            # Size
            f.write("### Summary Size\n\n")
            
            f.write("Relative size measures the ratio of the summary graph size to the original graph size. ")
            f.write("Lower values indicate better compression.\n\n")
            
            size_summary = results_df.groupby(['Algorithm', 'Dataset'])['Relative_Size'].mean().reset_index()
            size_summary = size_summary.pivot(index='Dataset', columns='Algorithm', values='Relative_Size')
            
            f.write("Average Relative Size by Algorithm and Dataset:\n\n")
            f.write("```\n")
            f.write(tabulate(size_summary, headers="keys", tablefmt="pipe"))
            f.write("\n```\n\n")
            
            # Trade-off analysis
            f.write("## Trade-off Analysis\n\n")
            
            f.write("The ideal graph summarization algorithm would minimize both reconstruction error and summary size ")
            f.write("while maintaining reasonable runtime and memory usage. Below we analyze the trade-offs between these metrics.\n\n")
            
            # Extract challenges
            challenges = self.identify_implementation_challenges()
            
            f.write("### Error vs. Size Trade-off\n\n")
            
            f.write("The fundamental trade-off in graph summarization is between reconstruction error and summary size. ")
            f.write("Algorithms that achieve lower error with smaller summary sizes are generally preferable.\n\n")
            
            # Create recommendations section
            f.write("## Recommendations\n\n")
            
            f.write("Based on the comparative analysis, here are algorithm recommendations for different scenarios:\n\n")
            
            f.write("### For Small to Medium Graphs (< 10,000 nodes)\n\n")
            
            # Determine best algorithm for small graphs based on error and runtime
            small_results = results_df.groupby('Algorithm').agg({
                'L1_Error': 'mean',
                'Runtime_sec': 'mean',
                'Relative_Size': 'mean'
            }).reset_index()
            
            # Simple scoring (lower is better)
            small_results['score'] = (
                small_results['L1_Error'] / small_results['L1_Error'].max() +
                small_results['Runtime_sec'] / small_results['Runtime_sec'].max() +
                small_results['Relative_Size'] / small_results['Relative_Size'].max()
            )
            
            best_small = small_results.loc[small_results['score'].idxmin()]['Algorithm']
            
            f.write(f"**Recommended Algorithm**: {best_small}\n\n")
            f.write("**Rationale**: Best balance of accuracy, speed, and output size for smaller graphs.\n\n")
            
            f.write("### For Large Graphs (> 10,000 nodes)\n\n")
            
            # For large graphs, prioritize scalability (runtime and memory)
            # Since we may not have large graphs in the test, use scaling behavior
            scaling_scores = {}
            for alg, alg_challenges in challenges.items():
                # Count scaling issues (lower is better)
                scaling_issues = sum(1 for challenge in alg_challenges if "scaling" in challenge.lower())
                scaling_scores[alg] = scaling_issues
            
            best_large = min(scaling_scores.items(), key=lambda x: x[1])[0]
            
            f.write(f"**Recommended Algorithm**: {best_large}\n\n")
            f.write("**Rationale**: Better scaling behavior for larger graphs with reasonable accuracy-size trade-off.\n\n")
            
            f.write("### For Accuracy-Critical Applications\n\n")
            
            best_accuracy = results_df.groupby('Algorithm')['L1_Error'].mean().idxmin()
            
            f.write(f"**Recommended Algorithm**: {best_accuracy}\n\n")
            f.write("**Rationale**: Achieves the lowest reconstruction error, preserving the graph structure most accurately.\n\n")
            
            f.write("### For Space-Critical Applications\n\n")
            
            best_size = results_df.groupby('Algorithm')['Relative_Size'].mean().idxmin()
            
            f.write(f"**Recommended Algorithm**: {best_size}\n\n")
            f.write("**Rationale**: Produces the most compact summaries while maintaining acceptable accuracy.\n\n")
            
            # Implementation challenges
            f.write("## Implementation Challenges for Larger Datasets\n\n")
            
            for algorithm, alg_challenges in challenges.items():
                f.write(f"### {algorithm}\n\n")
                if alg_challenges:
                    for challenge in alg_challenges:
                        f.write(f"- {challenge}\n")
                else:
                    f.write("No specific challenges identified.\n")
                f.write("\n")
            
            # Future improvements
            f.write("## Potential Improvements\n\n")
            
            f.write("### k-Gs\n\n")
            f.write("- Implement edge sparsification to reduce summary size\n")
            f.write("- Explore more efficient pair selection strategies\n")
            f.write("- Implement parallel processing for candidate pair evaluation\n\n")
            
            f.write("### S2L\n\n")
            f.write("- Use sparse matrix operations to reduce memory usage\n")
            f.write("- Implement more advanced dimensionality reduction techniques\n")
            f.write("- Improve clustering initialization for better convergence\n\n")
            
            f.write("### SAA-Gs\n\n")
            f.write("- Optimize weight tree updates for faster iteration\n")
            f.write("- Implement adaptive count-min sketch dimensions\n")
            f.write("- Explore more sophisticated error estimation methods\n\n")
            
            # Conclusion
            f.write("## Conclusion\n\n")
            
            f.write("This comparative analysis reveals the strengths and weaknesses of different graph summarization approaches. ")
            f.write("Each algorithm offers a different trade-off between accuracy, runtime, memory usage, and summary size. ")
            f.write("The choice of algorithm should be guided by the specific requirements of the application and the characteristics of the input graph.\n\n")
            
            f.write("For further improvements, focus should be placed on enhancing the scalability of these algorithms ")
            f.write("through the use of sparse data structures, approximation techniques, and parallel processing strategies. ")
            f.write("Additionally, hybrid approaches that combine the strengths of multiple algorithms may offer promising directions for future research.\n")
        
        print(f"Comprehensive report generated and saved to {report_path}")
        return report_path

def run_comparative_analysis():
    """
    Run a full comparative analysis of graph summarization algorithms.
    """
    # Create analysis object
    analysis = ComparativeAnalysis()
    
    # Load datasets
    datasets = analysis.load_datasets()
    
    # Generate synthetic datasets
    synthetic_datasets = analysis.generate_synthetic_datasets()
    
    # Combine all datasets
    all_datasets = {**datasets, **synthetic_datasets}
    
    # Compute baseline statistics
    analysis.compute_baseline_statistics(all_datasets)
    
    # Run comparative analysis
    analysis.run_comparative_analysis(all_datasets)
    
    # Create visualizations
    analysis.create_visualizations()
    
    # Identify implementation challenges
    analysis.identify_implementation_challenges()
    
    # Generate comprehensive report
    report_path = analysis.generate_report()
    
    print(f"\nComparative analysis complete! Full report available at: {report_path}")

if __name__ == "__main__":
    run_comparative_analysis()