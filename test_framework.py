#!/usr/bin/env python3
"""
Comprehensive testing framework for SSumM algorithm.

This script provides a complete testing framework for the SSumM algorithm,
including unit tests, integration tests, performance benchmarks, and visualizations.
It helps verify the correctness and efficiency of the SSumM implementation.
"""

import os
import sys
import unittest
import time
import logging
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

# Import our implementations
from graph_utils import GraphUtils
from graph_summary import GraphSummary
from ssumm_algorithm import SSumM
from ssumm_optimized import OptimizedSSumM
from kgs_algorithm import KGs
from s2l_algorithm import S2L
from saa_gs_algorithm import SAA_Gs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('testing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TestFramework')

class SSumMUnitTests(unittest.TestCase):
    """Unit tests for the SSumM algorithm and its components."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create small test graphs
        self.karate = nx.karate_club_graph()
        self.barbell = nx.barbell_graph(10, 2)
        self.ba_graph = nx.barabasi_albert_graph(50, 2, seed=42)
        
        # Initialize algorithms
        self.ssumm = SSumM(max_iterations=10)
        self.optimized_ssumm = OptimizedSSumM(max_iterations=10)
    
    def test_trivial_summary(self):
        """Test creation of a trivial summary."""
        summary = GraphSummary(self.karate)
        
        # In a trivial summary, each node should be in its own supernode
        self.assertEqual(len(summary.supernodes), self.karate.number_of_nodes())
        
        # The number of superedges should match the number of edges
        self.assertEqual(len(summary.superedge_counts), self.karate.number_of_edges())
        
        # Reconstruction error should be 0
        self.assertAlmostEqual(summary.compute_reconstruction_error(), 0.0)
    
    def test_merge_supernodes(self):
        """Test merging of supernodes."""
        summary = GraphSummary(self.karate)
        
        # Get initial counts
        initial_supernodes = len(summary.supernodes)
        initial_superedges = len(summary.superedge_counts)
        
        # Merge two supernodes
        supernode1, supernode2 = list(summary.supernodes.keys())[:2]
        new_supernode = summary.merge_supernodes(supernode1, supernode2)
        
        # Check that the number of supernodes decreased by 1
        self.assertEqual(len(summary.supernodes), initial_supernodes - 1)
        
        # Check that the new supernode exists
        self.assertIn(new_supernode, summary.supernodes)
        
        # Check that the original supernodes no longer exist
        self.assertNotIn(supernode1, summary.supernodes)
        self.assertNotIn(supernode2, summary.supernodes)
        
        # Check that the new supernode contains all nodes from the original supernodes
        original_nodes = set()
        original_nodes.update(summary.original_graph.nodes[supernode1])
        original_nodes.update(summary.original_graph.nodes[supernode2])
        
        self.assertEqual(summary.supernodes[new_supernode], original_nodes)
    
    def test_compute_reconstruction_error(self):
        """Test computation of reconstruction error."""
        # Start with a trivial summary (zero error)
        summary = GraphSummary(self.karate)
        self.assertAlmostEqual(summary.compute_reconstruction_error(), 0.0)
        
        # Merge two supernodes and check that error increases
        supernode1, supernode2 = list(summary.supernodes.keys())[:2]
        summary.merge_supernodes(supernode1, supernode2)
        
        # Error should be non-zero after merging
        error = summary.compute_reconstruction_error()
        self.assertGreater(error, 0.0)
    
    def test_size_in_bits(self):
        """Test calculation of summary size in bits."""
        summary = GraphSummary(self.karate)
        
        # Calculate expected size
        n = self.karate.number_of_nodes()
        m = self.karate.number_of_edges()
        
        # Size of a trivial summary
        # Each superedge: 2*log2(n) + log2(1) bits for source, destination, and weight
        # Each node mapping: log2(n) bits
        expected_size = m * (2 * np.ceil(np.log2(n)) + 1) + n * np.ceil(np.log2(n))
        
        self.assertEqual(summary.size_in_bits(), expected_size)
    
    def test_optimize_superedges(self):
        """Test the selective creation of superedges in OptimizedSSumM."""
        # Start with a trivial summary
        summary = GraphSummary(self.barbell)
        
        # Merge some supernodes (e.g., all nodes in one clique)
        clique_nodes = list(range(10))  # First clique in barbell graph
        first_node = clique_nodes[0]
        
        for node in clique_nodes[1:]:
            summary.merge_supernodes(first_node, node)
        
        # Create cost function
        cost_function = OptimizedSSumM()._create_optimized_cost_function(self.barbell)
        
        # Optimize superedges for the merged supernode
        OptimizedSSumM()._optimize_superedges(summary, first_node, cost_function)
        
        # Check that the superedges were created selectively
        self.assertLess(len(summary.superedge_counts), self.barbell.number_of_edges())
        self.assertGreater(len(summary.superedge_counts), 0)

    def test_further_sparsify(self):
        """Test further sparsification to meet target size."""
        # Start with a summary that doesn't meet the target size
        summary = GraphSummary(self.ba_graph)
        current_size = summary.size_in_bits()
        
        # Set target size to 70% of current size
        target_size = int(0.7 * current_size)
        
        # Sparsify to meet target size
        OptimizedSSumM()._further_sparsify(summary, target_size)
        
        # Check that the new size meets the target
        self.assertLessEqual(summary.size_in_bits(), target_size)
    
    def test_algorithm_execution(self):
        """Test execution of the full SSumM algorithm."""
        # Calculate target size
        original_size = 2 * self.karate.number_of_edges() * np.ceil(np.log2(self.karate.number_of_nodes()))
        target_size = int(0.5 * original_size)
        
        # Run the algorithm
        summary = self.ssumm.summarize(self.karate, target_size)
        
        # Check that the summary meets the target size
        self.assertLessEqual(summary.size_in_bits(), target_size)
        
        # Check that the summary is a valid summarization
        self.assertGreaterEqual(len(summary.supernodes), 1)
        self.assertLessEqual(len(summary.supernodes), self.karate.number_of_nodes())
        
        # Check that the reconstruction error is reasonable
        error = summary.compute_reconstruction_error()
        self.assertGreaterEqual(error, 0.0)
        self.assertLess(error, 0.5)  # This is a reasonable threshold for this small graph
    
    def test_optimized_algorithm(self):
        """Test execution of the optimized SSumM algorithm."""
        # Calculate target size
        original_size = 2 * self.ba_graph.number_of_edges() * np.ceil(np.log2(self.ba_graph.number_of_nodes()))
        target_size = int(0.5 * original_size)
        
        # Run the optimized algorithm
        summary, performance_data = self.optimized_ssumm.summarize(self.ba_graph, target_size, track_convergence=True)
        
        # Check that the summary meets the target size
        self.assertLessEqual(summary.size_in_bits(), target_size)
        
        # Check that performance data was collected
        self.assertIn('performance', performance_data)
        self.assertIn('convergence', performance_data)
        
        # Check that the convergence data has the expected fields
        convergence = performance_data['convergence']
        self.assertIn('iteration', convergence)
        self.assertIn('reconstruction_error_l1', convergence)
        self.assertIn('size_in_bits', convergence)


class SSumMIntegrationTests(unittest.TestCase):
    """Integration tests for the SSumM algorithm."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Load a medium-sized real-world graph
        self.graph = nx.karate_club_graph()  # For quick testing
        
        try:
            # Try to load a larger graph if available
            data_dir = "./data"
            for dataset in ["Ego-Facebook", "Email-Enron"]:
                for ext in [".txt", ".edges", ".gpickle"]:
                    path = os.path.join(data_dir, f"{dataset}{ext}")
                    if os.path.exists(path):
                        if ext == ".gpickle":
                            with open(path, 'rb') as f:
                                self.graph = pickle.load(f)
                        else:
                            self.graph = nx.read_edgelist(path)
                        logger.info(f"Loaded {dataset} with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
                        break
                if self.graph.number_of_nodes() > 100:
                    break
        except Exception as e:
            logger.warning(f"Could not load larger graph, using Karate Club: {e}")
        
        # Sample the graph if it's too large for quick testing
        if self.graph.number_of_nodes() > 500:
            nodes = list(self.graph.nodes())
            sampled_nodes = np.random.choice(nodes, 500, replace=False)
            self.graph = self.graph.subgraph(sampled_nodes).copy()
            logger.info(f"Sampled graph to 500 nodes")
        
        # Initialize algorithms
        self.algorithms = {
            'SSumM': SSumM(max_iterations=10),
            'OptimizedSSumM': OptimizedSSumM(max_iterations=10),
            'k-Gs': KGs(),
            'S2L': S2L(),
            'SAA-Gs': SAA_Gs()
        }
    
    def test_algorithm_comparison(self):
        """Test and compare all graph summarization algorithms."""
        # Calculate target parameters
        original_size = 2 * self.graph.number_of_edges() * np.ceil(np.log2(self.graph.number_of_nodes()))
        target_size = int(0.3 * original_size)
        target_k = int(0.3 * self.graph.number_of_nodes())
        
        results = []
        
        for algo_name, algorithm in self.algorithms.items():
            try:
                logger.info(f"Testing {algo_name}...")
                
                # Start timing
                start_time = time.time()
                
                # Run appropriate algorithm
                if algo_name in ['SSumM', 'OptimizedSSumM']:
                    summary = algorithm.summarize(self.graph, target_size)
                elif algo_name == 'k-Gs':
                    summary = algorithm.sample_pairs(self.graph, target_k, c=1.0)
                elif algo_name == 'S2L':
                    summary = algorithm.summarize(self.graph, target_k, method='kmeans')
                elif algo_name == 'SAA-Gs':
                    summary = algorithm.summarize(self.graph, target_k, log_n_sampling=True)
                
                # End timing
                runtime = time.time() - start_time
                
                # Calculate metrics
                size_bits = summary.size_in_bits()
                relative_size = size_bits / original_size
                l1_error = summary.compute_reconstruction_error(p=1)
                l2_error = summary.compute_reconstruction_error(p=2)
                
                # Store results
                results.append({
                    'algorithm': algo_name,
                    'runtime_sec': runtime,
                    'size_bits': size_bits,
                    'relative_size': relative_size,
                    'l1_error': l1_error,
                    'l2_error': l2_error,
                    'num_supernodes': len(summary.supernodes),
                    'num_superedges': len(summary.superedge_counts),
                    'status': 'success'
                })
                
                logger.info(f"  {algo_name} completed in {runtime:.2f}s")
                logger.info(f"  Size: {size_bits} bits ({relative_size:.2%} of original)")
                logger.info(f"  L1 Error: {l1_error:.6f}, L2 Error: {l2_error:.6f}")
                logger.info(f"  {len(summary.supernodes)} supernodes, {len(summary.superedge_counts)} superedges")
                
            except Exception as e:
                logger.error(f"Error testing {algo_name}: {e}")
                
                # Store error result
                results.append({
                    'algorithm': algo_name,
                    'runtime_sec': None,
                    'size_bits': None,
                    'relative_size': None,
                    'l1_error': None,
                    'l2_error': None,
                    'num_supernodes': None,
                    'num_superedges': None,
                    'status': 'error',
                    'error': str(e)
                })
        
        # Convert to DataFrame for easier analysis
        results_df = pd.DataFrame(results)
        
        # Verify that we have at least some successful runs
        successful = results_df[results_df['status'] == 'success']
        self.assertGreater(len(successful), 0, "All algorithms failed to run")
        
        # Check that OptimizedSSumM is faster than regular SSumM
        if 'OptimizedSSumM' in successful['algorithm'].values and 'SSumM' in successful['algorithm'].values:
            ssumm_time = successful[successful['algorithm'] == 'SSumM']['runtime_sec'].iloc[0]
            opt_time = successful[successful['algorithm'] == 'OptimizedSSumM']['runtime_sec'].iloc[0]
            
            # Verify optimization (allow some margin for measurement error)
            self.assertLessEqual(opt_time, ssumm_time * 1.1, 
                               "OptimizedSSumM should be at least as fast as SSumM")
        
        # Print comparison table
        if len(successful) > 0:
            logger.info("\nAlgorithm Comparison Results:")
            print(successful[['algorithm', 'runtime_sec', 'relative_size', 'l1_error']])


class SSumMPerformanceTests(unittest.TestCase):
    """Performance tests for the SSumM algorithm."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Generate synthetic graphs of different sizes
        self.graphs = {
            'small': nx.barabasi_albert_graph(100, 3, seed=42),
            'medium': nx.barabasi_albert_graph(500, 3, seed=42),
            'large': nx.barabasi_albert_graph(1000, 3, seed=42)
        }
        
        # Create algorithm instances
        self.ssumm = SSumM(max_iterations=10)
        self.optimized_ssumm = OptimizedSSumM(max_iterations=10)
    
    def test_scaling_behavior(self):
        """Test how the algorithm scales with graph size."""
        results = []
        
        for size, graph in self.graphs.items():
            # Calculate target size
            original_size = 2 * graph.number_of_edges() * np.ceil(np.log2(graph.number_of_nodes()))
            target_size = int(0.3 * original_size)
            
            # Skip large graph for regular SSumM to save time
            algorithms = [('OptimizedSSumM', self.optimized_ssumm)]
            if size != 'large':
                algorithms.append(('SSumM', self.ssumm))
            
            for algo_name, algorithm in algorithms:
                try:
                    logger.info(f"Testing {algo_name} on {size} graph...")
                    
                    # Start timing
                    start_time = time.time()
                    
                    # Run the algorithm
                    summary = algorithm.summarize(graph, target_size)
                    
                    # End timing
                    runtime = time.time() - start_time
                    
                    # Calculate metrics
                    size_bits = summary.size_in_bits()
                    relative_size = size_bits / original_size
                    l1_error = summary.compute_reconstruction_error(p=1)
                    
                    # Store results
                    results.append({
                        'algorithm': algo_name,
                        'graph_size': size,
                        'nodes': graph.number_of_nodes(),
                        'edges': graph.number_of_edges(),
                        'runtime_sec': runtime,
                        'relative_size': relative_size,
                        'l1_error': l1_error
                    })
                    
                    logger.info(f"  {algo_name} completed in {runtime:.2f}s")
                    logger.info(f"  Size: {size_bits} bits ({relative_size:.2%} of original)")
                    logger.info(f"  L1 Error: {l1_error:.6f}")
                
                except Exception as e:
                    logger.error(f"Error testing {algo_name} on {size} graph: {e}")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Verify that we have results
        self.assertGreater(len(results_df), 0, "No successful performance tests")
        
        # Plot scaling behavior if we have enough data
        if len(results_df) >= 3:
            plt.figure(figsize=(12, 5))
            
            # Runtime vs. Graph Size
            plt.subplot(1, 2, 1)
            for algo in results_df['algorithm'].unique():
                algo_data = results_df[results_df['algorithm'] == algo]
                plt.plot(algo_data['nodes'], algo_data['runtime_sec'], 'o-', label=algo)
            
            plt.xlabel('Number of Nodes')
            plt.ylabel('Runtime (seconds)')
            plt.title('Runtime Scaling')
            plt.legend()
            plt.grid(alpha=0.3)
            
            # Error vs. Graph Size
            plt.subplot(1, 2, 2)
            for algo in results_df['algorithm'].unique():
                algo_data = results_df[results_df['algorithm'] == algo]
                plt.plot(algo_data['nodes'], algo_data['l1_error'], 'o-', label=algo)
            
            plt.xlabel('Number of Nodes')
            plt.ylabel('L1 Reconstruction Error')
            plt.title('Error Scaling')
            plt.legend()
            plt.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig("scaling_behavior.png", dpi=300)
            plt.close()
            
            logger.info("Scaling behavior visualization saved to scaling_behavior.png")
    
    def test_memory_usage(self):
        """Test memory usage of the algorithm."""
        # Skip this test if memory_profiler is not available
        try:
            from memory_profiler import memory_usage
        except ImportError:
            logger.warning("memory_profiler not available, skipping memory usage test")
            return
        
        results = []
        
        for size, graph in self.graphs.items():
            # Calculate target size
            original_size = 2 * graph.number_of_edges() * np.ceil(np.log2(graph.number_of_nodes()))
            target_size = int(0.3 * original_size)
            
            # Skip large graph for regular SSumM to save time
            algorithms = [('OptimizedSSumM', self.optimized_ssumm)]
            if size != 'large':
                algorithms.append(('SSumM', self.ssumm))
            
            for algo_name, algorithm in algorithms:
                try:
                    logger.info(f"Testing memory usage of {algo_name} on {size} graph...")
                    
                    # Define function to measure
                    def run_algorithm():
                        algorithm.summarize(graph, target_size)
                    
                    # Measure memory usage
                    mem_usage = memory_usage(run_algorithm, interval=0.1)
                    
                    # Calculate memory statistics
                    max_mem = max(mem_usage)
                    avg_mem = sum(mem_usage) / len(mem_usage)
                    
                    # Store results
                    results.append({
                        'algorithm': algo_name,
                        'graph_size': size,
                        'nodes': graph.number_of_nodes(),
                        'edges': graph.number_of_edges(),
                        'max_memory_mb': max_mem,
                        'avg_memory_mb': avg_mem
                    })
                    
                    logger.info(f"  {algo_name} used max {max_mem:.2f} MB, avg {avg_mem:.2f} MB")
                
                except Exception as e:
                    logger.error(f"Error testing memory usage of {algo_name} on {size} graph: {e}")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Verify that we have results
        self.assertGreater(len(results_df), 0, "No successful memory usage tests")
        
        # Plot memory usage if we have enough data
        if len(results_df) >= 3:
            plt.figure(figsize=(10, 6))
            
            # Memory vs. Graph Size
            for algo in results_df['algorithm'].unique():
                algo_data = results_df[results_df['algorithm'] == algo]
                plt.plot(algo_data['nodes'], algo_data['max_memory_mb'], 'o-', label=f"{algo} (Max)")
                plt.plot(algo_data['nodes'], algo_data['avg_memory_mb'], 's--', label=f"{algo} (Avg)")
            
            plt.xlabel('Number of Nodes')
            plt.ylabel('Memory Usage (MB)')
            plt.title('Memory Scaling')
            plt.legend()
            plt.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig("memory_scaling.png", dpi=300)
            plt.close()
            
            logger.info("Memory scaling visualization saved to memory_scaling.png")


class TestHarness:
    """Test harness to run all tests and report results."""
    
    def __init__(self, output_dir="./test_results"):
        """
        Initialize the test harness.
        
        Args:
            output_dir: Directory to save test results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure logger to also write to file
        file_handler = logging.FileHandler(os.path.join(output_dir, "test_results.log"))
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    def run_all_tests(self):
        """Run all tests and report results."""
        logger.info("Starting SSumM algorithm test suite")
        
        # Create test suite
        suite = unittest.TestSuite()
        
        # Add unit tests
        suite.addTest(unittest.makeSuite(SSumMUnitTests))
        
        # Add integration tests
        suite.addTest(unittest.makeSuite(SSumMIntegrationTests))
        
        # Add performance tests
        suite.addTest(unittest.makeSuite(SSumMPerformanceTests))
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Report results
        logger.info(f"Tests complete: {result.testsRun} tests run")
        logger.info(f"Successes: {result.testsRun - len(result.errors) - len(result.failures)}")
        logger.info(f"Failures: {len(result.failures)}")
        logger.info(f"Errors: {len(result.errors)}")
        
        # Create detailed report
        self._create_test_report(result)
        
        return result
    
    def _create_test_report(self, result):
        """Create a detailed test report."""
        report_path = os.path.join(self.output_dir, "test_report.md")
        
        with open(report_path, 'w') as f:
            f.write("# SSumM Algorithm Test Report\n\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overview
            f.write("## Test Summary\n\n")
            f.write(f"- Total tests: {result.testsRun}\n")
            f.write(f"- Successful: {result.testsRun - len(result.errors) - len(result.failures)}\n")
            f.write(f"- Failures: {len(result.failures)}\n")
            f.write(f"- Errors: {len(result.errors)}\n\n")
            
            # Details of failures
            if result.failures:
                f.write("## Test Failures\n\n")
                for i, (test, trace) in enumerate(result.failures):
                    f.write(f"### Failure {i+1}: {test}\n\n")
                    f.write("```\n")
                    f.write(trace)
                    f.write("```\n\n")
            
            # Details of errors
            if result.errors:
                f.write("## Test Errors\n\n")
                for i, (test, trace) in enumerate(result.errors):
                    f.write(f"### Error {i+1}: {test}\n\n")
                    f.write("```\n")
                    f.write(trace)
                    f.write("```\n\n")
            
            # Visualizations
            f.write("## Performance Visualizations\n\n")
            
            if os.path.exists("scaling_behavior.png"):
                f.write("### Scaling Behavior\n\n")
                f.write("![Scaling Behavior](scaling_behavior.png)\n\n")
            
            if os.path.exists("memory_scaling.png"):
                f.write("### Memory Scaling\n\n")
                f.write("![Memory Scaling](memory_scaling.png)\n\n")
        
        logger.info(f"Test report saved to {report_path}")
        
        return report_path


def main():
    """Main function to run the test framework."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run tests for SSumM algorithm')
    parser.add_argument('--output-dir', type=str, default='./test_results',
                        help='Directory to save test results')
    parser.add_argument('--unit-only', action='store_true',
                        help='Run only unit tests')
    parser.add_argument('--skip-performance', action='store_true',
                        help='Skip performance tests')
    args = parser.parse_args()
    
    # Create test harness
    harness = TestHarness(output_dir=args.output_dir)
    
    # Run appropriate tests
    if args.unit_only:
        suite = unittest.makeSuite(SSumMUnitTests)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
    elif args.skip_performance:
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(SSumMUnitTests))
        suite.addTest(unittest.makeSuite(SSumMIntegrationTests))
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
    else:
        result = harness.run_all_tests()
    
    # Return success/failure
    return 0 if len(result.failures) + len(result.errors) == 0 else 1

if __name__ == "__main__":
    sys.exit(main())