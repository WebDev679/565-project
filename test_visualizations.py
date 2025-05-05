import networkx as nx
import numpy as np
import os
import pickle
from graph_visualizer import GraphVisualizer
import matplotlib.pyplot as plt
import random

def create_test_data(output_dir="./test_data"):
    """
    Create test data for visualization testing.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a larger random graph
    # Using Barabasi-Albert model with 500 nodes and 5 edges per new node
    G_original = nx.barabasi_albert_graph(500, 5, seed=42)
    
    # Create a summary graph (simplified version)
    # Group nodes into supernodes
    supernode_mapping = {}
    for i in range(0, 500, 10):  # Create 50 supernodes with ~10 nodes each
        supernode_id = i // 10
        nodes = list(range(i, min(i+10, 500)))
        supernode_mapping[supernode_id] = nodes
    
    # Create summary graph
    G_summary = nx.Graph()
    for supernode_id in supernode_mapping:
        G_summary.add_node(supernode_id)
    
    # Add superedges based on connections in original graph
    for supernode_i in supernode_mapping:
        nodes_i = supernode_mapping[supernode_i]
        for supernode_j in supernode_mapping:
            if supernode_i >= supernode_j:  # Avoid duplicates
                continue
            nodes_j = supernode_mapping[supernode_j]
            
            # Count edges between supernodes
            edge_count = sum(1 for u in nodes_i for v in nodes_j if G_original.has_edge(u, v))
            
            if edge_count > 0:
                G_summary.add_edge(supernode_i, supernode_j, weight=edge_count)
    
    # Create reconstructed graph
    G_reconstructed = nx.Graph()
    G_reconstructed.add_nodes_from(G_original.nodes())
    
    # Add edges based on summary graph
    for supernode_i, supernode_j in G_summary.edges():
        weight = G_summary[supernode_i][supernode_j]['weight'] / (len(supernode_mapping[supernode_i]) * len(supernode_mapping[supernode_j]))
        
        for u in supernode_mapping[supernode_i]:
            for v in supernode_mapping[supernode_j]:
                if np.random.random() < weight:
                    G_reconstructed.add_edge(u, v)
    
    # Generate more realistic algorithm results
    # SSumM should generally perform better but not unrealistically so
    algorithm_results = {
        'algorithm': ['SSumM', 'k-Gs', 'S2L', 'SAA-Gs'],
        'dataset': ['test'] * 4,
        'reconstruction_error_l1': [0.0123, 0.0192, 0.0174, 0.0145],
        'reconstruction_error_l2': [0.0078, 0.0112, 0.0098, 0.0083],
        'size_in_bits': [42500, 68750, 51200, 57800],
        'runtime': [27.4, 119.8, 93.5, 42.3],
        'relative_size': [0.17, 0.275, 0.205, 0.231]
    }
    
    # Create more realistic convergence data with smoother curves
    iterations = 20
    # Create exponentially decreasing error
    base_error = 0.08
    decay_rate = 0.85
    reconstruction_error_l1 = [base_error * (decay_rate ** i) for i in range(iterations)]
    
    # Create decreasing size with diminishing returns
    initial_size = 75000
    final_size = 42500
    size_decrease = initial_size - final_size
    size_in_bits = [initial_size - size_decrease * (1 - decay_rate ** i) / (1 - decay_rate ** (iterations-1)) for i in range(iterations)]
    
    # Create decreasing number of supernodes
    initial_nodes = 200
    final_nodes = 50
    nodes_decrease = initial_nodes - final_nodes
    num_supernodes = [initial_nodes - nodes_decrease * (1 - decay_rate ** i) / (1 - decay_rate ** (iterations-1)) for i in range(iterations)]
    num_supernodes = [round(n) for n in num_supernodes]
    
    convergence_data = {
        'iteration': list(range(iterations)),
        'reconstruction_error_l1': reconstruction_error_l1,
        'size_in_bits': size_in_bits,
        'num_supernodes': num_supernodes
    }
    
    # Save all data
    with open(os.path.join(output_dir, 'original_graph.gpickle'), 'wb') as f:
        pickle.dump(G_original, f)

    with open(os.path.join(output_dir, 'summary_graph.gpickle'), 'wb') as f:
        pickle.dump(G_summary, f)

    with open(os.path.join(output_dir, 'reconstructed_graph.gpickle'), 'wb') as f:
        pickle.dump(G_reconstructed, f)
    
    with open(os.path.join(output_dir, 'supernode_mapping.pickle'), 'wb') as f:
        pickle.dump(supernode_mapping, f)
    
    with open(os.path.join(output_dir, 'algorithm_results.pickle'), 'wb') as f:
        pickle.dump(algorithm_results, f)
    
    with open(os.path.join(output_dir, 'convergence_data.pickle'), 'wb') as f:
        pickle.dump(convergence_data, f)
    
    print(f"Test data created in {output_dir}")
    return {
        'original_graph': os.path.join(output_dir, 'original_graph.gpickle'),
        'summary_graph': os.path.join(output_dir, 'summary_graph.gpickle'),
        'reconstructed_graph': os.path.join(output_dir, 'reconstructed_graph.gpickle'),
        'supernode_mapping': os.path.join(output_dir, 'supernode_mapping.pickle'),
        'algorithm_results': os.path.join(output_dir, 'algorithm_results.pickle'),
        'convergence_data': os.path.join(output_dir, 'convergence_data.pickle')
    }

def test_visualizations():
    """
    Test the visualization capabilities with sample data.
    """
    # Create test data
    test_dir = "./test_data"
    vis_dir = "./test_visualizations"
    
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    test_files = create_test_data(test_dir)
    
    # Initialize visualizer
    visualizer = GraphVisualizer(output_dir=vis_dir)
    
    # Load test data
    with open(test_files['original_graph'], 'rb') as f:
        G_original = pickle.load(f)

    with open(test_files['summary_graph'], 'rb') as f:
        G_summary = pickle.load(f)

    with open(test_files['reconstructed_graph'], 'rb') as f:
        G_reconstructed = pickle.load(f)
    
    with open(test_files['supernode_mapping'], 'rb') as f:
        supernode_info = pickle.load(f)
    
    with open(test_files['algorithm_results'], 'rb') as f:
        algorithm_results = pickle.load(f)
    
    with open(test_files['convergence_data'], 'rb') as f:
        convergence_data = pickle.load(f)
    
    # Test all visualization functions
    visualizations = []
    
    # 1. Original graph
    viz_path = visualizer.visualize_original_graph(G_original, "TestData")
    visualizations.append(viz_path)
    
    # 2. Summary graph
    viz_path = visualizer.visualize_summary_graph(G_summary, "TestData", 
                                                original_graph=G_original,
                                                supernode_info=supernode_info)
    visualizations.append(viz_path)
    
    # 3. Graph comparison
    viz_path = visualizer.visualize_graph_comparison(G_original, G_summary, G_reconstructed, 
                                                   "TestData", supernode_info=supernode_info)
    visualizations.append(viz_path)
    
    # 4. Error distribution
    original_adj = nx.to_numpy_array(G_original)
    reconstructed_adj = nx.to_numpy_array(G_reconstructed)
    viz_path = visualizer.visualize_error_distribution(original_adj, reconstructed_adj, "TestData")
    visualizations.append(viz_path)
    
    # 5. Algorithm comparison
    viz_paths = visualizer.visualize_algorithm_comparison(algorithm_results, "TestData")
    visualizations.extend(viz_paths)
    
    # 6. Convergence
    viz_path = visualizer.visualize_convergence(convergence_data, "TestData")
    visualizations.append(viz_path)
    
    # 7. Supernode content
    viz_path = visualizer.visualize_supernode_content(G_original, G_summary, supernode_info, "TestData")
    visualizations.append(viz_path)
    
    # 8. Create dashboard
    dashboard_path = visualizer.create_dashboard("TestData", visualizations)
    
    print(f"Test visualizations created in {vis_dir}")
    print(f"Dashboard available at {dashboard_path}")
    
    return dashboard_path

if __name__ == "__main__":
    test_visualizations()