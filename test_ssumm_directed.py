import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from ssumm_directed import SSuMMDirected, GraphSummaryDirected

def create_directed_test_graphs():
    """Create various test directed graphs."""
    
    # 1. Simple directed cycle
    G1 = nx.DiGraph()
    G1.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])
    
    # 2. Directed network with fan-out
    G2 = nx.DiGraph()
    G2.add_edges_from([
        (1, 2), (1, 3), (1, 4),
        (2, 5), (3, 5), (4, 5),
        (5, 6), (5, 7)
    ])
    
    # 3. Bipartite-like directed graph
    G3 = nx.DiGraph()
    # Source group
    sources = range(1, 4)
    # Sink group  
    sinks = range(4, 8)
    # Connect sources to sinks with directed edges
    for s in sources:
        for t in sinks:
            if (s + t) % 2 == 0:
                G3.add_edge(s, t)
    
    # 4. Web-like graph with backlinks
    G4 = nx.DiGraph()
    G4.add_edges_from([
        (1, 2), (1, 3), (2, 3), (2, 4),
        (3, 4), (3, 5), (4, 5), (4, 6),
        (5, 6), (5, 1), (6, 1)  # Back edges
    ])
    
    return {
        'cycle': G1,
        'fan_out': G2,
        'bipartite': G3,
        'web': G4
    }

def visualize_directed_graph(G, filename, title="Directed Graph"):
    """Visualize a directed graph."""
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
    
    # Draw edges with arrows
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                          arrowsize=20, arrowstyle='->', 
                          connectionstyle='arc3,rad=0.1')
    
    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=12)
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_summary_graph(summary, filename, title="Summary Graph"):
    """Visualize a summary graph."""
    G_summary = summary.to_networkx_graph()
    
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G_summary, seed=42)
    
    # Node sizes based on supernode size
    node_sizes = [500 + 500 * G_summary.nodes[n]['size'] for n in G_summary.nodes()]
    
    # Node colors based on size
    node_colors = [G_summary.nodes[n]['size'] for n in G_summary.nodes()]
    
    # Draw nodes
    nodes = nx.draw_networkx_nodes(G_summary, pos, node_size=node_sizes, 
                                 node_color=node_colors, cmap='viridis')
    
    # Draw edges with arrows and weights
    edges = nx.draw_networkx_edges(G_summary, pos, edge_color='gray', 
                                 arrows=True, arrowsize=20, arrowstyle='->', 
                                 connectionstyle='arc3,rad=0.1')
    
    # Add node labels
    nx.draw_networkx_labels(G_summary, pos, font_size=10)
    
    # Add edge labels (weights)
    edge_labels = nx.get_edge_attributes(G_summary, 'weight')
    nx.draw_networkx_edge_labels(G_summary, pos, edge_labels, font_size=8)
    
    # Add colorbar
    plt.colorbar(nodes, label='Supernode size')
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def compare_with_undirected(G_directed):
    """Compare directed and undirected summarization."""
    # Convert to undirected for comparison
    G_undirected = G_directed.to_undirected()
    
    # Parameters
    k_bits = 50  # Target size in bits
    
    # Summarize directed graph
    ssumm_directed = SSuMMDirected(num_iterations=10)
    summary_directed = ssumm_directed.summarize(G_directed, k_bits, verbose=True)
    
    # Get statistics
    stats_directed = summary_directed.summary_stats()
    
    print(f"\nDirected Summary Statistics:")
    print(f"Number of supernodes: {stats_directed['num_supernodes']}")
    print(f"Number of superedges: {stats_directed['num_superedges']}")
    print(f"Reconstruction error: {stats_directed['reconstruction_error']:.6f}")
    print(f"Size in bits: {summary_directed.size_in_bits():.2f}")
    
    return summary_directed

def test_reconstruction_error():
    """Test reconstruction error calculation for directed graphs."""
    # Create a simple directed graph with known properties
    G = nx.DiGraph()
    edges = [(1, 2), (2, 3), (3, 1), (2, 4), (4, 3)]
    G.add_edges_from(edges)
    
    # Initialize trivial summary
    summary = GraphSummaryDirected(G)
    
    # Test initial error (should be 0 since trivial summary is perfect)
    initial_error = summary.compute_reconstruction_error()
    print(f"Initial reconstruction error: {initial_error:.6f}")
    
    # Merge two nodes and test error
    new_id = summary.merge_supernodes(1, 2)
    merge_error = summary.compute_reconstruction_error()
    print(f"Error after merging nodes 1 and 2: {merge_error:.6f}")
    
    # Show expected adjacency matrix
    A_expected = summary.get_expected_adjacency_matrix()
    print("\nExpected adjacency matrix:")
    print(A_expected)
    
    return summary

def compare_different_candidates():
    """Compare performance with different candidate generation strategies."""
    G = nx.DiGraph()
    # Create a more complex directed graph
    for i in range(20):
        for j in range(20):
            if i != j and np.random.random() < 0.1:
                G.add_edge(i, j)
    
    # Test different number of iterations
    k_bits = 100
    iterations_list = [5, 10, 20]
    
    results = []
    for iterations in iterations_list:
        ssumm = SSuMMDirected(num_iterations=iterations)
        summary = ssumm.summarize(G, k_bits, verbose=False)
        
        stats = summary.summary_stats()
        results.append({
            'iterations': iterations,
            'reconstruction_error': stats['reconstruction_error'],
            'size_in_bits': summary.size_in_bits(),
            'num_supernodes': stats['num_supernodes']
        })
    
    # Display results
    print("\nPerformance with different iteration counts:")
    print("Iterations | Error | Size (bits) | Supernodes")
    print("-" * 45)
    for r in results:
        print(f"{r['iterations']:9d} | {r['reconstruction_error']:.6f} | {r['size_in_bits']:10.2f} | {r['num_supernodes']:9d}")
    
    return results

def main():
    """Main test function."""
    print("Testing SSumM for Directed Graphs")
    print("=" * 40)
    
    # Create test graphs
    graphs = create_directed_test_graphs()
    
    # Test each graph
    for name, G in graphs.items():
        print(f"\nTesting {name} graph:")
        print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        
        # Visualize original graph
        visualize_directed_graph(G, f"{name}_original.png", f"Original {name} Graph")
        
        # Set target size as 30% of original
        original_size = G.number_of_edges() * 2 * np.ceil(np.log2(G.number_of_nodes()))
        k_bits = int(0.3 * original_size)
        
        print(f"Original size: {original_size:.2f} bits")
        print(f"Target size: {k_bits:.2f} bits")
        
        # Run SSumM
        ssumm = SSuMMDirected(num_iterations=15)
        summary = ssumm.summarize(G, k_bits, verbose=False)
        
        # Visualize summary
        visualize_summary_graph(summary, f"{name}_summary.png", f"Summary {name} Graph")
        
        # Print statistics
        stats = summary.summary_stats()
        print(f"Summary statistics:")
        print(f"  Supernodes: {stats['num_supernodes']}")
        print(f"  Superedges: {stats['num_superedges']}")
        print(f"  Reconstruction error: {stats['reconstruction_error']:.6f}")
        print(f"  Final size: {summary.size_in_bits():.2f} bits")
        print(f"  Compression ratio: {stats['size_reduction']:.2%}")
    
    # Test reconstruction error
    print("\n" + "=" * 40)
    print("Testing reconstruction error...")
    test_reconstruction_error()
    
    # Compare different iteration counts
    print("\n" + "=" * 40)
    print("Comparing different iteration counts...")
    compare_different_candidates()
    
    # Create a large directed graph for scalability test
    print("\n" + "=" * 40)
    print("Testing scalability with larger graph...")
    G_large = nx.generators.random_graphs.fast_gnp_random_graph(100, 0.05, directed=True)
    ssumm_large = SSuMMDirected(num_iterations=20)
    
    # Test different target sizes
    original_size = G_large.number_of_edges() * 2 * np.ceil(np.log2(G_large.number_of_nodes()))
    target_ratios = [0.5, 0.3, 0.1]
    
    print(f"Large graph: {G_large.number_of_nodes()} nodes, {G_large.number_of_edges()} edges")
    print(f"Original size: {original_size:.2f} bits")
    print("\nTarget Ratio | Final Size | Error | Supernodes")
    print("-" * 45)
    
    for ratio in target_ratios:
        k_bits = int(ratio * original_size)
        summary = ssumm_large.summarize(G_large, k_bits, verbose=False)
        stats = summary.summary_stats()
        
        print(f"{ratio:11.1f} | {summary.size_in_bits():10.2f} | {stats['reconstruction_error']:.6f} | {stats['num_supernodes']:9d}")

if __name__ == "__main__":
    main()