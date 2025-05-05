import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from ssumm_directed import GraphSummaryDirected

def create_comparison_figure():
    """Create a visual comparison of directed graph summarization."""
    
    # Create a directed graph with interesting structure
    G = nx.DiGraph()
    # Add some nodes with different roles
    sources = [1, 2, 3]
    middlayers = [4, 5, 6, 7]
    sinks = [8, 9, 10]
    
    # Create edges with directional patterns
    # Sources to middle layer
    for s in sources:
        for m in middlayers:
            if np.random.random() < 0.6:
                G.add_edge(s, m)
    
    # Middle layer to sinks
    for m in middlayers:
        for sink in sinks:
            if np.random.random() < 0.5:
                G.add_edge(m, sink)
    
    # Add some feedback edges
    G.add_edge(5, 2)
    G.add_edge(6, 3)
    G.add_edge(9, 4)
    
    # Create a summary where we merge similar nodes
    summary = GraphSummaryDirected(G)
    
    # Merge sources into one supernode
    summary.merge_supernodes(1, 2)
    summary.merge_supernodes(1, 3)
    
    # Merge middle layers
    summary.merge_supernodes(4, 5)
    summary.merge_supernodes(4, 6)
    
    # Merge sinks
    summary.merge_supernodes(8, 9)
    
    # Create visualization
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1])
    
    # Original graph
    ax1 = fig.add_subplot(gs[0, 0])
    pos_orig = nx.spring_layout(G, seed=42)
    
    # Draw original graph with node coloring by role
    node_colors = []
    for node in G.nodes():
        if node in sources:
            node_colors.append('lightgreen')
        elif node in middlayers:
            node_colors.append('lightblue')
        elif node in sinks:
            node_colors.append('lightcoral')
        else:
            node_colors.append('lightgray')
    
    nx.draw_networkx_nodes(G, pos_orig, node_color=node_colors, node_size=500, ax=ax1)
    nx.draw_networkx_edges(G, pos_orig, edge_color='gray', arrows=True, 
                          arrowsize=20, arrowstyle='->', ax=ax1)
    nx.draw_networkx_labels(G, pos_orig, font_size=10, ax=ax1)
    ax1.set_title('Original Directed Graph')
    ax1.axis('off')
    
    # Summary graph
    ax2 = fig.add_subplot(gs[0, 1])
    G_summary = summary.to_networkx_graph()
    pos_summary = nx.spring_layout(G_summary, seed=42)
    
    # Node sizes based on supernode size
    node_sizes = [800 + 400 * G_summary.nodes[n]['size'] for n in G_summary.nodes()]
    
    # Draw summary graph
    nx.draw_networkx_nodes(G_summary, pos_summary, node_size=node_sizes, 
                          node_color='orange', ax=ax2)
    nx.draw_networkx_edges(G_summary, pos_summary, edge_color='darkred', 
                          arrows=True, arrowsize=20, arrowstyle='->', 
                          width=2, ax=ax2)
    
    # Add supernode labels showing contained nodes
    labels = {}
    for node in G_summary.nodes():
        original_nodes = G_summary.nodes[node]['original_nodes']
        labels[node] = str(sorted(original_nodes))
    nx.draw_networkx_labels(G_summary, pos_summary, labels, font_size=8, ax=ax2)
    
    # Add edge weights
    edge_labels = nx.get_edge_attributes(G_summary, 'weight')
    nx.draw_networkx_edge_labels(G_summary, pos_summary, edge_labels, font_size=8, ax=ax2)
    
    ax2.set_title('Summary Graph (Supernodes)')
    ax2.axis('off')
    
    # Adjacency matrices
    ax3 = fig.add_subplot(gs[0, 2])
    A_orig = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
    im = ax3.imshow(A_orig, cmap='Blues')
    ax3.set_title('Original Adjacency Matrix')
    plt.colorbar(im, ax=ax3, fraction=0.046)
    
    ax4 = fig.add_subplot(gs[1, 0])
    A_expected = summary.get_expected_adjacency_matrix()
    im = ax4.imshow(A_expected, cmap='Blues')
    ax4.set_title('Expected Adjacency (Reconstructed)')
    plt.colorbar(im, ax=ax4, fraction=0.046)
    
    # Error visualization
    ax5 = fig.add_subplot(gs[1, 1])
    error_matrix = np.abs(A_orig - A_expected)
    im = ax5.imshow(error_matrix, cmap='Reds')
    ax5.set_title('Reconstruction Error Matrix')
    plt.colorbar(im, ax=ax5, fraction=0.046)
    
    # Statistics panel
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Calculate statistics
    stats = summary.summary_stats()
    original_size = G.number_of_edges() * 2 * np.ceil(np.log2(G.number_of_nodes()))
    summary_size = summary.size_in_bits()
    
    # Create statistics text
    stats_text = f"""Summary Statistics:
    
Original Graph:
  Nodes: {G.number_of_nodes()}
  Edges: {G.number_of_edges()}
  Size: {original_size:.0f} bits

Summary Graph:
  Supernodes: {stats['num_supernodes']}
  Superedges: {stats['num_superedges']}
  Size: {summary_size:.0f} bits
  
Compression:
  Ratio: {original_size/summary_size:.2f}x
  Error: {stats['reconstruction_error']:.4f}

Key Features:
  • Preserves directionality
  • Asymmetric adjacency
  • Feedback edges detected
  • Role-based merging"""
    
    ax6.text(0.1, 0.5, stats_text, transform=ax6.transAxes, 
             fontsize=11, verticalalignment='center', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('directed_graph_summarization_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a separate figure for edge direction preservation
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Show original edge directions
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=400, ax=ax1)
    edges = nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                                  arrowsize=15, arrowstyle='->', ax=ax1)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax1)
    ax1.set_title('Original: All Edge Directions')
    ax1.axis('off')
    
    # Show summarized edge directions with weights
    pos_sum = nx.circular_layout(G_summary)
    node_sizes = [800 + 400 * G_summary.nodes[n]['size'] for n in G_summary.nodes()]
    nx.draw_networkx_nodes(G_summary, pos_sum, node_size=node_sizes, 
                          node_color='orange', ax=ax2)
    
    # Draw edges with varying widths based on weight
    edge_weights = [G_summary[u][v]['weight'] for u, v in G_summary.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [4 * w / max_weight for w in edge_weights]
    
    nx.draw_networkx_edges(G_summary, pos_sum, edge_color='darkred', 
                          arrows=True, arrowsize=15, arrowstyle='->', 
                          width=edge_widths, ax=ax2)
    nx.draw_networkx_labels(G_summary, pos_sum, labels, font_size=8, ax=ax2)
    
    # Add edge weights as labels
    nx.draw_networkx_edge_labels(G_summary, pos_sum, edge_labels, font_size=8, ax=ax2)
    
    ax2.set_title('Summary: Aggregated Directions')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('edge_direction_preservation.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_comparison_figure()