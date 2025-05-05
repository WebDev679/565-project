import networkx as nx
import numpy as np
import math

class SsummCostFunction:
    """
    Implementation of the cost function for SSumM, based on the Minimum Description Length (MDL) principle.
    The cost function balances between the size of the summary and the reconstruction error.
    """
    
    def __init__(self, original_graph):
        """
        Initialize the cost function with the original graph.
        
        Args:
            original_graph (nx.Graph): The original input graph
        """
        self.original_graph = original_graph
        self.num_nodes = original_graph.number_of_nodes()
        self.num_edges = original_graph.number_of_edges()
        
    def calculate_total_cost(self, summary):
        """
        Calculate the total description cost of the summary graph.
        
        Args:
            summary (GraphSummary): The summary graph
            
        Returns:
            float: Total description cost in bits
        """
        # Calculate model cost (i.e., cost of describing the summary)
        model_cost = self.calculate_model_cost(summary)
        
        # Calculate data cost (i.e., cost of describing the original graph given the summary)
        data_cost = self.calculate_data_cost(summary)
        
        # Total cost is the sum of model cost and data cost
        return model_cost + data_cost
    
    def calculate_model_cost(self, summary):
        """
        Calculate the model cost, which is the cost of describing the summary graph.
        
        Args:
            summary (GraphSummary): The summary graph
            
        Returns:
            float: Model cost in bits
        """
        # Number of supernodes
        k = len(summary.supernodes)
        
        # Number of superedges
        p = len(summary.superedge_counts)
        
        # Cost to describe each superedge (source, destination, weight)
        # Using log2(|V|) as an upper bound for log2(k)
        edge_cost = p * (2 * np.ceil(np.log2(self.num_nodes)) + np.ceil(np.log2(self.num_edges)))
        
        # Cost to describe the membership of each node in a supernode
        membership_cost = self.num_nodes * np.ceil(np.log2(k))
        
        return edge_cost + membership_cost
    
    def calculate_data_cost(self, summary):
        """
        Calculate the data cost, which is the cost of describing the original graph given the summary.
        
        Args:
            summary (GraphSummary): The summary graph
            
        Returns:
            float: Data cost in bits
        """
        return sum(self.calculate_cost_for_supernode_pair(summary, supernode_pair) 
                  for supernode_pair in self.get_all_supernode_pairs(summary))
    
    def get_all_supernode_pairs(self, summary):
        """
        Get all pairs of supernodes, including self-pairs.
        
        Args:
            summary (GraphSummary): The summary graph
            
        Returns:
            list: List of all supernode pairs
        """
        supernodes = list(summary.supernodes.keys())
        pairs = []
        
        # Add all possible pairs
        for i in range(len(supernodes)):
            for j in range(i, len(supernodes)):
                pairs.append((supernodes[i], supernodes[j]))
                
        return pairs
    
    def calculate_cost_for_supernode_pair(self, summary, supernode_pair):
        """
        Calculate the cost for a pair of supernodes.
        
        Args:
            summary (GraphSummary): The summary graph
            supernode_pair (tuple): Pair of supernodes (A, B)
            
        Returns:
            float: Cost in bits for this supernode pair
        """
        A, B = supernode_pair
        key = (min(A, B), max(A, B))
        
        # Check if there's a superedge between A and B
        has_superedge = key in summary.superedge_counts
        
        # Get the subedges between nodes in A and B
        E_AB = self.get_subedges_between_supernodes(summary, A, B)
        
        # Get all possible subedges between nodes in A and B
        Pi_AB = self.get_possible_subedges_between_supernodes(summary, A, B)
        
        if has_superedge:
            # Cost when using the first encoding method (with superedge)
            return self.calculate_cost_method1(summary, A, B, E_AB, Pi_AB)
        else:
            # Cost when using the second encoding method (without superedge)
            return self.calculate_cost_method2(E_AB)
    
    def get_subedges_between_supernodes(self, summary, A, B):
        """
        Get the set of subedges between nodes in supernodes A and B in the original graph.
        
        Args:
            summary (GraphSummary): The summary graph
            A: First supernode
            B: Second supernode
            
        Returns:
            set: Set of subedges between A and B
        """
        nodes_A = summary.supernodes[A]
        nodes_B = summary.supernodes[B]
        
        # If A==B, we need to handle it differently to avoid counting edges twice
        if A == B:
            edges = set()
            for u in nodes_A:
                for v in nodes_A:
                    if u < v and self.original_graph.has_edge(u, v):
                        edges.add((u, v))
            return edges
        else:
            edges = set()
            for u in nodes_A:
                for v in nodes_B:
                    if self.original_graph.has_edge(u, v):
                        edges.add((u, v))
            return edges
    
    def get_possible_subedges_between_supernodes(self, summary, A, B):
        """
        Get the set of all possible subedges between nodes in supernodes A and B.
        
        Args:
            summary (GraphSummary): The summary graph
            A: First supernode
            B: Second supernode
            
        Returns:
            set: Set of all possible subedges between A and B
        """
        nodes_A = summary.supernodes[A]
        nodes_B = summary.supernodes[B]
        
        # If A==B, we need to handle it differently
        if A == B:
            possible_edges = set()
            for u in nodes_A:
                for v in nodes_A:
                    if u < v:  # Avoid counting pairs twice and self-loops
                        possible_edges.add((u, v))
            return possible_edges
        else:
            possible_edges = set()
            for u in nodes_A:
                for v in nodes_B:
                    possible_edges.add((u, v))
            return possible_edges
    
    def calculate_cost_method1(self, summary, A, B, E_AB, Pi_AB):
        """
        Calculate the cost using the first encoding method (with superedge).
        
        Args:
            summary (GraphSummary): The summary graph
            A: First supernode
            B: Second supernode
            E_AB: Set of subedges between A and B
            Pi_AB: Set of all possible subedges between A and B
            
        Returns:
            float: Cost in bits
        """
        # Cost for the superedge
        superedge_cost = 2 * np.ceil(np.log2(self.num_nodes)) + np.ceil(np.log2(self.num_edges))
        
        # Calculate sigma (proportion of existing subedges)
        sigma = len(E_AB) / len(Pi_AB) if len(Pi_AB) > 0 else 0
        
        # Shannon entropy for encoding the subedges
        if sigma == 0 or sigma == 1:
            entropy_cost = 0  # If sigma is 0 or 1, entropy is 0
        else:
            entropy_cost = -len(Pi_AB) * (sigma * np.log2(sigma) + (1 - sigma) * np.log2(1 - sigma))
        
        return superedge_cost + entropy_cost
    
    def calculate_cost_method2(self, E_AB):
        """
        Calculate the cost using the second encoding method (without superedge).
        
        Args:
            E_AB: Set of subedges between supernodes
            
        Returns:
            float: Cost in bits
        """
        # Simply encode each subedge directly
        return 2 * len(E_AB) * np.ceil(np.log2(self.num_nodes))
    
    def find_optimal_superedges(self, summary):
        """
        Find the optimal set of superedges given the supernodes.
        
        Args:
            summary (GraphSummary): The summary graph with supernodes defined
            
        Returns:
            set: Set of optimal superedges (pairs of supernodes)
        """
        optimal_superedges = set()
        
        # For each pair of supernodes, determine if adding a superedge reduces the cost
        for supernode_pair in self.get_all_supernode_pairs(summary):
            A, B = supernode_pair
            key = (min(A, B), max(A, B))
            
            # Get the subedges between nodes in A and B
            E_AB = self.get_subedges_between_supernodes(summary, A, B)
            
            # Get all possible subedges between nodes in A and B
            Pi_AB = self.get_possible_subedges_between_supernodes(summary, A, B)
            
            # Calculate cost with and without the superedge
            cost_with_superedge = self.calculate_cost_method1(summary, A, B, E_AB, Pi_AB)
            cost_without_superedge = self.calculate_cost_method2(E_AB)
            
            # Add superedge if it reduces the cost
            if cost_with_superedge <= cost_without_superedge and len(E_AB) > 0:
                optimal_superedges.add(key)
                
        return optimal_superedges
    
    def calculate_relative_reduction(self, summary, supernode1, supernode2):
        """
        Calculate the relative reduction in cost when merging two supernodes.
        This is used in the greedy merging phase.
        
        Args:
            summary (GraphSummary): The current summary graph
            supernode1: First supernode to merge
            supernode2: Second supernode to merge
            
        Returns:
            float: Relative reduction in cost (higher is better)
        """
        # Calculate the cost of describing the original supernodes and their connections
        original_cost = self.calculate_cost_for_supernode(summary, supernode1)
        original_cost += self.calculate_cost_for_supernode(summary, supernode2)
        original_cost -= self.calculate_cost_for_supernode_pair(summary, (supernode1, supernode2))
        
        # Create a temporary summary with the merged supernode
        temp_summary = self.create_merged_summary(summary, supernode1, supernode2)
        
        # Calculate the cost of describing the merged supernode and its connections
        merged_supernode = min(supernode1, supernode2)  # The ID of the merged supernode
        merged_cost = self.calculate_cost_for_supernode(temp_summary, merged_supernode)
        
        # Calculate relative reduction
        if original_cost == 0:
            return 0
        else:
            return 1 - (merged_cost / original_cost)
    
    def calculate_cost_for_supernode(self, summary, supernode):
        """
        Calculate the total cost for a supernode and all its connections.
        
        Args:
            summary (GraphSummary): The summary graph
            supernode: The supernode ID
            
        Returns:
            float: Total cost in bits
        """
        total_cost = 0
        
        # Add cost for the self-loop
        total_cost += self.calculate_cost_for_supernode_pair(summary, (supernode, supernode))
        
        # Add costs for connections to other supernodes
        for other_supernode in summary.supernodes:
            if other_supernode != supernode:
                total_cost += self.calculate_cost_for_supernode_pair(summary, (supernode, other_supernode))
                
        return total_cost
    
    def create_merged_summary(self, summary, supernode1, supernode2):
        """
        Create a new summary with two supernodes merged.
        
        Args:
            summary (GraphSummary): The original summary graph
            supernode1: First supernode to merge
            supernode2: Second supernode to merge
            
        Returns:
            GraphSummary: New summary with merged supernodes
        """
        import copy
        
        # Create a deep copy of the original summary
        new_summary = copy.deepcopy(summary)
        
        # Merge the supernodes in the copy
        new_summary.merge_supernodes(supernode1, supernode2)
        
        # Find optimal superedges for the new summary
        optimal_superedges = self.find_optimal_superedges(new_summary)
        
        # Update the superedges in the new summary
        new_superedge_counts = {}
        for key in optimal_superedges:
            A, B = key
            # Count the subedges between these supernodes
            E_AB = self.get_subedges_between_supernodes(new_summary, A, B)
            new_superedge_counts[key] = len(E_AB)
        
        new_summary.superedge_counts = new_superedge_counts
        
        return new_summary