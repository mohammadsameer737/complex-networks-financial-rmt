import numpy as np
import networkx as nx
import pytest
from finnet_backbone.markov_analysis import MarketRegimeChain, random_walk_centrality

class TestMarketRegimeChain:
    """Tests for discrete-time Markov chain modeling of financial regimes."""

    def test_simulation_trajectory_length(self):
        """
        Verifies that the simulate method returns a trajectory 
        of exactly n_steps + 1 (including the initial state).
        """
        chain = MarketRegimeChain()
        trajectory = chain.simulate(n_steps=100)
        
        assert len(trajectory) == 101
        assert all(state in chain.states for state in trajectory)

    def test_stationary_distribution_properties(self):
        """
        Verifies that the stationary distribution (left eigenvector for eigenvalue 1)
        is a valid probability distribution: non-negative and sums to 1.0.
        This represents the long-run proportion of time spent in each market regime.
        """
        chain = MarketRegimeChain()
        pi = chain.stationary_distribution()
        
        assert np.isclose(np.sum(pi), 1.0, atol=1e-6), "Probabilities must sum to 1"
        assert np.all(pi >= 0), "Probabilities must be non-negative"


class TestRandomWalkCentrality:
    """Tests for random walk centrality on financial networks."""

    def test_centrality_normalization(self):
        """
        Verifies that the random walk centrality values sum to 1.0,
        representing a valid probability distribution of a walker's location 
        in the long run (ergodic theorem).
        """
        G = nx.karate_club_graph()
        # Add some dummy weights to simulate financial correlations
        for u, v in G.edges():
            G[u][v]['weight'] = np.random.rand()
            
        centrality = random_walk_centrality(G, n_steps=1000, n_walkers=10)
        
        assert np.isclose(sum(centrality.values()), 1.0, atol=1e-6)
        assert len(centrality) == G.number_of_nodes()

    def test_centrality_disconnected_graph(self):
        """
        Verifies that the random walk handles disconnected components 
        (isolated nodes) without crashing, utilizing the teleportation 
        mechanism to ensure all nodes are visited.
        """
        G = nx.Graph()
        G.add_edge(1, 2, weight=1.0)
        G.add_node(3) # Isolated node
        
        centrality = random_walk_centrality(G, n_steps=500, n_walkers=5)
        
        assert 3 in centrality
        assert centrality[3] > 0.0 # Teleportation ensures it gets visited
        assert np.isclose(sum(centrality.values()), 1.0, atol=1e-6)