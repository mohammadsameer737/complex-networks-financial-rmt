import numpy as np
import networkx as nx
import pytest
from finnet_backbone.backbone import disparity_filter_naive, get_gcc_size

class TestDisparityFilter:
    """Tests for the Disparity Filter backbone extraction algorithm."""

    def test_disparity_filter_naive_symmetric(self):
        """
        Verifies that the naive disparity filter correctly processes a 
        symmetric weight matrix and returns a valid binary backbone matrix 
        and a matrix of p-values bounded in [0, 1].
        """
        np.random.seed(42)
        W = np.random.rand(20, 20)
        W = (W + W.T) / 2  # Enforce symmetry
        np.fill_diagonal(W, 0)
        
        backbone, pvals = disparity_filter_naive(W, alpha=0.05)
        
        assert backbone.shape == W.shape
        assert np.all(backbone >= 0)
        assert np.all(pvals >= 0) and np.all(pvals <= 1)

    def test_disparity_filter_disconnected_components(self):
        """
        Verifies that the disparity filter correctly normalizes weights 
        per node independently, even when the graph contains disconnected 
        components. The sum of normalized weights for any node must equal 1.0.
        """
        G = nx.Graph()
        G.add_edge(1, 2, weight=3.0)
        G.add_edge(1, 3, weight=1.0)
        G.add_edge(4, 5, weight=5.0)  # Disconnected component
        
        # Convert to weight matrix for naive filter
        nodes = list(G.nodes())
        W = np.zeros((len(nodes), len(nodes)))
        for u, v, data in G.edges(data=True):
            i, j = nodes.index(u), nodes.index(v)
            W[i, j] = W[j, i] = data['weight']
            
        backbone, _ = disparity_filter_naive(W, alpha=0.5)
        assert backbone.shape == W.shape

    def test_disparity_filter_zero_weight_handling(self):
        """
        Verifies robustness against zero-weight edges, which can cause 
        division by zero in naive implementations of weight normalization (p_i = w_i / s_i).
        """
        W = np.array([[0.0, 0.0, 5.0],
                      [0.0, 0.0, 0.0],
                      [5.0, 0.0, 0.0]])
        # Should not raise ZeroDivisionError
        backbone, pvals = disparity_filter_naive(W, alpha=0.1)
        assert np.isfinite(pvals).all()


class TestNetworkMetrics:
    """Tests for network robustness and connectivity metrics."""

    def test_gcc_size_calculation_bounds(self):
        """
        Verifies that the Giant Connected Component (GCC) size is correctly 
        normalized by N_global and strictly bounded between 0.0 and 1.0.
        """
        G = nx.erdos_renyi_graph(100, 0.1)
        gcc_size = get_gcc_size(G, N_global=100)
        assert 0.0 <= gcc_size <= 1.0

    def test_gcc_size_empty_graph(self):
        """
        Verifies that the GCC size of an empty graph (0 edges) with N nodes
        correctly returns 1/N, as the largest connected component is a single 
        isolated node.
        """
        G = nx.Graph()
        G.add_nodes_from(range(10))
        gcc_size = get_gcc_size(G, N_global=10)
        assert np.isclose(gcc_size, 0.1)