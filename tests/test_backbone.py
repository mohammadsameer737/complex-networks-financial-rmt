import numpy as np
from src.finnet_backbone.backbone import disparity_filter_naive, get_gcc_size
import networkx as nx

def test_disparity_filter_basic():
    """Test disparity filter returns valid backbone."""
    W = np.random.rand(20, 20)
    W = (W + W.T) / 2  # Symmetric
    np.fill_diagonal(W, 0)
    backbone, pvals = disparity_filter_naive(W, alpha=0.05)
    assert backbone.shape == W.shape
    assert np.all(backbone >= 0)
    assert np.all(pvals >= 0) and np.all(pvals <= 1)

def test_gcc_size_calculation():
    """Test GCC size calculation."""
    G = nx.erdos_renyi_graph(100, 0.1)
    gcc_size = get_gcc_size(G, N_global=100)
    assert 0 <= gcc_size <= 1.0