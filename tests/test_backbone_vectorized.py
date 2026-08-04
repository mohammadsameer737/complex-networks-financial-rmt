import numpy as np
import pytest
from finnet_backbone.backbone_vectorized import disparity_filter_vectorized, bootstrap_parallel

class TestVectorizedBackbone:
    """Tests for the vectorized disparity filter and bootstrap implementations."""

    def test_disparity_filter_vectorized_symmetry(self):
        """
        Verifies that the vectorized disparity filter produces a symmetric 
        backbone matrix and valid p-values bounded in [0, 1].
        """
        np.random.seed(42)
        W = np.random.rand(10, 10)
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0)
        
        backbone, pvals = disparity_filter_vectorized(W, alpha=0.05)
        
        assert backbone.shape == W.shape
        assert np.allclose(backbone, backbone.T), "Backbone must be symmetric"
        assert np.all((pvals >= 0) & (pvals <= 1)), "P-values must be in [0, 1]"

    def test_disparity_filter_vectorized_isolated_nodes(self):
        """
        Verifies that the vectorized filter handles isolated nodes (zero strength)
        without raising division-by-zero errors. 
        Since p_matrix is initialized to zeros, isolated nodes retain p-value 0.0.
        """
        W = np.zeros((5, 5))
        W[0, 1] = W[1, 0] = 0.5  # Only one edge
        
        backbone, pvals = disparity_filter_vectorized(W, alpha=0.05)
        
        # Isolated nodes (indices 2, 3, 4) should have p-value 0.0 (initialized value)
        assert np.all(pvals[2:, 2:] == 0.0)
        assert backbone.shape == W.shape

    def test_bootstrap_parallel_execution(self):
        """
        Verifies that the parallel bootstrap resampling executes without 
        crashing and returns a survival probability matrix of the correct shape.
        """
        np.random.seed(42)
        W = np.random.rand(10, 10)
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0)
        
        # Use small n_reps to keep test fast
        survival = bootstrap_parallel(W, alpha=0.05, n_reps=2, block_size=2)
        
        assert survival.shape == W.shape
        assert np.all((survival >= 0) & (survival <= 1)), "Survival probs must be in [0, 1]"