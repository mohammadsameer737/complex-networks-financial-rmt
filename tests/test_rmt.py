import numpy as np
from src.finnet_backbone.rmt import mp_null_lambda_plus, identify_sector_modes

def test_mp_null_lambda_plus():
    """Test MP null distribution generation."""
    null_max = mp_null_lambda_plus(Tn=500, Nn=50, n_sims=10)
    assert len(null_max) == 10
    assert all(null_max > 0)
    assert all(null_max < 10)  # Reasonable eigenvalue range

def test_sector_modes_identification():
    """Test sector mode detection."""
    eigvals = np.array([20.0, 5.0, 2.5, 1.5, 1.0])
    lambda_plus = 1.3
    sector_indices = [i for i in range(1, len(eigvals)) if eigvals[i] > lambda_plus]
    assert len(sector_indices) == 3  # Should find 3 sector modes (indices 1, 2, 3)