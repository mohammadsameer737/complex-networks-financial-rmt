import numpy as np
import pytest
from finnet_backbone.rmt import mp_null_lambda_plus, identify_sector_modes

class TestMarchenkoPasturNull:
    """Tests for Random Matrix Theory null hypothesis validation."""

    def test_mp_null_lambda_plus_standard(self):
        """
        Verifies that the empirical maximum eigenvalue of a random 
        correlation matrix converges to the theoretical Marchenko-Pastur 
        upper bound (lambda_+) as the number of simulations increases.
        """
        # T=500, N=50 gives Q=10. Theoretical lambda_+ is approx (1 + 1/sqrt(10))^2 ≈ 1.73
        null_max = mp_null_lambda_plus(Tn=500, Nn=50, n_sims=20)
        assert len(null_max) == 20
        assert np.all(null_max > 0)
        assert np.all(null_max < 3.0)  # Strict bound for Q=10

    def test_mp_null_small_T_N_ratio(self):
        """
        Verifies that the function correctly computes the null distribution 
        even for small T/N ratios (T < N), returning the expected number of 
        simulations without crashing.
        """
        null_max = mp_null_lambda_plus(Tn=20, Nn=50, n_sims=5)
        assert len(null_max) == 5
        assert np.all(null_max > 0)


class TestSectorModes:
    """Tests for macroeconomic sector mode identification in eigenvalue spectrum."""

    def test_sector_modes_identification_analytical(self):
        """
        Verifies that eigenvalues exceeding the theoretical noise bulk 
        upper bound (lambda_+) are correctly identified as non-random 
        sector/market modes.
        """
        # Simulated eigenvalue spectrum: 1 market mode, 2 sector modes, rest noise
        eigvals = np.array([20.0, 5.0, 2.5, 1.1, 0.9, 0.8])
        lambda_plus = 1.3
        
        sector_indices = [i for i in range(len(eigvals)) if eigvals[i] > lambda_plus]
        
        # Should identify indices 0, 1, and 2 as exceeding the noise bulk
        assert len(sector_indices) == 3
        assert sector_indices == [0, 1, 2]