import numpy as np

def test_log_return_calculation():
    """Test log-return calculation."""
    prices = np.array([100, 101, 102, 103])
    log_returns = np.log(prices[1:] / prices[:-1])
    assert len(log_returns) == 3
    assert all(log_returns > 0)  # All positive for increasing prices

def test_correlation_to_distance():
    """Test distance metric d = sqrt(2*(1-rho))."""
    rho = 1.0  # Perfect correlation
    distance = np.sqrt(2 * (1 - rho))
    assert distance == 0.0
    
    rho = -1.0  # Perfect anti-correlation
    distance = np.sqrt(2 * (1 - rho))
    assert abs(distance - 2.0) < 1e-10