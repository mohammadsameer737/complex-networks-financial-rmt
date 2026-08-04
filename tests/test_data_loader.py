import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from finnet_backbone.data_loader import load_market_data

class TestDataLoader:
    """Tests for financial data ingestion and preprocessing robustness."""

    @patch('finnet_backbone.data_loader.urllib.request.urlopen')
    @patch('finnet_backbone.data_loader.yf.download')
    @patch('finnet_backbone.data_loader.os.path.exists', return_value=False)
    @patch('finnet_backbone.data_loader.pd.DataFrame.to_csv')
    def test_load_market_data_fallback_mechanism(self, mock_to_csv, mock_exists, mock_download, mock_urlopen):
        """
        Verifies that load_market_data gracefully falls back to a hardcoded 
        representative dataset when the Wikipedia scrape fails.
        """
        # Force Wikipedia fetch to fail
        mock_urlopen.side_effect = Exception("Network failure")
        
        # Mock yfinance to return a valid MultiIndex DataFrame (mimicking real yfinance output)
        mock_dates = pd.date_range("2023-01-01", periods=10, freq='D')
        mock_data = pd.DataFrame(
            np.random.rand(10, 2),
            index=mock_dates,
            columns=pd.MultiIndex.from_product([['Adj Close'], ['AAPL', 'MSFT']])
        )
        mock_download.return_value = mock_data
        
        data, tickers, sectors, n_global, mapping = load_market_data()
        
        # Verify the fallback dataset structure is correctly returned
        assert n_global == 2
        assert "AAPL" in tickers
        assert sectors["AAPL"] == "Technology"
        assert data.shape == (10, 2)

    def test_log_return_calculation_properties(self):
        """
        Verifies the mathematical properties of log-return calculation.
        Log returns of monotonically increasing prices must be strictly positive.
        """
        prices = np.array([100.0, 101.0, 102.0, 103.0])
        log_returns = np.log(prices[1:] / prices[:-1])
        
        assert len(log_returns) == 3
        assert np.all(log_returns > 0)

    def test_correlation_to_distance_metric_bounds(self):
        """
        Verifies the metric distance formula d = sqrt(2 * (1 - rho)).
        - Perfect correlation (rho=1) must yield distance 0.0.
        - Perfect anti-correlation (rho=-1) must yield distance 2.0.
        """
        rho_perfect = 1.0
        distance_perfect = np.sqrt(2 * (1 - rho_perfect))
        assert np.isclose(distance_perfect, 0.0)
        
        rho_anti = -1.0
        distance_anti = np.sqrt(2 * (1 - rho_anti))
        assert np.isclose(distance_anti, 2.0)