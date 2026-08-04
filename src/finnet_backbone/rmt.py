"""
Random Matrix Theory (RMT) module for financial correlation analysis.

Implements eigenvalue decomposition, Marchenko-Pastur bounds calculation,
and sector mode identification via eigenvector projection.

References:
    - Marchenko-Pastur distribution (Course slides: Spectral analysis)
    - Serrano et al., PNAS 2009 (Disparity Filter)
"""
import numpy as np
import pandas as pd
from typing import Tuple, List


def marchenko_pastur_bounds(T: int, N: int, sigma_sq: float = 1.0) -> Tuple[float, float]:
    """
    Calculate theoretical Marchenko-Pastur bounds (lambda_-, lambda_+) for noise bulk.
    
    The MP distribution describes the eigenvalue spectrum of a random correlation
    matrix with aspect ratio Q = T/N. Eigenvalues within [lambda_-, lambda_+] are
    consistent with pure noise; eigenvalues above lambda_+ represent genuine
    macroeconomic factors (market mode, sector modes).
    
    Formula:
        lambda_pm = sigma^2 * (1 pm sqrt(1/Q))^2
        where Q = T/N >= 1
    
    Args:
        T: Number of time steps (observations)
        N: Number of assets (variables)
        sigma_sq: Variance of the random matrix elements (default: 1.0)
    
    Returns:
        Tuple[float, float]: (lambda_minus, lambda_plus) bounds
    
    Raises:
        ValueError: If T < N (Q < 1 violates MP assumptions)
    
    Example:
        >>> lambda_minus, lambda_plus = marchenko_pastur_bounds(T=500, N=50)
        >>> print(f"Noise bulk: [{lambda_minus:.3f}, {lambda_plus:.3f}]")
    """
    if T < N:
        raise ValueError(f"T ({T}) must be >= N ({N}) for valid MP bounds. Q = T/N < 1")
    
    Q = T / N
    lambda_minus = sigma_sq * (1.0 - 1.0 / np.sqrt(Q))**2
    lambda_plus = sigma_sq * (1.0 + 1.0 / np.sqrt(Q))**2
    
    return lambda_minus, lambda_plus


def mp_null_lambda_plus(Tn: int, Nn: int, n_sims: int = 200) -> np.ndarray:
    """
    Generate null distribution of maximum eigenvalues via Monte Carlo simulation.
    
    Creates an ensemble of random Gaussian correlation matrices to empirically
    estimate the distribution of lambda_max under the null hypothesis of
    uncorrelated variables. This validates the theoretical MP upper bound.
    
    Physics Interpretation:
        For a random matrix with aspect ratio Q = T/N, the eigenvalue spectrum
        follows the Marchenko-Pastur distribution. By simulating many random
        matrices, we build the empirical distribution of lambda_max to test
        whether observed eigenvalues exceed what pure noise would produce.
    
    Args:
        Tn: Number of time steps (T) in the random matrix
        Nn: Number of assets (N) in the random matrix
        n_sims: Number of Monte Carlo simulations (default: 200)
    
    Returns:
        np.ndarray: Array of maximum eigenvalues from each simulation,
                    shape (n_sims,)
    
    Example:
        >>> null_max = mp_null_lambda_plus(Tn=500, Nn=50, n_sims=100)
        >>> empirical_pval = (null_max >= 2.0).sum() / len(null_max)
        >>> print(f"P(lambda_max >= 2.0): {empirical_pval:.3f}")
    """
    null_max = []
    for _ in range(n_sims):
        # Generate Tn x Nn random Gaussian matrix (i.i.d. normal variables)
        # This represents pure noise with no true cross-correlations
        X = np.random.normal(size=(Tn, Nn))
        
        # Compute correlation matrix of the random data
        # rowvar=False ensures each column is treated as a variable (asset)
        C = np.corrcoef(X, rowvar=False)
        
        # Calculate eigenvalues using eigvalsh (optimized for symmetric matrices)
        # eigvalsh is ~2x faster than eig for real symmetric/Hermitian matrices
        ev = np.linalg.eigvalsh(C)
        
        # Store the maximum eigenvalue for this simulation
        # This builds the empirical null distribution of lambda_max
        null_max.append(ev.max())
    
    return np.array(null_max)


def detect_sector_modes(eigvals: np.ndarray, lambda_plus: float) -> List[int]:
    """
    Identify eigenvalues exceeding the Marchenko-Pastur upper bound.
    
    Eigenvalues above lambda_+ represent genuine macroeconomic factors
    (market mode at index 0, sector modes at indices 1, 2, ...) rather
    than random noise. These modes contain systemic information about
    economic sector correlations.
    
    Args:
        eigvals: Sorted eigenvalues in descending order
        lambda_plus: MP theoretical upper bound
    
    Returns:
        List[int]: Indices of sector modes (excluding market mode at index 0)
    
    Example:
        >>> eigvals = np.array([20.0, 5.0, 2.5, 1.1, 0.9])
        >>> sector_idx = detect_sector_modes(eigvals, lambda_plus=1.3)
        >>> print(f"Detected {len(sector_idx)} sector modes at indices {sector_idx}")
        [1, 2]  # Modes 2 and 3 exceed noise bulk
    """
    # Find all indices where eigenvalue exceeds lambda+ (excluding index 0 = market mode)
    return [i for i in range(1, len(eigvals)) if eigvals[i] > lambda_plus]


def calculate_ipr(eigvecs: np.ndarray) -> np.ndarray:
    """
    Calculate Inverse Participation Ratio (IPR) for eigenvector localization.
    
    IPR measures how localized or delocalized an eigenvector is across assets:
        IPR_k = sum_i (u_k,i)^4
    
    - IPR ~ 1/N: Eigenvector is delocalized (uniformly distributed across all assets)
    - IPR ~ 1: Eigenvector is localized (concentrated on few assets)
    
    Localized eigenvectors indicate sector-specific correlations, while
    delocalized eigenvectors represent systemic market-wide factors.
    
    Args:
        eigvecs: Eigenvectors matrix, shape (N, N), columns are eigenvectors
    
    Returns:
        np.ndarray: IPR values for each eigenvector, shape (N,)
    
    Example:
        >>> _, eigvecs = np.linalg.eigh(corr_matrix)
        >>> ipr = calculate_ipr(eigvecs)
        >>> print(f"Market mode IPR: {ipr[0]:.4f} (should be ~1/N for delocalized)")
    """
    # Sum of fourth powers of eigenvector components
    # High IPR = localized (few large components), Low IPR = delocalized
    return np.sum(eigvecs**4, axis=0)


def remove_market_mode(log_returns: pd.DataFrame, market_vec: np.ndarray) -> np.ndarray:
    """
    Project out the dominant market mode to isolate idiosyncratic correlations.
    
    The first eigenvector (market mode) typically absorbs 30-40% of total variance
    and represents systemic market-wide movements. By removing this mode, we
    isolate sector-specific and idiosyncratic correlations that are hidden
    by the dominant market signal.
    
    Projection formula:
        residuals = X - (X · u_1) * u_1^T
        where u_1 is the market mode eigenvector
    
    Args:
        log_returns: Log returns DataFrame, shape (T, N)
        market_vec: Market mode eigenvector (first eigenvector), shape (N,)
    
    Returns:
        np.ndarray: Residual returns with market mode removed, shape (T, N)
    
    Example:
        >>> residuals = remove_market_mode(log_returns, market_vec=eigvecs[:, 0])
        >>> C_res = np.corrcoef(residuals, rowvar=False)
        >>> # Now analyze residual correlation matrix without market contamination
    """
    # Project returns onto market mode: (X · u_1) gives time series of market factor
    market_proj = log_returns.values.dot(market_vec)[:, None] * market_vec[None, :]
    
    # Subtract market projection to get residuals
    residuals = log_returns.values - market_proj
    
    # Standardize residuals to zero mean, unit variance
    residuals = (residuals - residuals.mean(axis=0)) / (residuals.std(axis=0, ddof=1) + 1e-12)
    
    return residuals


def validate_rmt_assumptions(log_returns: pd.DataFrame) -> None:
    """
    Validate RMT assumptions: Gaussianity and independence of returns.
    
    The Marchenko-Pastur distribution assumes:
    1. Returns are i.i.d. Gaussian (zero mean, finite variance)
    2. No temporal autocorrelation
    3. No cross-sectional correlations (under null hypothesis)
    
    Real financial data violates these assumptions (fat tails, volatility clustering),
    so MP bounds are approximate. This function quantifies the deviations.
    
    Args:
        log_returns: Log returns DataFrame, shape (T, N)
    
    Prints:
        - Mean excess kurtosis (Gaussian = 0, fat tails > 0)
        - Number of assets with |kurtosis| > 3 (significant non-Gaussianity)
        - Mean lag-1 autocorrelation (i.i.d. = 0)
        - Warning if deviations exceed thresholds
    
    Example:
        >>> validate_rmt_assumptions(log_returns)
        RMT Assumption Check:
        Mean excess kurtosis: 18.554 (Gaussian = 0)
        Assets with |kurtosis| > 3: 499/499
        Mean lag-1 autocorrelation: -0.0377 (i.i.d. = 0)
        WARNING: Returns deviate from i.i.d. Gaussian assumption. RMT bounds may be approximate.
    """
    # Calculate excess kurtosis for each asset (Gaussian has kurtosis = 3, excess = 0)
    kurtosis_vals = log_returns.kurtosis(axis=0)
    
    print("\nRMT Assumption Check:")
    print(f"Mean excess kurtosis: {kurtosis_vals.mean():.3f} (Gaussian = 0)")
    print(f"Assets with |kurtosis| > 3: {(np.abs(kurtosis_vals) > 3).sum()}/{len(log_returns.columns)}")
    
    # Calculate lag-1 autocorrelation for each asset
    acf_lag1 = log_returns.apply(lambda x: np.corrcoef(x[:-1], x[1:])[0, 1])
    print(f"Mean lag-1 autocorrelation: {acf_lag1.mean():.4f} (i.i.d. = 0)")
    
    # Warn if assumptions are severely violated
    if np.abs(kurtosis_vals.mean()) > 1 or np.abs(acf_lag1.mean()) > 0.05:
        print("WARNING: Returns deviate from i.i.d. Gaussian assumption. RMT bounds may be approximate.")


def identify_sector_modes(eigvals_res_sorted: np.ndarray, 
                          eigvecs_res_sorted: np.ndarray, 
                          sector_indices: List[int], 
                          ticker_sectors: dict, 
                          effective_tickers: List[str]) -> None:
    """
    Identify dominant economic sector for each significant eigenmode.
    
    Projects squared eigenvector components onto sector membership masks to
    determine which economic sectors contribute most strongly to each eigenmode.
    This reveals the sector-specific correlations that emerge after removing
    market-wide noise.
    
    Methodology:
        For each sector mode k:
        1. Square eigenvector components: u_k,i^2 (weight distribution)
        2. Create binary mask for each sector (1 if asset in sector, 0 otherwise)
        3. Calculate projection: sum_i (u_k,i^2 * mask_sector,i)
        4. Sector with highest projection dominates the mode
    
    Args:
        eigvals_res_sorted: Sorted eigenvalues of residual correlation matrix
        eigvecs_res_sorted: Sorted eigenvectors of residual correlation matrix
        sector_indices: Indices of eigenvalues exceeding lambda+
        ticker_sectors: Dict mapping ticker symbol -> GICS sector
        effective_tickers: List of ticker symbols in the analysis
    
    Prints:
        For each of top 5 sector modes:
        - Mode number and eigenvalue
        - Dominant sector name
        - Projection strength (0-1 scale)
    
    Example:
        >>> identify_sector_modes(eigvals_res, eigvecs_res, sector_indices, 
        ...                       ticker_sectors, tickers)
        --- SECTOR MODE IDENTIFICATION ---
        Mode 2 (λ = 31.0038): Dominant Sector: Financials (Projection: 0.291)
        Mode 3 (λ = 17.4587): Dominant Sector: Real Estate (Projection: 0.258)
        Mode 4 (λ = 10.6415): Dominant Sector: Energy (Projection: 0.408)
    """
    print("\n--- SECTOR MODE IDENTIFICATION ---")
    unique_sectors = sorted(list(set(ticker_sectors.values())))
    
    # Analyze top 5 sector modes (excluding market mode at index 0)
    for count, mode_idx in enumerate(sector_indices[:5]):
        eigval = eigvals_res_sorted[mode_idx]
        
        # Square eigenvector components to get weight distribution
        # u_k,i^2 represents how much asset i contributes to mode k
        eigvec_sq = eigvecs_res_sorted[:, mode_idx]**2
        
        # Calculate projection of eigenvector onto each sector
        projections = {}
        for sector in unique_sectors:
            # Binary mask: 1 if asset belongs to sector, 0 otherwise
            mask = np.array([1 if ticker_sectors.get(t) == sector else 0 
                            for t in effective_tickers])
            
            # Dot product gives total weight of eigenvector in this sector
            # High projection = eigenvector concentrated in this sector
            projections[sector] = np.dot(eigvec_sq, mask)
        
        # Sector with highest projection dominates this eigenmode
        best_sector = max(projections, key=projections.get)
        print(f"Mode {count+2} (λ = {eigval:.4f}): Dominant Sector: {best_sector} "
              f"(Projection: {projections[best_sector]:.3f})")