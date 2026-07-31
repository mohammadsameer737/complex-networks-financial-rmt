import numpy as np

def mp_null_lambda_plus(Tn, Nn, n_sims=200):
    """
    Generate the null distribution of the maximum eigenvalue (lambda+) 
    using random Gaussian matrices (Marchenko-Pastur law).
    """
    null_max = []
    for _ in range(n_sims):
        # Random Gaussian matrix (Tn x Nn) representing pure noise
        X = np.random.normal(size=(Tn, Nn))
        
        # Correlation matrix (rowvar=False because columns are the assets)
        C = np.corrcoef(X, rowvar=False)
        
        # Get eigenvalues
        ev = np.linalg.eigvalsh(C)
        
        # Store the max eigenvalue for this simulation to build the null distribution
        null_max.append(ev.max())
        
    return np.array(null_max)


def identify_sector_modes(eigvals_res_sorted, eigvecs_res_sorted, sector_indices, ticker_sectors, effective_tickers):
    """
    Identify the dominant economic sector for each significant eigenmode.
    We do this by projecting the squared eigenvector components onto sector masks.
    """
    print("\n--- SECTOR MODE IDENTIFICATION ---")
    unique_sectors = sorted(list(set(ticker_sectors.values())))
    
    # Check the top 5 sector modes (excluding the market mode at index 0)
    for count, mode_idx in enumerate(sector_indices[:5]):
        eigval = eigvals_res_sorted[mode_idx]
        
        # Square eigenvector components to get the weight distribution
        eigvec_sq = eigvecs_res_sorted[:, mode_idx]**2
        
        # Calculate projection of the eigenvector onto each sector
        projections = {}
        for sector in unique_sectors:
            # Binary mask: 1 if asset belongs to sector, 0 otherwise
            mask = np.array([1 if ticker_sectors.get(t) == sector else 0 for t in effective_tickers])
            
            # Dot product gives the total weight of the eigenvector in this sector
            projections[sector] = np.dot(eigvec_sq, mask)
            
        # The sector with the highest projection is the dominant mode
        best_sector = max(projections, key=projections.get)
        print(f"Mode {count+2} (λ = {eigval:.4f}): Dominant Sector: {best_sector} (Projection: {projections[best_sector]:.3f})")