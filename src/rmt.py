import numpy as np

def mp_null_lambda_plus(Tn, Nn, n_sims=200):
    null_max = []
    for _ in range(n_sims):
        X = np.random.normal(size=(Tn, Nn))
        C = np.corrcoef(X, rowvar=False)
        ev = np.linalg.eigvalsh(C)
        null_max.append(ev.max())
    return np.array(null_max)

def identify_sector_modes(eigvals_res_sorted, eigvecs_res_sorted, sector_indices, ticker_sectors, effective_tickers):
    print("\n--- SECTOR MODE IDENTIFICATION ---")
    unique_sectors = sorted(list(set(ticker_sectors.values())))
    for count, mode_idx in enumerate(sector_indices[:5]):
        eigval = eigvals_res_sorted[mode_idx]
        eigvec_sq = eigvecs_res_sorted[:, mode_idx]**2
        projections = {}
        for sector in unique_sectors:
            mask = np.array([1 if ticker_sectors.get(t) == sector else 0 for t in effective_tickers])
            projections[sector] = np.dot(eigvec_sq, mask)
        best_sector = max(projections, key=projections.get)
        print(f"Mode {count+2} (λ = {eigval:.4f}): Dominant Sector: {best_sector} (Projection: {projections[best_sector]:.3f})")