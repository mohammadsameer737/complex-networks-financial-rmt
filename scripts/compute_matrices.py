"""
Compute Pearson correlation and distance matrices from log-returns.
"""
import pandas as pd
import numpy as np
import os

def main():
    if 'snakemake' in globals():
        input_returns = snakemake.input.returns
        output_corr = snakemake.output.correlation
        output_dist = snakemake.output.distance
    else:
        input_returns = "data/log_returns.csv"
        output_corr = "data/correlation_matrix.npy"
        output_dist = "data/distance_matrix.npy"

    os.makedirs(os.path.dirname(output_corr), exist_ok=True)

    log_returns = pd.read_csv(input_returns, index_col=0, parse_dates=True)
    
    # Pearson correlation
    rho = log_returns.corr(method='pearson').values
    np.save(output_corr, rho)
    print(f"Saved correlation matrix to {output_corr}")

    # Distance metric: d = sqrt(2*(1-rho))
    rho_clipped = np.clip(rho, -1.0, 1.0)
    distance = np.sqrt(2 * (1 - rho_clipped))
    np.save(output_dist, distance)
    print(f"Saved distance matrix to {output_dist}")

if __name__ == "__main__":
    main()