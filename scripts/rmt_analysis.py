"""
Random Matrix Theory analysis: eigenvalue spectrum and Marchenko-Pastur bounds.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.finnet_backbone.visualization import plot_rmt_spectrum

def main():
    if 'snakemake' in globals():
        input_corr = snakemake.input.correlation
        output_spectrum = snakemake.output.spectrum
        output_eigvals = snakemake.output.eigenvalues
        q = snakemake.params.q
    else:
        input_corr = "data/correlation_matrix.npy"
        output_spectrum = "results/RMT_Spectrum.png"
        output_eigvals = "data/eigenvalues.npy"
        q = 5.53

    os.makedirs(os.path.dirname(output_spectrum), exist_ok=True)
    os.makedirs(os.path.dirname(output_eigvals), exist_ok=True)

    rho = np.load(input_corr)
    
    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals_sorted = eigvals[::-1]
    
    np.save(output_eigvals, eigvals_sorted)
    print(f"Saved eigenvalues to {output_eigvals}")

    # Marchenko-Pastur bounds (unit variance)
    lambda_minus = (1 - 1/np.sqrt(q))**2
    lambda_plus = (1 + 1/np.sqrt(q))**2

    print(f"MP bounds: lambda- = {lambda_minus:.4f}, lambda+ = {lambda_plus:.4f}")
    print(f"Market mode: lambda_1 = {eigvals_sorted[0]:.4f}")

    # Plot
    plot_rmt_spectrum(eigvals_sorted, lambda_minus, lambda_plus, q, sigma_sq=1.0)
    plt.savefig(output_spectrum, dpi=300, bbox_inches='tight')
    print(f"Saved spectrum plot to {output_spectrum}")

if __name__ == "__main__":
    main()