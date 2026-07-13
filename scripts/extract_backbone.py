"""
Apply the Disparity Filter to extract the network backbone.
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.finnet_backbone.backbone import disparity_filter_naive
from src.finnet_backbone.visualization import plot_network_backbone

def main():
    if 'snakemake' in globals():
        input_corr = snakemake.input.correlation
        output_backbone = snakemake.output.backbone
        output_viz = snakemake.output.viz
        alpha = snakemake.params.alpha
    else:
        input_corr = "data/correlation_matrix.npy"
        output_backbone = "data/backbone.npy"
        output_viz = "results/Network_Backbone.png"
        alpha = 0.05

    os.makedirs(os.path.dirname(output_backbone), exist_ok=True)
    os.makedirs(os.path.dirname(output_viz), exist_ok=True)

    rho = np.load(input_corr)
    W = np.abs(rho).copy()
    np.fill_diagonal(W, 0.0)

    backbone, pvals = disparity_filter_naive(W, alpha=alpha)
    np.save(output_backbone, backbone)
    print(f"Saved backbone to {output_backbone}")

    # Build graph for visualization
    G = nx.from_numpy_array(backbone)
    G.remove_nodes_from(list(nx.isolates(G)))
    
    print(f"Backbone: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Dummy sector mapping for standalone execution
    ticker_sectors = {i: "Sector" for i in range(G.number_of_nodes())}
    
    plot_network_backbone(G, ticker_sectors)
    plt.savefig(output_viz, dpi=300, bbox_inches='tight')
    print(f"Saved network plot to {output_viz}")

if __name__ == "__main__":
    main()