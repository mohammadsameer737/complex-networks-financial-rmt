"""
Simulate targeted attacks and random failures to assess network robustness.
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.finnet_backbone.backbone import robustness_curve_ensemble
from src.finnet_backbone.visualization import plot_perturbation_analysis

def main():
    if 'snakemake' in globals():
        input_backbone = snakemake.input.backbone
        output_plot = snakemake.output.plot
        output_metrics = snakemake.output.metrics
        n_sims = snakemake.params.n_sims
    else:
        input_backbone = "data/backbone.npy"
        output_plot = "results/Perturbation_Analysis.png"
        output_metrics = "results/robustness_metrics.json"
        n_sims = 100

    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    os.makedirs(os.path.dirname(output_metrics), exist_ok=True)

    backbone = np.load(input_backbone)
    G = nx.from_numpy_array(backbone)
    G.remove_nodes_from(list(nx.isolates(G)))
    
    N_global = backbone.shape[0]

    print(f"Running {n_sims} robustness simulations...")
    x_td, t_d, x_rd, r_d_mean, r_d_std = robustness_curve_ensemble(G, N_global, num_sims=n_sims)

    metrics = {
        "initial_gcc_size": float(t_d[0]),
        "targeted_decay": [float(x) for x in t_d],
        "random_mean_decay": [float(x) for x in r_d_mean],
        "random_std_decay": [float(x) for x in r_d_std],
        "n_simulations": n_sims
    }
    
    with open(output_metrics, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {output_metrics}")

    plot_perturbation_analysis(x_td, t_d, x_rd, r_d_mean, r_d_std)
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"Saved perturbation plot to {output_plot}")

if __name__ == "__main__":
    main()