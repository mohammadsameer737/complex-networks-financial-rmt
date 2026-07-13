import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

def plot_ipr(IPR, N, sector_indices):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, N+1), IPR, 'o-', ms=4, color='teal', label='Empirical IPR')
    ax.axhline(1/N, color='red', linestyle='--', lw=2, label=f'Theoretical Delocalization (1/N = {1/N:.3f})')
    ax.axvline(1, color='purple', linestyle=':', lw=2, label='Market Mode (λ₁)')
    for idx_sec in sector_indices[:5]:
        ax.axvline(idx_sec + 1, color='orange', linestyle=':', alpha=0.5, label='Sector Modes' if idx_sec == sector_indices[0] else "")
    ax.set_xlabel('Eigenmode Index (sorted by λ)', fontsize=12)
    ax.set_ylabel('Inverse Participation Ratio (IPR)', fontsize=12)
    ax.set_title('Eigenvector Localization: Market vs. Sector Modes', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig('IPR_Analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_parameter_sweep(alphas, gcc_vals, mod_vals):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(alphas, gcc_vals, 'g-o', label='GCC Size', linewidth=2)
    ax1.set_xlabel('Significance Threshold ($\\alpha$)', fontsize=12)
    ax1.set_ylabel('GCC Fraction', color='g', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='g')
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(alphas, mod_vals, 'b-s', label='Modularity', linewidth=2)
    ax2.set_ylabel('Modularity ($Q_{mod}$)', color='b', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='b')
    plt.title('Parameter Sweep: Network Sensitivity to $\\alpha$', fontsize=14)
    fig.tight_layout()
    plt.savefig('Parameter_Sweep_Alpha.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_er_phase_transition(p_values, gcc_er, N_er):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p_values, gcc_er, 'b-', linewidth=2)
    ax.axvline(x=1/N_er, color='r', linestyle='--', label=f'Critical p_c = 1/N = {1/N_er:.4f}')
    ax.set_xlabel('Connection Probability p', fontsize=12)
    ax.set_ylabel('GCC Size', fontsize=12)
    ax.set_title('Erdős-Rényi Phase Transition', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig('ER_Phase_Transition.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_jaccard_overlap(overlap_matrix, methods):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(overlap_matrix, annot=True, fmt=".3f", cmap='Blues',
                xticklabels=list(methods.keys()), yticklabels=list(methods.keys()), ax=ax)
    ax.set_title('Jaccard Edge Overlap Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig('Jaccard_Overlap.png', dpi=300)
    plt.show()

def plot_backbone_comparison(mst, G_thresh, G_backbone, corr_threshold):
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Backbone Extraction Comparison: Topology & Metrics", fontsize=16, fontweight='bold')
    seed = 42

    ax1 = fig.add_subplot(2, 2, 1)
    pos_mst = nx.spring_layout(mst, seed=seed)
    nx.draw(mst, pos_mst, ax=ax1, node_size=100, node_color='red', edge_color='red', width=1.5, with_labels=True, font_size=7)
    ax1.set_title("MST (Drastic: No Cycles)", fontsize=12)

    ax2 = fig.add_subplot(2, 2, 2)
    pos_thresh = nx.spring_layout(G_thresh, seed=seed)
    nx.draw(G_thresh, pos_thresh, ax=ax2, node_size=100, node_color='blue', edge_color='blue', width=1.5, with_labels=True, font_size=7)
    ax2.set_title(f"Global Threshold (ρ > {corr_threshold})", fontsize=12)

    ax3 = fig.add_subplot(2, 2, 3)
    pos_disp = nx.spring_layout(G_backbone, seed=seed)
    nx.draw(G_backbone, pos_disp, ax=ax3, node_size=100, node_color='green', edge_color='green', width=1.5, with_labels=True, font_size=7)
    ax3.set_title("Disparity Filter (Local Heterogeneity)", fontsize=12)

    ax4 = fig.add_subplot(2, 2, 4)
    for G, color, label in [(mst, 'red', 'MST'), (G_thresh, 'blue', 'Threshold'), (G_backbone, 'green', 'Disparity')]:
        degrees = [d for n, d in G.degree()]
        if degrees and max(degrees) > 0:
            ax4.hist(degrees, bins=range(max(degrees)+2), density=True, alpha=0.5, color=color, label=label, log=True)
    ax4.set_yscale('log')
    ax4.set_xlabel('Degree (k)')
    ax4.set_ylabel('Probability P(k)')
    ax4.set_title('Degree Distribution (Log-Log Scale)')
    ax4.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('Backbone_Comparison_Composite.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_perturbation_analysis(x_axis_targeted, gcc_targeted, x_axis_random, mean_gcc_random, std_gcc_random):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_axis_targeted, gcc_targeted, 'r-', lw=2.5, label='Targeted Attack (Highest Degree First)')
    ax.plot(x_axis_random, mean_gcc_random, 'b-', lw=2.5, label='Random Failure (Ensemble Mean)')
    ax.fill_between(x_axis_random, mean_gcc_random - std_gcc_random, mean_gcc_random + std_gcc_random,
                    color='blue', alpha=0.2, label='Random Failure (±1 Std Dev Band)')
    ax.set_xlabel('Fraction of Total Market Assets Removed', fontsize=12)
    ax.set_ylabel('Relative Size of Giant Connected Component (GCC / N_global)', fontsize=12)
    ax.set_title('Network Robustness: Targeted Attacks vs. Ensemble Failures', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    plt.savefig('Perturbation_Analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_robustness_comparison(x_td, t_d, x_rd, r_d_mean, r_d_std, x_tm, t_m, x_rm, r_m_mean, r_m_std, x_tt, t_t, x_rt, r_t_mean, r_t_std):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_td, t_d, 'g-', lw=2, label='Disparity (Targeted)')
    ax.plot(x_rd, r_d_mean, 'g--', lw=2, label='Disparity (Random Mean)')
    ax.fill_between(x_rd, r_d_mean - r_d_std, r_d_mean + r_d_std, color='green', alpha=0.1)
    ax.plot(x_tm, t_m, 'r-', lw=2, label='MST (Targeted)')
    ax.plot(x_rm, r_m_mean, 'r--', lw=2, label='MST (Random Mean)')
    ax.fill_between(x_rm, r_m_mean - r_m_std, r_m_mean + r_m_std, color='red', alpha=0.1)
    ax.plot(x_tt, t_t, 'b-', lw=2, label='Threshold (Targeted)')
    ax.plot(x_rt, r_t_mean, 'b--', lw=2, label='Threshold (Random Mean)')
    ax.fill_between(x_rt, r_t_mean - r_t_std, r_t_mean + r_t_std, color='blue', alpha=0.1)
    ax.set_xlabel('Fraction of Market Assets Removed (Global Baseline)', fontsize=12)
    ax.set_ylabel('GCC Size / N_global', fontsize=12)
    ax.set_title('Systemic Robustness Comparison: Disparity vs MST vs Threshold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    plt.savefig('Robustness_Comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_rmt_spectrum(eigvals_res_sorted, lambda_minus, lambda_plus, Q, sigma_sq):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(eigvals_res_sorted[1:], bins=30, density=True, alpha=0.6, color='steelblue', label='Empirical Noise Bulk')
    x_mp = np.linspace(lambda_minus, lambda_plus, 200)
    mp_pdf = (Q / (2 * np.pi * sigma_sq * x_mp)) * np.sqrt((lambda_plus - x_mp) * (x_mp - lambda_minus))
    ax.plot(x_mp, mp_pdf, 'r-', lw=2.5, label='De-contaminated Marchenko-Pastur Theory')
    ax.axvline(lambda_plus, color='red', linestyle='--', lw=2, label=f'Cleaned λ+ = {lambda_plus:.3f}')
    ax.set_xlim(0, lambda_plus + 0.5)
    ax.set_xlabel('Eigenvalue (λ)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('RMT: Market-Cleaned Spectrum vs. Theoretical Bulk', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig('RMT_Spectrum.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_network_backbone(G_backbone, ticker_sectors):
    fig, ax = plt.subplots(figsize=(13, 9))
    unique_sectors = sorted(list(set(ticker_sectors.values())))
    color_palette = sns.color_palette("Set1", len(unique_sectors))
    sector_colors = dict(zip(unique_sectors, color_palette))
    backbone_nodes = list(G_backbone.nodes())
    node_color_list = [sector_colors.get(ticker_sectors.get(node, ''), (0.5,0.5,0.5)) for node in backbone_nodes]
    pos = nx.spring_layout(G_backbone, seed=42, k=0.55)
    nx.draw_networkx_nodes(G_backbone, pos, node_size=280, node_color=node_color_list, ax=ax, edgecolors='black', linewidths=0.5)
    nx.draw_networkx_edges(G_backbone, pos, alpha=0.35, edge_color='dimgray', ax=ax)
    nx.draw_networkx_labels(G_backbone, pos, font_size=8, font_weight='bold', ax=ax)
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=sector, markerfacecolor=color, markersize=10, markeredgecolor='black') for sector, color in sector_colors.items()]
    ax.legend(handles=legend_elements, title="GICS Economic Sectors", loc="upper right", fontsize=10, title_fontsize=11)
    ax.set_title(f"Financial Correlation Backbone (Disparity Filter)\nSector Clustering | Edges: {G_backbone.number_of_edges()}", fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('Network_Backbone.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_temporal_evolution(time_windows, modularities_over_time, largest_comm_sizes_over_time):
    fig, ax1 = plt.subplots(figsize=(12,6))
    ax1.plot(time_windows, modularities_over_time, 'b-o', markersize=4)
    ax1.set_xlabel('Time Window End')
    ax1.set_ylabel('Modularity', color='b')
    ax2 = ax1.twinx()
    ax2.plot(time_windows, largest_comm_sizes_over_time, 'r-s', markersize=4)
    ax2.set_ylabel('Largest Community Fraction', color='r')
    plt.title('Temporal Evolution of Community Structure')
    plt.tight_layout()
    plt.savefig('Temporal_Evolution_Communities.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_temporal_centrality(centralities_over_time):
    all_top_nodes = set()
    for entry in centralities_over_time:
        for date, top_nodes in entry.items():
            for node, val in top_nodes:
                all_top_nodes.add(node)
    all_top_nodes = sorted(list(all_top_nodes))
    window_dates = [list(entry.keys())[0] for entry in centralities_over_time]
    cent_matrix = np.zeros((len(all_top_nodes), len(window_dates)))
    for j, entry in enumerate(centralities_over_time):
        for date, top_nodes in entry.items():
            for node, cent_val in top_nodes:
                if node in all_top_nodes:
                    i_idx = all_top_nodes.index(node)
                    cent_matrix[i_idx, j] = cent_val
    fig, ax = plt.subplots(figsize=(14,8))
    date_labels = [d.strftime('%Y-%m') for d in window_dates]
    sns.heatmap(cent_matrix, xticklabels=date_labels, yticklabels=all_top_nodes, cmap='YlOrRd', ax=ax)
    ax.set_title('Top Central Firms Over Time')
    ax.set_xlabel('Time Window')
    ax.set_ylabel('Firm Ticker')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('Temporal_Centrality_Heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()