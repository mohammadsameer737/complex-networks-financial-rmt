import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
from scipy.linalg import eigh
from scipy import integrate

# Import custom modules
from src.data_loader import load_market_data
from src.rmt import mp_null_lambda_plus, identify_sector_modes
from src.backbone import (disparity_filter_naive, disparity_null_fp_rate, bootstrap_backbone_prices,
                          get_gcc_size, jaccard, safe_modularity, safe_algebraic_connectivity,
                          track_gcc_decay, robustness_curve_ensemble)
from src.visualization import (plot_ipr, plot_parameter_sweep, plot_er_phase_transition,
                               plot_jaccard_overlap, plot_backbone_comparison,
                               plot_perturbation_analysis, plot_robustness_comparison,
                               plot_rmt_spectrum, plot_network_backbone,
                               plot_temporal_evolution, plot_temporal_centrality)

np.random.seed(42)
random.seed(42)

# 1. Data Pipeline
data, effective_tickers, ticker_sectors, N_global, mapping = load_market_data()

# 2. Log-returns and correlation
log_returns = np.log(data / data.shift(1)).dropna()
T, N = log_returns.shape
Q = T / N
print(f"Time Steps (T): {T} | Aspect Ratio (Q = T/N): {Q:.2f}")

rho_matrix = log_returns.corr(method="pearson")
rho_clipped = np.clip(rho_matrix.values, -1.0, 1.0)
distance_matrix = np.sqrt(2 * (1 - rho_clipped))

# 3. RMT and Market Mode cleaning
eigvals, eigvecs = np.linalg.eigh(rho_matrix.values)
idx = np.argsort(eigvals)[::-1]
eigvals_sorted = eigvals[idx]
eigvecs_sorted = eigvecs[:, idx]

lambda_1 = eigvals_sorted[0]
market_vec = eigvecs_sorted[:, 0]
market_proj = log_returns.values.dot(market_vec)[:, None] * market_vec[None, :]
residuals = log_returns.values - market_proj
residuals = (residuals - residuals.mean(axis=0)) / (residuals.std(axis=0, ddof=1) + 1e-12)
C_res = np.corrcoef(residuals, rowvar=False)
eigvals_res, eigvecs_res = np.linalg.eigh(C_res)
idx_res = np.argsort(eigvals_res)[::-1]
eigvals_res_sorted = eigvals_res[idx_res]
eigvecs_res_sorted = eigvecs_res[:, idx_res]

T_eff = residuals.shape[0]
N_eff = residuals.shape[1]
Q = T_eff / N_eff if N_eff > 0 else 1.0
sigma_sq = 1.0
lambda_minus = sigma_sq * (1 - 1.0 / np.sqrt(Q))**2 if Q > 1 else 0.0
lambda_plus = sigma_sq * (1 + 1.0 / np.sqrt(Q))**2 if Q > 1 else np.max(eigvals_res_sorted) * 1.1

null_max = mp_null_lambda_plus(T_eff, N_eff, n_sims=200)
emp_pval_lambda_plus = (null_max >= lambda_plus).sum() / null_max.size
sector_indices = [i for i in range(1, len(eigvals_res_sorted)) if eigvals_res_sorted[i] > lambda_plus]

print(f"\nRMT Cleaned Bounds: λ- = {lambda_minus:.4f}, λ+ = {lambda_plus:.4f}")
print(f"Market Mode: λ1 = {lambda_1:.4f} (Absorbs {(lambda_1/N)*100:.2f}% of Total Variance)")
print(f"Significant Sector Modes Detected: {len(sector_indices)}")
print(f"MP null test: empirical p-value for λ+ = {emp_pval_lambda_plus:.3f}")

kurtosis_vals = log_returns.kurtosis(axis=0)
print(f"\nRMT Assumption Check:")
print(f"Mean excess kurtosis: {kurtosis_vals.mean():.3f} (Gaussian = 0)")
print(f"Assets with |kurtosis| > 3: {(np.abs(kurtosis_vals) > 3).sum()}/{N}")

acf_lag1 = log_returns.apply(lambda x: np.corrcoef(x[:-1], x[1:])[0,1])
print(f"Mean lag-1 autocorrelation: {acf_lag1.mean():.4f} (i.i.d. = 0)")

if np.abs(kurtosis_vals.mean()) > 1 or np.abs(acf_lag1.mean()) > 0.05:
    print("WARNING: Returns deviate from i.i.d. Gaussian assumption. RMT bounds may be approximate.")

IPR = np.sum(eigvecs_sorted**4, axis=0)
plot_ipr(IPR, N, sector_indices)

identify_sector_modes(eigvals_res_sorted, eigvecs_res_sorted, sector_indices, ticker_sectors, effective_tickers)

# 4. Disparity Filter
print("\n--- SENSITIVITY ANALYSIS ---")
W_pure = np.abs(rho_matrix.values)
np.fill_diagonal(W_pure, 0.0)
alphas = np.linspace(0.01, 0.20, 25)
gcc_vals, mod_vals = [], []
for a in alphas:
    backbone_temp, pvals_temp = disparity_filter_naive(W_pure, alpha=a)
    G_temp = nx.from_numpy_array(backbone_temp)
    G_temp.remove_nodes_from(list(nx.isolates(G_temp)))
    if G_temp.number_of_nodes() > 0:
        gcc_vals.append(len(max(nx.connected_components(G_temp), key=len)) / N_global)
        communities = list(nx.algorithms.community.greedy_modularity_communities(G_temp))
        mod_vals.append(nx.algorithms.community.modularity(G_temp, communities))
    else:
        gcc_vals.append(0.0)
        mod_vals.append(0.0)

plot_parameter_sweep(alphas, gcc_vals, mod_vals)

backbone_np, pvals_matrix = disparity_filter_naive(W_pure, alpha=0.05)
G_backbone = nx.from_numpy_array(backbone_np)
G_backbone = nx.relabel_nodes(G_backbone, mapping)
G_backbone.remove_nodes_from(list(nx.isolates(G_backbone)))

fp_mean, fp_std = disparity_null_fp_rate(log_returns, n_sims=50, alpha=0.05)
print(f"Disparity null edges (mean ± std) at alpha=0.05: {fp_mean:.1f} ± {fp_std:.1f}")

try:
    from statsmodels.stats.multitest import multipletests
    iu = np.triu_indices_from(pvals_matrix, k=1)
    pvals_raw = pvals_matrix[iu]
    valid_mask = np.isfinite(pvals_raw) & (pvals_raw >= 0) & (pvals_raw <= 1)
    pvals_valid = pvals_raw[valid_mask]
    if pvals_valid.size > 0:
        rej, pvals_adj_valid, _, _ = multipletests(pvals_valid, alpha=0.05, method='fdr_bh')
        p_adj_full = np.ones_like(pvals_raw)
        p_adj_full[valid_mask] = pvals_adj_valid
        p_adj_matrix = np.zeros_like(pvals_matrix)
        p_adj_matrix[iu] = p_adj_full
        p_adj_matrix = p_adj_matrix + p_adj_matrix.T
        np.fill_diagonal(p_adj_matrix, 1.0)
        backbone_fdr = np.zeros_like(W_pure)
        mask_keep = p_adj_matrix < 0.05
        backbone_fdr[mask_keep] = W_pure[mask_keep]
        G_backbone_fdr = nx.from_numpy_array(backbone_fdr)
        G_backbone_fdr = nx.relabel_nodes(G_backbone_fdr, mapping)
        G_backbone_fdr.remove_nodes_from(list(nx.isolates(G_backbone_fdr)))
        print(f"Edges after FDR correction (q<0.05): {G_backbone_fdr.number_of_edges()} | Nodes: {G_backbone_fdr.number_of_nodes()}")
        if G_backbone.number_of_edges() > 0:
            fdr_edges = set(G_backbone_fdr.edges())
            disp_edges = set(G_backbone.edges())
            overlap = len(fdr_edges & disp_edges)
            print(f"Edge stability: {overlap}/{G_backbone.number_of_edges()} ({100*overlap/G_backbone.number_of_edges():.1f}%) of disparity edges survive FDR correction")
    else:
        print("No valid p-values for FDR correction.")
except ModuleNotFoundError:
    print("statsmodels not installed; skipping FDR correction.")
except Exception as e:
    print(f"FDR correction failed: {e}")

G_temp = nx.from_numpy_array(backbone_np)
if G_temp.number_of_edges() < 15:
    print("\n[WARNING] Pure filter yielded too few edges. Reverting to optimal sweep parameters.")
    W_pure_temp = W_pure.copy()
    W_pure_temp[W_pure_temp < 0.2] = 0.0
    backbone_np, pvals_matrix = disparity_filter_naive(W_pure_temp, alpha=0.2)
    G_backbone = nx.from_numpy_array(backbone_np)
    G_backbone = nx.relabel_nodes(G_backbone, mapping)
    G_backbone.remove_nodes_from(list(nx.isolates(G_backbone)))

print(f"\nFinal Disparity Filter Backbone: {G_backbone.number_of_edges()} edges retained.")
print(f"Nodes in Backbone: {G_backbone.number_of_nodes()}")
print(f"Network Density: {nx.density(G_backbone):.4f}")

# 4.6 Baseline graphs
G_full_dist = nx.from_numpy_array(distance_matrix)
G_full_dist = nx.relabel_nodes(G_full_dist, mapping)
mst = nx.minimum_spanning_tree(G_full_dist, weight='weight')

corr_threshold = 0.45
thresh_matrix = rho_matrix.where(rho_matrix.abs() > corr_threshold, 0.0)
thresh_array = np.array(thresh_matrix.values, dtype=float, copy=True)
np.fill_diagonal(thresh_array, 0.0)
G_thresh = nx.from_numpy_array(np.abs(thresh_array))
G_thresh = nx.relabel_nodes(G_thresh, mapping)
G_thresh.remove_nodes_from(list(nx.isolates(G_thresh)))

print(f"MST Edges: {mst.number_of_edges()} | Global Threshold Edges: {G_thresh.number_of_edges()}")

block_sizes = [1, 5, 10]
bootstrap_summary = {}
for bs in block_sizes:
    survival_bs = bootstrap_backbone_prices(data, mapping, n_reps=200, block_size=bs, alpha=0.05)
    stable50 = [e for e, f in survival_bs.items() if f > 0.5]
    stable75 = [e for e, f in survival_bs.items() if f > 0.75]
    bootstrap_summary[bs] = (len(stable50), len(stable75))
    print(f"Bootstrap block_size={bs}: edges >50%: {len(stable50)} | >75%: {len(stable75)}")
survival = survival_bs
stable_edges_50 = [e for e, f in survival.items() if f > 0.5]
stable_edges_75 = [e for e, f in survival.items() if f > 0.75]

print("\n--- ERDŐS-RÉNYI PHASE TRANSITION ---")
N_er = 500
p_values = np.linspace(0.001, 0.01, 50)
gcc_er = []
for p in p_values:
    G_er = nx.erdos_renyi_graph(N_er, p)
    if G_er.number_of_edges() > 0:
        gcc_size = len(max(nx.connected_components(G_er), key=len)) / N_er
    else:
        gcc_size = 0.0
    gcc_er.append(gcc_size)

plot_er_phase_transition(p_values, gcc_er, N_er)

# 5. Robustness and GCC analysis
initial_gcc = get_gcc_size(G_backbone, N_global)
print(f"Initial Backbone GCC Size: {initial_gcc * 100:.2f}% of the global network.")

edges_mst = set(mst.edges())
edges_thresh = set(G_thresh.edges())
edges_disp = set(G_backbone.edges())

methods = {'MST': edges_mst, 'Threshold': edges_thresh, 'Disparity': edges_disp}
overlap_matrix = np.zeros((3, 3))
for i, (name1, set1) in enumerate(methods.items()):
    for j, (name2, set2) in enumerate(methods.items()):
        overlap_matrix[i, j] = jaccard(set1, set2)

print("\nJACCARD EDGE OVERLAP MATRIX:")
print(pd.DataFrame(overlap_matrix, index=methods.keys(), columns=methods.keys()).round(3))
plot_jaccard_overlap(overlap_matrix, methods)

plot_backbone_comparison(mst, G_thresh, G_backbone, corr_threshold)

mod_score = safe_modularity(G_backbone)
if mod_score is None:
    mod_score = 0.0
else:
    print(f"\nBackbone Modularity (Q): {mod_score:.4f}")

try:
    mst_mod = nx.algorithms.community.modularity(mst, list(nx.algorithms.community.greedy_modularity_communities(mst)))
except Exception:
    mst_mod = 0.0
try:
    thresh_mod = nx.algorithms.community.modularity(G_thresh, list(nx.algorithms.community.greedy_modularity_communities(G_thresh)))
except Exception:
    thresh_mod = 0.0
print(f"MST Modularity: {mst_mod:.4f}")
print(f"Threshold Modularity: {thresh_mod:.4f}")

null_mods = []
for _ in range(100):
    degs = [d for n, d in G_backbone.degree()]
    if not degs:
        continue
    G_null = nx.configuration_model(degs, create_using=nx.Graph)
    G_null.remove_edges_from(nx.selfloop_edges(G_null))
    if G_null.number_of_edges() > 0:
        comm_null = list(nx.algorithms.community.greedy_modularity_communities(G_null))
        null_mod = nx.algorithms.community.modularity(G_null, comm_null)
        null_mods.append(null_mod)

if null_mods:
    null_mean = np.mean(null_mods)
    null_std = np.std(null_mods)
    z_score = (mod_score - null_mean) / null_std if null_std > 0 else 0
    print(f"Null model modularity: {null_mean:.4f} +/- {null_std:.4f}")
    print(f"Modularity Z-score: {z_score:.2f}")

mod_boot = []
for _ in range(100):
    idx = np.random.choice(T, size=T, replace=True)
    lr_s = log_returns.iloc[idx]
    rho_s = np.array(lr_s.corr().values, copy=True)
    np.fill_diagonal(rho_s, 0.0)
    W_s = np.abs(rho_s)
    bb_s, _ = disparity_filter_naive(W_s, alpha=0.05)
    temp_G = nx.from_numpy_array(bb_s)
    if temp_G.number_of_edges() < 15:
        W_s_temp = W_s.copy()
        W_s_temp[W_s_temp < 0.2] = 0.0
        bb_s, _ = disparity_filter_naive(W_s_temp, alpha=0.2)
    G_s = nx.from_numpy_array(bb_s)
    G_s.remove_nodes_from(list(nx.isolates(G_s)))
    if G_s.number_of_edges() > 0 and G_s.number_of_nodes() > 1:
        try:
            c_s = list(nx.algorithms.community.greedy_modularity_communities(G_s))
            mod_val = nx.algorithms.community.modularity(G_s, c_s)
            mod_boot.append(mod_val)
        except:
            pass

if mod_boot:
    ci = np.percentile(mod_boot, [2.5, 97.5])
    print(f"Modularity 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

if G_backbone.number_of_nodes() > 0:
    L_full = nx.laplacian_matrix(G_backbone).toarray()
    eig_L_full = np.sort(eigh(L_full, eigvals_only=True))
    zero_mult = int(np.sum(np.isclose(eig_L_full, 0.0)))
    gcc_nodes = max(nx.connected_components(G_backbone), key=len)
    G_gcc = G_backbone.subgraph(gcc_nodes)
    if G_gcc.number_of_nodes() > 1:
        L_gcc = nx.laplacian_matrix(G_gcc).toarray()
        eig_L_gcc = np.sort(eigh(L_gcc, eigvals_only=True))
        fiedler_gcc = float(eig_L_gcc[1])
    else:
        fiedler_gcc = 0.0
else:
    zero_mult = 0
    fiedler_gcc = 0.0

fiedler_mst = safe_algebraic_connectivity(mst)
fiedler_thresh = safe_algebraic_connectivity(G_thresh)
print(f"Backbone Laplacian zero eigenvalue multiplicity: {zero_mult}")
print(f"Backbone Algebraic Connectivity (GCC local Fiedler λ2): {fiedler_gcc:.4f}")
print(f"Comparison Fiedler Values -> MST: {fiedler_mst:.4f} | Threshold: {fiedler_thresh:.4f} | Disparity (GCC): {fiedler_gcc:.4f}")

degrees = dict(G_backbone.degree())
targeted_order = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
gcc_targeted = track_gcc_decay(G_backbone, targeted_order, N_global)

random.seed(42)
num_simulations = 100
nodes_list = list(G_backbone.nodes())
ensemble_random_curves = []
for _ in range(num_simulations):
    shuffled_nodes = nodes_list.copy()
    random.shuffle(shuffled_nodes)
    ensemble_random_curves.append(track_gcc_decay(G_backbone, shuffled_nodes, N_global))

ensemble_random_curves = np.array(ensemble_random_curves)
mean_gcc_random = np.mean(ensemble_random_curves, axis=0)
std_gcc_random = np.std(ensemble_random_curves, axis=0)

x_axis_targeted = np.linspace(0, len(targeted_order) / N_global, len(gcc_targeted))
x_axis_random = np.linspace(0, len(nodes_list) / N_global, len(mean_gcc_random))

auc_targeted = integrate.trapezoid(gcc_targeted, x_axis_targeted)
auc_random = integrate.trapezoid(mean_gcc_random, x_axis_random)

initial_gcc_nodes = int(len(max(nx.connected_components(G_backbone), key=len))) if G_backbone.number_of_nodes() > 0 else 0
print(f"\nInitial Backbone GCC Size: {initial_gcc_nodes} nodes ({initial_gcc*100:.2f}% of global network).")

thresholds = [0.5, 0.25, 0.75]
for thr in thresholds:
    t_idx = next((i for i, x in enumerate(gcc_targeted) if x < thr * initial_gcc), None)
    r_idx = next((i for i, x in enumerate(mean_gcc_random) if x < thr * initial_gcc), None)
    t_nodes = t_idx if t_idx is not None else None
    r_nodes = r_idx if r_idx is not None else None
    t_frac = (t_nodes / N_global) if t_nodes is not None else 1.0
    r_frac = (r_nodes / N_global) if r_nodes is not None else 1.0
    print(f"Fragility at {int(thr*100)}% of initial GCC: targeted -> {t_nodes} nodes removed ({t_frac*100:.2f}%), random -> {r_nodes} nodes removed ({r_frac*100:.2f}%)")

fc_targeted = next((i / N_global for i, x in enumerate(gcc_targeted) if x < 0.5 * initial_gcc), 1.0)
fc_random = next((i / N_global for i, x in enumerate(mean_gcc_random) if x < 0.5 * initial_gcc), 1.0)
print(f"\n--- FRAGILITY THRESHOLD (summary) ---")
print(f"Initial GCC Size: {initial_gcc*100:.2f}% of global network.")
print(f"Targeted Attack Fragility Threshold (50% initial GCC loss): {fc_targeted*100:.2f}% of nodes removed.")
print(f"Random Failure Fragility Threshold (50% initial GCC loss): {fc_random*100:.2f}% of nodes removed.")

plot_perturbation_analysis(x_axis_targeted, gcc_targeted, x_axis_random, mean_gcc_random, std_gcc_random)

x_td, t_d, x_rd, r_d_mean, r_d_std = robustness_curve_ensemble(G_backbone, N_global)
x_tm, t_m, x_rm, r_m_mean, r_m_std = robustness_curve_ensemble(mst, N_global)
x_tt, t_t, x_rt, r_t_mean, r_t_std = robustness_curve_ensemble(G_thresh, N_global)

plot_robustness_comparison(x_td, t_d, x_rd, r_d_mean, r_d_std, x_tm, t_m, x_rm, r_m_mean, r_m_std, x_tt, t_t, x_rt, r_t_mean, r_t_std)

# 6. Final visualizations
plot_rmt_spectrum(eigvals_res_sorted, lambda_minus, lambda_plus, Q, sigma_sq)
plot_network_backbone(G_backbone, ticker_sectors)

# 7. Temporal Network Analysis
print("\n--- STARTING TEMPORAL NETWORK ANALYSIS ---")
window_size = 252 
step_size = 21 
time_windows = []
centralities_over_time = []
modularities_over_time = []
largest_comm_sizes_over_time = []
gcc_sizes_over_time = []

T_total, N_total = log_returns.shape
num_windows = max(0, (T_total - window_size) // step_size + 1)

for i in range(num_windows):
    start_idx = i * step_size
    end_idx = start_idx + window_size
    if end_idx > T_total:
        break
    window_returns = log_returns.iloc[start_idx:end_idx]
    window_date = window_returns.index[-1]
    time_windows.append(window_date)
    rho_window = window_returns.corr(method="pearson")
    rho_window_clipped = np.clip(rho_window.values, -1.0, 1.0)
    corr_threshold_temporal = 0.5 
    abs_rho = np.abs(rho_window_clipped)
    backbone_window = np.zeros_like(abs_rho)
    mask = abs_rho > corr_threshold_temporal
    backbone_window[mask] = abs_rho[mask]
    np.fill_diagonal(backbone_window, 0.0)
    
    G_window = nx.from_numpy_array(backbone_window)
    G_window = nx.relabel_nodes(G_window, mapping)
    G_window.remove_nodes_from(list(nx.isolates(G_window)))
    
    if i % 10 == 0 or i == num_windows - 1:
        print(f"Processed window {i+1}/{num_windows} | Date: {window_date.strftime('%Y-%m-%d')} | Edges retained: {G_window.number_of_edges()}")
    if G_window.number_of_nodes() > 0:
        gcc_nodes = max(nx.connected_components(G_window), key=len)
        gcc_fraction = len(gcc_nodes) / G_window.number_of_nodes()
        gcc_sizes_over_time.append(gcc_fraction)
    else:
        gcc_sizes_over_time.append(0.0)
    if G_window.number_of_nodes() > 1:
        try:
            gcc_nodes = max(nx.connected_components(G_window), key=len)
            G_gcc = G_window.subgraph(gcc_nodes).copy()
            if len(gcc_nodes) >= 10:
                cent = nx.eigenvector_centrality_numpy(G_gcc, weight='weight', max_iter=1000)
            else:
                cent = nx.degree_centrality(G_gcc)
            full_cent = {node: 0.0 for node in G_window.nodes()}
            full_cent.update(cent)
            top_nodes = sorted(full_cent.items(), key=lambda x: x[1], reverse=True)[:5]
            centralities_over_time.append({window_date: top_nodes})
        except Exception:
            centralities_over_time.append({window_date: []})
    else:
        centralities_over_time.append({window_date: []})
    if G_window.number_of_edges() > 0:
        communities_window = list(nx.algorithms.community.greedy_modularity_communities(G_window))
        mod_window = nx.algorithms.community.modularity(G_window, communities_window)
        largest_comm_size = max(len(c) for c in communities_window) / G_window.number_of_nodes()
        modularities_over_time.append(mod_window)
        largest_comm_sizes_over_time.append(largest_comm_size)
    else:
        modularities_over_time.append(0.0)
        largest_comm_sizes_over_time.append(0.0)

if time_windows:
    plot_temporal_evolution(time_windows, modularities_over_time, largest_comm_sizes_over_time)
    plot_temporal_centrality(centralities_over_time)

print("Execution finished. Output images exported successfully.")
print(f"MP test p-value for λ+: {emp_pval_lambda_plus:.3f}")
print(f"Disparity null mean edges at alpha=0.05: {fp_mean:.1f} ± {fp_std:.1f}")
print(f"Bootstrap summary (block_size -> (>50%, >75%)): {bootstrap_summary}")