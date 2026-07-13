import numpy as np
import pandas as pd
import networkx as nx
import random

def disparity_filter_naive(weight_matrix_pd, alpha=0.05):
    W = np.array(weight_matrix_pd, dtype=np.float64, copy=True)
    W = np.abs(W)
    np.fill_diagonal(W, 0.0)
    n = W.shape[0]
    backbone = np.zeros_like(W)
    pvals = np.ones_like(W)
    s = np.sum(W, axis=1)
    k = np.sum(W > 0, axis=1)
    eps = 1e-12
    s_safe = np.maximum(s, eps)
    for i in range(n):
        if k[i] <= 1:
            continue
        for j in range(i + 1, n):
            if W[i, j] <= 0:
                continue
            alpha_ij = (1.0 - (W[i, j] / s_safe[i])) ** max(int(k[i]) - 1, 1)
            alpha_ji = (1.0 - (W[i, j] / s_safe[j])) ** max(int(k[j]) - 1, 1) if k[j] > 1 else 1.0
            p_ij = min(alpha_ij, alpha_ji)
            pvals[i, j] = pvals[j, i] = p_ij
            if p_ij < alpha:
                backbone[i, j] = backbone[j, i] = W[i, j]
    return backbone, pvals

def disparity_null_fp_rate(returns_df, n_sims=50, alpha=0.05):
    Tn = returns_df.shape[0]
    fp_counts = []
    for _ in range(n_sims):
        shuffled = returns_df.apply(np.random.permutation)
        rho_sh = shuffled.corr().values.copy()  
        np.fill_diagonal(rho_sh, 0.0)
        bb_sh, _ = disparity_filter_naive(np.abs(rho_sh), alpha=alpha)
        G_sh = nx.from_numpy_array(bb_sh)
        G_sh.remove_nodes_from(list(nx.isolates(G_sh)))
        fp_counts.append(G_sh.number_of_edges())
    return np.mean(fp_counts), np.std(fp_counts)

def bootstrap_backbone_prices(prices_df, mapping, n_reps=200, block_size=5, alpha=0.05):
    Tn = prices_df.shape[0]
    edges_counts = {}
    for rep in range(n_reps):
        if rep % 20 == 0:
            print(f"Bootstrap rep {rep}/{n_reps}...")
        starts = np.random.randint(0, Tn - block_size + 1, size=int(np.ceil(Tn / block_size)))
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:Tn]
        sample = prices_df.iloc[idx].reset_index(drop=True)
        lr = np.log(sample / sample.shift(1)).dropna()
        if lr.shape[0] < 2:
            continue
        rho = lr.corr()
        Wp = np.abs(rho.values)
        np.fill_diagonal(Wp, 0.0)
        backbone_rep, _ = disparity_filter_naive(Wp, alpha=alpha)
        G_rep = nx.from_numpy_array(backbone_rep)
        G_rep = nx.relabel_nodes(G_rep, mapping)
        G_rep.remove_nodes_from(list(nx.isolates(G_rep)))
        for e in G_rep.edges():
            edges_counts[e] = edges_counts.get(e, 0) + 1
    survival = {e: edges_counts.get(e, 0) / n_reps for e in edges_counts}
    return survival

def get_gcc_size(graph, N_global):
    if graph.number_of_nodes() == 0:
        return 0.0
    connected_components = list(nx.connected_components(graph))
    largest_cc = max(connected_components, key=len)
    return len(largest_cc) / N_global

def jaccard(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def safe_modularity(G):
    if G.number_of_nodes() < 10 or G.number_of_edges() < 5:
        print("Modularity unreliable: graph too small. Skipping.")
        return None
    comms = list(nx.algorithms.community.greedy_modularity_communities(G))
    if len(comms) < 2:
        print("Modularity unreliable: fewer than 2 communities.")
        return None
    return nx.algorithms.community.modularity(G, comms)

def safe_algebraic_connectivity(G):
    try:
        if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
            return 0.0
        return nx.algebraic_connectivity(G)
    except Exception:
        return 0.0

def track_gcc_decay(graph, removal_order, N_global):
    G_sim = graph.copy()
    gcc_sizes = [get_gcc_size(G_sim, N_global)]
    for node in removal_order:
        if node in G_sim:
            G_sim.remove_node(node)
        gcc_sizes.append(get_gcc_size(G_sim, N_global))
    return gcc_sizes

def robustness_curve_ensemble(graph, N_global, num_sims=100):
    degrees = dict(graph.degree())
    targeted_order_local = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
    gcc_targeted_local = track_gcc_decay(graph, targeted_order_local, N_global)
    all_random_runs = []
    graph_nodes = list(graph.nodes())
    for _ in range(num_sims):
        shuffled = graph_nodes.copy()
        random.shuffle(shuffled)
        all_random_runs.append(track_gcc_decay(graph, shuffled, N_global))
    all_random_runs = np.array(all_random_runs)
    mean_random = np.mean(all_random_runs, axis=0)
    std_random = np.std(all_random_runs, axis=0)
    x_targeted = np.linspace(0, len(targeted_order_local) / N_global, len(gcc_targeted_local))
    x_random = np.linspace(0, len(graph_nodes) / N_global, len(mean_random))
    return x_targeted, gcc_targeted_local, x_random, mean_random, std_random