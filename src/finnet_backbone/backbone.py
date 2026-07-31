import numpy as np
import pandas as pd
import networkx as nx
import random

def disparity_filter_naive(weight_matrix_pd, alpha=0.05):
    """
    Extract the multiscale backbone using the Disparity Filter.
    
    Null hypothesis: The weights of a node's links are uniformly distributed 
    (equivalent to randomly splitting a unit segment into k_i parts).
    We keep an edge only if its weight is statistically significant at level alpha.
    """
    W = np.array(weight_matrix_pd, dtype=np.float64, copy=True)
    W = np.abs(W)
    np.fill_diagonal(W, 0.0)
    n = W.shape[0]
    backbone = np.zeros_like(W)
    pvals = np.ones_like(W)
    
    # Node strength (sum of weights) and degree (number of links)
    s = np.sum(W, axis=1)
    k = np.sum(W > 0, axis=1)
    eps = 1e-12
    s_safe = np.maximum(s, eps) # Prevent division by zero
    
    for i in range(n):
        if k[i] <= 1:
            continue # Skip isolated nodes or leaves (no meaningful disparity)
            
        for j in range(i + 1, n):
            if W[i, j] <= 0:
                continue
                
            # Calculate p-value from node i's perspective
            # Formula: alpha_ij = (1 - w_ij / s_i)^(k_i - 1)
            alpha_ij = (1.0 - (W[i, j] / s_safe[i])) ** max(int(k[i]) - 1, 1)
            
            # Calculate p-value from node j's perspective
            alpha_ji = (1.0 - (W[i, j] / s_safe[j])) ** max(int(k[j]) - 1, 1) if k[j] > 1 else 1.0
            
            # Symmetric test: take the minimum p-value from both endpoints
            p_ij = min(alpha_ij, alpha_ji)
            pvals[i, j] = pvals[j, i] = p_ij
            
            # Keep edge only if p-value is below the significance threshold alpha
            if p_ij < alpha:
                backbone[i, j] = backbone[j, i] = W[i, j]
                
    return backbone, pvals

def disparity_null_fp_rate(returns_df, n_sims=50, alpha=0.05):
    """
    Estimate the false positive rate of the Disparity Filter on uncorrelated data.
    We shuffle the returns to destroy any real correlation, then apply the filter.
    """
    Tn = returns_df.shape[0]
    fp_counts = []
    for _ in range(n_sims):
        # Shuffle each column independently to break correlations
        shuffled = returns_df.apply(np.random.permutation)
        rho_sh = shuffled.corr().values.copy()  
        np.fill_diagonal(rho_sh, 0.0)
        
        bb_sh, _ = disparity_filter_naive(np.abs(rho_sh), alpha=alpha)
        G_sh = nx.from_numpy_array(bb_sh)
        G_sh.remove_nodes_from(list(nx.isolates(G_sh)))
        fp_counts.append(G_sh.number_of_edges())
        
    return np.mean(fp_counts), np.std(fp_counts)

def bootstrap_backbone_prices(prices_df, mapping, n_reps=200, block_size=5, alpha=0.05):
    """
    Assess edge stability using block bootstrap resampling.
    This preserves short-term temporal correlations in the financial time series.
    """
    Tn = prices_df.shape[0]
    edges_counts = {}
    for rep in range(n_reps):
        if rep % 20 == 0:
            print(f"Bootstrap rep {rep}/{n_reps}...")
            
        # Resample blocks of time steps to preserve local temporal structure
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
            
    # Calculate survival probability for each edge
    survival = {e: edges_counts.get(e, 0) / n_reps for e in edges_counts}
    return survival

def get_gcc_size(graph, N_global):
    """
    Calculate the relative size of the Giant Connected Component (GCC).
    This is the primary order parameter for network connectivity and resilience.
    """
    if graph.number_of_nodes() == 0:
        return 0.0
    connected_components = list(nx.connected_components(graph))
    largest_cc = max(connected_components, key=len)
    return len(largest_cc) / N_global

def jaccard(set1, set2):
    """Calculate Jaccard similarity coefficient between two sets of edges."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def safe_modularity(G):
    """
    Calculate network modularity with safety checks.
    Modularity is unreliable or undefined for very small or disconnected graphs.
    """
    if G.number_of_nodes() < 10 or G.number_of_edges() < 5:
        print("Modularity unreliable: graph too small. Skipping.")
        return None
    comms = list(nx.algorithms.community.greedy_modularity_communities(G))
    if len(comms) < 2:
        print("Modularity unreliable: fewer than 2 communities.")
        return None
    return nx.algorithms.community.modularity(G, comms)

def safe_algebraic_connectivity(G):
    """
    Calculate the Fiedler value (second smallest Laplacian eigenvalue).
    It measures the algebraic connectivity of the graph. Returns 0 if disconnected.
    """
    try:
        if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
            return 0.0
        return nx.algebraic_connectivity(G)
    except Exception:
        return 0.0

def track_gcc_decay(graph, removal_order, N_global):
    """
    Simulate node removal and track the relative GCC size at each step.
    Used to measure network robustness against targeted or random failures.
    """
    G_sim = graph.copy()
    gcc_sizes = [get_gcc_size(G_sim, N_global)]
    for node in removal_order:
        if node in G_sim:
            G_sim.remove_node(node)
        gcc_sizes.append(get_gcc_size(G_sim, N_global))
    return gcc_sizes

def robustness_curve_ensemble(graph, N_global, num_sims=100):
    """
    Generate robustness curves for targeted attacks (by degree) and random failures.
    Returns the fractional nodes removed (x-axis) vs. relative GCC size (y-axis).
    """
    degrees = dict(graph.degree())
    
    # Targeted attack: remove nodes in descending order of degree
    targeted_order_local = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
    gcc_targeted_local = track_gcc_decay(graph, targeted_order_local, N_global)
    
    # Random failures: average over multiple random permutations
    all_random_runs = []
    graph_nodes = list(graph.nodes())
    for _ in range(num_sims):
        shuffled = graph_nodes.copy()
        random.shuffle(shuffled)
        all_random_runs.append(track_gcc_decay(graph, shuffled, N_global))
        
    all_random_runs = np.array(all_random_runs)
    mean_random = np.mean(all_random_runs, axis=0)
    std_random = np.std(all_random_runs, axis=0)
    
    # Normalize x-axis to fraction of total nodes removed
    x_targeted = np.linspace(0, len(targeted_order_local) / N_global, len(gcc_targeted_local))
    x_random = np.linspace(0, len(graph_nodes) / N_global, len(mean_random))
    
    return x_targeted, gcc_targeted_local, x_random, mean_random, std_random