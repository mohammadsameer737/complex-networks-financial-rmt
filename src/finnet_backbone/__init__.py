"""
finnet_backbone: Financial Network Backbone Extraction.

This package implements Random Matrix Theory (RMT) filtering and the 
Disparity Filter algorithm to extract the multiscale backbone of 
financial correlation networks, along with tools for robustness analysis 
and stochastic modeling.
"""

# Data ingestion
from .data_loader import load_market_data

# Random Matrix Theory
from .rmt import mp_null_lambda_plus, identify_sector_modes

# Backbone extraction and network metrics (Naive & Vectorized)
from .backbone import (
    disparity_filter_naive,
    disparity_null_fp_rate,
    bootstrap_backbone_prices,
    get_gcc_size,
    jaccard,
    safe_modularity,
    safe_algebraic_connectivity,
    track_gcc_decay,
    robustness_curve_ensemble
)
from .backbone_vectorized import disparity_filter_vectorized, bootstrap_parallel

# Stochastic processes and random walks
from .markov_analysis import MarketRegimeChain, random_walk_centrality

# Visualization utilities
from .visualization import (
    plot_ipr,
    plot_parameter_sweep,
    plot_er_phase_transition,
    plot_jaccard_overlap,
    plot_backbone_comparison,
    plot_perturbation_analysis,
    plot_robustness_comparison,
    plot_rmt_spectrum,
    plot_network_backbone,
    plot_temporal_evolution,
    plot_temporal_centrality
)

# Explicitly define the public API to prevent namespace pollution
__all__ = [
    # Data
    "load_market_data",
    # RMT
    "mp_null_lambda_plus",
    "identify_sector_modes",
    # Backbone
    "disparity_filter_naive",
    "disparity_filter_vectorized",
    "disparity_null_fp_rate",
    "bootstrap_backbone_prices",
    "bootstrap_parallel",
    # Metrics
    "get_gcc_size",
    "jaccard",
    "safe_modularity",
    "safe_algebraic_connectivity",
    "track_gcc_decay",
    "robustness_curve_ensemble",
    # Stochastic
    "MarketRegimeChain",
    "random_walk_centrality",
    # Visualization
    "plot_ipr",
    "plot_parameter_sweep",
    "plot_er_phase_transition",
    "plot_jaccard_overlap",
    "plot_backbone_comparison",
    "plot_perturbation_analysis",
    "plot_robustness_comparison",
    "plot_rmt_spectrum",
    "plot_network_backbone",
    "plot_temporal_evolution",
    "plot_temporal_centrality"
]

__version__ = "0.1.0"