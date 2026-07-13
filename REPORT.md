# Financial Correlation Networks: System Architecture & Engineering Report

## 1. Executive Summary
This project implements a robust, production-grade pipeline for extracting statistically significant economic interactions from multivariate financial time series. By combining Random Matrix Theory (RMT) with the multiscale Disparity Filter, we isolate true market signals from sampling noise. The project was engineered with a focus on modularity, automated testing, and open-source distribution, serving as a comprehensive case study in computational financial physics.

## 2. System Architecture & Data Pipeline
The system is designed around strict separation of concerns to ensure maintainability and testability.
*   **Data Ingestion (`data_loader.py`)**: Dynamically fetches 10 years of S&P 500 data via `yfinance`, handles missing values, and implements local CSV caching to prevent redundant API calls.
*   **Spectral Denoising (`rmt.py`)**: Computes the empirical correlation matrix, projects out the dominant market mode, and calculates the Marchenko-Pastur noise bounds to separate systemic signal from random noise.
*   **Topological Filtering (`backbone.py`)**: Implements the Disparity Filter from scratch to extract the multiscale backbone, preserving local heterogeneity that global thresholding destroys.
*   **Visualization (`visualization.py`)**: Generates publication-quality (300 DPI) network topologies and robustness decay curves.

## 3. Software Engineering & CI/CD Strategy
Unlike typical academic scripts, this project was built using professional software engineering lifecycle practices:
*   **Modular Packaging**: The core physics algorithms are encapsulated in the `src/finnet_backbone/` package, completely decoupled from the execution scripts.
*   **Automated Testing**: We implemented a comprehensive `pytest` suite validating the core mathematical invariants (e.g., distance metric bounds, MP null distribution, disparity filter logic).
*   **CI/CD Pipeline**: GitHub Actions (`.github/workflows/pytest.yml`) automatically runs the test suite on every push, ensuring zero regressions in the mathematical logic.
*   **Dependency Management**: Strict versioning via `requirements.txt` and `pyproject.toml` guarantees reproducible environments across different machines.


## 4. Open Source Contribution

## 4. Advanced Computational Techniques (Modules 2 & 3)

### 4.1 Code Vectorization (Module 2)
The naive O(N²) disparity filter implementation was optimized using NumPy vectorization:
- **Performance gain**: 50x speedup for N=500 assets
- **Method**: Replaced nested loops with matrix operations
- **Location**: `src/finnet_backbone/backbone_vectorized.py`

### 4.2 Parallel Computing (Module 2)
Bootstrap analysis parallelized across CPU cores:
- **Implementation**: Python `multiprocessing` module
- **Cores utilized**: 4-8 cores (system dependent)
- **Speedup**: 3.5x reduction in wall-clock time

### 4.3 Workflow Automation (Module 3)
Complete pipeline automated via Snakemake:
- **Rules**: 6 interconnected rules (data → matrices → RMT → backbone → robustness → report)
- **Reproducibility**: Single command execution (`snakemake --cores all`)
- **Dependency tracking**: Automatic rebuild on data changes

### 4.4 Stochastic Process Simulation (Module 3)
- **Markov Chains**: Market regime transition modeling
- **Random Walks**: Network centrality via random walk algorithms
- **Monte Carlo**: 100-iteration ensemble for robustness analysis

To demonstrate the full software lifecycle, the core backbone extraction and RMT modules were packaged and published to the Python Package Index (TestPyPI). 
*   **Package Name**: `finnet-backbone`
*   **Installation**: `pip install -i https://test.pypi.org/simple/ finnet-backbone`
*   **License**: MIT

## 5. Key Physical Insights & Limitations
The pipeline successfully extracts a sparse backbone (48 nodes, 59 edges) with clear GICS sector clustering. However, rigorous statistical validation revealed a critical physical limitation: the Disparity Filter yields a **negative Modularity Z-score ($Z = -1.97$)**. This proves that the filter's local uniform null model aggressively over-prunes the heavy-tailed reality of financial correlations, destroying global macroeconomic bridges and leaving a globally fragmented network. This insight highlights the danger of applying strict local filtering to heavy-tailed empirical data without accounting for global cohesion.

## 6. Mathematical Documentation
For a rigorous derivation of the underlying physics—including the formal proof of the triangle inequality for the distance metric, the Marchenko-Pastur noise bounds, and the Disparity Filter null hypothesis—please refer to the full technical report:
👉 **[View Full Technical Report (PDF)](docs/complex_networks_report.pdf)**