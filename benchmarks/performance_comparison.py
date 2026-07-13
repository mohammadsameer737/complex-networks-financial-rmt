"""
Benchmark naive vs. vectorized Disparity Filter implementations.
"""
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.finnet_backbone.backbone import disparity_filter_naive
from src.finnet_backbone.backbone_vectorized import disparity_filter_vectorized

def benchmark(sizes=[50, 100, 200, 500]):
    print("=" * 70)
    print("DISPARITY FILTER PERFORMANCE BENCHMARK")
    print("=" * 70)
    print(f"{'N':<10} | {'Naive (s)':<15} | {'Vectorized (s)':<15} | {'Speedup'}")
    print("-" * 70)
    
    for n in sizes:
        W = np.random.rand(n, n)
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0)
        
        start = time.time()
        disparity_filter_naive(W, alpha=0.05)
        t_naive = time.time() - start
        
        start = time.time()
        disparity_filter_vectorized(W, alpha=0.05)
        t_vec = time.time() - start
        
        speedup = t_naive / t_vec if t_vec > 0 else float('inf')
        print(f"{n:<10} | {t_naive:<15.5f} | {t_vec:<15.5f} | {speedup:.2f}x")

if __name__ == "__main__":
    benchmark()