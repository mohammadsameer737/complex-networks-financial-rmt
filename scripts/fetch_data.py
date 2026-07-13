"""
Fetch S&P 500 price data and compute log-returns.
"""
import pandas as pd
import numpy as np
import yfinance as yf
import os

def main():
    if 'snakemake' in globals():
        output_cache = snakemake.output.cache
        output_returns = snakemake.output.returns
        start = snakemake.params.start
        end = snakemake.params.end
    else:
        output_cache = "data/sp500_cache.csv"
        output_returns = "data/log_returns.csv"
        start = "2014-01-01"
        end = "2024-12-31"

    os.makedirs(os.path.dirname(output_cache), exist_ok=True)
    os.makedirs(os.path.dirname(output_returns), exist_ok=True)

    # Download S&P 500 data
    tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
    tickers = [t.replace('.', '-') for t in tickers][:500]  # Limit to 500

    print(f"Fetching data for {len(tickers)} tickers...")
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)['Close']
    
    # Forward/backward fill missing values
    data = data.ffill().bfill()
    data.to_csv(output_cache)
    print(f"Saved price data to {output_cache}")

    # Compute log-returns
    log_returns = np.log(data / data.shift(1)).dropna()
    log_returns.to_csv(output_returns)
    print(f"Saved log-returns to {output_returns}")

if __name__ == "__main__":
    main()