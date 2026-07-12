import numpy as np
import pandas as pd
import yfinance as yf
import os
import urllib.request
import ssl
import io

def load_market_data():
    try:
        context = ssl._create_unverified_context()
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urllib.request.urlopen(req, context=context).read()
        sp500_table = pd.read_html(io.StringIO(html.decode('utf-8')))[0]
        tickers = sp500_table['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers]
        ticker_sectors = dict(zip(sp500_table['Symbol'].tolist(), sp500_table['GICS Sector'].tolist()))
        ticker_sectors = {k.replace('.', '-'): v for k, v in ticker_sectors.items()}
    except Exception as e:
        print(f"wiki fetch failed: {e}")
        tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "CSCO", "ADBE", "NFLX",
            "JPM", "BAC", "WFC", "MS", "GS", "AXP", "BLK", "C",
            "JNJ", "LLY", "VRTX", "MRK", "PFE", "ABBV", "TMO", "UNH",
            "CAT", "GE", "HON", "UNP", "UPS", "MMM", "LIN", "FCX",
            "TSLA", "HD", "MCD", "NKE", "SBUX", "PG", "KO", "PEP", "WMT", "COST",
            "XOM", "CVX", "COP", "NEE", "DUK", "SO"
        ]
        ticker_sectors = {
            "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Communications", "AMZN": "Consumer Disc",
            "META": "Communications", "NVDA": "Technology", "AVGO": "Technology", "CSCO": "Technology",
            "ADBE": "Technology", "NFLX": "Communications",
            "JPM": "Financials", "BAC": "Financials", "WFC": "Financials", "MS": "Financials",
            "GS": "Financials", "AXP": "Financials", "BLK": "Financials", "C": "Financials",
            "JNJ": "Healthcare", "LLY": "Healthcare", "VRTX": "Healthcare", "MRK": "Healthcare",
            "PFE": "Healthcare", "ABBV": "Healthcare", "TMO": "Healthcare", "UNH": "Healthcare",
            "CAT": "Industrials", "GE": "Industrials", "HON": "Industrials", "UNP": "Industrials",
            "UPS": "Industrials", "MMM": "Industrials", "LIN": "Materials", "FCX": "Materials",
            "TSLA": "Consumer Disc", "HD": "Consumer Disc", "MCD": "Consumer Disc", "NKE": "Consumer Disc", "SBUX": "Consumer Disc",
            "PG": "Consumer Staples", "KO": "Consumer Staples", "PEP": "Consumer Staples", "WMT": "Consumer Staples", "COST": "Consumer Staples",
            "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
            "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities"
        }

    start_date = "2014-01-01"
    end_date = "2024-12-31"
    cache_file = 'sp500_10yr_cache.csv'
    
    if os.path.exists(cache_file):
        data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    else:
        print(f"downloading {len(tickers)} tickers...")
        chunk_size = 50
        all_data = []
        for i in range(0, len(tickers), chunk_size):
            batch = tickers[i:i+chunk_size]
            try:
                batch_data = yf.download(batch, start=start_date, end=end_date, auto_adjust=False, progress=False, timeout=30)
                if isinstance(batch_data.columns, pd.MultiIndex):
                    batch_close = batch_data['Adj Close'] if 'Adj Close' in batch_data.columns.levels[0] else batch_data['Close']
                else:
                    batch_close = batch_data['Adj Close'] if 'Adj Close' in batch_data.columns else batch_data['Close']
                valid_cols = batch_close.columns[batch_close.notna().any()]
                all_data.append(batch_close[valid_cols])
            except:
                pass
        data = pd.concat(all_data, axis=1)
        data = data.dropna(axis=1, how='all').ffill().bfill().dropna(axis=1, how='any')
        data.to_csv(cache_file)

    effective_tickers = list(data.columns)
    ticker_sectors = {k: v for k, v in ticker_sectors.items() if k in effective_tickers}
    N_global = len(effective_tickers)
    mapping = {i: effective_tickers[i] for i in range(N_global)}
    print(f"Effective Assets (N_global): {N_global}")
    
    return data, effective_tickers, ticker_sectors, N_global, mapping