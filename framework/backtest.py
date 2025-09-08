import pandas as pd
from data_ingest import fetch_prices
from signals import generate_signal

def backtest(ticker: str):
    df = fetch_prices(ticker)
    signal = generate_signal(df)
    print(f"📊 {ticker} latest signal: {signal}")

if __name__ == "__main__":
    for ticker in ["NVDA", "TSLA", "AAPL", "AMD"]:
        backtest(ticker)
