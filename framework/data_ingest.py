import yfinance as yf
import pandas as pd
from config import TRADING, DATA
from pathlib import Path

Path("data").mkdir(exist_ok=True)

def fetch_prices(ticker):
    df = yf.Ticker(ticker).history(
        period=DATA["period"], interval=DATA["interval"]
    )
    df.reset_index(inplace=True)
    df.to_csv(f"data/{ticker}.csv", index=False)
    return df

if __name__ == "__main__":
    for t in TRADING["watchlist"]:
        df = fetch_prices(t)
        print(f"✅ {t}: {len(df)} rows saved to data/{t}.csv")
