import yfinance as yf
import pandas as pd
from datetime import timedelta

# -----------------------------
# Parameters
# -----------------------------
TICKERS = ["NVDA", "TSLA", "AMD", "META", "GOOGL", "AMZN", "PLTR", "AAPL", "MSFT", "INTC", "QCOM", "IBM", "ORCL", "NBIS"]
LOOKBACK_DAYS = 5
INTERVAL = "30m"
RVOL_LOOKBACK = 20
RVOL_THRESHOLD = 2.5
ATR_LOOKBACK = 14

# -----------------------------
# Dummy option contract
# -----------------------------
def get_option_contract(ticker, spot_price):
    expiry = pd.Timestamp.today() + timedelta(days=14)  # 2 weeks
    strike = round(spot_price)  # ATM
    return f"{ticker} {expiry.strftime('%Y-%m-%d')} {strike}C"

# -----------------------------
# Exit strategy calculator
# -----------------------------
def compute_exit_strategy(df, entry_idx, entry_price):
    # ATR-based target
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = (df["High"] - df["Close"].shift()).abs()
    df["L-PC"] = (df["Low"] - df["Close"].shift()).abs()
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    atr = df["TR"].rolling(ATR_LOOKBACK).mean().iloc[entry_idx]

    target_price = entry_price * (1 + atr / entry_price * 1.5)  # ATR × 1.5
    stop_price = entry_price * (1 - 0.003)  # initial stop ~0.3%

    # Trailing stop ratio = 50% of unrealized gain
    tsl_ratio = 0.5  

    return round(target_price, 2), round(stop_price, 2), tsl_ratio

# -----------------------------
# Signal detection
# -----------------------------
def find_signals(df, ticker):
    signals = []
    for i in range(1, len(df)):
        ts = df.index[i]
        close = df["Close"].iloc[i]
        high_prev = df["High"].iloc[i - 1]
        rvol = df["RVOL"].iloc[i]

        if ts.hour < 9 or (ts.hour == 9 and ts.minute < 30) or ts.hour > 12:
            continue

        if close > high_prev and rvol > RVOL_THRESHOLD:
            option = get_option_contract(ticker, close)
            tp, sl, tsl = compute_exit_strategy(df, i, close)

            signals.append({
                "Ticker": ticker,
                "SignalTime": ts,
                "SpotPrice": round(close, 2),
                "RVOL": round(rvol, 2),
                "Option": option,
                "TakeProfit": tp,
                "StopLoss": sl,
                "TrailingStopRatio": tsl
            })
    return signals

# -----------------------------
# Run across tickers
# -----------------------------
all_signals = []

for ticker in TICKERS:
    print(f"Scanning {ticker}...")
    df = yf.download(ticker, interval=INTERVAL, period=f"{LOOKBACK_DAYS}d")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")
    else:
        df.index = df.index.tz_convert("US/Eastern")

    # Indicators
    df["Vol20"] = df["Volume"].rolling(RVOL_LOOKBACK).mean()
    df["RVOL"] = df["Volume"] / df["Vol20"]

    signals = find_signals(df, ticker)
    all_signals.extend(signals)

# -----------------------------
# Pick Top 5
# -----------------------------
signals_df = pd.DataFrame(all_signals)

if signals_df.empty:
    print("\n⚠️ No option signals found today.")
else:
    top5 = signals_df.sort_values(by="RVOL", ascending=False).head(10)
    print("\n=== Top 5 Option Recommendations for Today ===")
    print(top5.to_string(index=False))
