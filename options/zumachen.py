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
# Get real option contract
# -----------------------------
def get_option_contract(ticker, spot_price):
    try:
        opt = yf.Ticker(ticker)
        expiries = opt.options
        if not expiries:
            return None

        today = pd.Timestamp.today().normalize()

        # Pick the nearest expiry at least 7 days away (avoid same-week expiry)
        valid_expiries = [pd.Timestamp(e) for e in expiries if pd.Timestamp(e) > today + pd.Timedelta(days=7)]
        expiry = min(valid_expiries) if valid_expiries else pd.Timestamp(expiries[0])

        chain = opt.option_chain(expiry.strftime("%Y-%m-%d"))
        calls = chain.calls
        if calls.empty:
            return None

        # Find nearest strike to spot price
        strike = calls.iloc[(calls['strike'] - spot_price).abs().argmin()]['strike']
        return f"{ticker} {expiry.strftime('%Y-%m-%d')} {int(strike)}C"
    except Exception as e:
        print(f"⚠️ Could not fetch option chain for {ticker}: {e}")
        return None

# -----------------------------
# Exit strategy calculator
# -----------------------------
def compute_exit_strategy(df, entry_idx, entry_price):
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = (df["High"] - df["Close"].shift()).abs()
    df["L-PC"] = (df["Low"] - df["Close"].shift()).abs()
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    atr = df["TR"].rolling(ATR_LOOKBACK).mean().iloc[entry_idx]

    target_price = entry_price * (1 + atr / entry_price * 1.5)
    stop_price = entry_price * (1 - 0.003)
    tsl_ratio = 0.5  # 50% of unrealized gain

    return round(target_price, 2), round(stop_price, 2), tsl_ratio

# -----------------------------
# Find most recent signal
# -----------------------------
def find_latest_signal(df, ticker):
    if len(df) < 2:
        return None

    ts = df.index[-1]
    close = df["Close"].iloc[-1]
    high_prev = df["High"].iloc[-2]
    rvol = df["RVOL"].iloc[-1]

    # Only check signals during morning session
    if ts.hour < 9 or (ts.hour == 9 and ts.minute < 30) or ts.hour > 12:
        return None

    if close > high_prev and rvol > RVOL_THRESHOLD:
        option = get_option_contract(ticker, close)
        if option:
            tp, sl, tsl = compute_exit_strategy(df, len(df) - 1, close)
            return {
                "Ticker": ticker,
                "SignalTime": ts,
                "SpotPrice": round(close, 2),
                "RVOL": round(rvol, 2),
                "Option": option,
                "TakeProfit": tp,
                "StopLoss": sl,
                "TrailingStopRatio": tsl
            }
    return None

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

    df["Vol20"] = df["Volume"].rolling(RVOL_LOOKBACK).mean()
    df["RVOL"] = df["Volume"] / df["Vol20"]

    sig = find_latest_signal(df, ticker)
    if sig:
        all_signals.append(sig)

# -----------------------------
# Pick Top 5
# -----------------------------
signals_df = pd.DataFrame(all_signals)

if signals_df.empty:
    print("\n⚠️ No option signals found on latest bar.")
else:
    top5 = signals_df.sort_values(by="RVOL", ascending=False).head(5)
    print("\n=== Top 5 Option Recommendations (Latest Signals) ===")
    print(top5.to_string(index=False))
