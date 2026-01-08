import yfinance as yf
import pandas as pd
import itertools

# -----------------------------
# Parameters
# -----------------------------
TICKERS = ["NVDA", "TSLA", "AMD", "AAPL", "META"]
LOOKBACK_DAYS = 30
INTERVALS = ["15m", "30m"]   # timescale grid
RVOL_LOOKBACK = 20
INITIAL_EQUITY = 100000

# Grid search ranges
RVOL_THRESHOLDS = [2.0, 2.5, 3.0]
TARGET_PCTS = [0.005, 0.01, 0.015]    # stock %
STOP_PCTS = [-0.003, -0.005, -0.007]
MAX_HOLDS = [15, 30, 45]              # minutes
DELTAS = [0.5, 0.6, 0.7]
PREMIUMS = [0.02, 0.03, 0.05]
LEVERAGE = 5.0                        # keep fixed

# -----------------------------
# Backtest function
# -----------------------------
def run_backtest(df, ticker, rvol_thr, tgt, stop, max_hold, delta, premium):
    equity = INITIAL_EQUITY
    in_trade = False
    entry_price, entry_time = 0, None
    wins, losses = [], []
    trades = 0

    for i in range(1, len(df)):
        ts = df.index[i]
        close = df["Close"].iloc[i]
        high_prev = df["High"].iloc[i - 1]
        rvol = df["RVOL"].iloc[i]

        # Morning session filter
        if ts.hour < 9 or (ts.hour == 9 and ts.minute < 30) or ts.hour > 12:
            continue

        # Entry
        if not in_trade and pd.notna(rvol):
            if close > high_prev and rvol > rvol_thr:
                in_trade = True
                entry_price = close
                entry_time = ts
                trades += 1

        # Exit
        elif in_trade:
            change_stock = (close - entry_price) / entry_price
            hold_time = (ts - entry_time).total_seconds() / 60

            if change_stock >= tgt or change_stock <= stop or hold_time >= max_hold:
                in_trade = False

                # Option return model
                change_option = change_stock * (delta * LEVERAGE) - premium
                equity *= (1 + change_option)

                if change_option > 0:
                    wins.append(change_option)
                else:
                    losses.append(change_option)

    return {
        "Ticker": ticker,
        "RVOL": rvol_thr,
        "Target%": tgt,
        "Stop%": stop,
        "MaxHold": max_hold,
        "Delta": delta,
        "Premium": premium,
        "Trades": trades,
        "WinRate": (len(wins) / trades * 100) if trades > 0 else 0,
        "AvgWin%": (sum(wins) / len(wins) * 100) if wins else 0,
        "AvgLoss%": (sum(losses) / len(losses) * 100) if losses else 0,
        "FinalEquity": equity,
        "PnL%": (equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100,
    }

# -----------------------------
# Grid Search Execution
# -----------------------------
all_results = []

for ticker in TICKERS:
    for interval in INTERVALS:
        print(f"\n=== {ticker} ({interval}) ===")
        df = yf.download(ticker, interval=interval, period=f"{LOOKBACK_DAYS}d")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        # Localize timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")
        else:
            df.index = df.index.tz_convert("US/Eastern")

        # Indicators
        df["Vol20"] = df["Volume"].rolling(RVOL_LOOKBACK).mean()
        df["RVOL"] = df["Volume"] / df["Vol20"]

        # Grid loop
        for rvol_thr, tgt, stop, max_hold, delta, premium in itertools.product(
            RVOL_THRESHOLDS, TARGET_PCTS, STOP_PCTS, MAX_HOLDS, DELTAS, PREMIUMS
        ):
            stats = run_backtest(df, ticker, rvol_thr, tgt, stop, max_hold, delta, premium)
            stats["Interval"] = interval
            all_results.append(stats)

# -----------------------------
# Collect Results
# -----------------------------
results_df = pd.DataFrame(all_results)

# Sort by PnL% per ticker
summary = results_df.groupby("Ticker").apply(
    lambda x: x.sort_values("PnL%", ascending=False).head(10)
)

print("\n=== Top Strategies Per Ticker ===")
print(summary.to_string(index=False))
