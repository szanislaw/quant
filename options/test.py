import yfinance as yf
import pandas as pd
import itertools

# -----------------------------
# Parameters
# -----------------------------
TICKER = "NVDA"
LOOKBACK_DAYS = 30
INITIAL_EQUITY = 100000
RVOL_LOOKBACK = 20

# Parameter grids
INTERVALS = ["5m", "15m", "30m"]        # Timescales to test
RVOL_THRESHOLDS = [2.5, 3.0, 3.5]
TARGET_PCTS = [0.005, 0.008, 0.01]      # 0.5%, 0.8%, 1.0%
STOP_PCTS = [-0.003, -0.005, -0.007]    # -0.3%, -0.5%, -0.7%
MAX_HOLDS = [15, 30, 45]                # minutes

# -----------------------------
# Backtest Function
# -----------------------------
def run_backtest(df, rvol_thresh, target_pct, stop_pct, max_hold):
    equity = INITIAL_EQUITY
    in_trade = False
    entry_price, entry_time = 0, None
    wins, losses = [], []
    trades = 0

    for i in range(1, len(df)):
        ts = df.index[i]
        close = df["Close"].iloc[i]
        high_prev = df["High"].iloc[i-1]
        rvol = df["RVOL"].iloc[i]

        # Only trade morning session
        if ts.hour < 9 or (ts.hour == 9 and ts.minute < 30) or ts.hour > 10:
            continue

        # Entry
        if not in_trade and pd.notna(rvol):
            if close > high_prev and rvol > rvol_thresh:
                in_trade = True
                entry_price = close
                entry_time = ts
                trades += 1

        # Exit
        elif in_trade:
            change = (close - entry_price) / entry_price
            hold_time = (ts - entry_time).total_seconds() / 60

            if change >= target_pct or change <= stop_pct or hold_time >= max_hold:
                in_trade = False
                equity *= (1 + change)
                if change > 0:
                    wins.append(change)
                else:
                    losses.append(change)

    return {
        "RVOL": rvol_thresh,
        "Target%": target_pct,
        "Stop%": stop_pct,
        "MaxHold": max_hold,
        "Trades": trades,
        "WinRate": (len(wins) / trades * 100) if trades > 0 else 0,
        "AvgWin": (sum(wins)/len(wins)*100) if wins else 0,
        "AvgLoss": (sum(losses)/len(losses)*100) if losses else 0,
        "FinalEquity": equity,
        "PnL%": (equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100,
    }

# -----------------------------
# Grid Search with Timescale
# -----------------------------
results = []

for interval in INTERVALS:
    print(f"\nDownloading {TICKER} data ({interval}, {LOOKBACK_DAYS}d)...")
    df = yf.download(TICKER, interval=interval, period=f"{LOOKBACK_DAYS}d")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")
    else:
        df.index = df.index.tz_convert("US/Eastern")

    # Indicators
    df["Vol20"] = df["Volume"].rolling(RVOL_LOOKBACK).mean()
    df["RVOL"] = df["Volume"] / df["Vol20"]

    # Run backtest across all parameter combinations
    for rvol_thresh, target_pct, stop_pct, max_hold in itertools.product(
        RVOL_THRESHOLDS, TARGET_PCTS, STOP_PCTS, MAX_HOLDS
    ):
        stats = run_backtest(df, rvol_thresh, target_pct, stop_pct, max_hold)
        stats["Interval"] = interval
        results.append(stats)

# -----------------------------
# Results
# -----------------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("PnL%", ascending=False)

print("\n=== Top 15 Strategies Across All Timescales ===")
print(results_df.head(15).to_string(index=False))

results_df.to_csv("grid_results_timescale.csv", index=False)
print("\nFull results saved to grid_results_timescale.csv")
