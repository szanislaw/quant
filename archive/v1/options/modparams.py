import yfinance as yf
import pandas as pd

# -----------------------------
# Parameters
# -----------------------------
TICKERS = ["NVDA", "TSLA", "AMD", "AAPL", "META"]
LOOKBACK_DAYS = 30
INTERVAL = "30m"       # best interval
RVOL_LOOKBACK = 20
RVOL_THRESHOLD = 2.5
TARGET_PCT = 0.01      # +1.0% profit target
STOP_PCT = -0.003      # -0.3% stop
MAX_HOLD = 45          # minutes
INITIAL_EQUITY = 100000

# Options parameters (approximation)
OPTION_MULTIPLIER = 8   # leverage: option moves ~8x stock
DELTA = 0.7             # assume ATM call delta ~0.7

# -----------------------------
# Backtest function
# -----------------------------
def run_backtest(df, ticker):
    stock_equity = INITIAL_EQUITY
    option_equity = INITIAL_EQUITY
    in_trade = False
    entry_price, entry_time = 0, None
    wins, losses = [], []
    trades = 0

    for i in range(1, len(df)):
        ts = df.index[i]
        close = df["Close"].iloc[i]
        high_prev = df["High"].iloc[i-1]
        rvol = df["RVOL"].iloc[i]

        # Restrict to morning session (spike focus)
        if ts.hour < 9 or (ts.hour == 9 and ts.minute < 30) or ts.hour > 12:
            continue

        # Entry
        if not in_trade and pd.notna(rvol):
            if close > high_prev and rvol > RVOL_THRESHOLD:
                in_trade = True
                entry_price = close
                entry_time = ts
                trades += 1
                print(f"[{ts}] BUY {ticker} @ {close:.2f} (RVOL={rvol:.2f})")

        # Exit
        elif in_trade:
            change = (close - entry_price) / entry_price
            hold_time = (ts - entry_time).total_seconds() / 60

            if change >= TARGET_PCT or change <= STOP_PCT or hold_time >= MAX_HOLD:
                in_trade = False

                # Stock PnL
                stock_equity *= (1 + change)

                # Option PnL (approx)
                option_change = change * OPTION_MULTIPLIER * DELTA
                option_equity *= (1 + option_change)

                if change > 0:
                    wins.append(change)
                else:
                    losses.append(change)

                print(
                    f"[{ts}] SELL {ticker} @ {close:.2f} → "
                    f"Stock PnL {change*100:.2f}% | Option PnL {option_change*100:.2f}% | "
                    f"Equity S:{stock_equity:.2f} O:{option_equity:.2f}"
                )

    return {
        "Ticker": ticker,
        "Trades": trades,
        "WinRate": (len(wins) / trades * 100) if trades > 0 else 0,
        "AvgWin%": (sum(wins)/len(wins)*100) if wins else 0,
        "AvgLoss%": (sum(losses)/len(losses)*100) if losses else 0,
        "StockFinal": stock_equity,
        "StockPnL%": (stock_equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100,
        "OptionFinal": option_equity,
        "OptionPnL%": (option_equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100,
    }

# -----------------------------
# Run for multiple tickers
# -----------------------------
results = []

for ticker in TICKERS:
    print(f"\n=== Running {ticker} ({INTERVAL}, {LOOKBACK_DAYS}d) ===")
    df = yf.download(ticker, interval=INTERVAL, period=f"{LOOKBACK_DAYS}d")

    # Clean columns
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

    stats = run_backtest(df, ticker)
    results.append(stats)

# -----------------------------
# Summary Table
# -----------------------------
summary = pd.DataFrame(results)
print("\n=== Summary Across Tickers ===")
print(summary[["Ticker","Trades","StockPnL%","OptionPnL%"]].to_string(index=False))
