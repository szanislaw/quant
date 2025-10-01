import yfinance as yf
import pandas as pd
import numpy as np

# ========================
# CONFIG
# ========================
TICKERS = ["NVDA", "TSLA", "AMD", "AAPL", "META"]
INTERVAL = "30m"          # best from grid search
LOOKBACK_DAYS = 30
START_EQUITY = 100000

# Strategy params (from your grid search)
RVOL_THRESHOLD = 2.5
TARGET_PCT = 0.01
STOP_PCT = -0.003
MAX_HOLD = 45  # bars

# Option assumptions
OPTION_PREMIUM_PCT = 0.02   # 2% of stock price
OPTION_DELTA = 0.5
CONTRACT_MULTIPLIER = 100

# ========================
# STRATEGY FUNCTIONS
# ========================
def clean_dataframe(df):
    """Flattens MultiIndex columns (if present) so we always have single-level names."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_dataframe(df)
    vol = df["Volume"].astype(float)  # ensure Series
    df["Vol20"] = vol.rolling(20).mean()
    df["RVOL"] = vol / df["Vol20"]
    return df

def run_backtest(ticker):
    print(f"\nDownloading {ticker} data ({INTERVAL}, {LOOKBACK_DAYS}d)...")
    df = yf.download(ticker, interval=INTERVAL, period=f"{LOOKBACK_DAYS}d")
    if df.empty:
        print("No data retrieved.")
        return [], START_EQUITY, START_EQUITY

    df = compute_indicators(df).dropna()

    stock_equity = START_EQUITY
    option_equity = START_EQUITY

    trades = []
    position = None
    hold_bars = 0

    for i in range(len(df)):
        price = df["Close"].iloc[i]
        rvol = df["RVOL"].iloc[i]
        date = df.index[i]

        # ============ ENTRY ============
        if position is None and rvol > RVOL_THRESHOLD:
            entry_price = price
            position = {"entry": entry_price, "entry_time": date}
            hold_bars = 0

        # ============ EXIT ============
        elif position is not None:
            hold_bars += 1
            change = (price - position["entry"]) / position["entry"]

            if change >= TARGET_PCT or change <= STOP_PCT or hold_bars >= MAX_HOLD:
                exit_price = price
                stock_pnl = (exit_price - position["entry"]) / position["entry"]

                # Option approximation
                premium = OPTION_PREMIUM_PCT * position["entry"]
                option_return = stock_pnl * (position["entry"] / premium) * OPTION_DELTA

                stock_equity *= (1 + stock_pnl)
                option_equity *= (1 + option_return)

                trades.append({
                    "entry_time": position["entry_time"],
                    "exit_time": date,
                    "entry_price": position["entry"],
                    "exit_price": exit_price,
                    "stock_pnl%": round(stock_pnl * 100, 2),
                    "option_pnl%": round(option_return * 100, 2),
                    "stock_equity": round(stock_equity, 2),
                    "option_equity": round(option_equity, 2)
                })

                position = None

    return trades, stock_equity, option_equity

# ========================
# RUN ACROSS TICKERS
# ========================
summary = []
for ticker in TICKERS:
    trades, stock_final, option_final = run_backtest(ticker)
    df_trades = pd.DataFrame(trades)
    if not df_trades.empty:
        print(df_trades[["entry_time","exit_time","entry_price","exit_price","stock_pnl%","option_pnl%"]])
    else:
        print("No trades executed.")

    stock_pnl = (stock_final - START_EQUITY) / START_EQUITY * 100
    option_pnl = (option_final - START_EQUITY) / START_EQUITY * 100
    summary.append([ticker, len(trades), round(stock_pnl,2), round(option_pnl,2)])

summary_df = pd.DataFrame(summary, columns=["Ticker","Trades","StockPnL%","OptionPnL%"])
print("\n=== Summary Across Tickers ===")
print(summary_df)
