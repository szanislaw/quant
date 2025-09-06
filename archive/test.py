import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from math import log, sqrt, exp
from scipy.stats import norm

# ----------------------------
# CONFIGURATION
# ----------------------------
TRADING_SLEEVE = 7000
RISK_PCT = 0.20
MAX_RISK = 5000
LOSS_CAP = 3000         # new hard per-trade loss cap
DTE = 5
R = 0.05
IV = 0.60
YEARS = 2

TICKERS = ["NVDA", "TSLA", "AMD", "META", "GOOGL",
           "MSFT", "AMZN", "AAPL", "PLTR", "SHOP"]

# ----------------------------
# BLACK-SCHOLES CALL PRICER
# ----------------------------
def bs_call_price(S, K, T, r, sigma):
    if T <= 0:
        return max(0.0, S - K)
    d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    call = S*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)
    return call

# ----------------------------
# SIGNAL GENERATOR
# ----------------------------
def signal_filter(df):
    df["SMA10"] = df["Close"].rolling(10).mean()
    df["High20"] = df["High"].rolling(20).max().shift(1)

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["DailyReturn"] = df["Close"].pct_change()

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACDsig"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACDhist"] = df["MACD"] - df["MACDsig"]

    breakout = (df["Close"] > df["High20"]) & (df["Close"] > df["SMA10"])
    momentum = (df["DailyReturn"] > 0.03) & (df["RSI"] > 60)
    macd_flip = df["MACDhist"] > 0

    df["Signal"] = breakout | momentum | macd_flip
    return df

# ----------------------------
# SIMULATOR
# ----------------------------
def simulate_stock(symbol, sleeve, trades):
    df = yf.download(symbol, period=f"{YEARS}y", interval="1d", auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = signal_filter(df)

    last_trade_date = None

    for i in range(20, len(df) - DTE):
        if not df["Signal"].iloc[i]:
            continue
        if last_trade_date and (df.index[i] - last_trade_date).days < 10:
            continue
        last_trade_date = df.index[i]

        entry_price = df["Close"].iloc[i]
        breakout_level = df["High20"].iloc[i]
        strike = entry_price * 1.05

        # Risk per trade with hard cap
        risk_amount = min(sleeve * RISK_PCT, MAX_RISK, LOSS_CAP)

        entry_opt = bs_call_price(entry_price, strike, DTE/252, R, IV)
        if entry_opt <= 0:
            continue
        contracts = risk_amount / (entry_opt * 100)
        if contracts < 1:
            continue

        exit_price = entry_opt
        holding_days = 0

        for j in range(1, DTE + 1):
            if i + j >= len(df):
                break
            holding_days = j
            close_price = df["Close"].iloc[i + j]

            if close_price < breakout_level:
                break

            T = (DTE - j) / 252
            est_price = bs_call_price(close_price, strike, T, R, IV)

            if est_price >= entry_opt * 11:
                exit_price = entry_opt * 11
                break
            elif est_price >= entry_opt * 5:
                exit_price = entry_opt * 5
                break
            elif est_price >= entry_opt * 3:
                exit_price = entry_opt * 3
                break
            else:
                exit_price = est_price

        pnl_trade = (exit_price - entry_opt) * contracts * 100
        pnl_trade = max(pnl_trade, -LOSS_CAP)   # enforce hard loss cap
        sleeve += pnl_trade

        trades.append({
            "date": df.index[i].date(),
            "symbol": symbol,
            "pnl": pnl_trade,
            "sleeve_after": sleeve,
            "holding_days": holding_days
        })

    return sleeve, trades

def simulate_universe_monthly(tickers=TICKERS):
    sleeve = TRADING_SLEEVE
    trades = []
    for t in tickers:
        sleeve, trades = simulate_stock(t, sleeve, trades)

    df_trades = pd.DataFrame(trades)
    if df_trades.empty:
        return sleeve, trades, pd.DataFrame()

    df_trades["month"] = pd.to_datetime(df_trades["date"]).dt.to_period("M")
    monthly = df_trades.groupby("month").agg(
        trades=("pnl", "count"),
        pnl=("pnl", "sum"),
        sleeve=("sleeve_after", "last")
    ).reset_index()
    monthly["return_pct"] = monthly["pnl"] / TRADING_SLEEVE * 100

    return sleeve, trades, monthly

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    final_sleeve, trades, monthly = simulate_universe_monthly(TICKERS)

    print("=== Monthly Performance ===")
    if monthly.empty:
        print("No trades triggered.")
    else:
        print(monthly)

    print("\n=== Final Summary ===")
    total_pnl = final_sleeve - TRADING_SLEEVE
    print(f"Initial trading capital: ${TRADING_SLEEVE:,.2f}")
    print(f"Final trading capital: ${final_sleeve:,.2f}")
    print(f"Total PnL: ${total_pnl:,.2f}")
    print(f"Total trades: {len(trades)}")
    win_rate = (np.array([t['pnl'] for t in trades]) > 0).mean() if trades else 0
    expectancy = np.mean([t['pnl'] for t in trades]) if trades else 0
    avg_holding = np.mean([t['holding_days'] for t in trades]) if trades else 0
    print(f"Win rate: {win_rate:.2%}")
    print(f"Avg PnL per trade: ${expectancy:,.2f}")
    print(f"Avg holding time: {avg_holding:.1f} days")
