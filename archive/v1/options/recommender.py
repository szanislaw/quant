import yfinance as yf
import pandas as pd
from datetime import timedelta

# -----------------------------
# Parameters (best from NVDA grid search)
# -----------------------------
TICKERS = ["NVDA", "TSLA", "AMD", "AAPL", "META"]
LOOKBACK_DAYS = 30
INTERVAL = "30m"       # best interval
RVOL_LOOKBACK = 20
RVOL_THRESHOLD = 2.5
TARGET_PCT = 0.01      # +1.0% profit target
STOP_PCT = -0.003      # -0.3% stop
MAX_HOLD = 45          # minutes
INITIAL_EQUITY = 1000

# -----------------------------
# Option Chain Fetcher
# -----------------------------
def get_option_contract(ticker, spot_price):
    tk = yf.Ticker(ticker)
    expiries = tk.options
    if not expiries:
        return None

    # Choose expiry ~2–3 weeks ahead
    today = pd.Timestamp.today()
    valid_exps = [pd.Timestamp(e) for e in expiries]
    valid_exps = [e for e in valid_exps if 14 <= (e - today).days <= 21]
    if not valid_exps:
        expiry = pd.Timestamp(expiries[0])  # fallback earliest
    else:
        expiry = min(valid_exps, key=lambda x: abs((x - today).days))

    # Get option chain for expiry
    chain = tk.option_chain(expiry.strftime("%Y-%m-%d"))
    calls = chain.calls

    # Pick strike nearest to spot
    calls["diff"] = abs(calls["strike"] - spot_price)
    opt = calls.loc[calls["diff"].idxmin()]

    option = {
        "expiry": expiry.strftime("%Y-%m-%d"),
        "strike": float(opt["strike"]),
        "lastPrice": float(opt["lastPrice"]),
        "bid": float(opt["bid"]),
        "ask": float(opt["ask"]),
        "impliedVol": float(opt["impliedVolatility"]),
        "delta": 0.5 if abs(opt["strike"] - spot_price) < 1 else 0.4,  # rough approx
    }
    return option

# -----------------------------
# Outcome projection
# -----------------------------
def project_outcomes(entry_price, option, df, entry_idx):
    delta = option["delta"]

    outcomes = {"Target": None, "Stop": None, "Flat": None}
    entry_time = df.index[entry_idx]

    for i in range(entry_idx + 1, len(df)):
        ts = df.index[i]
        hold_time = (ts - entry_time).total_seconds() / 60
        change = (df["Close"].iloc[i] - entry_price) / entry_price

        if change >= TARGET_PCT and outcomes["Target"] is None:
            outcomes["Target"] = round(delta * TARGET_PCT * 100, 2)
        if change <= STOP_PCT and outcomes["Stop"] is None:
            outcomes["Stop"] = round(delta * STOP_PCT * 100, 2)

        if hold_time >= MAX_HOLD:
            flat_change = (df["Close"].iloc[i] - entry_price) / entry_price
            outcomes["Flat"] = round(delta * flat_change * 100, 2)
            break

    for k in outcomes:
        if outcomes[k] is None:
            outcomes[k] = 0.0

    return outcomes

# -----------------------------
# Recommender + Equity Curve
# -----------------------------
def run_recommender(ticker):
    print(f"\n=== Running {ticker} ({INTERVAL}, {LOOKBACK_DAYS}d) ===")
    df = yf.download(ticker, interval=INTERVAL, period=f"{LOOKBACK_DAYS}d")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")
    else:
        df.index = df.index.tz_convert("US/Eastern")

    df["Vol20"] = df["Volume"].rolling(RVOL_LOOKBACK).mean()
    df["RVOL"] = df["Volume"] / df["Vol20"]

    signals = []
    equity = INITIAL_EQUITY

    for i in range(1, len(df)):
        ts = df.index[i]
        close = df["Close"].iloc[i]
        high_prev = df["High"].iloc[i - 1]
        rvol = df["RVOL"].iloc[i]

        if ts.hour < 9 or (ts.hour == 9 and ts.minute < 30) or ts.hour > 12:
            continue

        if close > high_prev and rvol > RVOL_THRESHOLD:
            option = get_option_contract(ticker, close)
            if not option:
                continue

            outcomes = project_outcomes(close, option, df, i)

            # Conservative: prefer Flat outcome
            realized = outcomes["Flat"] if outcomes["Flat"] != 0 else outcomes["Stop"] or outcomes["Target"]
            equity *= (1 + realized/100)

            signals.append({
                "Ticker": ticker,
                "SignalTime": ts,
                "SpotPrice": round(close, 2),
                "RVOL": round(rvol, 2),
                "Option": f"{option['expiry']} {option['strike']}C",
                "LastPrice": option["lastPrice"],
                "Bid": option["bid"],
                "Ask": option["ask"],
                "IV": round(option["impliedVol"], 3),
                "Target%": outcomes["Target"],
                "Stop%": outcomes["Stop"],
                "Flat%": outcomes["Flat"],
                "EquityAfter": round(equity, 2),
            })

    return pd.DataFrame(signals), equity

# -----------------------------
# Run across tickers
# -----------------------------
all_signals = []
finals = []

for ticker in TICKERS:
    df_signals, eq_final = run_recommender(ticker)
    all_signals.append(df_signals)
    finals.append({"Ticker": ticker, "FinalEquity": eq_final, "PnL%": (eq_final - INITIAL_EQUITY) / INITIAL_EQUITY * 100})

summary = pd.concat(all_signals, ignore_index=True)
# -----------------------------
# Rank recommendations by best payoff potential
# -----------------------------
ranked = summary.sort_values(by="Target%", ascending=False).reset_index(drop=True)

print("\n=== Ranked Option Recommendations (by Target% payoff) ===")
print(ranked[["Ticker", "SignalTime", "SpotPrice", "RVOL", "Option", 
              "LastPrice", "Bid", "Ask", "IV", "Target%", "Stop%", "Flat%", "EquityAfter"]]
      .to_string(index=False))

results = pd.DataFrame(finals)

# print("\n=== Current Recommendations (per trade) ===")
# print(summary.to_string(index=False))

print("\n=== Equity Curve Summary ===")
print(results.to_string(index=False))
