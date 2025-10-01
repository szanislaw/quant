#!/usr/bin/env python3
# quantml_pseudo_backtest.py
"""
NVDA Pseudo-Backtest using Black-Scholes option pricing (historical BS proxy)
- Uses realized vol as a proxy for implied vol (rolling window)
- Prices ATM calls with Black-Scholes at entry and during lookahead
- Labels trades if BS-call price hits PROFIT_TARGET within LOOKAHEAD_DAYS
- Trains XGBoost classifier and backtests simple equity curve
"""

import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy.stats import norm
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
import matplotlib.pyplot as plt
import joblib
import os

# ----------------------------
# PARAMETERS (tweak as needed)
# ----------------------------
TICKER = "NVDA"
HISTORY_PERIOD = "2y"          # how much stock history to pull
INTERVAL = "1d"
ROLLING_IV_DAYS = 20           # window to estimate realized vol -> proxy IV
LOOKAHEAD_DAYS = 5             # check if option hits profit within this many trading days
TARGET_DTE = 10                # option term (days) at entry (used to compute time decay)
PROFIT_TARGET = 0.30           # 30% option return
RISK_FREE_RATE = 0.02          # annualized risk-free rate used in BS
MODEL_PATH = "nvda_bs_model.pkl"
MIN_TRADE_PROB = 0.7           # ML threshold for taking a trade in backtest
INITIAL_CAPITAL = 10000
RISK_PER_TRADE_PCT = 0.10      # fraction of capital risked per trade as exposure proxy

# ----------------------------
# Utilities: Black-Scholes call
# ----------------------------
def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call option price.
    S: spot price
    K: strike
    T: time to expiry in years (>=0)
    r: annual risk-free rate (decimal)
    sigma: annual volatility (decimal)
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    c = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return float(c)

# ----------------------------
# 1. Fetch stock data
# ----------------------------
print("Fetching historical data for", TICKER)
df = yf.download(TICKER, period=HISTORY_PERIOD, interval=INTERVAL, auto_adjust=True)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = [c[0] for c in df.columns]

# ensure index is business-day like; we'll use integer indexing for lookahead
df = df.sort_index().copy()

# ----------------------------
# 2. Feature engineering (same as your quant)
# ----------------------------
df["SMA10"] = df["Close"].rolling(10).mean()
df["SMA20"] = df["Close"].rolling(20).mean()
df["High20"] = df["High"].rolling(20).max().shift(1)

delta = df["Close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
df["RSI"] = 100 - (100 / (1 + rs))

ema12 = df["Close"].ewm(span=12).mean()
ema26 = df["Close"].ewm(span=26).mean()
df["MACD"] = ema12 - ema26
df["MACDsig"] = df["MACD"].ewm(span=9).mean()
df["MACDhist"] = df["MACD"] - df["MACDsig"]

df["H-L"] = df["High"] - df["Low"]
df["H-PC"] = (df["High"] - df["Close"].shift(1)).abs()
df["L-PC"] = (df["Low"] - df["Close"].shift(1)).abs()
df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
df["ATR14"] = df["TR"].rolling(14).mean()

# VAMS
df["VAMS"] = (df["Close"] - df["SMA20"]) / df["ATR14"]

# ----------------------------
# 3. Estimate historical "IV" proxy using realized vol
# ----------------------------
# log returns
df["logret"] = np.log(df["Close"] / df["Close"].shift(1))
# rolling realized vol (annualized)
df["realized_vol"] = df["logret"].rolling(ROLLING_IV_DAYS).std() * np.sqrt(252)

# Drop rows without enough history
df = df.dropna().reset_index(drop=False)  # keep date column

# For reproducibility, keep date column name
df.rename(columns={"index": "date"}, inplace=True)

# ----------------------------
# 4. Generate labels using BS pseudo-pricing
# ----------------------------
print("Generating labels with BS pseudo-pricing...")
labels = []
entry_call_price_list = []
max_future_call_price_list = []
entry_times = []
entry_spots = []
entry_sigmas = []

n = len(df)
for i in range(n):
    # must have LOOKAHEAD_DAYS ahead and enough DTE remaining
    if i + LOOKAHEAD_DAYS >= n:
        labels.append(np.nan)
        entry_call_price_list.append(np.nan)
        max_future_call_price_list.append(np.nan)
        entry_times.append(np.nan)
        entry_spots.append(np.nan)
        entry_sigmas.append(np.nan)
        continue

    S0 = df.loc[i, "Close"]
    strike = round(S0, 2)                  # ATM: use spot rounded as strike
    sigma_entry = float(df.loc[i, "realized_vol"])  # annualized
    # entry time to expiry in years
    T_entry = max(TARGET_DTE / 252.0, 1/252.0)
    c0 = bs_call_price(S0, strike, T_entry, RISK_FREE_RATE, sigma_entry)

    # record entry
    entry_call_price_list.append(c0)
    entry_times.append(T_entry)
    entry_spots.append(S0)
    entry_sigmas.append(sigma_entry)

    # compute future call prices for the lookahead window
    future_prices = []
    for t in range(1, LOOKAHEAD_DAYS + 1):
        S_t = df.loc[i + t, "Close"]
        # remaining time to expiry in years:
        T_t = max((TARGET_DTE - t) / 252.0, 1/252.0)
        # update sigma: assume sigma evolves to realized vol at that future date
        sigma_t = float(df.loc[i + t, "realized_vol"])
        c_t = bs_call_price(S_t, strike, T_t, RISK_FREE_RATE, sigma_t)
        future_prices.append(c_t)

    max_future = max(future_prices) if len(future_prices) else 0.0
    max_future_call_price_list.append(max_future)

    # profit check:
    profit_pct = (max_future - c0) / c0 if c0 > 0 else 0.0
    labels.append(1 if profit_pct >= PROFIT_TARGET else 0)

# append to df
df["bs_call_entry_price"] = entry_call_price_list
df["bs_call_max_future"] = max_future_call_price_list
df["bs_entry_T"] = entry_times
df["bs_entry_sigma"] = entry_sigmas
df["label"] = labels

# drop last rows with NaN labels
df = df.dropna(subset=["label"]).reset_index(drop=True)
df["label"] = df["label"].astype(int)

print(f"Total labeled rows: {len(df)}")

# ----------------------------
# 5. Prepare ML dataset
# ----------------------------
features = ["SMA10", "SMA20", "High20", "RSI", "MACD", "MACDhist", "ATR14", "VAMS", "realized_vol", "bs_call_entry_price"]
# drop rows where any feature is NaN
ml_df = df.dropna(subset=features + ["label"]).copy()

X = ml_df[features].values
y = ml_df["label"].values

# train/test split (time-series style)
split_idx = int(0.8 * len(ml_df))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# ----------------------------
# 6. Train XGBoost
# ----------------------------
print("Training XGBoost classifier...")
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))

# save model
joblib.dump(model, MODEL_PATH)
print("Model saved to", MODEL_PATH)

# ----------------------------
# 7. Backtest simple equity curve
# ----------------------------
print("Running backtest (paper trades)...")
capital = INITIAL_CAPITAL
equity = []
trade_records = []

# mapping test indices to ml_df rows: test part is ml_df.iloc[split_idx:]
test_df = ml_df.iloc[split_idx:].reset_index(drop=True)

for idx in range(len(test_df)):
    prob = float(y_prob[idx])
    row = test_df.iloc[idx]
    date = row["date"]
    entry_call = row["bs_call_entry_price"]
    entry_spot = row["Close"]
    # decide
    if prob >= MIN_TRADE_PROB and entry_call > 0:
        # simulate buying exposure scaled to RISK_PER_TRADE_PCT of capital (this is proxy: not buying option contracts)
        exposure = capital * RISK_PER_TRADE_PCT
        # compute quantity in terms of option notional: number of "contracts" proxy = exposure / (entry_call * 100)
        # we'll compute PnL using bs_call_max_future (which exists in df)
        exit_price = row["bs_call_max_future"]
        pnl = exposure * ((exit_price - entry_call) / entry_call) if entry_call > 0 else 0.0
        capital += pnl
        trade_records.append({
            "date": date,
            "prob": prob,
            "entry_spot": entry_spot,
            "entry_call": entry_call,
            "exit_call": exit_price,
            "pnl": pnl,
            "capital": capital
        })
    equity.append(capital)

# convert trade_records to df for inspection
trades_df = pd.DataFrame(trade_records)
print(f"Trades taken: {len(trades_df)}")
if not trades_df.empty:
    print(trades_df.head())

# Plot equity
plt.figure(figsize=(10,5))
plt.plot(equity, label="Equity (paper)")
plt.axhline(INITIAL_CAPITAL, linestyle="--", color="red", label="Start")
plt.title("Pseudo-Backtest Equity Curve (NVDA, BS proxy)")
plt.xlabel("Test step")
plt.ylabel("Capital ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------
# 8. Inference on latest row
# ----------------------------
latest_row = ml_df.iloc[-1]
latest_features = latest_row[features].values.reshape(1, -1)
latest_prob = float(model.predict_proba(latest_features)[:, 1])

print("\nLatest date:", latest_row["date"])
print(f"ML probability of profitable option (BS proxy) within {LOOKAHEAD_DAYS} days: {latest_prob:.2%}")

if latest_prob >= MIN_TRADE_PROB:
    print("✅ Paper trade strong signal")
elif latest_prob >= 0.5:
    print("⚠️ Paper trade medium signal")
else:
    print("❌ No paper trade signal")

# Save the dataset used
ds_fname = "nvda_bs_pseudo_dataset.csv"
ml_df.to_csv(ds_fname, index=False)
print("Saved dataset to", ds_fname)
