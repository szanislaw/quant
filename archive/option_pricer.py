# option_pricer.py
# True option P/L simulation using Black–Scholes
# Used by backtester to compute real option returns over time

import numpy as np
import pandas as pd
from math import log, sqrt, exp
from scipy.stats import norm
import yfinance as yf



# =========================================================
# BLACK–SCHOLES FORMULA FOR CALL OPTIONS
# =========================================================
def bs_call_price(S, K, T, r, iv):
    """
    Black–Scholes price for a call option.

    S  = underlying price
    K  = strike
    T  = time to expiry in years
    r  = risk-free rate
    iv = implied volatility (annualized)
    """
    if T <= 0 or iv <= 0:
        return max(S - K, 0)

    d1 = (np.log(S / K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)

    return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)


# =========================================================
# INTERPOLATE DAILY IV TO 4H CANDLES
# =========================================================
def interpolate_iv(df_daily, df_intraday):
    daily_iv = df_daily["Implied Volatility"].ffill().bfill()

    # Convert daily IV to time-indexed series
    iv_daily = daily_iv.reindex(df_intraday.index, method="ffill")

    return iv_daily


# =========================================================
# SIMULATE OPTION PRICE PATH
# =========================================================
def simulate_option_path(df, sig_row, r=0.04):
    """
    df       = 4H price DataFrame
    sig_row  = row from generate_signals()
    r        = risk-free rate

    Returns a DataFrame with:
    - option price at each candle
    - MFE option
    - MAE option
    - return at expiry
    """
    strike = sig_row["strike"]
    t0 = sig_row["timestamp"]
    expiry_date = sig_row["expiry"]

    # Build forward window
    window = df[df.index >= t0]

    # If no future data exists:
    if window.empty:
        return None

    # Load daily implied volatility (Yahoo)
    hist = yf.Ticker(sig_row["ticker"]).history(period="120d")
    if "Implied Volatility" not in hist.columns:
        # fallback IV = 30% annualized
        hist["Implied Volatility"] = 0.30

    iv_series = interpolate_iv(hist, window)

    option_prices = []
    expiries = pd.Timestamp(expiry_date)

    for timestamp, row in window.iterrows():
        S = float(row["Close"])
        iv = float(iv_series.loc[timestamp])
        T_days = (expiries - timestamp).total_seconds() / 86400

        if T_days <= 0:
            T = 0
        else:
            T = T_days / 365

        opt_val = bs_call_price(S, strike, T, r, iv)
        option_prices.append(opt_val)

    window = window.copy()
    window["OptionPrice"] = option_prices

    # Compute MFE/MAE
    entry_price = window["OptionPrice"].iloc[0]
    window["ReturnPct"] = (window["OptionPrice"] - entry_price) / entry_price * 100

    mfe = window["ReturnPct"].max()
    mae = window["ReturnPct"].min()
    final_ret = window["ReturnPct"].iloc[-1]

    summary = {
        "entry_price": entry_price,
        "max_favorable_excursion_pct": mfe,
        "max_adverse_excursion_pct": mae,
        "return_at_expiry_pct": final_ret
    }

    return window, summary
