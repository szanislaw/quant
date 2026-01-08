# backtest.py
# Full 90-day 4H quant backtester with REAL option P/L simulation using Black–Scholes

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os

# Scanner imports
from qqscanner import (
    add_indicators,
    detect_structure,
    compute_confidence,
    choose_strike,
    choose_expiry,
    options_flow_score,
    VOL_Z_THRESHOLD,
    ATR_MULTIPLIER,
)

# Option simulation
from option_pricer import simulate_option_path


# ------------- CONFIG -------------
TICKERS = [
    "NVDA", "AMD", "TSLA", "META",
    "GOOGL", "AAPL", "MSFT", "PLTR"
]

LOOKBACK = "90d"
INTERVAL = "4h"

OUT_SIGNALS = "backtest_signals.csv"
OUT_FORWARD = "backtest_forward_stats.csv"
OUT_SUMMARY = "backtest_summary.csv"
OUT_OPTIONS = "backtest_options.csv"
PATH_DIR = "option_paths"

os.makedirs(PATH_DIR, exist_ok=True)
# -----------------------------------


# =========================================================
# LOAD DATA
# =========================================================
def load_data(ticker):
    df = yf.download(
        ticker,
        period=LOOKBACK,
        interval=INTERVAL,
        auto_adjust=True,
        progress=False
    )
    if df.empty:
        return None

    df = add_indicators(df)
    df.dropna(inplace=True)

    return df


# =========================================================
# GENERATE SIGNALS LIKE SCANNER (CANDLE-BY-CANDLE)
# =========================================================
def generate_signals(ticker, df):
    rows = []

    for i in range(50, len(df)):
        sub = df.iloc[:i].copy()
        last = sub.iloc[-1]

        close = last["Close"].item()
        atr   = last["ATR"].item()
        rsi   = last["RSI"].item()
        macdh = last["MACD_H"].item()
        volz  = last["Vol_Z"].item()
        ema20 = last["EMA20"].item()
        ema50 = last["EMA50"].item()

        hl, hh, breakout, breakout_level = detect_structure(sub)

        trend_up = ema20 > ema50
        volume_ok = volz > VOL_Z_THRESHOLD
        structure_ok = hl and (hh or breakout)
        optflow = options_flow_score(ticker)

        confidence = compute_confidence(
            trend_up, macdh, volume_ok, structure_ok, optflow
        )

        strike = choose_strike(ticker, close, atr, breakout_level)
        expiry = choose_expiry()

        rows.append({
            "timestamp": last.name,
            "ticker": ticker,
            "close": close,
            "atr": atr,
            "rsi": rsi,
            "macdh": macdh,
            "volz": volz,
            "trend_up": trend_up,
            "structure_ok": structure_ok,
            "volume_ok": volume_ok,
            "optflow": optflow,
            "breakout_level": breakout_level,
            "strike": strike,
            "expiry": expiry,
            "confidence": confidence
        })

    return pd.DataFrame(rows)


# =========================================================
# FORWARD EVAL (stock-based metrics)
# =========================================================
def forward_eval(df, sig):
    t = sig["timestamp"]
    strike = sig["strike"]
    close = sig["close"]
    atr = sig["atr"]
    breakout_level = sig["breakout_level"]

    expiry_ts = pd.Timestamp(sig["expiry"]).tz_localize("UTC")
    if t.tzinfo is None:
        t = t.tz_localize("UTC")

    win30 = t + timedelta(days=30)
    win60 = t + timedelta(days=60)

    future = df[df.index > t]
    if future.empty:
        return {k: np.nan for k in [
            "strike_expiry", "strike_30", "strike_60",
            "atr_expiry", "atr_30", "atr_60",
            "ret_expiry", "ret_30", "ret_60",
            "mfe_expiry", "mae_expiry"
        ]}

    f_exp = future[future.index <= expiry_ts]
    f_30  = future[future.index <= win30]
    f_60  = future[future.index <= win60]

    getp = lambda f: f["Close"].values if not f.empty else np.array([])

    p_exp = getp(f_exp)
    p_30  = getp(f_30)
    p_60  = getp(f_60)

    atr_target = breakout_level + ATR_MULTIPLIER * atr

    strike_expiry = np.any(p_exp >= strike)
    strike_30 = np.any(p_30 >= strike)
    strike_60 = np.any(p_60 >= strike)

    atr_expiry = np.any(p_exp >= atr_target)
    atr_30 = np.any(p_30 >= atr_target)
    atr_60 = np.any(p_60 >= atr_target)

    ret_expiry = p_exp[-1] - close if p_exp.size else np.nan
    ret_30 = p_30[-1] - close if p_30.size else np.nan
    ret_60 = p_60[-1] - close if p_60.size else np.nan

    mfe = (p_exp.max() - close) if p_exp.size else np.nan
    mae = (p_exp.min() - close) if p_exp.size else np.nan

    return {
        "strike_expiry": strike_expiry,
        "strike_30": strike_30,
        "strike_60": strike_60,
        "atr_expiry": atr_expiry,
        "atr_30": atr_30,
        "atr_60": atr_60,
        "ret_expiry": ret_expiry,
        "ret_30": ret_30,
        "ret_60": ret_60,
        "mfe_expiry": mfe,
        "mae_expiry": mae,
    }


# =========================================================
# MAIN BACKTEST RUNNER — WITH OPTION SIMULATION
# =========================================================
def run_backtest():
    all_sigs = []
    all_fwd = []
    all_opt = []

    for ticker in TICKERS:
        print(f"\nProcessing {ticker}...")

        df = load_data(ticker)
        if df is None:
            print("No data.")
            continue

        sigs = generate_signals(ticker, df)

        for idx, row in sigs.iterrows():
            # ---- Stock forward eval ----
            fstats = forward_eval(df, row)
            fstats["ticker"] = ticker
            fstats["timestamp"] = row["timestamp"]
            fstats["confidence"] = row["confidence"]

            # ---- Option simulation ----
            opt = simulate_option_path(df, row)

            if opt is not None:
                path_df, opt_summary = opt

                trade_id = f"{ticker}_{row['timestamp']:%Y%m%d_%H%M}"
                path_file = f"{PATH_DIR}/{trade_id}.csv"

                path_df.to_csv(path_file)

                opt_summary.update({
                    "ticker": ticker,
                    "timestamp": row["timestamp"],
                    "confidence": row["confidence"],
                    "strike": row["strike"],
                    "expiry": row["expiry"],
                    "path_file": path_file
                })

                all_opt.append(opt_summary)

            all_fwd.append(fstats)

        all_sigs.append(sigs)

    sig_df = pd.concat(all_sigs, ignore_index=True)
    fwd_df = pd.concat(all_fwd, ignore_index=True)
    opt_df = pd.DataFrame(all_opt)

    sig_df.to_csv(OUT_SIGNALS, index=False)
    fwd_df.to_csv(OUT_FORWARD, index=False)
    opt_df.to_csv(OUT_OPTIONS, index=False)

    summary = {
        "signals_total": len(sig_df),
        "avg_confidence": sig_df["confidence"].mean(),
        "strike_expiry_rate": fwd_df["strike_expiry"].mean(),
        "atr_expiry_rate": fwd_df["atr_expiry"].mean(),
        "option_avg_return": opt_df["return_at_expiry_pct"].mean(),
        "option_avg_mfe": opt_df["max_favorable_excursion_pct"].mean(),
        "option_avg_mae": opt_df["max_adverse_excursion_pct"].mean(),
    }

    pd.DataFrame([summary]).to_csv(OUT_SUMMARY, index=False)

    print("\n=== BACKTEST COMPLETE ===")
    print(f"Signals          → {OUT_SIGNALS}")
    print(f"Forward stats    → {OUT_FORWARD}")
    print(f"Option summaries → {OUT_OPTIONS}")
    print(f"Option paths     → {PATH_DIR}/")
    print(f"Summary          → {OUT_SUMMARY}")


if __name__ == "__main__":
    run_backtest()
