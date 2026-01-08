import yfinance as yf
import pandas as pd
import numpy as np
from math import ceil
from datetime import datetime
from colorama import Fore, Style, init

# -------------------------------------------------------------------
# Init colorama
# -------------------------------------------------------------------
init(autoreset=True)

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
TICKERS = [
    "NVDA", "AMD", "TSLA", "META",
    "GOOGL", "AAPL", "MSFT", "PLTR"
]

INTERVAL = "4h"
LOOKBACK = "90d"

VOL_Z_THRESHOLD = 1.5
ATR_MULTIPLIER = 1.8
MAX_OTM_PERCENT = 0.10
CSV_OUTPUT = "scan_results.csv"

MIN_WEEKS = 4
MAX_WEEKS = 8

STRIKE_INCREMENTS = {
    "NVDA": 5, "AMD": 2.5, "TSM": 5, "ASML": 5, "AVGO": 5,
    "QCOM": 2.5, "MU": 2.5, "INTC": 1, "ARM": 2.5, "SMCI": 5,
    "MRVL": 2.5, "AMAT": 2.5, "LRCX": 5, "KLAC": 5, "TER": 2.5,
    "NXPI": 5, "ADI": 2.5,
    "CSCO": 1, "ANET": 5, "JNPR": 1, "HPE": 1, "DELL": 1,
    "NTNX": 1, "FFIV": 5, "GLW": 1,
    "WDC": 1, "STX": 2.5,
    "AAPL": 5, "MSFT": 5, "GOOGL": 5, "META": 5, "AMZN": 5,
    "TSLA": 2.5, "PLTR": 1,
    "PFE": 1, "MRK": 2.5, "LLY": 5, "JNJ": 2.5, "ABBV": 2.5,
    "BMY": 1, "AMGN": 5, "REGN": 5, "GILD": 2.5, "VRTX": 5
}

# -------------------------------------------------------------------
# Color display
# -------------------------------------------------------------------
def colorize_conf(conf):
    if conf >= 70:
        return f"{Fore.GREEN}{conf}{Style.RESET_ALL}"
    elif conf >= 40:
        return f"{Fore.YELLOW}{conf}{Style.RESET_ALL}"
    return f"{Fore.RED}{conf}{Style.RESET_ALL}"

# -------------------------------------------------------------------
# Expiry selection (4–8 weeks)
# -------------------------------------------------------------------
def choose_expiry(start_date=datetime.now(),
                  min_weeks=MIN_WEEKS, max_weeks=MAX_WEEKS):
    min_days = min_weeks * 7
    max_days = max_weeks * 7

    target_days = (min_days + max_days) // 2
    target_date = start_date + pd.Timedelta(days=target_days)

    weekday = target_date.weekday()
    expiry = target_date + pd.Timedelta(days=(4 - weekday) % 7)

    if expiry < start_date + pd.Timedelta(days=min_days):
        expiry += pd.Timedelta(days=7)
    if expiry > start_date + pd.Timedelta(days=max_days):
        expiry -= pd.Timedelta(days=7)

    return expiry.date()

# -------------------------------------------------------------------
# Indicators
# -------------------------------------------------------------------
def add_indicators(df):
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_H"] = df["MACD"] - df["Signal"]

    df["Vol_Z"] = (
        df["Volume"] - df["Volume"].rolling(30).mean()
    ) / df["Volume"].rolling(30).std()

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    return df

# -------------------------------------------------------------------
# Market structure (scalar-safe)
# -------------------------------------------------------------------
def detect_structure(df):
    swing_highs = df["High"].rolling(3).max().shift(1)
    swing_lows = df["Low"].rolling(3).min().shift(1)

    last_close = float(df["Close"].iloc[-1])
    last_high = float(df["High"].iloc[-1])
    last_low = float(df["Low"].iloc[-1])

    prev_sh = float(swing_highs.iloc[-2])
    prev_sl = float(swing_lows.iloc[-2])

    higher_low = bool(last_low > prev_sl)
    higher_high = bool(last_high > prev_sh)
    breakout = bool(last_close > prev_sh)

    return higher_low, higher_high, breakout, prev_sh

# -------------------------------------------------------------------
# Options flow signal
# -------------------------------------------------------------------
def options_flow_score(ticker):
    try:
        chain = yf.Ticker(ticker).option_chain()
        calls, puts = chain.calls, chain.puts

        call_vol = float(calls["volume"].sum())
        put_vol = float(puts["volume"].sum())
        call_oi = float(calls["openInterest"].sum())

        if call_oi <= 0:
            return 1.0

        cp_ratio = call_vol / (put_vol + 1)
        vo_ratio = call_vol / call_oi

        score = 1.0
        if cp_ratio > 1.2: score *= 1.2
        if vo_ratio > 0.8: score *= 1.2
        if call_vol > put_vol * 2: score *= 1.2

        return min(score, 1.5)

    except:
        return 1.0

# -------------------------------------------------------------------
# Strike selection
# -------------------------------------------------------------------
def choose_strike(ticker, close, atr, breakout_level):
    close = float(close)
    atr = float(atr)
    breakout_level = float(breakout_level)

    expected = close + ATR_MULTIPLIER * atr
    raw_target = max(expected, breakout_level)

    upper = raw_target * (1 + MAX_OTM_PERCENT)
    inc = STRIKE_INCREMENTS[ticker]

    return ceil(upper / inc) * inc

# -------------------------------------------------------------------
# Confidence scoring
# -------------------------------------------------------------------
def compute_confidence(trend, momentum, volume_ok, structure_ok, optflow):
    tr = 1.0 if bool(trend) else 0.5
    mo = max(0.1, min(1.0, momentum + 0.5))
    vo = 1.0 if bool(volume_ok) else 0.5
    st = 1.0 if bool(structure_ok) else 0.5
    op = float(optflow)

    score = tr * mo * vo * st * op
    return min(100, round(score * 20, 2))

# -------------------------------------------------------------------
# MAIN SCAN ENGINE
# -------------------------------------------------------------------
def scan():
    rows = []

    print("\n=== QUANT STOCK SCANNER (4H) ===")
    print("Scan Time:", datetime.now(), "\n")

    for ticker in TICKERS:
        print(f"Scanning {ticker}...")

        df = yf.download(
            ticker, period=LOOKBACK, interval=INTERVAL,
            auto_adjust=True, progress=False
        )

        if df.empty:
            print(" ! No data\n")
            continue

        df = add_indicators(df)
        df.dropna(inplace=True)

        last = df.iloc[-1]

        close = float(last["Close"])
        atr = float(last["ATR"])
        rsi = float(last["RSI"])
        momentum = float(last["MACD_H"])
        volz = float(last["Vol_Z"])
        ema20 = float(last["EMA20"])
        ema50 = float(last["EMA50"])

        hl, hh, breakout, breakout_level = detect_structure(df)

        trend = bool(ema20 > ema50)
        volume_ok = bool(volz > VOL_Z_THRESHOLD)
        structure_ok = bool(hl and (hh or breakout))
        optflow = options_flow_score(ticker)

        strike = choose_strike(ticker, close, atr, breakout_level)
        expiry = choose_expiry()

        confidence = compute_confidence(
            trend, momentum, volume_ok, structure_ok, optflow
        )

        cc = colorize_conf(confidence)
        print(f"{ticker}: {cc}/100 | Close {close:.2f} | Strike {strike} | Exp {expiry}")

        rows.append({
            "Ticker": ticker,
            "Close": close,
            "EMA20>EMA50": trend,
            "RSI": rsi,
            "MACD_H": momentum,
            "Vol_Z": volz,
            "ATR": atr,
            "Higher Low": hl,
            "Higher High": hh,
            "Breakout": breakout,
            "Breakout Level": breakout_level,
            "Chosen Strike": strike,
            "Expiry": expiry,
            "Options Flow Score": optflow,
            "Confidence": confidence
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(CSV_OUTPUT, index=False)

    print(f"\nSaved results → {CSV_OUTPUT}")

    df_ranked = df_out.sort_values("Confidence", ascending=False).head(3)

    print("\n=== TOP 3 PICKS ===")
    for _, row in df_ranked.iterrows():
        print(f"{row['Ticker']}: {colorize_conf(row['Confidence'])}/100 | "
              f"Strike {row['Chosen Strike']} | Exp {row['Expiry']}")

    return df_out


if __name__ == "__main__":
    scan()
