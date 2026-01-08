import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# ========================================================================
# 1. DATA LOADING (Daily + 4H)
# ========================================================================
def load_multi_tf(ticker, days=120):
    """Loads daily and 4h timeframe candles."""
    df_d = yf.download(ticker, period=f"{days}d", interval="1d", auto_adjust=True)
    df_4h = yf.download(ticker, period=f"{days}d", interval="4h", auto_adjust=True)

    for df in (df_d, df_4h):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    return df_d, df_4h


# ========================================================================
# 2. INDICATOR ENGINE
# ========================================================================
def compute_indicators(df):
    df = df.copy()

    # Moving Averages
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["EMA21"] = df["Close"].ewm(span=21).mean()

    # ATR
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACDsig"] = df["MACD"].ewm(span=9).mean()
    df["MACDhist"] = df["MACD"] - df["MACDsig"]

    # VWAP for intraday only
    if "Volume" in df.columns:
        typical = (df["High"] + df["Low"] + df["Close"]) / 3
        df["VWAP"] = (typical * df["Volume"]).cumsum() / (df["Volume"].cumsum() + 1e-9)

    # Relative Volume
    df["RVOL"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)

    return df


# ========================================================================
# 3. REGIME DETECTION (Daily candles)
# ========================================================================
def detect_regime(df_d):
    last = df_d.iloc[-1]
    trend_strength = (last["EMA21"] - last["SMA50"]) / (last["ATR14"] + 1e-9)

    # Volatility compression
    bb_width = (
        df_d["Close"].rolling(20).std() /
        df_d["Close"].rolling(20).mean()
    ).iloc[-1]

    if trend_strength > 1 and bb_width < 0.03:
        return "BULL_BREAKOUT"
    if trend_strength > 0.5:
        return "UPTREND"
    if trend_strength < -0.5:
        return "DOWNTREND"
    return "CHOP"


# ========================================================================
# 4. BREAKOUT + MOMENTUM ENGINE (4h candles)
# ========================================================================
def compute_signal(df_4h):
    last = df_4h.iloc[-1]
    prev = df_4h.iloc[-2]

    signals = {}

    # VWAP reclaim
    signals["vwap_reclaim"] = (last["Close"] > last["VWAP"]) and (prev["Close"] < prev["VWAP"])

    # MACD flip bullish
    signals["macd_flip"] = last["MACDhist"] > 0

    # Bullish RSI zone
    signals["rsi_bull"] = last["RSI"] > 55

    # Breakout through 20-period high (4h)
    high20 = df_4h["High"].rolling(20).max().iloc[-2]
    signals["breakout"] = last["Close"] > high20 * 0.998

    # Volume participation
    signals["volume_expansion"] = last["RVOL"] > 1.4

    score = sum(int(v) for v in signals.values())
    return score >= 3, score, signals


# ========================================================================
# 5. OPTION FLOW ENGINE (Predictive Factors)
# ========================================================================
def compute_option_flow(ticker):
    try:
        tkr = yf.Ticker(ticker)
        expiries = tkr.options
        if not expiries:
            return None

        today = datetime.today().date()
        nearest = min(
            [e for e in expiries if datetime.strptime(e, "%Y-%m-%d").date() > today],
            key=lambda e: abs((datetime.strptime(e, "%Y-%m-%d").date() - today).days)
        )

        chain = tkr.option_chain(nearest).calls.copy()
        spot = yf.download(ticker, period="1d")["Close"].iloc[-1]

        chain["RVOL"] = chain["volume"] / (chain["volume"].rolling(20).mean() + 1e-9)
        chain["VOI"] = chain["volume"] / (chain["openInterest"] + 1)

        otm = chain[chain["strike"] > spot * 1.03]
        otm_vol = otm["volume"].sum()

        chain["IV_spike"] = chain["impliedVolatility"].pct_change()

        return {
            "call_rvol": chain["RVOL"].mean(),
            "call_voi": chain["VOI"].mean(),
            "otm_volume": otm_vol,
            "iv_spike": chain["IV_spike"].iloc[-1],
        }
    except:
        return None


# ========================================================================
# 6. DELTA-BASED OPTION SELECTOR
# ========================================================================
def select_option_contract(ticker, target_dte=20):
    """
    Select a liquidity-safe, delta-targeted call option contract.
    Ensures:
    - Only near-term expiries (7–60 DTE)
    - No LEAPS
    - NaN-safe volume, OI, IV
    """
    try:
        tkr = yf.Ticker(ticker)
        expiries = tkr.options
        if not expiries:
            return None

        today = datetime(2025, 12, 2).date()   # Override today's date

        # -----------------------------------------
        # Convert expiries to date objects
        # -----------------------------------------
        expiry_dates = [
            datetime.strptime(e, "%Y-%m-%d").date()
            for e in expiries
        ]

        # Compute DTE
        expiry_dtes = [(e, (e - today).days) for e in expiry_dates]

        # -----------------------------------------
        # Keep expiries ONLY within 7–60 DTE
        # -----------------------------------------
        valid = [e for e, dte in expiry_dtes if 7 <= dte <= 60]

        # Fallback: choose nearest expiry ABOVE today
        if not valid:
            valid = [
                min(
                    [e for e in expiry_dates if (e - today).days > 0],
                    key=lambda d: (d - today).days
                )
            ]

        # Choose expiry closest to target_dte
        expiry = min(valid, key=lambda e: abs((e - today).days - target_dte))
        expiry_str = expiry.strftime("%Y-%m-%d")

        # -----------------------------------------
        # Fetch call chain
        # -----------------------------------------
        chain = tkr.option_chain(expiry_str).calls.copy()
        if chain.empty:
            return None

        # -----------------------------------------
        # Remove rows missing key fields
        # -----------------------------------------
        chain = chain.dropna(subset=["strike", "lastPrice", "bid", "ask"], how="any")
        if chain.empty:
            return None

        # -----------------------------------------
        # Fill NaNs safely
        # -----------------------------------------
        chain["volume"] = chain["volume"].fillna(0).astype(float)
        chain["openInterest"] = chain["openInterest"].fillna(0).astype(float)
        chain["impliedVolatility"] = chain["impliedVolatility"].fillna(0).astype(float)

        # -----------------------------------------
        # Spread sanity check
        # -----------------------------------------
        chain["spread"] = (chain["ask"] - chain["bid"]) / chain["lastPrice"].replace(0, np.nan)
        chain = chain[chain["spread"] < 0.5]   # 50% limit to avoid garbage data
        if chain.empty:
            return None

        # -----------------------------------------
        # Primary selection: DELTA targeting
        # -----------------------------------------
        if "delta" in chain.columns:
            chain["delta"] = chain["delta"].fillna(0.25)
            chain["delta_diff"] = (chain["delta"] - 0.35).abs()   # target Δ ≈ 0.35
            sel = chain.sort_values("delta_diff").iloc[0]

        else:
            # Fallback: ATM
            spot = yf.download(ticker, period="1d", interval="1m")["Close"].iloc[-1]
            chain["dist"] = (chain["strike"] - spot).abs()
            sel = chain.sort_values("dist").iloc[0]

        # -----------------------------------------
        # Return clean contract info
        # -----------------------------------------
        return {
            "ticker": ticker,
            "expiry": expiry_str,
            "strike": float(sel["strike"]),
            "price": float(sel["lastPrice"]),
            "volume": int(sel["volume"]),
            "oi": int(sel["openInterest"]),
            "spread": float(sel["spread"]),
            "iv": float(sel.get("impliedVolatility", 0)),
            "delta": float(sel.get("delta", 0.0))
        }

    except Exception as e:
        return {"error": str(e)}


# ========================================================================
# 7. CONFIDENCE MODEL (Regime + 4h Momentum + Option Flow)
# ========================================================================
def compute_confidence(regime, signal_score, option_flow):
    base = 40 + signal_score * 10

    # Regime weighting
    if regime == "BULL_BREAKOUT":
        base += 20
    elif regime == "UPTREND":
        base += 10
    elif regime == "CHOP":
        base -= 5

    # Option Flow predictive adjustments
    if option_flow:
        if option_flow["call_rvol"] > 3:
            base += 8
        if option_flow["call_voi"] > 1:
            base += 8
        if option_flow["otm_volume"] > 5000:
            base += 5
        if option_flow["iv_spike"] > 0.05:
            base += 6  # IV rising → bullish

    return int(np.clip(base, 0, 100))


# ========================================================================
# 8. BREAKOUT CHARTING MARKERS (for Streamlit)
# ========================================================================
def detect_chart_breakout(df_4h):
    df = df_4h.copy()
    df["High20"] = df["High"].rolling(20).max()
    df["Breakout"] = df["Close"] > df["High20"].shift(1)
    return df
