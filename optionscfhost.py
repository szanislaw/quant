import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import time
import re
import os
from math import log, sqrt, exp
from scipy.stats import norm

# ----------------------------
# STREAMLIT HOST CONFIG
# ----------------------------
st.set_page_config(page_title="Options Signal Dashboard", layout="wide")

# ----------------------------
# SIMPLE PASSWORD PROTECTION
# ----------------------------
PASSWORD = os.getenv("STREAMLIT_PASSWORD", "celshawn00")

def check_password():
    def password_entered():
        if st.session_state["password"] == PASSWORD:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False
    if "password_correct" not in st.session_state:
        st.text_input("🔒 Enter password", type="password",
                      on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("🔒 Enter password", type="password",
                      on_change=password_entered, key="password")
        st.error("❌ Wrong password")
        return False
    else:
        return True

if not check_password():
    st.stop()

# ----------------------------
# CONFIG
# ----------------------------
TRADING_SLEEVE = 7000
RISK_PCT = 0.20
MAX_RISK = 5000
LOSS_CAP = 3000
TARGET_DTE = 10
TICKERS = ["NVDA", "TSLA", "AMD", "META", "GOOGL", "AMZN",
           "PLTR", "AAPL", "MSFT", "INTC", "QCOM", "IBM", "ORCL"]

# ----------------------------
# LOAD LOCAL LLM
# ----------------------------
@st.cache_resource
def load_llm():
    model_name = "microsoft/phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    return pipeline("text-generation", model=model,
                    tokenizer=tokenizer, return_full_text=False)

llm = load_llm()

# ----------------------------
# QUANT HELPERS
# ----------------------------
def compute_confidence(latest):
    score = 50
    if 55 <= latest["RSI"] <= 70: score += 20
    elif latest["RSI"] > 70: score -= 10
    elif latest["RSI"] < 40: score -= 20
    if latest["Close"] > latest["SMA10"]: score += 15
    else: score -= 10
    if latest["MACDhist"] > 0: score += 15
    else: score -= 10
    return max(0, min(100, score))

def bs_greeks(S, K, T, r, sigma, option_type="call"):
    d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    delta = norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
    return delta, gamma

def analyze_skew(chain, spot):
    otm_calls = chain.calls[chain.calls["strike"] > spot]
    otm_puts = chain.puts[chain.puts["strike"] < spot]
    if otm_calls.empty or otm_puts.empty: return None
    call_iv = otm_calls.iloc[0].get("impliedVolatility", np.nan)
    put_iv = otm_puts.iloc[-1].get("impliedVolatility", np.nan)
    if np.isnan(call_iv) or np.isnan(put_iv): return None
    return call_iv - put_iv

def compute_iv_rank(ticker, expiry, window_days=180):
    tkr = yf.Ticker(ticker)
    hist = tkr.history(period=f"{window_days}d", interval="1d")
    if "Implied Volatility" not in hist.columns: return None
    iv_hist = hist["Implied Volatility"]
    if iv_hist.empty: return None
    iv_now = iv_hist.iloc[-1]
    iv_min, iv_max = iv_hist.min(), iv_hist.max()
    iv_rank = (iv_now - iv_min) / (iv_max - iv_min)
    return iv_now, iv_rank, iv_hist

# ----------------------------
# SIGNAL GENERATOR
# ----------------------------
def signal_filter(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df["SMA10"] = df["Close"].rolling(10).mean()
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
    df = df.dropna(subset=["SMA10", "High20", "RSI", "MACDhist"])
    breakout = (df["Close"] > df["SMA10"]) | (df["Close"] > df["High20"])
    momentum = (df["Close"].pct_change() > 0.015) & (df["RSI"] > 50)
    macd_flip = df["MACDhist"] > 0
    df.loc[:, "Signal"] = (breakout.astype(int) +
                           momentum.astype(int) +
                           macd_flip.astype(int)) >= 2
    return df

# ----------------------------
# APP BODY
# ----------------------------
st.title("📈 Options Signal Dashboard with Quant + LLM Commentary")

for ticker in TICKERS:
    df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=True)
    df = signal_filter(df)
    latest = df.iloc[-1]

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"{ticker} — Close: {latest['Close']:.2f}")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            name="Price", increasing_line_color="green",
            decreasing_line_color="red"
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA10"],
            mode="lines", name="SMA10",
            line=dict(color="blue", width=2, dash="dot")
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if latest["Signal"]:
            st.success(f"🚨 Trade Signal Triggered at {latest['Close']:.2f}")
            confidence = compute_confidence(latest)
            st.metric("Quant Confidence Score", f"{confidence}/100")

            try:
                tkr = yf.Ticker(ticker)
                expiries = tkr.options
                today = datetime.today().date()
                valid_expiries = [e for e in expiries
                                  if datetime.strptime(e, "%Y-%m-%d").date() > today]
                if not valid_expiries:
                    st.warning("No future expiries available.")
                    continue

                expiry_dates = [datetime.strptime(e, "%Y-%m-%d").date() for e in valid_expiries]
                best_expiry = min(expiry_dates, key=lambda d: abs((d - today).days - TARGET_DTE))
                expiry = best_expiry.strftime("%Y-%m-%d")

                chain = tkr.option_chain(expiry)
                calls = chain.calls.copy()
                for colx in ["lastPrice", "volume", "openInterest", "strike", "impliedVolatility"]:
                    if colx in calls.columns:
                        calls[colx] = pd.to_numeric(calls[colx], errors="coerce").fillna(0)

                spot = latest["Close"]
                target_strike = spot * 1.05
                calls["dist"] = (calls["strike"] - target_strike).abs()
                calls = calls.dropna(subset=["strike", "lastPrice"])
                if calls.empty:
                    st.warning("No valid call contracts available.")
                    continue

                selected_call = calls.sort_values("dist").iloc[0]
                strike = float(selected_call["strike"])
                option_price = float(selected_call["lastPrice"])
                iv = float(selected_call.get("impliedVolatility", np.nan))
                volume = int(selected_call.get("volume", 0))
                oi = int(selected_call.get("openInterest", 0))

                # -----------------------------
                # DIAGNOSTICS EXPANDER
                # -----------------------------
                with st.expander("🔍 Options Diagnostics"):
                    # ---- IV Rank ----
                    iv_info = compute_iv_rank(ticker, expiry)
                    if iv_info:
                        iv_now, iv_rank, iv_hist = iv_info
                        st.write(f"Current IV: {iv_now:.2%}, IV Rank: {iv_rank:.2f}")
                        if iv_rank > 0.9:
                            st.error("❌ Trade blocked: IV Rank extremely high (>0.9).")
                            continue
                        elif iv_rank > 0.7:
                            st.warning("⚠️ Elevated IV Rank (0.7–0.9).")
                        st.line_chart(iv_hist)

                    # ---- Delta/Gamma Curve ----
                    if not np.isnan(iv) and iv > 0:
                        days_to_expiry = (best_expiry - today).days / 365
                        greeks_data = []
                        for _, row in calls.iterrows():
                            d, g = bs_greeks(spot, row["strike"], days_to_expiry, 0.05, row["impliedVolatility"])
                            greeks_data.append((row["strike"], d, g))
                        greeks_df = pd.DataFrame(greeks_data, columns=["Strike", "Delta", "Gamma"])
                        st.write("Delta/Gamma across strikes:")
                        st.dataframe(greeks_df.set_index("Strike"))
                        delta, gamma = bs_greeks(spot, strike, days_to_expiry, 0.05, iv)
                        st.write(f"Selected Contract → Delta: {delta:.2f}, Gamma: {gamma:.4f}")
                        if not (0.25 <= delta <= 0.5):
                            st.error("❌ Trade blocked: Delta outside 0.25–0.5.")
                            continue
                        elif not (0.3 <= delta <= 0.4):
                            st.warning("⚠️ Delta not in sweet spot (0.3–0.4).")

                    # ---- Skew ----
                    skew_val = analyze_skew(chain, spot)
                    if skew_val is not None:
                        st.write(f"Skew (Call IV − Put IV): {skew_val:.2%}")
                        if skew_val < -0.05:
                            st.error("❌ Trade blocked: Skew strongly bearish (< -0.05).")
                            continue
                        elif skew_val < 0:
                            st.warning("⚠️ Skew slightly bearish (puts > calls).")
                        skew_plot = pd.DataFrame({
                            "Strike": pd.concat([calls["strike"], chain.puts["strike"]]),
                            "IV": pd.concat([calls["impliedVolatility"], chain.puts["impliedVolatility"]]),
                            "Type": ["Call"]*len(calls) + ["Put"]*len(chain.puts)
                        })
                        st.line_chart(skew_plot.pivot(index="Strike", columns="Type", values="IV"))

                # ✅ All gates passed
                st.info(f"📊 Suggested Contract: {ticker} {expiry} {strike}C @ ${option_price:.2f}")
                st.write(f"**Volume:** {volume} | **Open Interest:** {oi}")

            except Exception as e:
                st.error(f"⚠️ Option chain fetch failed for {ticker}: {e}")
        else:
            st.info("No signal today.")
