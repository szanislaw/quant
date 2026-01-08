# mac-newsfingptquantcfhost.py
# ---------------------------------------------
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
import re
from gnews import GNews
from textblob import TextBlob
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(page_title="Options Signal Dashboard", layout="wide")
st.title("📈 Options Signal Dashboard + finance-LLM Commentary")

# ============================================================
# MODEL LOADING (Transformers, finance-LLM)
# ============================================================
@st.cache_resource
def load_finance_llm():
    model_name = "AdaptLLM/finance-LLM"
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "mps" else torch.float32,
        device_map={"": device},
    )

    return model, tokenizer, device

model, tokenizer, device = load_finance_llm()


# ============================================================
# CONFIG
# ============================================================
TICKERS = [
    "NVDA", "AMD", "GOOGL", "META", "NBIS", "CRWV",
    "AMZN", "PLTR", "AAPL", "MSFT", "INTC", "QCOM", "IBM", "ORCL"
]
TARGET_DTE = 10


# ============================================================
# NEWS SENTIMENT
# ============================================================
def fetch_news_sentiment(ticker, max_items=5):
    google_news = GNews(language="en", max_results=max_items, period="7d")
    articles = google_news.get_news(ticker)
    sentiments = []
    for article in articles:
        title = article.get("title", "")
        if not title:
            continue
        polarity = TextBlob(title).sentiment.polarity
        sentiments.append(polarity)

    if not sentiments:
        return 0

    return int(np.mean(sentiments) * 20)


# ============================================================
# QUANT CONFIDENCE SCORE
# ============================================================
def compute_confidence(latest, ticker):
    score = 50

    # RSI
    if 55 <= latest["RSI"] <= 70:
        score += 20
    elif latest["RSI"] > 70:
        score -= 10
    elif latest["RSI"] < 40:
        score -= 20

    # SMA trend
    if latest["Close"] > latest["SMA10"]:
        score += 15
    else:
        score -= 10

    # MACD momentum
    if latest["MACDhist"] > 0:
        score += 15
    else:
        score -= 10

    # Keltner channels
    if latest["Close"] > latest["KC_Upper"]:
        score += 10
    elif latest["Close"] < latest["KC_Lower"]:
        score -= 10

    # News sentiment adjustment
    news_adj = 0
    if ticker in ["NVDA", "AMD"]:
        news_adj = fetch_news_sentiment(ticker)
        score += news_adj

    return max(0, min(100, score)), news_adj


# ============================================================
# QUANT EXIT LOGIC
# ============================================================
def quant_exit_logic(entry_price, option_price, expiry_date, latest):
    today = datetime.today().date()
    dte = (expiry_date - today).days
    profit_mult = option_price / entry_price if entry_price > 0 else 1.0

    reasons, decision = [], "HOLD"

    if profit_mult >= 2.5:
        decision = "SELL"
        reasons.append("Profit target reached (≥2.5x).")

    elif latest["Close"] < latest["SMA10"]:
        decision = "SELL"
        reasons.append("Broke below SMA10 support.")

    elif dte <= 5:
        decision = "SELL"
        reasons.append("Contract near expiry (≤ 5 DTE).")

    if not reasons:
        reasons.append("Maintain position — no exit triggers fired.")

    return decision, reasons, dte, profit_mult


# ============================================================
# LLM COMMENTARY (Transformers)
# ============================================================
def llm_commentary(model, tokenizer, device, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with st.spinner("🧠 Generating commentary from finance-LLM..."):
        output = model.generate(
            **inputs,
            max_new_tokens=800,
            temperature=0.45,
            top_p=0.9
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


# ============================================================
# TECHNICAL SIGNALS
# ============================================================
def signal_filter(df):
    df["SMA10"] = df["Close"].rolling(10).mean()
    df["High20"] = df["High"].rolling(20).max().shift(1)

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACDsig"] = df["MACD"].ewm(span=9).mean()
    df["MACDhist"] = df["MACD"] - df["MACDsig"]

    df["ATR14"] = df["Close"].diff().abs().rolling(14).mean()

    ema20 = df["Close"].ewm(span=20).mean()
    df["KC_Mid"] = ema20
    df["KC_Upper"] = ema20 + df["ATR14"] * 2
    df["KC_Lower"] = ema20 - df["ATR14"] * 2

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["VAMS"] = (df["Close"] - df["SMA20"]) / df["ATR14"]

    breakout = (df["Close"] > df["SMA10"]) | (df["Close"] > df["High20"])
    momentum = (df["Close"].pct_change() > 0.015) & (df["RSI"] > 50)
    macd_flip = df["MACDhist"] > 0
    vams_signal = df["VAMS"] > 2

    df["Signal"] = (
        breakout.astype(int) +
        momentum.astype(int) +
        macd_flip.astype(int) +
        vams_signal.astype(int)
    ) >= 2

    return df


# ============================================================
# MAIN UI
# ============================================================
st.header("📊 Quant Confidence Dashboard (Phase 1)")

quant_data = {}
progress = st.progress(0)

for i, ticker in enumerate(TICKERS):
    try:
        df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=True)
        if df.empty:
            continue

        df = signal_filter(df)
        latest = df.iloc[-1]

        confidence, news_adj = compute_confidence(latest, ticker)

        quant_data[ticker] = {
            "df": df,
            "latest": latest,
            "confidence": confidence,
            "news_adj": news_adj,
            "signal": latest["Signal"]
        }

    except Exception as e:
        quant_data[ticker] = {"error": str(e)}

    progress.progress((i + 1) / len(TICKERS))

progress.empty()

rows = []
for t, v in quant_data.items():
    if "error" in v:
        rows.append([t, "⚠️", "-", "-", "-", v["error"]])
    else:
        l = v["latest"]
        rows.append([
            t,
            f"{l['Close']:.2f}",
            f"{l['RSI']:.1f}",
            f"{l['MACDhist']:.2f}",
            f"{v['confidence']}/100",
            "✅" if v["signal"] else "❌",
        ])

st.dataframe(pd.DataFrame(rows, columns=["Ticker", "Close", "RSI", "MACDhist", "Confidence", "Signal"]), use_container_width=True)
st.divider()

# ============================================================
# PHASE 2 — LLM Commentary
# ============================================================
st.header("🧠 finance-LLM Commentary (Phase 2)")

for ticker, info in quant_data.items():

    if not info.get("signal"):
        continue

    latest = info["latest"]
    confidence = info["confidence"]
    news_adj = info["news_adj"]
    df = info["df"]

    st.subheader(f"🚨 Signal Triggered: {ticker}")
    st.metric("Quant Confidence Score", f"{confidence}/100")

    # ---------------------------------------------------------
    # Select nearest ~10 DTE call contract
    # ---------------------------------------------------------
    tkr = yf.Ticker(ticker)
    expiries = tkr.options
    today = datetime.today().date()

    future_exp = [e for e in expiries if datetime.strptime(e, "%Y-%m-%d").date() > today]
    if not future_exp:
        st.warning(f"No future expiries for {ticker}.")
        continue

    expiry_dates = [datetime.strptime(e, "%Y-%m-%d").date() for e in future_exp]
    best_expiry = min(expiry_dates, key=lambda d: abs((d - today).days - TARGET_DTE))
    expiry = best_expiry.strftime("%Y-%m-%d")

    chain = tkr.option_chain(expiry)
    calls = chain.calls.dropna(subset=["strike", "lastPrice"])
    calls["dist"] = (calls["strike"] - latest["Close"] * 1.05).abs()

    sel = calls.sort_values("dist").iloc[0]

    strike = float(sel["strike"])
    option_price = float(sel["lastPrice"])
    volume = int(sel.get("volume", 0))
    oi = int(sel.get("openInterest", 0))

    st.markdown(f"""
### 📊 Suggested Contract  
**{ticker} {expiry} {strike}C — ${option_price:.2f}**  
- Volume: **{volume:,}**  
- Open Interest: **{oi:,}**  
- Confidence: **{confidence}/100**  
""")

    # ---------------------------------------------------------
    # Quant exit logic
    # ---------------------------------------------------------
    decision, reasons, dte, profit_mult = quant_exit_logic(option_price, option_price, best_expiry, latest)
    st.write(f"**Quant Decision:** {decision} — {', '.join(reasons)}")

    # ---------------------------------------------------------
    # Build LLM prompt
    # ---------------------------------------------------------
    prompt = f"""
You are a senior options strategist at a hedge fund.

Ticker: {ticker}
Close: {latest['Close']:.2f}
RSI: {latest['RSI']:.2f}
SMA10: {latest['SMA10']:.2f}
SMA20: {latest['SMA20']:.2f}
MACDhist: {latest['MACDhist']:.2f}
News Adjustment: {news_adj:+d}
Confidence Score: {confidence}/100
Suggested contract: {ticker} {expiry} {strike}C for ${option_price:.2f}
Time to expiry (DTE): {dte}
Quant Decision: {decision}
Reasons: {", ".join(reasons)}

Write the following:
1. Restate and interpret the quant’s signal.
2. Explain the technical indicators in professional financial language.
3. Highlight bullish and bearish factors.
4. Evaluate risk scenario and expected move.
5. End with one explicit rating:
   - "Proceed" (bullish)
   - "Wait" (neutral)
   - "Avoid" (bearish)
"""

    # ---------------------------------------------------------
    # Generate commentary
    # ---------------------------------------------------------
    commentary = llm_commentary(model, tokenizer, device, prompt)

    st.markdown("### 💬 LLM Commentary")
    st.markdown(commentary)
    st.divider()
