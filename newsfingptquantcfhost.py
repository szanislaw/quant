import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
import re
from gnews import GNews
from textblob import TextBlob
import random

# ----------------------------
# STREAMLIT HOST CONFIG
# ----------------------------
st.set_page_config(page_title="Options Signal Dashboard", layout="wide")

# ----------------------------
# CONFIG
# ----------------------------
TRADING_SLEEVE = 7000
RISK_PCT = 0.20
MAX_RISK = 5000
LOSS_CAP = 3000
TARGET_DTE = 10
REFRESH_INTERVAL = 15 * 60
TICKERS = ["NVDA", "AMD", "GOOGL", "META", "NBIS", "CRWV", "AMZN", "PLTR",
           "AAPL", "MSFT", "INTC", "QCOM", "IBM", "ORCL"]

# ----------------------------
# LOAD FIN-O1-8B
# ----------------------------
@st.cache_resource
def load_fino1():
    model_name = "TheFinAI/Fin-o1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True
    )
    return model, tokenizer

model, tokenizer = load_fino1()

# ----------------------------
# SENTIMENT ANALYSIS
# ----------------------------
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

# ----------------------------
# CONFIDENCE SCORE
# ----------------------------
def compute_confidence(latest, ticker):
    score = 50
    if 55 <= latest["RSI"] <= 70:
        score += 20
    elif latest["RSI"] > 70:
        score -= 10
    elif latest["RSI"] < 40:
        score -= 20
    if latest["Close"] > latest["SMA10"]:
        score += 15
    else:
        score -= 10
    if latest["MACDhist"] > 0:
        score += 15
    else:
        score -= 10
    if latest["Close"] > latest["KC_Upper"]:
        score += 10
    elif latest["Close"] < latest["KC_Lower"]:
        score -= 10
    news_adjust = 0
    if ticker in ["NVDA", "AMD"]:
        news_adjust = fetch_news_sentiment(ticker)
        score += news_adjust
    return max(0, min(100, score)), news_adjust

# ----------------------------
# QUANT EXIT LOGIC
# ----------------------------
def quant_exit_logic(entry_price, option_price, expiry_date, latest):
    today = datetime.today().date()
    dte = (expiry_date - today).days
    profit_mult = option_price / entry_price if entry_price > 0 else 1.0
    reasons, decision = [], "HOLD"
    if profit_mult >= 2.5:
        decision = "SELL"; reasons.append("Profit target reached (≥ 2.5x).")
    elif latest["Close"] < latest["SMA10"]:
        decision = "SELL"; reasons.append("Close fell below SMA10 support.")
    elif dte <= 5:
        decision = "SELL"; reasons.append("Contract near expiry (≤ 5 DTE).")
    if not reasons:
        reasons.append("No exit triggers hit; maintain position.")
    return decision, reasons, dte, profit_mult

# ----------------------------
# LLM COMMENTARY
# ----------------------------
def llm_commentary(model, tokenizer, ticker, latest, confidence, news_adjust,
                   decision, reasons, dte, profit_mult):
    signal_context = f"""
    Ticker: {ticker}
    Close: {latest['Close']:.2f}
    RSI: {latest['RSI']:.2f}
    SMA10: {latest['SMA10']:.2f}
    SMA20: {latest['SMA20']:.2f}
    MACDhist: {latest['MACDhist']:.2f}
    Confidence: {confidence}/100
    News Adjustment: {news_adjust:+d}
    Decision: {decision}
    Reasons: {', '.join(reasons)}
    DTE: {dte}
    Profit Multiple: {profit_mult:.2f}
    """
    system_prompt = """
    You are a senior options strategist at a hedge fund.
    Step 1 — Restate the quant decision.
    Step 2 — Explain technical reasoning.
    Step 3 — Discuss bullish vs bearish factors.
    Step 4 — Conclude explicitly:
        ✅ Proceed
        ⚠️ Wait
        ❌ Avoid
    Step 5 — End with one of these lines:
        ✅ LLM agrees with Quant — Enter trade now.
        ⚠️ LLM suggests caution — Wait or scale in.
        ❌ LLM disagrees with Quant — Avoid entry.
    """
    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": signal_context.strip()}
    ]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

    status = st.empty()
    progress = st.progress(0)
    for i in range(10):
        status.markdown(f"🧠 Running Fin-o1-8B reasoning {'.' * (i % 3)}")
        progress.progress((i + 1) / 10)
        time.sleep(0.4)
    outputs = model.generate(inputs, max_new_tokens=1500, temperature=0.7, top_p=0.9)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    progress.empty(); status.empty()
    match = re.search(r"(Step 1.*)", text, re.DOTALL)
    if match:
        text = match.group(1)
    st.success(f"💬 Advisory ready for {ticker}.")
    return text

# ----------------------------
# SIGNAL FILTER
# ----------------------------
def signal_filter(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
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
    df["KC_Upper"] = ema20 + (df["ATR14"] * 2)
    df["KC_Lower"] = ema20 - (df["ATR14"] * 2)
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

# ----------------------------
# APP BODY
# ----------------------------
st.title("📈 Options Signal Dashboard with Quant + Fin-o1-8B Commentary")

# ================================
# PHASE 1 — Load All Quant Scores
# ================================
st.header("📊 Quant Confidence Dashboard (Phase 1)")
quant_data = {}
progress = st.progress(0)
for i, ticker in enumerate(TICKERS):
    try:
        df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            continue
        df = signal_filter(df)
        latest = df.iloc[-1]
        confidence, news_adjust = compute_confidence(latest, ticker)
        quant_data[ticker] = {
            "df": df,
            "latest": latest,
            "confidence": confidence,
            "news_adjust": news_adjust,
            "signal": latest["Signal"]
        }
    except Exception as e:
        quant_data[ticker] = {"error": str(e)}
    progress.progress((i + 1) / len(TICKERS))
progress.empty()

summary_rows = []
for t, v in quant_data.items():
    if "error" in v:
        summary_rows.append([t, "⚠️ Error", "-", "-", "-", v["error"]])
        continue
    l = v["latest"]
    summary_rows.append([
        t,
        f"{l['Close']:.2f}",
        f"{l['RSI']:.1f}",
        f"{l['MACDhist']:.2f}",
        f"{v['confidence']}/100",
        "✅" if v["signal"] else "❌"
    ])
summary_df = pd.DataFrame(summary_rows, columns=["Ticker", "Close", "RSI", "MACDhist", "Confidence", "Signal"])
st.dataframe(summary_df, use_container_width=True)
st.divider()

# ================================
# PHASE 2 — LLM Analysis for Signals
# ================================
st.header("🧠 Fin-o1-8B Financial Commentary (Phase 2)")

for ticker, info in quant_data.items():
    if not info.get("signal"):
        continue

    latest = info["latest"]
    confidence = info["confidence"]
    news_adjust = info["news_adjust"]
    df = info["df"]

    st.subheader(f"🚨 Signal Triggered: {ticker}")
    st.metric("Quant Confidence", f"{confidence}/100")

    # Option Chain
    tkr = yf.Ticker(ticker)
    expiries = tkr.options
    today = datetime.today().date()
    valid_exp = [e for e in expiries if datetime.strptime(e, "%Y-%m-%d").date() > today]
    if not valid_exp:
        st.warning(f"No future expiries for {ticker}.")
        continue
    expiry_dates = [datetime.strptime(e, "%Y-%m-%d").date() for e in valid_exp]
    best_expiry = min(expiry_dates, key=lambda d: abs((d - today).days - TARGET_DTE))
    expiry = best_expiry.strftime("%Y-%m-%d")
    chain = tkr.option_chain(expiry)
    calls = chain.calls
    calls["dist"] = (calls["strike"] - latest["Close"] * 1.05).abs()
    calls = calls.dropna(subset=["strike", "lastPrice"])
    sel = calls.sort_values("dist").iloc[0]
    strike, option_price = float(sel["strike"]), float(sel["lastPrice"])
    volume = int(sel.get("volume", 0))
    oi = int(sel.get("openInterest", 0))

    # ----------------------------
    # SUGGESTED CONTRACT CARD
    # ----------------------------
    if confidence >= 75:
        grad = "linear-gradient(135deg,#052e16,#16a34a)"  # strong
    elif confidence >= 50:
        grad = "linear-gradient(135deg,#3b3000,#facc15)"  # medium
    else:
        grad = "linear-gradient(135deg,#2d0000,#ef4444)"  # weak

    st.markdown(f"""
    <div style="
        background:{grad};
        border-radius:16px;
        padding:18px 22px;
        margin-top:8px;margin-bottom:15px;
        box-shadow:0 4px 20px rgba(0,0,0,0.3);
        border:1px solid rgba(255,255,255,0.2);
    ">
        <h2 style="color:white;text-align:center;font-size:1.6rem;margin-bottom:6px;">
            📊 Suggested Contract
        </h2>
        <h1 style="color:#00FFAA;text-align:center;font-size:2.2rem;margin:4px 0;">
            {ticker} {expiry} {strike}C
        </h1>
        <p style="color:white;text-align:center;font-size:1.4rem;margin:0;">
            💵 <b>${option_price:.2f}</b> per contract
        </p>
        <p style="color:#cccccc;text-align:center;font-size:1.1rem;margin-top:10px;">
            📈 Volume: <b>{volume:,}</b> &nbsp;•&nbsp; 🧾 Open Interest: <b>{oi:,}</b>
        </p>
        <p style="color:white;text-align:center;font-size:1rem;margin-top:10px;">
            🎯 Confidence Level: <b>{confidence}/100</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ----------------------------
    # QUANT DECISION + LLM REVIEW
    # ----------------------------
    decision, reasons, dte, profit_mult = quant_exit_logic(
        option_price, option_price, best_expiry, latest
    )
    st.write(f"**Quant Decision:** {decision} — {', '.join(reasons)}")

    st.info("🧠 Running Fin-o1-8B Advisory Review...")
    commentary = llm_commentary(
        model, tokenizer, ticker, latest, confidence,
        news_adjust, decision, reasons, dte, profit_mult
    )
    st.markdown(commentary, unsafe_allow_html=True)
    st.divider()
