import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
import os
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
REFRESH_INTERVAL = 15 * 60  # 15 minutes
TICKERS = ["NVDA", "AMD", "BITF", "NBIS", "TSLA", "CRWV", "META", "GOOGL",
           "AMZN", "PLTR", "AAPL", "MSFT", "INTC", "QCOM", "IBM", "ORCL"]

# ----------------------------
# LOAD FIN-O1-8B
# ----------------------------
@st.cache_resource
def load_fino1():
    model_name = "TheFinAI/Fin-o1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quant_config
    )
    return model, tokenizer

model, tokenizer = load_fino1()

# ----------------------------
# NEWS SENTIMENT ANALYSIS
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
    avg_sentiment = np.mean(sentiments)
    sentiment_score = int(avg_sentiment * 20)  # scale to -20 … +20
    return sentiment_score

# ----------------------------
# CONFIDENCE SCORING
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
        st.caption(f"📰 News Sentiment Adjustment for {ticker}: {news_adjust:+d}")
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
        decision = "SELL"; reasons.append("Contract too close to expiry (≤ 5 DTE).")
    if not reasons:
        reasons.append("No exit triggers hit; maintain position.")
    return decision, reasons, dte, profit_mult

# ----------------------------
# LLM COMMENTARY (Enhanced Loader)
# ----------------------------
def llm_commentary(model, tokenizer, ticker, latest, confidence, news_adjust,
                   decision, reasons, dte, profit_mult):
    signal_context = f"""
    Ticker: {ticker}
    Close: {latest['Close']:.2f}
    SMA10: {latest['SMA10']:.2f}
    SMA20: {latest['SMA20']:.2f}
    RSI: {latest['RSI']:.2f}
    MACD: {latest['MACD']:.2f}
    MACD Signal: {latest['MACDsig']:.2f}
    MACD Histogram: {latest['MACDhist']:.2f}
    ATR14: {latest['ATR14']:.2f}
    High20: {latest['High20']:.2f}
    VAMS: {latest['VAMS']:.2f}
    KC Upper: {latest['KC_Upper']:.2f}
    KC Mid: {latest['KC_Mid']:.2f}
    KC Lower: {latest['KC_Lower']:.2f}
    Confidence Score: {confidence}/100
    News Sentiment Adjustment: {news_adjust:+d}
    Quant Decision: {decision}
    Quant Reasons: {', '.join(reasons)}
    Profit Multiple: {profit_mult:.2f}x
    Days to Expiry: {dte}
    """

    system_prompt = """
    You are a senior options strategist at a hedge fund.
    Advise a trader who closes positions at 15–20% profit.
    Step 1 — Restate the quant decision.
    Step 2 — Explain technical and sentiment reasoning.
    Step 3 — Give LLM Advisory Review:
        - Bullish factors
        - Bearish factors
    Step 4 — Conclude explicitly:
        ✅ Proceed now
        ⚠️ Wait for confirmation
        ❌ Avoid entry
    Step 5 — If you disagree with quant, explain briefly.
    Step 6 — End with:
        ✅ LLM agrees with Quant — Enter trade now.
        ⚠️ LLM suggests caution — Wait or scale in.
        ❌ LLM disagrees with Quant — Avoid entry.
    """

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": signal_context.strip()}
    ]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

    # Animated, randomized loading
    status = st.empty()
    progress = st.progress(0)
    total_steps = random.randint(8, 12)
    phases = [
        "🔍 Gathering latest price action...",
        "📊 Computing moving averages & RSI...",
        "⚙️ Evaluating MACD and volatility patterns...",
        "📈 Detecting breakout and momentum signals...",
        "💡 Running quant risk filters...",
        "📰 Pulling recent market sentiment & news...",
        f"🧮 Integrating Fin-o1-8B sentiment weighting for {ticker}...",
        "🧠 Building context window for reasoning...",
        "💬 Generating strategic advisory text...",
        "📑 Reviewing consistency with quant output...",
        "✅ Finalizing formatted response..."
    ]
    random.shuffle(phases)
    phases = phases[:total_steps]

    for i, phase in enumerate(phases):
        delay = random.uniform(0.4, 1.6)
        progress.progress((i + 1) / total_steps)
        for dots in range(random.randint(2, 4)):
            status.markdown(f"**{phase}{'.' * dots}**")
            time.sleep(delay / (random.randint(2, 3)))
        time.sleep(random.uniform(0.2, 0.6))

    outputs = model.generate(inputs, max_new_tokens=1500, temperature=0.7, top_p=0.9)
    progress.progress(1.0)
    status.markdown("**🧠 Generating final insights...**")
    time.sleep(random.uniform(0.6, 1.4))
    status.empty(); progress.empty()

    text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    match = re.search(r"(Step 1.*)", text, flags=re.DOTALL)
    if match:
        text = match.group(1).strip()

    st.success(random.choice([
        f"✅ Fin-o1-8B has completed its review for **{ticker}**.",
        f"🧩 Advisory generation finished for **{ticker}**.",
        f"💬 Final strategy note for **{ticker}** ready.",
        f"🧠 Fin-o1-8B commentary delivered successfully."
    ]))
    return text, ("⚠️" if "discrepancy" in text.lower() else "✅")

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
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = (df["High"] - df["Close"].shift(1)).abs()
    df["L-PC"] = (df["Low"] - df["Close"].shift(1)).abs()
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df["ATR14"] = df["TR"].rolling(14).mean()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["VAMS"] = (df["Close"] - df["SMA20"]) / df["ATR14"]
    ema20 = df["Close"].ewm(span=20).mean()
    df["KC_Mid"] = ema20
    df["KC_Upper"] = ema20 + (df["ATR14"] * 2)
    df["KC_Lower"] = ema20 - (df["ATR14"] * 2)
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
st.title("📈 Options Signal Dashboard with Quant + Fin-o1-8B Commentary + News Sentiment")

# Auto-refresh
if "last_refresh_time" not in st.session_state:
    st.session_state.last_refresh_time = time.time()
elapsed = time.time() - st.session_state.last_refresh_time
remaining = max(0, REFRESH_INTERVAL - int(elapsed))
progress_fraction = 1 - (remaining / REFRESH_INTERVAL)
st.caption(f"⏱️ Last refreshed at: **{datetime.now().strftime('%H:%M:%S')}**")
st.progress(progress_fraction, text=f"🔄 Auto-refresh in {remaining} s")
st.markdown(f"⏳ Next auto-refresh in **{remaining} s**")

if st.sidebar.button("🔄 Manual Refresh") or remaining == 0:
    st.session_state.last_refresh_time = time.time()
    st.experimental_rerun()

# Sidebar
st.sidebar.header("Capital")
st.session_state.setdefault("sleeve", TRADING_SLEEVE)
st.session_state.setdefault("trades", [])
st.sidebar.metric("Trading Sleeve", f"${st.session_state.sleeve:,.2f}")
st.sidebar.metric("Total Trades", len(st.session_state.trades))

# ----------------------------
# MAIN LOOP
# ----------------------------
for ticker in TICKERS:
    try:
        df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=True)
        if df.empty:
            st.warning(f"No data for {ticker}. Skipping."); continue
        df = signal_filter(df)
        latest = df.iloc[-1]

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader(f"{ticker} — Close: {latest['Close']:.2f}")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"],
                name="Price", increasing_line_color="green",
                decreasing_line_color="red"
            ))
            for name, line, style in [
                ("SMA10", df["SMA10"], "dot"),
                ("KC Upper", df["KC_Upper"], "solid"),
                ("KC Mid", df["KC_Mid"], "dot"),
                ("KC Lower", df["KC_Lower"], "solid")
            ]:
                fig.add_trace(go.Scatter(x=df.index, y=line, mode="lines",
                                         name=name, line=dict(width=2, dash=style)))
            colors = np.where(df["Close"] >= df["Open"], "rgba(0,200,0,0.5)", "rgba(200,0,0,0.5)")
            fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                                 marker=dict(color=colors), yaxis="y2", opacity=0.6))
            fig.update_layout(template="plotly_white", height=500,
                              xaxis=dict(rangeslider=dict(visible=False)),
                              yaxis=dict(title="Price"),
                              yaxis2=dict(title="Volume", overlaying="y",
                                          side="right", showgrid=False),
                              legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if latest["Signal"]:
                st.success(f"🚨 Trade Signal Triggered at {latest['Close']:.2f}")
                confidence, news_adjust = compute_confidence(latest, ticker)
                st.metric("Quant Confidence Score", f"{confidence}/100")

                tkr = yf.Ticker(ticker)
                expiries = tkr.options
                today = datetime.today().date()
                valid_expiries = [e for e in expiries if datetime.strptime(e, "%Y-%m-%d").date() > today]
                if not valid_expiries:
                    st.warning(f"No future expiries for {ticker}."); continue
                expiry_dates = [datetime.strptime(e, "%Y-%m-%d").date() for e in valid_expiries]
                best_expiry = min(expiry_dates, key=lambda d: abs((d - today).days - TARGET_DTE))
                expiry = best_expiry.strftime("%Y-%m-%d")
                chain = tkr.option_chain(expiry)
                calls = chain.calls.copy()
                for c in ["lastPrice", "volume", "openInterest", "strike"]:
                    if c in calls.columns:
                        calls[c] = pd.to_numeric(calls[c], errors="coerce").fillna(0)
                spot = latest["Close"]; target_strike = spot * 1.05
                calls["dist"] = (calls["strike"] - target_strike).abs()
                calls = calls.dropna(subset=["strike", "lastPrice"])
                if calls.empty:
                    st.warning(f"No valid call contracts for {ticker}."); continue
                sel = calls.sort_values("dist").iloc[0]
                strike, option_price = float(sel["strike"]), float(sel["lastPrice"])
                volume, oi = int(sel.get("volume", 0)), int(sel.get("openInterest", 0))

                # --- Adaptive color based on confidence ---
                if confidence >= 75:
                    grad = "linear-gradient(135deg,#052e16,#16a34a)"  # green strong
                elif confidence >= 50:
                    grad = "linear-gradient(135deg,#3b3000,#facc15)"  # yellow medium
                else:
                    grad = "linear-gradient(135deg,#2d0000,#ef4444)"  # red weak

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
                    <h1 style="color:#00FFAA;text-align:center;font-size:2.2rem;margin:4px 0
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

                # --- Position sizing ---
                risk_amount = min(st.session_state.sleeve * RISK_PCT, MAX_RISK, LOSS_CAP)
                contract_cost = option_price * 100
                if st.session_state.sleeve >= contract_cost:
                    qty = max(1, int(risk_amount / contract_cost))
                    trade_cost = contract_cost * qty
                    st.success(f"✅ You can buy {qty} contract(s) for ~${trade_cost:,.2f}")

                    # --- Quant Exit Logic ---
                    decision, reasons, dte, profit_mult = quant_exit_logic(
                        option_price, option_price, best_expiry, latest
                    )
                    st.markdown("### 📊 Quant Decision")
                    st.write(f"**{decision}** — {', '.join(reasons)}")

                    # --- LLM Commentary ---
                    st.markdown("### 🧠 Fin-o1-8B Advisory Review")
                    commentary_text, discrepancy_flag = llm_commentary(
                        model, tokenizer, ticker, latest, confidence,
                        news_adjust, decision, reasons, dte, profit_mult
                    )
                    st.markdown(commentary_text, unsafe_allow_html=True)
                    st.download_button(
                        label="💾 Download Full Report",
                        data=commentary_text,
                        file_name=f"{ticker}_analysis.md"
                    )
                else:
                    st.error(f"❌ Sleeve too small (need ${contract_cost:.2f} for 1 contract).")

            else:
                st.info("No signal today.")

    except Exception as e:
        st.error(f"⚠️ Error processing {ticker}: {e}")
