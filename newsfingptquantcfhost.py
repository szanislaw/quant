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

# ----------------------------
# STREAMLIT HOST CONFIG
# ----------------------------
st.set_page_config(page_title="Options Signal Dashboard", layout="wide")

# ----------------------------
# Optional: install if missing
# ----------------------------
try:
    from streamlit_autorefresh import st_autorefresh
    AUTORF_AVAILABLE = True
except ImportError:
    AUTORF_AVAILABLE = False
    st.warning("⚠️ streamlit-autorefresh not installed. Run: pip install streamlit-autorefresh")

# ----------------------------
# CONFIG
# ----------------------------
TRADING_SLEEVE = 7000
RISK_PCT = 0.20
MAX_RISK = 5000
LOSS_CAP = 3000
TARGET_DTE = 10
REFRESH_INTERVAL = 15 * 60 * 1000  # 5 minutes in ms
TICKERS = ["NVDA", "TSLA", "AMD", "META", "GOOGL", "AMZN", "PLTR", "AAPL", "MSFT", "INTC", "QCOM", "IBM", "ORCL", "NBIS", "CRWV"]

# ----------------------------
# LOAD FIN-O1-8B
# ----------------------------
@st.cache_resource
def load_fino1():
    model_name = "TheFinAI/Fin-o1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",             # ✅ safe for CPU offload
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
        polarity = TextBlob(title).sentiment.polarity  # -1 → +1
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
    # RSI rules
    if 55 <= latest["RSI"] <= 70:
        score += 20
    elif latest["RSI"] > 70:
        score -= 10
    elif latest["RSI"] < 40:
        score -= 20

    # SMA10 rule
    if latest["Close"] > latest["SMA10"]:
        score += 15
    else:
        score -= 10

    # MACD rule
    if latest["MACDhist"] > 0:
        score += 15
    else:
        score -= 10

    # Keltner influence
    if latest["Close"] > latest["KC_Upper"]:
        score += 10
    elif latest["Close"] < latest["KC_Lower"]:
        score -= 10

    # News sentiment adjustment (NVDA, AMD only)
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
# LLM COMMENTARY
# ----------------------------
def llm_commentary(model, tokenizer, ticker, latest, confidence, news_adjust, decision, reasons, dte, profit_mult):
    # Include ALL quant features
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
    High20 (breakout level): {latest['High20']:.2f}
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
    You are advising a short-term options trader who always closes positions when the option gains 15–20%.

    Your task:
    Step 1 — Restate the quant decision exactly as given above.
    Step 2 — Explain why quant made this call using the indicators provided (technical + sentiment).
    Step 3 — Provide deeper LLM Advisory Review:
    - Identify bullish/positive factors that could justify an immediate buy entry.
    - Identify bearish/negative factors that could weaken or delay entry.
    Step 4 — Conclude with an explicit recommendation:
    - ✅ Proceed with immediate entry (if strong technical + sentiment alignment).
    - ⚠️ Wait for confirmation (if indicators are mixed or overextended).
    - ❌ Avoid entry (if sentiment or indicators show near-term weakness).
    Step 5 — If your decision differs from the quant, explain why briefly.
    Step 6 — End with one of these final lines:
        ✅ LLM agrees with Quant — Enter trade now.
        ⚠️ LLM suggests caution — Wait or scale in.
        ❌ LLM disagrees with Quant — Avoid entry.
    """


    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": signal_context.strip()}
    ]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

    progress = st.progress(0, text=f"🧠 Generating Fin-o1-8B advisory for {ticker}...")
    for i in range(20):
        time.sleep(0.05)
        progress.progress((i+1)/20, text=f"🧠 Generating Fin-o1-8B advisory for {ticker}...")

    outputs = model.generate(
        inputs,
        max_new_tokens=2000,
        temperature=0.7,
        top_p=0.9
    )
    progress.empty()

    text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    match = re.search(r"(Step 1.*)", text, flags=re.DOTALL)
    if match:
        text = match.group(1).strip()

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

    df.loc[:, "Signal"] = (
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

if AUTORF_AVAILABLE:
    st_autorefresh(interval=REFRESH_INTERVAL, limit=None, key="refresh")

manual_refresh = st.sidebar.button("🔄 Manual Refresh")
if manual_refresh:
    st.session_state.last_refresh_time = time.time()
    st.experimental_rerun()

now = datetime.now().strftime("%H:%M:%S")
st.caption(f"⏱️ Last refreshed at: **{now}**")

seconds_remaining = REFRESH_INTERVAL // 1000
if "last_refresh_time" not in st.session_state:
    st.session_state.last_refresh_time = time.time()
elapsed = time.time() - st.session_state.last_refresh_time
remaining = max(0, seconds_remaining - int(elapsed))
st.markdown(f"⏳ Next auto-refresh in: **{remaining} seconds**")
if remaining == 0:
    st.session_state.last_refresh_time = time.time()

st.sidebar.header("Capital")
if "sleeve" not in st.session_state:
    st.session_state.sleeve = TRADING_SLEEVE
if "trades" not in st.session_state:
    st.session_state.trades = []
st.sidebar.metric("Trading Sleeve", f"${st.session_state.sleeve:,.2f}")
st.sidebar.metric("Total Trades", len(st.session_state.trades))

# ----------------------------
# MAIN LOOP
# ----------------------------
for ticker in TICKERS:
    try:
        df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=True)
        if df.empty:
            st.warning(f"No data for {ticker}. Skipping.")
            continue

        df = signal_filter(df)
        latest = df.iloc[-1]

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader(f"{ticker} — Close: {latest['Close']:.2f}")

            fig = go.Figure()

            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'],
                name="Price",
                increasing_line_color="green",
                decreasing_line_color="red"
            ))

            # SMA10
            fig.add_trace(go.Scatter(
                x=df.index, y=df["SMA10"],
                mode="lines", name="SMA10",
                line=dict(color="blue", width=2, dash="dot")
            ))

            # Keltner Channel
            fig.add_trace(go.Scatter(
                x=df.index, y=df["KC_Upper"],
                mode="lines", name="KC Upper",
                line=dict(color="orange", width=1)
            ))
            fig.add_trace(go.Scatter(
                x=df.index, y=df["KC_Mid"],
                mode="lines", name="KC Mid",
                line=dict(color="orange", width=1, dash="dot")
            ))
            fig.add_trace(go.Scatter(
                x=df.index, y=df["KC_Lower"],
                mode="lines", name="KC Lower",
                line=dict(color="orange", width=1)
            ))

            # Volume
            colors = np.where(df['Close'] >= df['Open'], "rgba(0,200,0,0.5)", "rgba(200,0,0,0.5)")
            fig.add_trace(go.Bar(
                x=df.index, y=df["Volume"],
                name="Volume",
                marker=dict(color=colors),
                yaxis="y2",
                opacity=0.6
            ))

            fig.update_layout(
                template="plotly_white",
                height=500,
                xaxis=dict(rangeslider=dict(visible=False)),
                yaxis=dict(title="Price"),
                yaxis2=dict(
                    title="Volume",
                    overlaying="y",
                    side="right",
                    showgrid=False
                ),
                legend=dict(orientation="h", y=-0.25)
            )

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
                    st.warning(f"No future expiries available for {ticker}.")
                    continue

                expiry_dates = [datetime.strptime(e, "%Y-%m-%d").date() for e in valid_expiries]
                best_expiry = min(expiry_dates, key=lambda d: abs((d - today).days - TARGET_DTE))
                expiry = best_expiry.strftime("%Y-%m-%d")

                chain = tkr.option_chain(expiry)
                calls = chain.calls.copy()
                for col in ["lastPrice", "volume", "openInterest", "strike"]:
                    if col in calls.columns:
                        calls[col] = pd.to_numeric(calls[col], errors="coerce").fillna(0)

                spot = latest["Close"]
                target_strike = spot * 1.05
                calls["dist"] = (calls["strike"] - target_strike).abs()
                calls = calls.dropna(subset=["strike", "lastPrice"])
                if calls.empty:
                    st.warning(f"No valid call contracts available for {ticker}.")
                    continue

                selected_call = calls.sort_values("dist").iloc[0]
                strike = float(selected_call["strike"])
                option_price = float(selected_call["lastPrice"])
                volume = int(selected_call.get("volume", 0))
                oi = int(selected_call.get("openInterest", 0))

                st.info(f"📊 Suggested Contract: {ticker} {expiry} {strike}C @ ${option_price:.2f}")
                st.write(f"**Volume:** {volume} | **Open Interest:** {oi}")

                risk_amount = min(st.session_state.sleeve * RISK_PCT, MAX_RISK, LOSS_CAP)
                contract_cost = option_price * 100
                if st.session_state.sleeve >= contract_cost:
                    qty = max(1, int(risk_amount / contract_cost))
                    trade_cost = contract_cost * qty
                    st.success(f"✅ You can buy {qty} contract(s) for ~${trade_cost:,.2f}")

                    decision, reasons, dte, profit_mult = quant_exit_logic(option_price, option_price, best_expiry, latest)
                    st.markdown("### 📊 Quant Decision")
                    st.write(f"**{decision}** — {', '.join(reasons)}")

                    st.markdown("### 🧠 Fin-o1-8B Advisory Review")
                    commentary_text, discrepancy_flag = llm_commentary(
                        model, tokenizer, ticker, latest, confidence, news_adjust, decision, reasons, dte, profit_mult
                    )
                    st.markdown(commentary_text, unsafe_allow_html=True)

                    # if discrepancy_flag == "⚠️":
                    #     st.warning("⚠️ Discrepancy Detected: LLM advisory does not fully align with quant rules.")
                    # else:
                    #     st.success("✅ LLM agrees with Quant.")

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

