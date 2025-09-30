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

# Optional: install if missing
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
REFRESH_INTERVAL = 5 * 60 * 1000  # 5 minutes in ms
TICKERS = ["NVDA", "TSLA", "AMD", "META", "GOOGL", "AMZN", "PLTR", "AAPL", "MSFT", "INTC", "QCOM", "IBM", "ORCL", "NBIS", "CRWV"]

# ----------------------------
# LOAD LOCAL LLM (Phi-3 Mini)
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
    return pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False)

llm = load_llm()

# ----------------------------
# CONFIDENCE SCORING
# ----------------------------
def compute_confidence(latest):
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
    return max(0, min(100, score))

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
# CLEANUP FUNCTION (improved)
# ----------------------------
def clean_llm_output(raw_text: str) -> str:
    """Remove leaked system prompts and keep only the assistant’s response."""
    text = raw_text.strip()

    if "Step 1" in text:
        text = "Step 1" + text.split("Step 1", 1)[1]

    text = re.split(r"(You are a|Step 1\s+— Restate.*again)", text)[0].strip()

    bad_patterns = [
        r"Step \d+\s+—.*", 
        r"Consider historical performance.*", 
        r"Evaluate the sentiment.*",
        r"Assess the impact.*"
    ]
    for pat in bad_patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    return text.strip()

# ----------------------------
# LLM COMMENTARY (progress bar)
# ----------------------------
def llm_commentary(ticker, latest, confidence, decision, reasons, dte, profit_mult):
    signal_context = f"""
    Ticker: {ticker}
    Close: {latest['Close']:.2f}
    SMA10: {latest['SMA10']:.2f}
    RSI: {latest['RSI']:.2f}
    MACD Histogram: {latest['MACDhist']:.2f}
    Confidence Score: {confidence}/100
    Quant Decision: {decision}
    Quant Reasons: {', '.join(reasons)}
    Profit Multiple: {profit_mult:.2f}x
    Days to Expiry: {dte}
    """

    system_prompt = f"""
    You are a senior options analyst.

    Step 1 — Restate the quant decision exactly as given above.  
    Step 2 — Explain in plain English why quant made this call, based only on the provided indicators.  
    Step 3 — Provide a deeper LLM Advisory Review:
      - Identify bullish/positive factors overlooked by quant.
      - Identify bearish/negative factors overlooked by quant.
      - Explicitly say whether the LLM would make the SAME decision or a DIFFERENT decision.  
    Step 4 — If different, explain the discrepancy clearly.  
    Step 5 — End with a line:  
        - If aligned: "✅ LLM agrees with Quant."  
        - If not aligned: "⚠️ LLM sees a discrepancy with Quant."
    """

    full_prompt = f"<|user|>\n{system_prompt}\n\nSignal Data:\n{signal_context}\n<|assistant|>\n"

    progress = st.progress(0, text=f"🧠 Generating advisory for {ticker}...")
    for i in range(20):
        time.sleep(0.05)
        progress.progress((i+1)/20, text=f"🧠 Generating advisory for {ticker}...")

    outputs = llm(full_prompt, max_new_tokens=500, temperature=0.7, do_sample=True, top_p=0.9)
    progress.empty()

    text = outputs[0].get("generated_text", "").strip()
    text = clean_llm_output(text)
    discrepancy_flag = "⚠️" if "discrepancy" in text.lower() else "✅"
    return text if text else "⚠️ No commentary generated.", discrepancy_flag

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
    df.loc[:, "Signal"] = (breakout.astype(int) + momentum.astype(int) + macd_flip.astype(int)) >= 2
    return df

# ----------------------------
# STREAMLIT APP
# ----------------------------
st.set_page_config(page_title="Options Signal Dashboard", layout="wide")
st.title("📈 Options Signal Dashboard with Quant + LLM Commentary")

# Auto-refresh
if AUTORF_AVAILABLE:
    st_autorefresh(interval=REFRESH_INTERVAL, limit=None, key="refresh")

manual_refresh = st.sidebar.button("🔄 Manual Refresh")
if manual_refresh:
    st.session_state.last_refresh_time = time.time()
    st.experimental_rerun()

# Last refreshed timestamp + countdown timer
now = datetime.now().strftime("%H:%M:%S")
st.caption(f"⏱️ Last refreshed at: **{now}**")

# Track refresh time for countdown
seconds_remaining = REFRESH_INTERVAL // 1000
if "last_refresh_time" not in st.session_state:
    st.session_state.last_refresh_time = time.time()

elapsed = time.time() - st.session_state.last_refresh_time
remaining = max(0, seconds_remaining - int(elapsed))
st.markdown(f"⏳ Next auto-refresh in: **{remaining} seconds**")

if remaining == 0:
    st.session_state.last_refresh_time = time.time()

# Diagnostics
st.sidebar.header("🔧 Diagnostics")
st.sidebar.write(f"PyTorch version: {torch.__version__}")
st.sidebar.write(f"Transformers version: {__import__('transformers').__version__}")
if torch.cuda.is_available():
    st.sidebar.write(f"CUDA: {torch.version.cuda}")
    st.sidebar.write(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    st.sidebar.warning("⚠️ CUDA not available, fallback to CPU/MPS.")

# Track price deltas
if "prev_prices" not in st.session_state:
    st.session_state.prev_prices = {}

# Capital
st.sidebar.header("Capital")
if "sleeve" not in st.session_state:
    st.session_state.sleeve = TRADING_SLEEVE
if "trades" not in st.session_state:
    st.session_state.trades = []
st.sidebar.metric("Trading Sleeve", f"${st.session_state.sleeve:,.2f}")
st.sidebar.metric("Total Trades", len(st.session_state.trades))

# ----------------------------
# MAIN LOOP OVER TICKERS
# ----------------------------
for ticker in TICKERS:
    df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=True)
    df = signal_filter(df)
    latest = df.iloc[-1]

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"{ticker} — Close: {latest['Close']:.2f}")

        # Price change metric
        latest_close = latest['Close']
        prev_close = st.session_state.prev_prices.get(ticker, latest_close)
        price_change = latest_close - prev_close
        pct_change = (price_change / prev_close) * 100 if prev_close else 0
        st.metric(f"{ticker} Price", f"${latest_close:.2f}", f"{price_change:.2f} ({pct_change:.2f}%)")
        st.session_state.prev_prices[ticker] = latest_close

        # Chart
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            name="Candlestick"
        )])
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA10"], mode="lines", name="SMA10", line=dict(color="blue")))
        fig.update_layout(xaxis_rangeslider_visible=False, height=400, template="plotly_white")
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
                valid_expiries = [e for e in expiries if datetime.strptime(e, "%Y-%m-%d").date() > today]

                if not valid_expiries:
                    st.warning(f"No future expiries available for {ticker}.")
                else:
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

                        st.markdown("### 🧠 LLM Advisory Review")
                        commentary_text, discrepancy_flag = llm_commentary(
                            ticker, latest, confidence, decision, reasons, dte, profit_mult
                        )
                        st.markdown(commentary_text, unsafe_allow_html=True)

                        if discrepancy_flag == "⚠️":
                            st.warning("⚠️ Discrepancy Detected: LLM advisory does not fully align with quant rules.")
                        else:
                            st.success("✅ LLM agrees with Quant.")

                        st.download_button(
                            label="💾 Download Full Report",
                            data=commentary_text,
                            file_name=f"{ticker}_analysis.md"
                        )
                    else:
                        st.error(f"❌ Sleeve too small (need ${contract_cost:.2f} for 1 contract).")

            except Exception as e:
                st.error(f"⚠️ Option chain fetch failed for {ticker}: {e}")
        else:
            st.info("No signal today.")
