import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ----------------------------
# CONFIG
# ----------------------------
TRADING_SLEEVE = 7000
RISK_PCT = 0.20
MAX_RISK = 5000
LOSS_CAP = 3000
TARGET_DTE = 10
TICKERS = ["NVDA", "TSLA", "AMD", "META", "GOOGL"]

# ----------------------------
# LOAD LOCAL LLM (Phi-3 Mini)
# ----------------------------
@st.cache_resource
def load_llm():
    model_name = "microsoft/phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",      # CUDA for 4070 Ti Super
        torch_dtype=torch.bfloat16
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False)

llm = load_llm()

# ----------------------------
# CONFIDENCE SCORING
# ----------------------------
def compute_confidence(latest):
    """Compute a 0–100 confidence score from signals."""
    score = 50  # base neutral

    # RSI sweet spot
    if 55 <= latest["RSI"] <= 70:
        score += 20
    elif latest["RSI"] > 70:
        score -= 10
    elif latest["RSI"] < 40:
        score -= 20

    # Price above SMA10
    if latest["Close"] > latest["SMA10"]:
        score += 15
    else:
        score -= 10

    # MACD histogram positive
    if latest["MACDhist"] > 0:
        score += 15
    else:
        score -= 10

    return max(0, min(100, score))

# ----------------------------
# LLM COMMENTARY
# ----------------------------
def llm_commentary(ticker: str, latest, confidence: int):
    """Generate structured Markdown commentary for the most recent signal of a ticker."""
    system_prompt = f"""
    You are a senior options analyst.
    Write a professional Markdown report analyzing the most recent signal for {ticker}.
    Use exactly these sections:

    ## Upside Potential
    - Explain bullish factors

    ## Risks / Downside
    - Explain bearish factors

    ## Suggested Action
    - Recommend option strategy (buy, hold, skip) with reasoning

    End the report after Suggested Action.
    Keep it concise, professional, and easy to understand.
    """

    signal_context = f"""
    Ticker: {ticker}
    Close: {latest['Close']:.2f}
    SMA10: {latest['SMA10']:.2f}
    RSI: {latest['RSI']:.2f}
    MACD Histogram: {latest['MACDhist']:.2f}
    Confidence Score: {confidence}/100
    """

    full_prompt = f"{system_prompt}\n\nSignal Data:\n{signal_context}"

    with st.spinner(f"🧠 Analyzing {ticker}..."):
        result = llm(full_prompt, max_new_tokens=800, do_sample=True, temperature=0.7)
        commentary_text = result[0]["generated_text"].strip() if result else "⚠️ No commentary generated."

    return commentary_text

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
st.title("📈 Options Signal Dashboard with LLM Commentary + Confidence Score")

# Sidebar diagnostics
st.sidebar.header("🔧 Diagnostics")
st.sidebar.write(f"PyTorch version: {torch.__version__}")
st.sidebar.write(f"Transformers version: {__import__('transformers').__version__}")
if torch.cuda.is_available():
    st.sidebar.write(f"CUDA: {torch.version.cuda}")
    st.sidebar.write(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    st.sidebar.warning("⚠️ CUDA not available, fallback to CPU/MPS.")

# Capital section
st.sidebar.header("Capital")
if "sleeve" not in st.session_state:
    st.session_state.sleeve = TRADING_SLEEVE
if "trades" not in st.session_state:
    st.session_state.trades = []
st.sidebar.metric("Trading Sleeve", f"${st.session_state.sleeve:,.2f}")
st.sidebar.metric("Total Trades", len(st.session_state.trades))

# Risk management explainer
with st.expander("📘 Risk Management Rules & Explanation"):
    st.markdown(f"""
    **Sleeve Size:** ${TRADING_SLEEVE:,.2f} allocated for options trading.  

    **Risk Per Trade:**  
    - Up to {RISK_PCT*100:.0f}% of sleeve per trade  
    - Capped at ${MAX_RISK:,} (absolute)  
    - Additional loss cap at ${LOSS_CAP:,}  

    **Exit Strategy:**  
    - 🎯 Take profits at +200–300%  
    - ❌ Cut if price closes below SMA10  

    This keeps drawdowns controlled while preserving upside.
    """)

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

        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            name="Candlestick"
        )])
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA10"], mode="lines", name="SMA10", line=dict(color="blue")))
        fig.update_layout(xaxis_rangeslider_visible=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if latest["Signal"]:
            st.success(f"🚨 Trade Signal Triggered at {latest['Close']:.2f}")

            confidence = compute_confidence(latest)
            st.metric("AI Confidence Score", f"{confidence}/100")

            try:
                # Option chain fetch
                tkr = yf.Ticker(ticker)
                expiries = tkr.options
                today = datetime.today().date()

                valid_expiries = [
                    e for e in expiries
                    if datetime.strptime(e, "%Y-%m-%d").date() > today
                ]
                if not valid_expiries:
                    st.warning(f"No future expiries available for {ticker}.")
                else:
                    # Pick expiry nearest TARGET_DTE
                    expiry_dates = [datetime.strptime(e, "%Y-%m-%d").date() for e in valid_expiries]
                    best_expiry = min(expiry_dates, key=lambda d: abs((d - today).days - TARGET_DTE))
                    expiry = best_expiry.strftime("%Y-%m-%d")

                    # Pull option chain
                    chain = tkr.option_chain(expiry)
                    calls = chain.calls
                    spot = latest["Close"]

                    # Pick 5% OTM call
                    target_strike = spot * 1.05
                    calls["dist"] = (calls["strike"] - target_strike).abs()
                    selected_call = calls.sort_values("dist").iloc[0]

                    strike = float(selected_call["strike"])
                    option_price = float(selected_call["lastPrice"])
                    volume = int(selected_call.get("volume", 0))
                    oi = int(selected_call.get("openInterest", 0))

                    st.info(f"📊 Suggested Contract: {ticker} {expiry} {strike}C @ ${option_price:.2f}")
                    st.write(f"**Volume:** {volume} | **Open Interest:** {oi}")

                    # Risk allocation
                    risk_amount = min(st.session_state.sleeve * RISK_PCT, MAX_RISK, LOSS_CAP)
                    contract_cost = option_price * 100
                    if st.session_state.sleeve >= contract_cost:
                        qty = max(1, int(risk_amount / contract_cost))
                        trade_cost = contract_cost * qty
                        st.success(f"✅ You can buy {qty} contract(s) for ~${trade_cost:,.2f}")
                    else:
                        st.error(f"❌ Sleeve too small (need ${contract_cost:.2f} for 1 contract).")

            except Exception as e:
                st.error(f"⚠️ Option chain fetch failed for {ticker}: {e}")

            # LLM commentary
            commentary_text = llm_commentary(ticker, latest, confidence)
            with st.expander("📊 LLM Commentary (Phi-3 Mini)"):
                st.markdown(commentary_text, unsafe_allow_html=True)
                st.download_button(
                    label="💾 Download Full Report",
                    data=commentary_text,
                    file_name=f"{ticker}_analysis.md"
                )
        else:
            st.info("No signal today.")