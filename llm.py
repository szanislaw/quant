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
        device_map="auto",   # prefer Apple Silicon GPU (MPS)
        dtype="auto"
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

llm = load_llm()

def llm_commentary(signal_context: str):
    """Generate commentary from local LLM (Phi-3 Mini)."""
    prompt = f"""
    You are a financial analyst. Based on these technical signals and option details,
    explain the potential risks and rewards in plain English.

    {signal_context}
    """
    try:
        result = llm(prompt, max_new_tokens=250, do_sample=True, temperature=0.7)
        if not result or "generated_text" not in result[0]:
            return "⚠️ No commentary generated."
        return result[0]["generated_text"]
    except Exception as e:
        return f"⚠️ LLM generation failed: {e}"

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
st.title("📈 Options Signal Dashboard with Local LLM Commentary (Phi-3 Mini)")

# Sidebar diagnostics
st.sidebar.header("🔧 Diagnostics")
st.sidebar.write(f"PyTorch version: {torch.__version__}")
st.sidebar.write(f"Transformers version: {__import__('transformers').__version__}")
st.sidebar.write(f"MPS available: {torch.backends.mps.is_available()}")
st.sidebar.write(f"MPS built: {torch.backends.mps.is_built()}")

device_status = "✅ Running on Apple Silicon GPU (MPS)" if torch.backends.mps.is_available() else "⚠️ Falling back to CPU"
st.sidebar.success(device_status if "✅" in device_status else device_status)

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

    **Contract Sizing:**  
    - At least 1 contract if sleeve can cover cost  
    - Warning if trade > allowed risk  

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

            try:
                tkr = yf.Ticker(ticker)
                expiries = tkr.options
                today = datetime.today().date()

                valid_expiries = [
                    e for e in expiries
                    if datetime.strptime(e, "%Y-%m-%d").date() > today
                ]

                if valid_expiries:
                    expiry_dates = [datetime.strptime(e, "%Y-%m-%d").date() for e in valid_expiries]
                    best_expiry = min(expiry_dates, key=lambda d: abs((d - today).days - TARGET_DTE))
                    expiry = best_expiry.strftime("%Y-%m-%d")

                    chain = tkr.option_chain(expiry)
                    calls = chain.calls

                    spot = latest["Close"]
                    target_strike = spot * 1.05
                    calls["dist"] = (calls["strike"] - target_strike).abs()
                    selected_call = calls.sort_values("dist").iloc[0]

                    option_price = float(selected_call["lastPrice"])
                    strike = float(selected_call["strike"])

                    st.info(f"📊 Suggested Contract: {ticker} {expiry} {strike}C @ ${option_price:.2f}")

                    risk_amount = min(st.session_state.sleeve * RISK_PCT, MAX_RISK, LOSS_CAP)
                    contract_cost = option_price * 100
                    if st.session_state.sleeve >= contract_cost:
                        qty = max(1, int(risk_amount / contract_cost))
                        trade_cost = contract_cost * qty
                        st.markdown(f"✅ Advice: Buy {qty} contract(s). Exit at +200–300%, cut if SMA10 breaks.")
                        if trade_cost > risk_amount:
                            st.warning(f"⚠️ Trade cost ${trade_cost:.2f} > allowed risk ${risk_amount:.2f}")
                    else:
                        st.error(f"❌ Skip: Sleeve too small (need ${contract_cost:.2f}).")

                    # LLM Commentary
                    signal_context = f"""
                    Ticker: {ticker}
                    Close: {latest['Close']:.2f}
                    SMA10: {latest['SMA10']:.2f}
                    RSI: {latest['RSI']:.2f}
                    MACD Histogram: {latest['MACDhist']:.2f}
                    Suggested Contract: {expiry} {strike}C at {option_price:.2f}
                    """
                    with st.expander("📊 LLM Commentary (Phi-3 Mini)"):
                        commentary = llm_commentary(signal_context)
                        st.write(commentary)

                else:
                    st.warning("No valid expiries available.")

            except Exception as e:
                st.error(f"Option chain fetch failed: {e}")
        else:
            st.info("No signal today.")
