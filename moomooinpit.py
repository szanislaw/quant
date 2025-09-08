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
RISK_PCT = 0.20
MAX_RISK = 5000
LOSS_CAP = 3000
TARGET_DTE = 10
DEFAULT_TICKERS = ["NVDA", "TSLA", "AMD", "META", "AAPL"]

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
# OPTION PICKER
# ----------------------------
def pick_otm_call(tkr, spot, target_dte=TARGET_DTE):
    expiries = tkr.options
    today = datetime.today().date()
    valid_expiries = [
        e for e in expiries
        if datetime.strptime(e, "%Y-%m-%d").date() > today
    ]
    if not valid_expiries:
        return None

    expiry_dates = [datetime.strptime(e, "%Y-%m-%d").date() for e in valid_expiries]
    best_expiry = min(expiry_dates, key=lambda d: abs((d - today).days - target_dte))
    expiry = best_expiry.strftime("%Y-%m-%d")

    chain = tkr.option_chain(expiry).calls
    if chain.empty:
        return None

    target_strike = spot * 1.05
    chain["dist"] = (chain["strike"] - target_strike).abs()
    selected = chain.sort_values("dist").iloc[0]

    return {
        "expiry": expiry,
        "strike": float(selected["strike"]),
        "lastPrice": float(selected["lastPrice"]),
        "volume": int(selected.get("volume", 0)),
        "openInterest": int(selected.get("openInterest", 0))
    }

# ----------------------------
# LLM COMMENTARY
# ----------------------------
def llm_commentary(ticker, latest, confidence, contract, portfolio_summary):
    system_prompt = """
    You are a senior options strategist.
    Write a Markdown analysis with these exact sections:
    ## Upside Potential
    ## Risks / Downside
    ## Suggested Action
    ## Suggested Option Contract
    Be concise and realistic. Explicitly recommend BUY / SKIP / HEDGE.
    """

    signal_context = {
        "ticker": ticker,
        "close": round(latest["Close"], 2),
        "SMA10": round(latest["SMA10"], 2),
        "RSI": round(latest["RSI"], 2),
        "MACD Histogram": round(latest["MACDhist"], 2),
        "confidence": confidence,
        "suggested_contract": contract
    }

    portfolio_context = {
        "total_value": portfolio_summary.get("total_value", 0),
        "sleeve": portfolio_summary.get("sleeve", 0),
        "positions": portfolio_summary.get("positions", [])
    }

    full_prompt = f"""
    {system_prompt}

    === Current Portfolio ===
    {portfolio_context}

    === Signal Data ===
    {signal_context}
    """

    with st.spinner(f"🧠 Analyzing {ticker} with portfolio context..."):
        result = llm(full_prompt, max_new_tokens=600, do_sample=True, temperature=0.6)
        return result[0]["generated_text"].strip() if result else "⚠️ No commentary generated."

# ----------------------------
# SIGNAL GENERATOR
# ----------------------------
def signal_filter(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.copy()
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

    breakout = (df["Close"] > df["SMA10"]) & (df["Close"] > df["High20"])
    momentum = (df["Close"].pct_change() > 0.015) & (df["RSI"] > 50)
    macd_flip = df["MACDhist"] > 0
    df["Signal"] = (breakout.astype(int) + momentum.astype(int) + macd_flip.astype(int)) >= 2

    return df.dropna()

# ----------------------------
# STREAMLIT APP
# ----------------------------
st.set_page_config(page_title="Options Quant Dashboard", layout="wide")
st.title("📈 Portfolio-Aware Options Quant Dashboard")

# Portfolio Upload
uploaded_file = st.sidebar.file_uploader("📂 Upload Portfolio CSV", type=["csv"])
portfolio_summary = {"positions": [], "total_value": 0, "sleeve": 7000}

if uploaded_file is not None:
    df_portfolio = pd.read_csv(uploaded_file)
    if "Market Value" in df_portfolio.columns:
        df_portfolio["Market Value"] = (
            df_portfolio["Market Value"]
            .astype(str)
            .str.replace(",", "")
            .str.replace("$", "")
            .str.replace("SGD", "")
            .str.replace("USD", "")
            .str.replace("HKD", "")
            .str.strip()
        )
        df_portfolio["Market Value"] = pd.to_numeric(df_portfolio["Market Value"], errors="coerce")

    total_value = df_portfolio["Market Value"].sum()
    sleeve = total_value * 0.20
    portfolio_summary = {
        "positions": df_portfolio[["Symbol", "Quantity", "Market Value"]].to_dict(orient="records"),
        "total_value": total_value,
        "sleeve": sleeve
    }
    st.sidebar.metric("Total Portfolio Value", f"${total_value:,.2f}")
    st.sidebar.metric("Trading Sleeve (20%)", f"${sleeve:,.2f}")

else:
    st.sidebar.warning("⚠️ No portfolio uploaded — using default sleeve = $7,000")

# Main loop
for ticker in DEFAULT_TICKERS:
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

            tkr = yf.Ticker(ticker)
            contract = pick_otm_call(tkr, latest["Close"])
            if contract:
                st.info(f"📊 Suggested Contract: {ticker} {contract['expiry']} {contract['strike']}C @ ${contract['lastPrice']:.2f}")
                st.caption(f"Volume: {contract['volume']} | Open Interest: {contract['openInterest']}")
            else:
                st.warning("⚠️ No valid option contract found.")

            commentary_text = llm_commentary(ticker, latest, confidence, contract, portfolio_summary)
            with st.expander("📊 LLM Commentary (Phi-3 Mini)"):
                st.markdown(commentary_text, unsafe_allow_html=True)
                st.download_button(
                    label="💾 Download Full Report",
                    data=commentary_text,
                    file_name=f"{ticker}_analysis.md"
                )
        else:
            st.info("No signal today.")