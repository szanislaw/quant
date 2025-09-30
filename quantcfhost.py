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

# ----------------------------
# STREAMLIT HOST CONFIG
# ----------------------------
st.set_page_config(page_title="Options Signal Dashboard", layout="wide")

# ----------------------------
# SIMPLE PASSWORD PROTECTION
# ----------------------------
PASSWORD = os.getenv("STREAMLIT_PASSWORD", "celshawn00")  # set via env var in production

def check_password():
    """Password check before running the app."""
    def password_entered():
        if st.session_state["password"] == PASSWORD:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # clean up
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("🔒 Enter password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("🔒 Enter password", type="password", on_change=password_entered, key="password")
        st.error("❌ Wrong password")
        return False
    else:
        return True

if not check_password():
    st.stop()

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
# CLEANUP FUNCTION
# ----------------------------
def clean_llm_output(raw_text: str) -> str:
    text = raw_text.strip()
    if "Step 1" in text:
        text = "Step 1" + text.split("Step 1", 1)[1]
    text = re.split(r"(You are a|Step 1\s+— Restate.*again)", text)[0].strip()
    return text.strip()

# ----------------------------
# LLM COMMENTARY
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

    system_prompt = """
    You are a senior options analyst.

    Step 1 — Restate the quant decision exactly as given above.
    Step 2 — Explain why quant made this call.
    Step 3 — Provide deeper LLM Advisory Review:
      - Identify bullish/positive factors overlooked by quant.
      - Identify bearish/negative factors overlooked by quant.
      - Explicitly say SAME decision or DIFFERENT decision.
    Step 4 — If different, explain the discrepancy.
    Step 5 — End with line:
        ✅ LLM agrees with Quant.
        ⚠️ LLM sees a discrepancy with Quant.
    """

    full_prompt = f"<|user|>\n{system_prompt}\n\nSignal Data:\n{signal_context}\n<|assistant|>\n"

    progress = st.progress(0, text=f"🧠 Generating advisory for {ticker}...")
    for i in range(20):
        time.sleep(0.05)
        progress.progress((i+1)/20, text=f"🧠 Generating advisory for {ticker}...")

    outputs = llm(full_prompt, max_new_tokens=500, temperature=0.7, do_sample=True, top_p=0.9)
    progress.empty()

    text = outputs[0].get("generated_text", "").strip()
    return clean_llm_output(text), ("⚠️" if "discrepancy" in text.lower() else "✅")

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
# APP BODY
# ----------------------------
st.title("📈 Options Signal Dashboard with Quant + LLM Commentary")

# Auto-refresh
if AUTORF_AVAILABLE:
    st_autorefresh(interval=REFRESH_INTERVAL, limit=None, key="refresh")

manual_refresh = st.sidebar.button("🔄 Manual Refresh")
if manual_refresh:
    st.session_state.last_refresh_time = time.time()
    st.experimental_rerun()

# Last refreshed timestamp
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

# Sidebar
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
        # Enhanced chart with volume and styling
        # Enhanced chart with colored volume
        fig = go.Figure()

        # Candlestick with colors
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            name="Price",
            increasing_line_color="green",
            decreasing_line_color="red",
            showlegend=True
        ))

        # SMA10 line
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA10"],
            mode="lines", name="SMA10",
            line=dict(color="blue", width=2, dash="dot")
        ))

        # Color volume bars based on up/down
        colors = np.where(df['Close'] >= df['Open'], "rgba(0,200,0,0.5)", "rgba(200,0,0,0.5)")

        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"],
            name="Volume",
            marker=dict(color=colors),
            yaxis="y2",
            opacity=0.6
        ))

        # Layout
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
