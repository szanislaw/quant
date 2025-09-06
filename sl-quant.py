import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

# ----------------------------
# CONFIG
# ----------------------------
TRADING_SLEEVE = 7000
RISK_PCT = 0.20       # 20% of sleeve per trade
MAX_RISK = 5000       # absolute cap
LOSS_CAP = 3000       # extra safety cap
TARGET_DTE = 10       # "best expiry" = closest to 10 days out
TICKERS = ["NVDA", "TSLA", "AMD", "META", "GOOGL", "AMZN", "AAPL", "MSFT", "NFLX", "INTC"]

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

    df["Signal"] = (breakout.astype(int) + momentum.astype(int) + macd_flip.astype(int)) >= 2
    return df

# ----------------------------
# STREAMLIT APP
# ----------------------------
st.set_page_config(page_title="Options Signal Dashboard", layout="wide")
st.title("📈 Options Signal Dashboard (with Advice)")

# Sidebar: interval + capital
st.sidebar.header("Settings")
interval = st.sidebar.selectbox(
    "Select Interval",
    options=["1d", "1h", "15m", "5m"],
    index=0,
    help="Choose candlestick timeframe"
)

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
    if interval == "1d":
        period = "6mo"
    elif interval == "1h":
        period = "2mo"
    else:
        period = "5d"

    df = yf.download(ticker, period=period, interval=interval)
    df = signal_filter(df)
    latest = df.iloc[-1]

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"{ticker} — Close: {latest['Close']:.2f}")

        # Candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Candlestick"
        )])

        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA10"],
            mode="lines", name="SMA10", line=dict(color="blue")
        ))

        fig.update_layout(
            title=f"{ticker} Candlestick Chart ({interval})",
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if latest["Signal"]:
            st.success(f"🚨 Trade Signal Triggered at {latest['Close']:.2f}")

            try:
                tkr = yf.Ticker(ticker)
                expiries = tkr.options
                today = datetime.today().date()

                # Filter out expired expiries
                valid_expiries = [
                    e for e in expiries
                    if datetime.strptime(e, "%Y-%m-%d").date() > today
                ]

                if valid_expiries:
                    # Convert to datetime for scoring
                    expiry_dates = [datetime.strptime(e, "%Y-%m-%d").date() for e in valid_expiries]
                    # Pick expiry closest to TARGET_DTE (10 days)
                    best_expiry = min(expiry_dates, key=lambda d: abs((d - today).days - TARGET_DTE))
                    expiry = best_expiry.strftime("%Y-%m-%d")

                    # Dropdown override
                    expiry = st.selectbox(
                        f"Select Expiry for {ticker} (best ≈ {TARGET_DTE}DTE → {expiry})",
                        options=valid_expiries,
                        index=valid_expiries.index(expiry),
                        key=f"expiry-{ticker}"
                    )

                    chain = tkr.option_chain(expiry)
                    calls = chain.calls

                    # Pick ~5% OTM strike
                    spot = latest["Close"]
                    target_strike = spot * 1.05
                    calls["dist"] = (calls["strike"] - target_strike).abs()
                    selected_call = calls.sort_values("dist").iloc[0]

                    option_price = float(selected_call["lastPrice"])
                    strike = float(selected_call["strike"])

                    st.info(f"📊 Suggested Contract: {ticker} {expiry} {strike}C @ ${option_price:.2f}")

                    # Advice block
                    risk_amount = min(st.session_state.sleeve * RISK_PCT, MAX_RISK, LOSS_CAP)
                    contract_cost = option_price * 100
                    advice_msg = ""

                    if st.session_state.sleeve >= contract_cost:
                        qty = max(1, int(risk_amount / contract_cost))
                        trade_cost = contract_cost * qty

                        if trade_cost > risk_amount:
                            advice_msg += f"⚠️ Size Warning: Trade cost ${trade_cost:.2f} > allowed risk ${risk_amount:.2f}\n"

                        advice_msg += f"✅ Advice: Buy {qty} contract(s) of {ticker} {expiry} {strike}C.\n"
                        advice_msg += f"🎯 Exit at +200–300%, cut if price closes below SMA10 ({latest['SMA10']:.2f})."
                    else:
                        advice_msg += f"❌ Skip: Sleeve too small for even 1 contract (need ${contract_cost:.2f})."

                    st.markdown(advice_msg)

                else:
                    st.warning("No valid expiries available (all expired).")

            except Exception as e:
                st.error(f"Option chain fetch failed: {e}")
        else:
            st.info("No signal today.")

# ----------------------------
# TRADE HISTORY
# ----------------------------
if st.session_state.trades:
    st.subheader("Trade History")
    st.dataframe(pd.DataFrame(st.session_state.trades))
