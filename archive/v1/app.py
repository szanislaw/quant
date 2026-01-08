import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime

# Import the new Quant Engine
from quant_engine_v2 import (
    load_multi_tf,
    compute_indicators,
    detect_regime,
    compute_signal,
    compute_option_flow,
    select_option_contract,
    compute_confidence,
    detect_chart_breakout
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Quant Options Dashboard",
    layout="wide",
)

st.title("⚡ Quant Options Signal Dashboard (4H Engine)")
st.caption("Minimal mode — actionable signals always shown at the top")

# ============================================================
# USER INPUTS (Top Panel)
# ============================================================
tickers = st.multiselect(
    "Select tickers",
    ["NVDA", "AMD", "GOOGL", "META", "AAPL", "MSFT", "AMZN", "TSLA", "PLTR"],
    default=["NVDA", "AMD", "GOOGL"]
)

target_dte = st.slider("Target DTE", 7, 45, 20)
run_btn = st.button("🔍 Run Signal Scan")

# Divider
st.markdown("---")


# ============================================================
# MAIN LOGIC
# ============================================================
if run_btn:

    action_rows = []   # store results to show at the top
    detail_tabs = {}   # chart + breakdown tabs

    progress_bar = st.progress(0)
    step = 0

    for ticker in tickers:
        step += 1
        progress_bar.progress(step / len(tickers))

        try:
            # 1. Load Data (Daily + 4H)
            df_d, df_4h = load_multi_tf(ticker)
            df_d = compute_indicators(df_d)
            df_4h = compute_indicators(df_4h)

            # 2. Regime (Daily)
            regime = detect_regime(df_d)

            # 3. Signal (4H)
            has_signal, score, signal_dict = compute_signal(df_4h)

            # 4. Option Flow Predictive Factors
            option_flow = compute_option_flow(ticker)

            # 5. Final Confidence
            confidence = compute_confidence(regime, score, option_flow)

            # 6. Option Contract Selection
            contract = select_option_contract(ticker, target_dte)

            # 7. Prepare Action Table Row
            action_rows.append([
                ticker,
                f"{df_4h['Close'].iloc[-1]:.2f}",
                regime,
                score,
                confidence,
                f"{contract['expiry']} {contract['strike']}C" if contract else "—",
                f"${contract['price']:.2f}" if contract else "—"
            ])

            # 8. Chart Data
            breakout_df = detect_chart_breakout(df_4h)
            detail_tabs[ticker] = (breakout_df, signal_dict, option_flow, contract, confidence)

        except Exception as e:
            action_rows.append([ticker, "-", "-", "-", "-", "-", f"Error: {e}"])

    progress_bar.empty()

    # ============================================================
    # ACTION TABLE (ALWAYS AT THE TOP)
    # ============================================================
    st.subheader("📌 Immediate Actions")
    st.dataframe(pd.DataFrame(
        action_rows,
        columns=["Ticker", "Price", "Regime", "SignalScore", "Confidence", "Suggested Contract", "Contract Price"]
    ), use_container_width=True)

    st.markdown("---")


    # ============================================================
    # DETAIL SECTIONS FOR EACH TICKER
    # ============================================================
    st.subheader("🔎 Detailed Breakdown (per ticker)")

    for ticker, bundle in detail_tabs.items():
        breakout_df, signal_dict, option_flow, contract, confidence = bundle

        with st.expander(f"📘 Details for {ticker}", expanded=False):

            # --- Quant Summary ---
            st.markdown(f"### 📊 Quant Summary — {ticker}")
            st.write(f"**Confidence:** {confidence}/100")
            st.write(f"**Signals:** {signal_dict}")

            if option_flow:
                st.write("**Option Flow Factors:**")
                st.json(option_flow)

            # --- Contract ---
            if contract:
                st.markdown("### 📝 Suggested Contract")
                st.write(contract)

            # --- Chart ---
            st.markdown("### 📈 4H Breakout Structure (minimal preview)")
            st.line_chart(breakout_df[["Close", "High20"]])

    st.success("Completed.")
