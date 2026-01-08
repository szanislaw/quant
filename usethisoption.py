import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time as dtime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import re

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(page_title="Options Signal Dashboard", layout="wide")

# =====================================================
# CONFIG
# =====================================================
TARGET_DTE = 10
TICKERS = [
    # Core AI / Mega-cap
    "NVDA", "AMD", "GOOGL", "META", "AMZN", "MSFT", "AAPL",

    # High-beta momentum
    "TSLA", "COIN", "SMCI",

    # Semis & AI volatility
    "ARM", "MU", "AVGO", "QCOM",

    # Software momentum
    "CRM", "SNOW", "NOW",

    # Optional high-risk
    "RBLX", "SHOP"
]


# =====================================================
# LOAD LLM
# =====================================================
@st.cache_resource
def load_llm():
    model_name = "TheFinAI/Fin-o1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quant,
        torch_dtype=torch.float16
    )
    return model, tokenizer

model, tokenizer = load_llm()

# =====================================================
# INDICATORS
# =====================================================
def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.copy()

    df["SMA10"] = df["Close"].rolling(10).mean()
    df["SMA20"] = df["Close"].rolling(20).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)

    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI_slope"] = df["RSI"].diff(3)

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACDsig"] = df["MACD"].ewm(span=9).mean()
    df["MACDhist"] = df["MACD"] - df["MACDsig"]
    df["MACDhist_slope"] = df["MACDhist"].diff()

    df["ATR14"] = df["Close"].diff().abs().rolling(14).mean()
    df["High20"] = df["High"].rolling(20).max().shift(1)
    df["RelVol"] = df["Volume"] / df["Volume"].rolling(20).mean()

    return df.dropna().copy()

# =====================================================
# SCORES (SCALAR-SAFE)
# =====================================================
def stock_score(latest: pd.Series) -> int:
    close = float(latest["Close"])
    sma10 = float(latest["SMA10"])
    rsi = float(latest["RSI"])
    rsi_slope = float(latest["RSI_slope"])
    macd = float(latest["MACDhist"])
    macd_slope = float(latest["MACDhist_slope"])
    high20 = float(latest["High20"])
    relvol = float(latest["RelVol"])

    s = 50
    if close > sma10:
        s += 10
    if rsi > 50 and rsi_slope > 0:
        s += 15
    if macd > 0 and macd_slope > 0:
        s += 15
    if close > high20 and relvol > 1.3:
        s += 20

    return min(100, max(0, s))

def timing_score(latest: pd.Series) -> int:
    close = float(latest["Close"])
    sma10 = float(latest["SMA10"])
    relvol = float(latest["RelVol"])

    s = 50
    now = datetime.now().time()

    if now < dtime(10, 0):
        s -= 10
    elif dtime(11,30) < now < dtime(13,30):
        s -= 15
    elif now > dtime(15,0):
        s += 10

    if close > sma10 and relvol > 1.2:
        s += 15

    return min(100, max(0, s))

def expected_move(price: float, iv: float, dte: int) -> float:
    return price * iv * np.sqrt(dte / 365)

# =====================================================
# OPTION SELECTION
# =====================================================
def select_best_call(ticker: str, latest: pd.Series):
    tkr = yf.Ticker(ticker)
    today = datetime.today().date()

    expiries = []
    for e in tkr.options:
        d = datetime.strptime(e, "%Y-%m-%d").date()
        if d > today:
            expiries.append(d)

    if not expiries:
        return None

    expiry = min(expiries, key=lambda d: abs((d - today).days - TARGET_DTE))
    dte = (expiry - today).days

    chain = tkr.option_chain(expiry.strftime("%Y-%m-%d")).calls
    if isinstance(chain.columns, pd.MultiIndex):
        chain.columns = chain.columns.get_level_values(0)

    chain = chain.dropna(subset=["strike","lastPrice","bid","ask","impliedVolatility"]).copy()

    chain["spread"] = (chain["ask"] - chain["bid"]) / chain["lastPrice"]

    chain = chain[
        (chain["spread"] <= 0.08) &
        (chain["volume"] >= 100) &
        (chain["openInterest"] >= 500) &
        (chain["lastPrice"] >= 0.3)
    ]

    if chain.empty:
        return None

    price = float(latest["Close"])
    atr = float(latest["ATR14"])

    chain["exp_move"] = chain["impliedVolatility"].apply(
        lambda iv: expected_move(price, iv, dte)
    )

    chain = chain[chain["exp_move"] >= atr * 1.2]
    if chain.empty:
        return None

    chain["strike_dist"] = abs(chain["strike"] - price)
    best = chain.sort_values(
        ["strike_dist","volume"],
        ascending=[True,False]
    ).iloc[0]

    return {
        "expiry": expiry,
        "strike": float(best["strike"]),
        "price": float(best["lastPrice"]),
        "iv": float(best["impliedVolatility"]),
        "volume": int(best["volume"]),
        "oi": int(best["openInterest"]),
        "dte": dte
    }

def option_score(opt: dict, latest: pd.Series) -> int:
    atr = float(latest["ATR14"])

    s = 50
    if opt["iv"] < 0.6:
        s += 15
    if opt["volume"] > 500:
        s += 10
    if opt["oi"] > 1000:
        s += 10
    if opt["price"] < atr:
        s += 10

    return min(100, s)

def buyability(stock: int, timing: int, option: int) -> int:
    return int(0.4 * stock + 0.3 * timing + 0.3 * option)

# =====================================================
# LLM REVIEW
# =====================================================
def llm_review(ticker, latest, scores, opt):
    prompt = f"""
Ticker: {ticker}
Close: {float(latest['Close']):.2f}
RSI: {float(latest['RSI']):.2f}
MACDhist: {float(latest['MACDhist']):.2f}

StockScore: {scores['stock']}
TimingScore: {scores['timing']}
OptionScore: {scores['option']}
Buyability: {scores['buyability']}

Option:
Strike: {opt['strike']}
Expiry: {opt['expiry']}
Price: {opt['price']}
IV: {opt['iv']:.2f}
DTE: {opt['dte']}

Explain whether this call option is justified today.
End with one of:
✅ Proceed
⚠️ Wait
❌ Avoid
"""

    messages = [
        {"role": "system", "content": "You are a senior options strategist."},
        {"role": "user", "content": prompt}
    ]

    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    out = model.generate(inputs, max_new_tokens=700)
    txt = tokenizer.decode(out[0], skip_special_tokens=True)

    m = re.search(r"(Explain.*)", txt, re.DOTALL)
    return m.group(1) if m else txt

# =====================================================
# APP
# =====================================================
st.title("📈 Options Signal Dashboard — Quant + LLM")

for ticker in TICKERS:
    df = yf.download(ticker, period="6mo", auto_adjust=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty or len(df) < 50:
        continue

    df = prepare_indicators(df)
    latest = df.iloc[-1].copy()

    ss = stock_score(latest)
    ts = timing_score(latest)

    if ts < 40:
        st.subheader(f"{ticker} — ⏱️ Timing poor")
        continue

    opt = select_best_call(ticker, latest)
    if not opt:
        st.subheader(f"{ticker} — ❌ No suitable option")
        continue

    os = option_score(opt, latest)
    bs = buyability(ss, ts, os)

    if bs < 50:
        continue

    st.subheader(f"🚨 {ticker} — Buyability {bs}/100")
    st.write(f"Stock: {ss} | Timing: {ts} | Option: {os}")

    st.markdown(f"""
**Suggested Call**
- Strike: `{opt['strike']}C`
- Expiry: `{opt['expiry']}`
- Price: `${opt['price']:.2f}`
- IV: `{opt['iv']:.2f}`
- Volume / OI: `{opt['volume']} / {opt['oi']}`
""")

    with st.expander("🧠 LLM Review"):
        st.write(
            llm_review(
                ticker,
                latest,
                {"stock": ss, "timing": ts, "option": os, "buyability": bs},
                opt
            )
        )

    st.divider()
