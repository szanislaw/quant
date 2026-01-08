import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time as dtime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import re
from scipy.stats import norm

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(page_title="Options Signal Dashboard", layout="wide")

# =====================================================
# CONFIG
# =====================================================
TICKERS = [
    "NVDA", "AMD", "GOOGL", "META", "AMZN", "MSFT", "AAPL",
    "TSLA", "COIN", "SMCI",
    "ARM", "MU", "AVGO", "QCOM",
    "CRM", "SNOW", "NOW",
    "RBLX", "SHOP"
]

# =====================================================
# REFRESH BUTTON
# =====================================================
col1, col2 = st.columns([1, 5])
with col1:
    if st.button("🔄 Refresh data"):
        st.rerun()
with col2:
    st.caption(f"Last refreshed: {datetime.now().strftime('%H:%M:%S')}")

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
# BLACK–SCHOLES DELTA
# =====================================================
def bs_call_delta(S, K, T, r, iv):
    if T <= 0 or iv <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    return norm.cdf(d1)

# =====================================================
# VOLATILITY-AWARE DTE
# =====================================================
def dynamic_target_dte(latest: pd.Series) -> int | None:
    atr_pct = float(latest["ATR14"]) / float(latest["Close"])

    if atr_pct >= 0.035:
        return 7
    elif atr_pct >= 0.02:
        return 10
    elif atr_pct >= 0.015:
        return 21
    else:
        return None  # dead volatility

# =====================================================
# SCORING
# =====================================================
def stock_score(latest: pd.Series) -> int:
    s = 50
    if latest["Close"] > latest["SMA10"]:
        s += 10
    if latest["RSI"] > 50 and latest["RSI_slope"] > 0:
        s += 15
    if latest["MACDhist"] > 0 and latest["MACDhist_slope"] > 0:
        s += 15
    if latest["Close"] > latest["High20"] and latest["RelVol"] > 1.3:
        s += 20
    return min(100, max(0, s))

def timing_score(latest: pd.Series) -> int:
    s = 50
    now = datetime.now().time()

    if now < dtime(10, 0):
        s -= 10
    elif dtime(11,30) < now < dtime(13,30):
        s -= 15
    elif now > dtime(15,0):
        s += 10

    if latest["Close"] > latest["SMA10"] and latest["RelVol"] > 1.2:
        s += 15

    return min(100, max(0, s))

def expected_move(price: float, iv: float, dte: int) -> float:
    return price * iv * np.sqrt(dte / 365)

# =====================================================
# DELTA-AWARE OPTION SELECTION (ROBUST)
# =====================================================
def select_best_call(ticker: str, latest: pd.Series):
    tkr = yf.Ticker(ticker)
    today = datetime.today().date()

    target_dte = dynamic_target_dte(latest)
    if target_dte is None:
        return None

    expiries = [
        datetime.strptime(e, "%Y-%m-%d").date()
        for e in tkr.options
        if datetime.strptime(e, "%Y-%m-%d").date() > today
    ]

    if not expiries:
        return None

    expiry = min(expiries, key=lambda d: abs((d - today).days - target_dte))
    dte = (expiry - today).days

    chain = tkr.option_chain(expiry.strftime("%Y-%m-%d")).calls
    if isinstance(chain.columns, pd.MultiIndex):
        chain.columns = chain.columns.get_level_values(0)

    chain = chain.dropna(
        subset=["strike", "lastPrice", "bid", "ask", "impliedVolatility"]
    ).copy()

    # Spread + liquidity
    chain["spread"] = (chain["ask"] - chain["bid"]) / chain["lastPrice"]
    chain = chain[
        (chain["spread"] <= 0.08) &
        (chain["volume"] >= 100) &
        (chain["openInterest"] >= 500) &
        (chain["lastPrice"] >= 0.3)
    ]

    if chain.empty:
        return None

    # === DELTA HANDLING ===
    if "delta" not in chain.columns:
        T = dte / 365
        r = 0.01
        S = float(latest["Close"])

        chain["delta"] = chain.apply(
            lambda row: bs_call_delta(
                S=S,
                K=float(row["strike"]),
                T=T,
                r=r,
                iv=float(row["impliedVolatility"])
            ),
            axis=1
        )

    chain = chain.dropna(subset=["delta"])
    chain = chain[(chain["delta"] >= 0.45) & (chain["delta"] <= 0.65)]

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

    chain["delta_dist"] = abs(chain["delta"] - 0.55)
    best = chain.sort_values(
        ["delta_dist", "volume"],
        ascending=[True, False]
    ).iloc[0]

    return {
        "expiry": expiry,
        "strike": float(best["strike"]),
        "price": float(best["lastPrice"]),
        "iv": float(best["impliedVolatility"]),
        "delta": float(best["delta"]),
        "volume": int(best["volume"]),
        "oi": int(best["openInterest"]),
        "dte": dte,
        "target_dte": target_dte
    }

def option_score(opt: dict, latest: pd.Series) -> int:
    s = 50
    if opt["iv"] < 0.6:
        s += 15
    if opt["volume"] > 500:
        s += 10
    if opt["oi"] > 1000:
        s += 10
    if opt["price"] < latest["ATR14"]:
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
Close: {latest['Close']:.2f}
RSI: {latest['RSI']:.2f}
MACDhist: {latest['MACDhist']:.2f}

StockScore: {scores['stock']}
TimingScore: {scores['timing']}
OptionScore: {scores['option']}
Buyability: {scores['buyability']}

Option:
Strike: {opt['strike']}
Expiry: {opt['expiry']}
Price: {opt['price']}
IV: {opt['iv']:.2f}
Delta: {opt['delta']:.2f}
DTE: {opt['dte']}

Explain whether this call option is justified today.
End with:
✅ Proceed
⚠️ Wait
❌ Avoid
"""

    inputs = tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are a senior options strategist with 50 years of experience trading options."},
         {"role": "user", "content": prompt}],
        return_tensors="pt"
    ).to(model.device)

    out = model.generate(inputs, max_new_tokens=600)
    txt = tokenizer.decode(out[0], skip_special_tokens=True)

    m = re.search(r"(Explain.*)", txt, re.DOTALL)
    return m.group(1) if m else txt

# =====================================================
# APP
# =====================================================
st.title("📈 Options Signal Dashboard — Quant + LLM")

for ticker in TICKERS:
    df = yf.download(ticker, period="6mo", auto_adjust=True, progress=False)
    if df.empty or len(df) < 50:
        continue

    df = prepare_indicators(df)
    latest = df.iloc[-1]

    ss = stock_score(latest)
    ts = timing_score(latest)
    if ts < 40:
        continue

    opt = select_best_call(ticker, latest)
    if not opt:
        continue

    os = option_score(opt, latest)
    bs = buyability(ss, ts, os)
    if bs < 50:
        continue

    st.subheader(f"🚨 {ticker} — Buyability {bs}/100")
    st.write(f"Stock: {ss} | Timing: {ts} | Option: {os}")
    st.write(f"🎯 Vol-adjusted target DTE: {opt['target_dte']} days")

    st.markdown(f"""
**Suggested Call**
- Strike: `{opt['strike']}C`
- Expiry: `{opt['expiry']}`
- Price: `${opt['price']:.2f}`
- Delta: `{opt['delta']:.2f}`
- IV: `{opt['iv']:.2f}`
- Volume / OI: `{opt['volume']} / {opt['oi']}`
""")

    with st.expander("🧠 LLM Review"):
        st.write(llm_review(
            ticker,
            latest,
            {"stock": ss, "timing": ts, "option": os, "buyability": bs},
            opt
        ))

    st.divider()
