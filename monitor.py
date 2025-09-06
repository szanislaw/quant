import datetime
import yfinance as yf
import pandas as pd
import ta
import os
from config import TRADING
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ----------------------------
# LLM LOADER
# ----------------------------
def load_llm():
    model_name = "microsoft/phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",   # Use GPU if available (MPS/CUDA)
        dtype="auto"
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

llm = load_llm()

def llm_commentary(ticker: str, context: str) -> str:
    system_prompt = f"""
    You are an expert options analyst.
    Analyze the latest signals for {ticker} and provide exactly:
    1. Upside Potential
    2. Risks / Downside
    3. Suggested Action
    """

    prompt = f"{system_prompt}\n\nContext:\n{context}"
    result = llm(prompt, max_new_tokens=400, temperature=0.7)
    return result[0]["generated_text"].strip()

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
def safe_series(serieslike):
    """Ensure TA output is a flat pandas Series."""
    return pd.Series(serieslike.values.ravel())

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure we don't mutate empty dataframes
    if df.empty:
        raise ValueError("No data received for ticker")

    # RSI
    rsi = ta.momentum.RSIIndicator(df["Close"]).rsi()
    df["rsi"] = pd.Series(rsi.values.ravel(), index=df.index)

    # MACD
    macd = ta.trend.MACD(df["Close"]).macd()
    df["macd"] = pd.Series(macd.values.ravel(), index=df.index)

    # ATR
    atr = ta.volatility.AverageTrueRange(
        df["High"], df["Low"], df["Close"]
    ).average_true_range()
    df["atr"] = pd.Series(atr.values.ravel(), index=df.index)

    return df.dropna()


# ----------------------------
# SIGNAL GENERATOR
# ----------------------------
def generate_signal(df: pd.DataFrame) -> str:
    latest = df.iloc[-1]
    if latest["rsi"] < 30 and latest["macd"] > 0:
        return "BUY"
    elif latest["rsi"] > 70:
        return "SELL"
    return "HOLD"

# ----------------------------
# MAIN MONITOR LOOP
# ----------------------------
def run_monitor():
    now = datetime.datetime.now()
    print(f"\n🕒 Running signals at {now}\n")

    # Ensure log directory exists
    os.makedirs("logs", exist_ok=True)
    log_file = "logs/signals_log.csv"

    # Initialize log file if not present
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("timestamp,ticker,signal,close,rsi,macd,atr,llm_commentary\n")

    for t in TRADING["watchlist"]:
        try:
            # Fetch historical data
            df = yf.download(t, period="6mo", interval="1d", auto_adjust=True, progress=False)
            df = add_features(df)
            sig = generate_signal(df)
            latest = df.iloc[-1]

            # Context block
            context = f"""
            Ticker: {t}
            Signal: {sig}
            Close: {latest['Close']:.2f}
            RSI: {latest['rsi']:.2f}
            MACD: {latest['macd']:.2f}
            ATR: {latest['atr']:.2f}
            """

            # Print raw signal
            print(f"📌 {t}: {sig}")
            print(f"   Close: {latest['Close']:.2f}")
            print(f"   RSI: {latest['rsi']:.2f} | MACD: {latest['macd']:.2f} | ATR: {latest['atr']:.2f}")

            # Generate LLM commentary
            try:
                commentary = llm_commentary(t, context)
                print("🧠 LLM Commentary:")
                print(commentary)
            except Exception as e:
                commentary = f"⚠️ LLM commentary failed: {e}"
                print(commentary)

            # Append to log
            with open(log_file, "a") as f:
                f.write(f"{now},{t},{sig},{latest['Close']:.2f},{latest['rsi']:.2f},{latest['macd']:.2f},{latest['atr']:.2f},\"{commentary.replace(',', ';')}\"\n")

            print("-" * 60)

        except Exception as e:
            print(f"⚠️ Failed to process {t}: {e}")

# ----------------------------
# ENTRY POINT
# ----------------------------
if __name__ == "__main__":
    run_monitor()
