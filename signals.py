import pandas as pd
from features import add_features

def generate_signal(df: pd.DataFrame) -> str:
    df = add_features(df)
    latest = df.iloc[-1]
    if latest["rsi"] < 30 and latest["macd"] > 0:
        return "BUY"
    elif latest["rsi"] > 70:
        return "SELL"
    return "HOLD"
