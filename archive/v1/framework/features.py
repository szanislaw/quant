import pandas as pd
import ta

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    df["macd"] = ta.trend.MACD(df["Close"]).macd()
    df["atr"] = ta.volatility.AverageTrueRange(
        df["High"], df["Low"], df["Close"]).average_true_range()
    return df.dropna()
