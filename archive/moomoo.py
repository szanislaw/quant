from futu import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ----------------------------
# CONFIGURATION
# ----------------------------
TRADING_SLEEVE = 7000
RISK_PCT = 0.20
MAX_RISK = 5000
LOSS_CAP = 3000
DTE = 5
LOOKBACK_DAYS = 5   # check last N days for signals

TICKERS = ["US.NVDA", "US.TSLA", "US.AMD", "US.META",
           "US.GOOGL", "US.MSFT", "US.AMZN", "US.AAPL",
           "US.PLTR", "US.SHOP"]

# ----------------------------
# CONNECT TO Moomoo OpenAPI
# ----------------------------
quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
trade_ctx = OpenSecTradeContext(filter_trdmarket=TrdMarket.US,
                                host='127.0.0.1', port=11111)

# ----------------------------
# SIGNAL GENERATOR (with debug prints)
# ----------------------------
def signal_filter(df, ticker):
    df["sma10"] = df["close"].rolling(10).mean()
    df["high20"] = df["high"].rolling(20).max().shift(1)

    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    df["dailyreturn"] = df["close"].pct_change()

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macdsig"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macdhist"] = df["macd"] - df["macdsig"]

    # Conditions
    breakout = (df["close"] > df["sma10"]) | (df["close"] > df["high20"])
    momentum = (df["dailyreturn"] > 0.015) & (df["rsi"] > 50)
    macd_flip = df["macdhist"] > 0

    conditions = breakout.astype(int) + momentum.astype(int) + macd_flip.astype(int)
    df["signal"] = conditions >= 2

    # Debug print: last 5 rows
    print(f"\n=== DEBUG SIGNALS for {ticker} ===")
    print(df.tail(5)[["close","sma10","high20","rsi","macdhist","signal"]])

    return df

# ----------------------------
# PICK OPTION CONTRACT
# ----------------------------
def pick_otm_call(ticker, spot):
    ret, expiries = quote_ctx.get_option_expiration_date(ticker)
    if ret != RET_OK or expiries.empty:
        print(f"[{ticker}] No expiries available.")
        return None
    today = datetime.today().date()
    expiry = None
    for e in expiries['strike_time'].tolist():
        e_date = datetime.strptime(e, "%Y-%m-%d").date()
        if e_date >= today + timedelta(days=DTE):
            expiry = e
            break
    if not expiry:
        print(f"[{ticker}] No expiry found for >= {DTE} days.")
        return None

    ret, chain = quote_ctx.get_option_chain(ticker, expiry_date=expiry, option_type=OptionType.CALL)
    if ret != RET_OK or chain.empty:
        print(f"[{ticker}] No option chain for expiry {expiry}.")
        return None

    target_strike = spot * 1.05
    chain["dist"] = (chain["strike_price"] - target_strike).abs()
    call = chain.sort_values("dist").iloc[0]
    print(f"[{ticker}] Selected call: strike={call['strike_price']} last_price={call['last_price']}")
    return call

# ----------------------------
# TRADE EXECUTION (PAPER, with debug)
# ----------------------------
def trade_signal(ticker, sleeve):
    start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    result = quote_ctx.request_history_kline(
        ticker,
        start=start_date,
        end=end_date,
        ktype=KLType.K_DAY,
        max_count=90
    )

    # Handle different return formats
    if isinstance(result, tuple):
        if len(result) == 2:
            ret, df = result
        elif len(result) == 3:
            ret, df, _ = result
        else:
            return sleeve, None
        if ret != RET_OK or df is None or df.empty:
            print(f"[{ticker}] No kline data.")
            return sleeve, None
        hist = df
    else:
        hist = result
        if hist is None or hist.empty:
            print(f"[{ticker}] Empty kline data.")
            return sleeve, None

    hist.rename(columns=lambda c: c.lower(), inplace=True)
    df = signal_filter(hist, ticker)

    recent = df.tail(LOOKBACK_DAYS)
    if not recent["signal"].any():
        print(f"[{ticker}] No signals in last {LOOKBACK_DAYS} days.")
        return sleeve, None

    signal_row = recent[recent["signal"]].iloc[-1]
    spot = signal_row["close"]
    print(f"[{ticker}] Signal triggered on {signal_row.name.date()} at close={spot}")

    call = pick_otm_call(ticker, spot)
    if call is None:
        return sleeve, None

    entry_price = call["last_price"]
    if entry_price <= 0:
        print(f"[{ticker}] Invalid entry price {entry_price}")
        return sleeve, None

    risk_amount = min(sleeve * RISK_PCT, MAX_RISK, LOSS_CAP)
    contracts = risk_amount / (entry_price * 100)
    if contracts < 1:
        print(f"[{ticker}] Not enough sleeve to buy contracts.")
        return sleeve, None

    code = call["code"]

    # Simulated BUY
    print(f"[{ticker}] BUY {int(contracts)} contracts of {code} at {entry_price}")
    ret, result = trade_ctx.place_order(
        price=entry_price,
        qty=int(contracts),
        code=code,
        trd_side=TrdSide.BUY,
        trd_env=TrdEnv.SIMULATE
    )
    if ret != RET_OK:
        print(f"[{ticker}] Trade failed: {result}")
        return sleeve, None

    trade_info = {
        "date": signal_row.name,
        "symbol": ticker,
        "option": code,
        "entry_price": entry_price,
        "contracts": int(contracts),
        "sleeve_before": sleeve,
    }

    sleeve -= entry_price * int(contracts) * 100
    return sleeve, trade_info

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    sleeve = TRADING_SLEEVE
    all_trades = []

    for t in TICKERS:
        sleeve, trade = trade_signal(t, sleeve)
        if trade:
            all_trades.append(trade)

    if all_trades:
        df_trades = pd.DataFrame(all_trades)
        print("\n=== Trades placed (last few days) ===")
        print(df_trades)
    else:
        print("\n=== No trades triggered in the last few days ===")

    print(f"\nRemaining sleeve: ${sleeve:,.2f}")

    quote_ctx.close()
    trade_ctx.close()
