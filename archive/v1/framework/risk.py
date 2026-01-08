from config import TRADING

def position_size(atr):
    risk_cap = TRADING["capital"] * TRADING["risk_per_trade"]
    return max(1, int(risk_cap / atr))  # number of shares/contracts
