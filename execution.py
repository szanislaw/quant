from ib_insync import IB, Stock, MarketOrder
from config import API_KEYS

def place_trade(ticker, action, qty):
    ib = IB()
    ib.connect(API_KEYS["ibkr"]["host"], API_KEYS["ibkr"]["port"], API_KEYS["ibkr"]["clientId"])
    contract = Stock(ticker, "SMART", "USD")
    order = MarketOrder(action, qty)
    trade = ib.placeOrder(contract, order)
    print(f"{action} {qty} {ticker}")
    return trade
