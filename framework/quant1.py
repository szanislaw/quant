import yfinance as yf

# Example: get NVDA option chain
nvda = yf.Ticker("NVDA")
options_dates = nvda.options
chain = nvda.option_chain(options_dates[0])
calls, puts = chain.calls, chain.puts
print(calls.head())
