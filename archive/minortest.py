from futu import *
quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

print("Testing kline fetch...")
ret, df, page_req_key = quote_ctx.request_history_kline(
    "US.NVDA", 
    start="2025-08-01", 
    end="2025-08-31", 
    ktype=KLType.K_DAY
)

if ret == RET_OK:
    print("SUCCESS: Got data")
    print(df.head())
else:
    print("ERROR:", df)  # df contains error message

quote_ctx.close()
