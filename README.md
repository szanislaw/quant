# quant-analysis-beta

A live options signal dashboard that combines quantitative technical analysis with an on-device financial LLM (Fin-o1-8B) to surface trade ideas and manage risk in real time.

---

## Core File: `newsfingptquantcfhost.py`

The primary application. Run it with Streamlit to get a full dashboard.

```bash
streamlit run newsfingptquantcfhost.py
```

### What it does

The app runs in two sequential phases on every refresh:

**Phase 1 — Quant Confidence Scan**  
Pulls 3 months of daily OHLCV data for every ticker in the watchlist via `yfinance`, computes technical indicators, and scores each ticker with a 0–100 confidence score. A live summary table is rendered showing price, change, RSI, MACD histogram, and whether a signal is active.

**Phase 2 — LLM Advisory (Fin-o1-8B)**  
For every ticker that triggered a signal in Phase 1, the app selects a liquid near-the-money call option (~10 DTE, ~2–3% OTM), runs a risk gate, then sends a structured prompt to the `TheFinAI/Fin-o1-8B` model. The model returns a step-by-step hedge-fund-style advisory that either agrees, cautions, or disagrees with the quant signal.

---

### Technical Indicators

| Indicator | How it's used |
|---|---|
| RSI (14) | Scores highest between 40–55; penalizes overbought (>70) |
| SMA10 / SMA20 | Price above SMA10 is bullish; SMA10 breakout is a signal condition |
| MACD Histogram | Positive histogram required for signal |
| Keltner Channels (EMA20 ± 2×ATR14) | Breakout above upper channel adds score |
| VAMS | Standardized distance from SMA20 in ATR units; >2 adds signal strength |
| Volume Surge | Requires >1.2× 20-period average volume |

A signal fires when **3 of 4** conditions are met (breakout, momentum, MACD flip, VAMS) **and** volume confirms **and** RSI < 70.

---

### Risk Management

Handled by `PositionManager`, which persists state to `position_log.json`.

| Parameter | Value |
|---|---|
| Max open positions | 5 |
| Max risk per trade | $500 |
| Daily loss limit | $1,000 |
| Stop loss | 50% of option premium |
| Profit target | 2.5× entry |
| Min confidence to enter | 65/100 |
| Options liquidity filter | Volume ≥ 100, OI ≥ 500, spread ≤ 15% |

Position sizing is auto-calculated as `floor($500 / (price × 100))`, capped between 1 and 10 contracts.

A pre-trade market check also runs: if SPY is below its 20-day MA or VIX is above 30, entries are blocked.

---

### LLM Setup

The app loads `TheFinAI/Fin-o1-8B` once at startup using 4-bit NF4 quantization via `bitsandbytes`, mapped automatically across available devices. This requires a CUDA-capable GPU with enough VRAM (~10 GB minimum recommended).

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
```

---

### Watchlist

```
NVDA, AMD, GOOGL, META, USAR, TSM, AVGO,
LMT, RTX, NOC, KTOS, AVAV, RKLB, GD, BA,
NBIS, CRWV, AMZN, PLTR, AAPL, MSFT,
INTC, QCOM, IBM, ORCL
```

News sentiment via Google News is fetched for a subset: `NVDA, AMD, GOOGL, META, TSLA`.

---

## Other Files

| File | Purpose |
|---|---|
| `instant.py` | Lighter / faster scanning variant |
| `usethisoption.py` | Standalone option selection utility |
| `dtefix.py` | DTE calculation helpers |
| `archive/` | Older iterations and experiments |
| `archive/v1/` | First-generation framework with backtesting |

---

## Requirements

```
streamlit
yfinance
pandas
numpy
plotly
torch
transformers
bitsandbytes
gnews
textblob
```

Install with:

```bash
pip install streamlit yfinance pandas numpy plotly torch transformers bitsandbytes gnews textblob
```

A Hugging Face account and acceptance of the `TheFinAI/Fin-o1-8B` model license may be required. Authenticate with:

```bash
huggingface-cli login
```

---

## Disclaimer

This tool is for research and educational purposes only. Nothing here constitutes financial advice. Options trading involves substantial risk of loss.
