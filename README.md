# Pairs Trading Analysis — Statistical Arbitrage

Interactive web application for Pairs Trading (Statistical Arbitrage) analysis between any two stocks.

Built by **Rhameyza Faiqo Susanto** as part of Quantitative Research Portfolio.

---

## Features

- **Any Stock Pair** — Input any two Yahoo Finance tickers to analyze
- **Cointegration Test** — Engle-Granger two-step method with auto-interpretation
- **OLS Hedge Ratio** — Ordinary Least Squares regression for optimal hedge ratio
- **Z-Score Signals** — Rolling Z-score with configurable window and threshold
- **Interactive Charts** — Plotly-powered visualizations with hover, zoom, and pan
- **Dark Theme** — Elegant deep blue/black design

## Tech Stack

- **Backend:** Python, Dash, Flask
- **Analysis:** statsmodels, pandas, numpy
- **Data:** yfinance (Yahoo Finance API)
- **Charts:** Plotly

## Quick Start

```bash
# Clone
git clone https://github.com/RhaVans/pairs-trading-analysis.git
cd pairs-trading-analysis

# Install
pip install -r requirements.txt

# Run
python app.py
```

Open **http://127.0.0.1:8050** in your browser.

## How to Use

1. Enter two stock tickers (e.g. `KO` and `PEP`, or `AAPL` and `MSFT`)
2. Select data period (1-5 years)
3. Adjust Z-Score window and threshold via sliders
4. Click **Run Analysis**
5. View metrics and interactive charts

## Analysis Pipeline

| Step | Method | Output |
|------|--------|--------|
| Data Fetch | `yfinance` | Adjusted close prices |
| Cointegration | Engle-Granger | p-value, t-statistic |
| Hedge Ratio | OLS Regression | Beta coefficient, R-squared |
| Spread | Price_1 - Beta * Price_2 - Alpha | Time series |
| Z-Score | Rolling (Spread - Mean) / Std | Trading signals |

## Trading Signals

- **BUY Spread** — Z-Score < -2.0 (Long Ticker 1, Short Ticker 2)
- **SELL Spread** — Z-Score > +2.0 (Short Ticker 1, Long Ticker 2)
- **NEUTRAL** — Z-Score between -2.0 and +2.0

## Files

| File | Description |
|------|-------------|
| `app.py` | Dash web application (main) |
| `run_analysis.py` | Standalone CLI script |
| `requirements.txt` | Python dependencies |

## Disclaimer

For educational and research purposes only. Not financial advice.
