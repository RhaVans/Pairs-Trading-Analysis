# ============================================================
# PAIRS TRADING ANALYSIS — Standalone Runner
# Statistical Arbitrage: KO vs PEP
# by Rhameyza Faiqo Susanto
# ============================================================
# Run: python run_analysis.py
# ============================================================

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.4f}'.format)


# ============================================================
# STEP 1: DATA FETCHING
# ============================================================

TICKER_1 = 'KO'
TICKER_2 = 'PEP'
PERIOD   = '5y'

print(f"\nFetching data for {TICKER_1} and {TICKER_2} ({PERIOD})...")

raw_data = yf.download(
    tickers=[TICKER_1, TICKER_2],
    period=PERIOD,
    interval='1d',
    auto_adjust=True
)

prices = raw_data['Close'][[TICKER_1, TICKER_2]].dropna()

print("\nDATA SUMMARY")
print("-" * 50)
print(f"  Period       : {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")
print(f"  Total Days   : {len(prices):,} trading days")
print(f"  {TICKER_1} Range    : ${prices[TICKER_1].min():.2f} - ${prices[TICKER_1].max():.2f}")
print(f"  {TICKER_2} Range   : ${prices[TICKER_2].min():.2f} - ${prices[TICKER_2].max():.2f}")
print("-" * 50)


# ============================================================
# STEP 2: COINTEGRATION TEST (Engle-Granger)
# ============================================================
# H0: Kedua aset TIDAK cointegrated
# H1: Kedua aset cointegrated
# p-value < 0.05 --> Tolak H0 --> COINTEGRATED

coint_t_stat, p_value, critical_values = coint(
    prices[TICKER_1],
    prices[TICKER_2]
)

correlation = prices[TICKER_1].corr(prices[TICKER_2])

print("\nCOINTEGRATION TEST RESULTS (Engle-Granger)")
print("=" * 50)
print(f"  t-Statistic     : {coint_t_stat:.4f}")
print(f"  p-Value         : {p_value:.6f}")
print(f"  Critical Values : 1%: {critical_values[0]:.4f} | "
      f"5%: {critical_values[1]:.4f} | "
      f"10%: {critical_values[2]:.4f}")
print(f"  Pearson Corr.   : {correlation:.4f}")
print("-" * 50)

if p_value < 0.05:
    print(f"  RESULT: {TICKER_1} dan {TICKER_2} TERBUKTI COINTEGRATED")
    print(f"  p-value ({p_value:.6f}) < 0.05")
    print(f"  Spread bersifat mean-reverting --> LAYAK untuk Pairs Trading")
else:
    print(f"  RESULT: Tidak cukup bukti bahwa {TICKER_1} dan {TICKER_2} cointegrated")
    print(f"  p-value ({p_value:.6f}) >= 0.05")
    print(f"  Pairs trading tetap bisa dilakukan, tetapi risiko lebih tinggi")

print("=" * 50)


# ============================================================
# STEP 3: HEDGE RATIO & SPREAD via OLS REGRESSION
# ============================================================
# Model: KO = alpha + beta * PEP + epsilon
# beta = hedge ratio, epsilon = spread

X = sm.add_constant(prices[TICKER_2])
Y = prices[TICKER_1]

ols_model   = sm.OLS(Y, X).fit()
alpha       = ols_model.params.iloc[0]
hedge_ratio = ols_model.params.iloc[1]
r_squared   = ols_model.rsquared

print("\nOLS REGRESSION RESULTS")
print("=" * 50)
print(f"  Model        : {TICKER_1} = alpha + beta * {TICKER_2}")
print(f"  Intercept    : {alpha:.4f}")
print(f"  Hedge Ratio  : {hedge_ratio:.4f}")
print(f"  R-Squared    : {r_squared:.4f} ({r_squared*100:.1f}%)")
print("=" * 50)

spread = prices[TICKER_1] - (hedge_ratio * prices[TICKER_2]) - alpha


# ============================================================
# STEP 4: ROLLING Z-SCORE & TRADING SIGNALS
# ============================================================
# Z = (Spread - Rolling_Mean) / Rolling_Std
# Z < -2.0 --> BUY spread  (Long KO, Short PEP)
# Z > +2.0 --> SELL spread (Short KO, Long PEP)

ZSCORE_WINDOW   = 30
ENTRY_THRESHOLD = 2.0

rolling_mean = spread.rolling(window=ZSCORE_WINDOW).mean()
rolling_std  = spread.rolling(window=ZSCORE_WINDOW).std()
z_score      = (spread - rolling_mean) / rolling_std

buy_signals  = z_score[z_score < -ENTRY_THRESHOLD]
sell_signals = z_score[z_score >  ENTRY_THRESHOLD]
current_z    = z_score.iloc[-1]

print("\nZ-SCORE & TRADING SIGNALS")
print("=" * 50)
print(f"  Rolling Window  : {ZSCORE_WINDOW} days")
print(f"  Entry Threshold : +/-{ENTRY_THRESHOLD}")
print(f"  Current Z-Score : {current_z:.4f}")
print("-" * 50)
print(f"  BUY Signals     : {len(buy_signals)} occurrences")
print(f"  SELL Signals    : {len(sell_signals)} occurrences")
print("-" * 50)

if current_z < -ENTRY_THRESHOLD:
    signal_str = f"BUY SPREAD (Long {TICKER_1}, Short {TICKER_2})"
elif current_z > ENTRY_THRESHOLD:
    signal_str = f"SELL SPREAD (Short {TICKER_1}, Long {TICKER_2})"
else:
    signal_str = "NEUTRAL -- No action required"

print(f"  CURRENT SIGNAL  : {signal_str}")
print("=" * 50)


# ============================================================
# STEP 5: INTERACTIVE VISUALIZATION (Plotly)
# ============================================================

prices_norm = prices / prices.iloc[0] * 100

# Color palette
C_BG      = '#050A18'
C_PAPER   = '#000000'
C_GRID    = '#0D1B3E'
C_TEXT    = '#8899BB'
C_TITLE   = '#E8EDF5'
C_CYAN    = '#00E5FF'
C_MAGENTA = '#FF3D7F'
C_GOLD    = '#FFD700'
C_GREEN   = '#00FF88'
C_RED     = '#FF3D5A'
C_WHITE   = 'rgba(255,255,255,0.4)'

fig = make_subplots(
    rows=2, cols=1,
    row_heights=[0.55, 0.45],
    vertical_spacing=0.08,
    subplot_titles=[
        'NORMALIZED PRICE MOVEMENT (Base = 100)',
        f'Z-SCORE TRADING SIGNALS (Window = {ZSCORE_WINDOW}d, Threshold = +/-{ENTRY_THRESHOLD})'
    ]
)

# Subplot 1: Normalized Prices
fig.add_trace(
    go.Scatter(
        x=prices_norm.index, y=prices_norm[TICKER_1],
        name=f'{TICKER_1} (Coca-Cola)',
        line=dict(color=C_CYAN, width=1.8),
        hovertemplate='%{x|%Y-%m-%d}<br>' + TICKER_1 + ': %{y:.2f}<extra></extra>'
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=prices_norm.index, y=prices_norm[TICKER_2],
        name=f'{TICKER_2} (PepsiCo)',
        line=dict(color=C_MAGENTA, width=1.8),
        hovertemplate='%{x|%Y-%m-%d}<br>' + TICKER_2 + ': %{y:.2f}<extra></extra>'
    ),
    row=1, col=1
)

# Subplot 2: Z-Score
fig.add_trace(
    go.Scatter(
        x=z_score.index, y=z_score,
        name='Z-Score',
        line=dict(color=C_CYAN, width=1.5),
        hovertemplate='%{x|%Y-%m-%d}<br>Z-Score: %{y:.3f}<extra></extra>'
    ),
    row=2, col=1
)

# Buy zones (Z < -2)
z_buy = z_score.copy()
z_buy[z_buy >= -ENTRY_THRESHOLD] = -ENTRY_THRESHOLD
fig.add_trace(
    go.Scatter(
        x=z_score.index, y=z_buy,
        fill='tonexty', fillcolor='rgba(0,255,136,0.08)',
        line=dict(width=0), showlegend=False,
        hoverinfo='skip'
    ),
    row=2, col=1
)

# Sell zones (Z > +2)
z_sell = z_score.copy()
z_sell[z_sell <= ENTRY_THRESHOLD] = ENTRY_THRESHOLD
fig.add_trace(
    go.Scatter(
        x=z_score.index, y=z_sell,
        fill='tonexty', fillcolor='rgba(255,61,90,0.08)',
        line=dict(width=0), showlegend=False,
        hoverinfo='skip'
    ),
    row=2, col=1
)

# Threshold lines
for val, label, color, dash in [
    ( ENTRY_THRESHOLD, f'SELL (+{ENTRY_THRESHOLD})', C_RED,   'dash'),
    ( 0,               'MEAN (0)',                   C_WHITE, 'dash'),
    (-ENTRY_THRESHOLD, f'BUY (-{ENTRY_THRESHOLD})',  C_GREEN, 'dash'),
]:
    fig.add_hline(
        y=val, row=2, col=1,
        line=dict(color=color, width=1, dash=dash),
        annotation_text=label,
        annotation_position='right',
        annotation_font=dict(color=color, size=10, family='Courier New')
    )

# Layout: Deep Blue & Black theme
fig.update_layout(
    title=dict(
        text=(
            'Statistical Arbitrage: Pairs Trading Analysis<br>'
            '<span style="font-size:13px;color:#8899BB">'
            'Quantitative Research Portfolio by Rhameyza Faiqo Susanto</span>'
        ),
        font=dict(size=18, color=C_GOLD, family='Courier New'),
        x=0.5, xanchor='center'
    ),
    height=750,
    plot_bgcolor=C_BG,
    paper_bgcolor=C_PAPER,
    font=dict(color=C_TEXT, family='Courier New', size=11),
    legend=dict(
        bgcolor='rgba(10,15,32,0.9)',
        bordercolor=C_GRID,
        borderwidth=1,
        font=dict(color=C_TITLE, size=11)
    ),
    hovermode='x unified',
    margin=dict(t=100, b=60, l=60, r=30)
)

for i in [1, 2]:
    fig.update_xaxes(
        row=i, col=1,
        gridcolor=C_GRID, gridwidth=0.5,
        linecolor=C_GRID, zerolinecolor=C_GRID,
        tickfont=dict(color=C_TEXT, size=9)
    )
    fig.update_yaxes(
        row=i, col=1,
        gridcolor=C_GRID, gridwidth=0.5,
        linecolor=C_GRID, zerolinecolor=C_GRID,
        tickfont=dict(color=C_TEXT, size=9)
    )

fig.update_annotations(
    font=dict(color=C_TITLE, size=12, family='Courier New')
)

fig.add_annotation(
    xref='x domain', yref='y domain',
    x=0.99, y=0.97,
    text=(
        f'Pearson r = {correlation:.4f}<br>'
        f'Coint. p  = {p_value:.4f}<br>'
        f'Hedge B   = {hedge_ratio:.4f}'
    ),
    showarrow=False,
    font=dict(color=C_GOLD, size=10, family='Courier New'),
    align='right',
    bgcolor='rgba(10,15,32,0.85)',
    bordercolor=C_GOLD,
    borderwidth=1,
    borderpad=6,
    row=1, col=1
)

# Save as HTML + open in browser
output_file = 'pairs_trading_analysis.html'
fig.write_html(output_file, auto_open=True)
print(f"\nInteractive chart saved and opened: {output_file}")


# ============================================================
# STEP 6: EXECUTIVE SUMMARY
# ============================================================

cointegrated_str = "YES" if p_value < 0.05 else "NO"

print()
print("=" * 60)
print("  EXECUTIVE SUMMARY -- Pairs Trading Analysis")
print("  Quantitative Research by Rhameyza Faiqo Susanto")
print("=" * 60)
print(f"  Asset Pair     : {TICKER_1} (Coca-Cola) vs {TICKER_2} (PepsiCo)")
print(f"  Data Period    : {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")
print(f"  Sample Size    : {len(prices):,} trading days")
print("-" * 60)
print(f"  Pearson Corr.  : {correlation:.4f}")
print(f"  Coint. p-value : {p_value:.6f}")
print(f"  Cointegrated?  : {cointegrated_str}")
print("-" * 60)
print(f"  Hedge Ratio    : {hedge_ratio:.4f}")
print(f"  Intercept      : {alpha:.4f}")
print(f"  R-Squared      : {r_squared:.4f} ({r_squared*100:.1f}%)")
print("-" * 60)
print(f"  Z-Score Window : {ZSCORE_WINDOW} days")
print(f"  Current Z      : {current_z:.4f}")
print(f"  BUY Signals    : {len(buy_signals)} occurrences")
print(f"  SELL Signals   : {len(sell_signals)} occurrences")
print("-" * 60)
print(f"  CURRENT SIGNAL : {signal_str}")
print("=" * 60)
print()
print("  Disclaimer: For educational & research purposes only.")
print("  Not financial advice. Conduct your own due diligence.")
