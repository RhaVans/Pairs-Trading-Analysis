import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Pairs Trading Analysis | Rhameyza Faiqo Susanto",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# COLOR PALETTE
# ============================================================
C = {
    'bg':       '#060B16',
    'surface':  '#0B1224',
    'surface2': '#101B33',
    'border':   '#1A2744',
    'text':     '#C8D6E5',
    'text_dim': '#5C6F8A',
    'text_brt': '#E8EDF5',
    'cyan':     '#00D4FF',
    'magenta':  '#FF3D7F',
    'gold':     '#FFB800',
    'green':    '#00E676',
    'red':      '#FF3D5A',
    'blue_acc': '#1E6FFF',
}

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown(f"""
<style>
    .stApp {{
        background-color: {C['bg']};
        color: {C['text']};
    }}
    
    section[data-testid="stSidebar"] {{
        background-color: {C['surface']};
        border-right: 1px solid {C['border']};
    }}
    
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] label {{
        color: {C['text_dim']} !important;
    }}
    
    .metric-card {{
        background-color: {C['surface']};
        border: 1px solid {C['border']};
        border-radius: 10px;
        padding: 18px 16px;
        text-align: left;
    }}
    .metric-label {{
        font-size: 11px;
        color: {C['text_dim']};
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }}
    .metric-value {{
        font-size: 22px;
        font-weight: 700;
        font-family: 'Courier New', monospace;
    }}
    .metric-sub {{
        font-size: 11px;
        color: {C['text_dim']};
        margin-top: 4px;
    }}
    
    .header-title {{
        font-size: 22px;
        font-weight: 600;
        color: {C['text_brt']};
        letter-spacing: -0.3px;
        margin: 0;
    }}
    .header-sub {{
        font-size: 12px;
        color: {C['text_dim']};
        letter-spacing: 0.5px;
        margin: 4px 0 0 0;
    }}
    
    .stButton > button {{
        width: 100%;
        background-color: {C['blue_acc']};
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px;
        font-weight: 600;
        font-size: 14px;
    }}
    .stButton > button:hover {{
        background-color: #1959CC;
        color: white;
        border: none;
    }}
    
    div[data-testid="stMetric"] {{
        background-color: {C['surface']};
        border: 1px solid {C['border']};
        border-radius: 10px;
        padding: 14px 16px;
    }}
    div[data-testid="stMetric"] label {{
        color: {C['text_dim']} !important;
    }}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: {C['text_brt']} !important;
        font-family: 'Courier New', monospace !important;
    }}

    header[data-testid="stHeader"] {{
        background-color: {C['bg']};
    }}
    
    .block-container {{
        padding-top: 2rem;
    }}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div style="padding-bottom: 16px; border-bottom: 1px solid #1A2744; margin-bottom: 24px;">
    <p class="header-title">Pairs Trading Analysis</p>
    <p class="header-sub">Statistical Arbitrage | Quantitative Research Portfolio by Rhameyza Faiqo Susanto</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### Configuration")
    
    st.markdown("**Asset Pair**")
    ticker1 = st.text_input("Ticker 1", value="KO", placeholder="e.g. KO")
    ticker2 = st.text_input("Ticker 2", value="PEP", placeholder="e.g. PEP")
    
    st.markdown("**Data Period**")
    period = st.selectbox("Period", ["1y", "2y", "3y", "5y"], index=3,
                          format_func=lambda x: {"1y": "1 Year", "2y": "2 Years",
                                                  "3y": "3 Years", "5y": "5 Years"}[x])
    
    st.markdown("**Z-Score Parameters**")
    z_window = st.slider("Rolling Window (days)", 10, 60, 30, 5)
    threshold = st.slider("Entry Threshold", 1.0, 3.0, 2.0, 0.25)
    
    st.markdown("---")
    run_btn = st.button("Run Analysis", type="primary", use_container_width=True)


# ============================================================
# ANALYSIS
# ============================================================
if run_btn:
    ticker1 = ticker1.strip().upper()
    ticker2 = ticker2.strip().upper()
    
    if not ticker1 or not ticker2:
        st.error("Please enter both tickers.")
        st.stop()
    
    if ticker1 == ticker2:
        st.error("Tickers must be different.")
        st.stop()
    
    with st.spinner(f"Fetching data for {ticker1} and {ticker2}..."):
        try:
            raw = yf.download(
                tickers=[ticker1, ticker2],
                period=period,
                interval='1d',
                auto_adjust=True,
            )
            
            if raw.empty:
                st.error(f"No data returned. Check that '{ticker1}' and '{ticker2}' are valid tickers.")
                st.stop()
            
            prices = raw['Close'][[ticker1, ticker2]].dropna()
            
            if len(prices) < 60:
                st.error(f"Not enough data ({len(prices)} days). Try a longer period.")
                st.stop()
            
            # Cointegration
            coint_t, p_val, crit_vals = coint(prices[ticker1], prices[ticker2])
            corr = prices[ticker1].corr(prices[ticker2])
            
            # OLS
            X = sm.add_constant(prices[ticker2])
            Y = prices[ticker1]
            model = sm.OLS(Y, X).fit()
            alpha_val = model.params.iloc[0]
            hedge = model.params.iloc[1]
            r_sq = model.rsquared
            
            # Spread & Z-Score
            spread = prices[ticker1] - (hedge * prices[ticker2]) - alpha_val
            roll_mean = spread.rolling(window=z_window).mean()
            roll_std = spread.rolling(window=z_window).std()
            z_score = (spread - roll_mean) / roll_std
            
            current_z = z_score.iloc[-1]
            buy_count = int((z_score < -threshold).sum())
            sell_count = int((z_score > threshold).sum())
            
            if current_z < -threshold:
                signal, sig_color = "BUY", C['green']
            elif current_z > threshold:
                signal, sig_color = "SELL", C['red']
            else:
                signal, sig_color = "NEUTRAL", C['text_dim']
            
            coint_color = C['green'] if p_val < 0.05 else C['gold']
            coint_sub = "Cointegrated" if p_val < 0.05 else "Not cointegrated"
        
        except Exception as e:
            err = str(e)
            if "No data" in err or "No timezone" in err or "list index" in err:
                st.error(f"Could not fetch data. Verify that '{ticker1}' and '{ticker2}' are valid Yahoo Finance tickers.")
            else:
                st.error(f"Error: {err}")
            st.stop()
    
    # ---- METRIC CARDS ----
    def metric_card(label, value, sub, color=None):
        vc = color or C['text_brt']
        return f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color: {vc};">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """
    
    cols = st.columns(5)
    with cols[0]:
        st.markdown(metric_card("Correlation", f"{corr:.4f}", f"{ticker1} vs {ticker2}"), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(metric_card("Coint. p-value", f"{p_val:.4f}", coint_sub, coint_color), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(metric_card("Hedge Ratio", f"{hedge:.4f}", f"R2 = {r_sq:.3f}"), unsafe_allow_html=True)
    with cols[3]:
        st.markdown(metric_card("Current Z", f"{current_z:.3f}", f"Window = {z_window}d"), unsafe_allow_html=True)
    with cols[4]:
        st.markdown(metric_card("Signal", signal, f"{buy_count} buy / {sell_count} sell", sig_color), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ---- CHARTS ----
    prices_norm = prices / prices.iloc[0] * 100
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.55, 0.45],
        vertical_spacing=0.08,
        subplot_titles=[
            f'NORMALIZED PRICE: {ticker1} vs {ticker2} (Base = 100)',
            f'Z-SCORE (Window = {z_window}d, Threshold = +/-{threshold})'
        ]
    )
    
    # Prices
    fig.add_trace(go.Scatter(
        x=prices_norm.index, y=prices_norm[ticker1],
        name=ticker1, line=dict(color=C['cyan'], width=1.8),
        hovertemplate='%{x|%Y-%m-%d}<br>' + ticker1 + ': %{y:.2f}<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=prices_norm.index, y=prices_norm[ticker2],
        name=ticker2, line=dict(color=C['magenta'], width=1.8),
        hovertemplate='%{x|%Y-%m-%d}<br>' + ticker2 + ': %{y:.2f}<extra></extra>'
    ), row=1, col=1)
    
    # Z-Score
    fig.add_trace(go.Scatter(
        x=z_score.index, y=z_score,
        name='Z-Score', line=dict(color=C['cyan'], width=1.4),
        hovertemplate='%{x|%Y-%m-%d}<br>Z: %{y:.3f}<extra></extra>'
    ), row=2, col=1)
    
    # Buy zone
    z_clamp_low = z_score.clip(upper=-threshold)
    fig.add_trace(go.Scatter(
        x=z_score.index, y=z_clamp_low,
        line=dict(width=0), showlegend=False, hoverinfo='skip',
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=z_score.index, y=np.full(len(z_score), -threshold),
        fill='tonexty', fillcolor='rgba(0,230,118,0.06)',
        line=dict(width=0), showlegend=False, hoverinfo='skip',
    ), row=2, col=1)
    
    # Sell zone
    z_clamp_hi = z_score.clip(lower=threshold)
    fig.add_trace(go.Scatter(
        x=z_score.index, y=z_clamp_hi,
        line=dict(width=0), showlegend=False, hoverinfo='skip',
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=z_score.index, y=np.full(len(z_score), threshold),
        fill='tonexty', fillcolor='rgba(255,61,90,0.06)',
        line=dict(width=0), showlegend=False, hoverinfo='skip',
    ), row=2, col=1)
    
    # Threshold lines
    for val, lbl, clr in [
        (threshold,  f'SELL (+{threshold})', C['red']),
        (0,          'MEAN',                 'rgba(255,255,255,0.2)'),
        (-threshold, f'BUY (-{threshold})',  C['green']),
    ]:
        fig.add_hline(
            y=val, row=2, col=1,
            line=dict(color=clr, width=1, dash='dash'),
            annotation_text=lbl,
            annotation_position='right',
            annotation_font=dict(color=clr, size=10),
        )
    
    # Layout
    fig.update_layout(
        height=680,
        plot_bgcolor=C['surface'],
        paper_bgcolor=C['bg'],
        font=dict(color=C['text'], family="Inter, sans-serif", size=11),
        legend=dict(
            bgcolor=C['surface'],
            bordercolor=C['border'],
            borderwidth=1,
            font=dict(color=C['text_brt'], size=11),
        ),
        hovermode='x unified',
        margin=dict(t=60, b=40, l=50, r=20),
    )
    
    for i in [1, 2]:
        fig.update_xaxes(
            row=i, col=1,
            gridcolor=C['border'], gridwidth=0.5,
            linecolor=C['border'], zerolinecolor=C['border'],
            tickfont=dict(color=C['text_dim'], size=9),
        )
        fig.update_yaxes(
            row=i, col=1,
            gridcolor=C['border'], gridwidth=0.5,
            linecolor=C['border'], zerolinecolor=C['border'],
            tickfont=dict(color=C['text_dim'], size=9),
        )
    
    fig.update_annotations(font=dict(color=C['text_brt'], size=12))
    
    fig.add_annotation(
        xref='x domain', yref='y domain', x=0.99, y=0.97,
        text=f'Corr = {corr:.4f}<br>p-val = {p_val:.4f}<br>Hedge = {hedge:.4f}',
        showarrow=False,
        font=dict(color=C['gold'], size=10, family="Courier New, monospace"),
        align='right',
        bgcolor=C['surface'], bordercolor=C['gold'],
        borderwidth=1, borderpad=6,
        row=1, col=1,
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ---- DATA TABLE ----
    with st.expander("View Raw Data"):
        st.dataframe(prices.tail(20), use_container_width=True)

else:
    st.markdown(f"""
    <div style="text-align: center; padding: 120px 40px; color: {C['text_dim']};">
        <p style="font-size: 18px; margin-bottom: 8px;">Enter tickers in the sidebar and click Run Analysis</p>
        <p style="font-size: 13px;">Compare any two stocks for cointegration and pairs trading signals</p>
    </div>
    """, unsafe_allow_html=True)
