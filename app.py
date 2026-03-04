# ============================================================
# PAIRS TRADING WEB APP
# Statistical Arbitrage Analysis — Any Stock Pair
# by Rhameyza Faiqo Susanto
# ============================================================
# Run: python app.py
# Open: http://127.0.0.1:8050
# ============================================================

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import warnings

from dash import Dash, html, dcc, Input, Output, State, callback, no_update

warnings.filterwarnings('ignore')

# ============================================================
# COLOR PALETTE
# ============================================================
C = {
    'bg':       '#060B16',
    'surface':  '#0B1224',
    'surface2': '#101B33',
    'border':   '#1A2744',
    'border_h': '#2A3F6A',
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
# HELPER FUNCTIONS (must be defined before layout)
# ============================================================

def _sidebar_section(title, children):
    return html.Div(
        style={
            'padding': '16px',
            'backgroundColor': C['surface'],
            'borderRadius': '10px',
            'border': f"1px solid {C['border']}",
        },
        children=[
            html.Div(
                title,
                style={
                    'fontSize': '11px',
                    'fontWeight': '600',
                    'color': C['text_dim'],
                    'textTransform': 'uppercase',
                    'letterSpacing': '0.8px',
                    'marginBottom': '12px',
                }
            ),
            html.Div(
                children=children,
                style={'display': 'flex', 'flexDirection': 'column', 'gap': '10px'}
            ),
        ]
    )


def _input_group(label, input_el):
    return html.Div([
        html.Label(
            label,
            style={
                'fontSize': '12px',
                'color': C['text_dim'],
                'marginBottom': '4px',
                'display': 'block',
            }
        ),
        input_el,
    ])


def _input_style():
    return {
        'width': '100%',
        'padding': '8px 12px',
        'backgroundColor': C['surface2'],
        'border': f"1px solid {C['border']}",
        'borderRadius': '6px',
        'color': C['text_brt'],
        'fontSize': '14px',
        'fontWeight': '500',
        'outline': 'none',
    }


def _metric_card(label, value, sub_text, color=None):
    val_color = color or C['text_brt']
    card_id = f"card-{label.lower().replace(' ', '-').replace('.', '')}"
    return html.Div(
        style={
            'padding': '18px 16px',
            'backgroundColor': C['surface'],
            'borderRadius': '10px',
            'border': f"1px solid {C['border']}",
        },
        children=[
            html.Div(label, style={
                'fontSize': '11px',
                'color': C['text_dim'],
                'textTransform': 'uppercase',
                'letterSpacing': '0.5px',
                'marginBottom': '8px',
            }),
            html.Div(str(value), style={
                'fontSize': '22px',
                'fontWeight': '700',
                'color': val_color,
                'fontFamily': "'Courier New', monospace",
            }),
            html.Div(sub_text, style={
                'fontSize': '11px',
                'color': C['text_dim'],
                'marginTop': '4px',
            })
        ]
    )


def _empty_chart():
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.55, 0.45],
        vertical_spacing=0.08,
        subplot_titles=['NORMALIZED PRICE MOVEMENT', 'Z-SCORE TRADING SIGNALS']
    )
    fig.update_layout(
        height=680,
        plot_bgcolor=C['surface'],
        paper_bgcolor=C['bg'],
        font=dict(color=C['text_dim'], family="Inter, sans-serif", size=11),
        margin=dict(t=60, b=40, l=50, r=20),
    )
    for i in [1, 2]:
        fig.update_xaxes(row=i, col=1, gridcolor=C['border'], linecolor=C['border'],
                         zerolinecolor=C['border'])
        fig.update_yaxes(row=i, col=1, gridcolor=C['border'], linecolor=C['border'],
                         zerolinecolor=C['border'])
    fig.update_annotations(font=dict(color=C['text_dim'], size=12))
    fig.add_annotation(
        text="Enter tickers and click Run Analysis",
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=16, color=C['text_dim']),
    )
    return fig


# ============================================================
# CUSTOM CSS
# ============================================================
CUSTOM_CSS = """
    * { box-sizing: border-box; }
    body { margin: 0; }

    .dash-dropdown .Select-control {
        background-color: #101B33 !important;
        border: 1px solid #1A2744 !important;
        border-radius: 6px !important;
    }
    .dash-dropdown .Select-menu-outer {
        background-color: #0B1224 !important;
        border: 1px solid #1A2744 !important;
    }
    .VirtualizedSelectOption {
        background-color: #0B1224 !important;
        color: #C8D6E5 !important;
    }
    .VirtualizedSelectFocusedOption {
        background-color: #1A2744 !important;
    }
    .Select-value-label { color: #C8D6E5 !important; }
    .Select-placeholder { color: #5C6F8A !important; }
    .Select-arrow-zone .Select-arrow {
        border-color: #5C6F8A transparent transparent !important;
    }

    .rc-slider-track { background-color: #1E6FFF !important; }
    .rc-slider-handle {
        background-color: #1E6FFF !important;
        border-color: #1E6FFF !important;
    }
    .rc-slider-rail { background-color: #1A2744 !important; }
    .rc-slider-dot { background-color: #1A2744 !important; border-color: #1A2744 !important; }
    .rc-slider-mark-text { color: #5C6F8A !important; font-size: 10px !important; }

    #analyze-btn:hover { opacity: 0.85; }

    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #060B16; }
    ::-webkit-scrollbar-thumb { background: #1A2744; border-radius: 3px; }
"""


# ============================================================
# DASH APP & LAYOUT
# ============================================================
app = Dash(__name__)
app.title = "Pairs Trading Analysis | Rhameyza Faiqo Susanto"

# Inject custom CSS via index_string (html.Style removed in Dash 4.0)
app.index_string = '''
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
''' + CUSTOM_CSS + '''
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
'''

app.layout = html.Div(
    style={
        'backgroundColor': C['bg'],
        'minHeight': '100vh',
        'fontFamily': "Inter, 'Segoe UI', system-ui, sans-serif",
        'color': C['text'],
    },
    children=[

        # ---- HEADER ----
        html.Div(
            style={
                'padding': '24px 40px',
                'borderBottom': f"1px solid {C['border']}",
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'space-between',
            },
            children=[
                html.Div([
                    html.H1(
                        "Pairs Trading Analysis",
                        style={
                            'margin': '0',
                            'fontSize': '22px',
                            'fontWeight': '600',
                            'color': C['text_brt'],
                            'letterSpacing': '-0.3px',
                        }
                    ),
                    html.P(
                        "Statistical Arbitrage | Quantitative Research Portfolio by Rhameyza Faiqo Susanto",
                        style={
                            'margin': '4px 0 0 0',
                            'fontSize': '12px',
                            'color': C['text_dim'],
                            'letterSpacing': '0.5px',
                        }
                    ),
                ]),
                html.Div(
                    id='status-badge',
                    children="Ready",
                    style={
                        'padding': '6px 16px',
                        'borderRadius': '20px',
                        'fontSize': '12px',
                        'fontWeight': '500',
                        'backgroundColor': C['surface2'],
                        'border': f"1px solid {C['border']}",
                        'color': C['text_dim'],
                    }
                ),
            ]
        ),

        # ---- MAIN CONTENT ----
        html.Div(
            style={
                'display': 'flex',
                'gap': '0',
                'minHeight': 'calc(100vh - 85px)',
            },
            children=[

                # ---- SIDEBAR ----
                html.Div(
                    style={
                        'width': '280px',
                        'minWidth': '280px',
                        'padding': '28px 24px',
                        'borderRight': f"1px solid {C['border']}",
                        'display': 'flex',
                        'flexDirection': 'column',
                        'gap': '20px',
                    },
                    children=[
                        _sidebar_section("Asset Pair", [
                            _input_group("Ticker 1", dcc.Input(
                                id='ticker-1', type='text', value='KO',
                                placeholder='e.g. KO',
                                style=_input_style(),
                            )),
                            _input_group("Ticker 2", dcc.Input(
                                id='ticker-2', type='text', value='PEP',
                                placeholder='e.g. PEP',
                                style=_input_style(),
                            )),
                        ]),

                        _sidebar_section("Data Period", [
                            dcc.Dropdown(
                                id='period',
                                options=[
                                    {'label': '1 Year',  'value': '1y'},
                                    {'label': '2 Years', 'value': '2y'},
                                    {'label': '3 Years', 'value': '3y'},
                                    {'label': '5 Years', 'value': '5y'},
                                ],
                                value='5y',
                                clearable=False,
                                style={
                                    'backgroundColor': C['surface2'],
                                    'border': 'none',
                                    'color': C['text'],
                                    'fontSize': '13px',
                                },
                                className='dark-dropdown',
                            ),
                        ]),

                        _sidebar_section("Z-Score Parameters", [
                            _input_group("Rolling Window (days)", html.Div([
                                dcc.Slider(
                                    id='zscore-window',
                                    min=10, max=60, step=5, value=30,
                                    marks={10: '10', 20: '20', 30: '30',
                                           40: '40', 50: '50', 60: '60'},
                                    tooltip={"placement": "bottom"},
                                ),
                            ])),
                            _input_group("Entry Threshold", html.Div([
                                dcc.Slider(
                                    id='threshold',
                                    min=1.0, max=3.0, step=0.25, value=2.0,
                                    marks={1.0: '1.0', 1.5: '1.5', 2.0: '2.0',
                                           2.5: '2.5', 3.0: '3.0'},
                                    tooltip={"placement": "bottom"},
                                ),
                            ])),
                        ]),

                        html.Button(
                            "Run Analysis",
                            id='analyze-btn',
                            n_clicks=0,
                            style={
                                'width': '100%',
                                'padding': '12px',
                                'backgroundColor': C['blue_acc'],
                                'color': '#FFFFFF',
                                'border': 'none',
                                'borderRadius': '8px',
                                'fontSize': '14px',
                                'fontWeight': '600',
                                'cursor': 'pointer',
                                'letterSpacing': '0.3px',
                                'marginTop': '4px',
                            }
                        ),

                        html.Div(id='error-msg', style={
                            'fontSize': '12px',
                            'color': C['red'],
                            'padding': '8px 0',
                            'minHeight': '20px',
                        }),
                    ]
                ),

                # ---- MAIN PANEL ----
                html.Div(
                    style={
                        'flex': '1',
                        'padding': '28px 32px',
                        'overflowY': 'auto',
                    },
                    children=[
                        dcc.Loading(
                            type='default',
                            color=C['cyan'],
                            children=[
                                html.Div(
                                    id='metrics-row',
                                    style={
                                        'display': 'grid',
                                        'gridTemplateColumns': 'repeat(5, 1fr)',
                                        'gap': '14px',
                                        'marginBottom': '24px',
                                    },
                                    children=[
                                        _metric_card("Correlation", "--", ""),
                                        _metric_card("Coint. p-value", "--", ""),
                                        _metric_card("Hedge Ratio", "--", ""),
                                        _metric_card("Current Z", "--", ""),
                                        _metric_card("Signal", "--", ""),
                                    ]
                                ),
                                dcc.Graph(
                                    id='main-chart',
                                    config={
                                        'displayModeBar': True,
                                        'displaylogo': False,
                                        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                                    },
                                    style={'height': '700px'},
                                    figure=_empty_chart(),
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        ),

    ]
)


# ============================================================
# CALLBACK: Run Analysis
# ============================================================

@callback(
    Output('metrics-row', 'children'),
    Output('main-chart', 'figure'),
    Output('error-msg', 'children'),
    Output('status-badge', 'children'),
    Output('status-badge', 'style'),
    Input('analyze-btn', 'n_clicks'),
    State('ticker-1', 'value'),
    State('ticker-2', 'value'),
    State('period', 'value'),
    State('zscore-window', 'value'),
    State('threshold', 'value'),
    prevent_initial_call=True,
)
def run_analysis(n_clicks, ticker1, ticker2, period, z_window, threshold):

    badge_base = {
        'padding': '6px 16px',
        'borderRadius': '20px',
        'fontSize': '12px',
        'fontWeight': '500',
    }

    if not ticker1 or not ticker2:
        return (no_update, no_update, "Please enter both tickers.",
                "Error", {**badge_base, 'backgroundColor': '#2A1015',
                          'border': f"1px solid {C['red']}", 'color': C['red']})

    ticker1 = ticker1.strip().upper()
    ticker2 = ticker2.strip().upper()

    if ticker1 == ticker2:
        return (no_update, no_update, "Tickers must be different.",
                "Error", {**badge_base, 'backgroundColor': '#2A1015',
                          'border': f"1px solid {C['red']}", 'color': C['red']})

    try:
        # ---- FETCH DATA ----
        raw = yf.download(
            tickers=[ticker1, ticker2],
            period=period,
            interval='1d',
            auto_adjust=True,
        )

        if raw.empty:
            raise ValueError(f"No data returned for {ticker1} and/or {ticker2}.")

        prices = raw['Close'][[ticker1, ticker2]].dropna()

        if len(prices) < 60:
            return (no_update, no_update,
                    f"Not enough data ({len(prices)} days). Try a longer period or different tickers.",
                    "Error", {**badge_base, 'backgroundColor': '#2A1015',
                              'border': f"1px solid {C['red']}", 'color': C['red']})

        # ---- COINTEGRATION TEST ----
        coint_t, p_val, crit_vals = coint(prices[ticker1], prices[ticker2])
        corr = prices[ticker1].corr(prices[ticker2])

        # ---- OLS HEDGE RATIO ----
        X = sm.add_constant(prices[ticker2])
        Y = prices[ticker1]
        model = sm.OLS(Y, X).fit()
        alpha_val = model.params.iloc[0]
        hedge = model.params.iloc[1]
        r_sq = model.rsquared

        # ---- SPREAD & Z-SCORE ----
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

        # ---- METRIC CARDS ----
        coint_color = C['green'] if p_val < 0.05 else C['gold']
        coint_sub = "Cointegrated" if p_val < 0.05 else "Not cointegrated"

        cards = [
            _metric_card("Correlation", f"{corr:.4f}", f"{ticker1} vs {ticker2}"),
            _metric_card("Coint. p-value", f"{p_val:.4f}", coint_sub, coint_color),
            _metric_card("Hedge Ratio", f"{hedge:.4f}", f"R2 = {r_sq:.3f}"),
            _metric_card("Current Z", f"{current_z:.3f}", f"Window = {z_window}d"),
            _metric_card("Signal", signal, f"{buy_count} buy / {sell_count} sell", sig_color),
        ]

        # ---- BUILD CHART ----
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

        # Subplot 1: Prices
        fig.add_trace(go.Scatter(
            x=prices_norm.index, y=prices_norm[ticker1],
            name=ticker1,
            line=dict(color=C['cyan'], width=1.8),
            hovertemplate='%{x|%Y-%m-%d}<br>' + ticker1 + ': %{y:.2f}<extra></extra>'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=prices_norm.index, y=prices_norm[ticker2],
            name=ticker2,
            line=dict(color=C['magenta'], width=1.8),
            hovertemplate='%{x|%Y-%m-%d}<br>' + ticker2 + ': %{y:.2f}<extra></extra>'
        ), row=1, col=1)

        # Subplot 2: Z-Score
        fig.add_trace(go.Scatter(
            x=z_score.index, y=z_score,
            name='Z-Score',
            line=dict(color=C['cyan'], width=1.4),
            hovertemplate='%{x|%Y-%m-%d}<br>Z: %{y:.3f}<extra></extra>'
        ), row=2, col=1)

        # Buy zone shading
        z_clamp_low = z_score.clip(upper=-threshold)
        fig.add_trace(go.Scatter(
            x=z_score.index, y=z_clamp_low,
            line=dict(width=0), showlegend=False, hoverinfo='skip',
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=z_score.index,
            y=np.full(len(z_score), -threshold),
            fill='tonexty', fillcolor='rgba(0,230,118,0.06)',
            line=dict(width=0), showlegend=False, hoverinfo='skip',
        ), row=2, col=1)

        # Sell zone shading
        z_clamp_hi = z_score.clip(lower=threshold)
        fig.add_trace(go.Scatter(
            x=z_score.index, y=z_clamp_hi,
            line=dict(width=0), showlegend=False, hoverinfo='skip',
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=z_score.index,
            y=np.full(len(z_score), threshold),
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

        # Layout styling
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

        # Stats box on price chart
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

        badge_ok = {
            **badge_base,
            'backgroundColor': '#0A2010',
            'border': f"1px solid {C['green']}",
            'color': C['green'],
        }

        return cards, fig, "", f"{ticker1}/{ticker2} analyzed", badge_ok

    except Exception as e:
        err = str(e)
        if "No data" in err or "No timezone" in err or "list index" in err:
            err = f"Could not fetch data. Verify that '{ticker1}' and '{ticker2}' are valid Yahoo Finance tickers."

        badge_err = {
            **badge_base,
            'backgroundColor': '#2A1015',
            'border': f"1px solid {C['red']}",
            'color': C['red'],
        }
        return no_update, no_update, err, "Error", badge_err


# ============================================================
# RUN SERVER
# ============================================================
if __name__ == '__main__':
    print()
    print("  Pairs Trading Analysis -- Web App")
    print("  http://127.0.0.1:8050")
    print("  Press Ctrl+C to stop")
    print()
    app.run(debug=True, host='127.0.0.1', port=8050)
