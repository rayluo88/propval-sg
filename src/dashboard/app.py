"""
Singapore HDB AVM — Gradio Dashboard

Three tabs:
  1. Property Valuation  — Enter flat details → get AVM price estimate
  2. Market Trends       — Historical price chart for any town/flat-type segment
  3. Price Forecast      — 12-month Prophet forecast with confidence bands
"""

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from src.avm.predict import get_predictor
from src.forecast.predict import (
    get_all_flat_types,
    get_all_towns,
    get_forecast,
    get_historical_trend,
)

# ── Pre-load ──────────────────────────────────────────────────────────────────
predictor = get_predictor()
TOWNS = get_all_towns()
FLAT_TYPES = get_all_flat_types()

FLAT_MODELS = [
    "IMPROVED", "NEW GENERATION", "MODEL A", "STANDARD", "SIMPLIFIED",
    "PREMIUM APARTMENT", "MAISONETTE", "APARTMENT", "MULTI GENERATION",
    "IMPROVED-MAISONETTE", "MODEL A-MAISONETTE", "TERRACE", "PREMIUM MAISONETTE",
    "2-ROOM", "MODEL A2", "DBSS", "TYPE S1", "TYPE S2",
]

# ── Design System ─────────────────────────────────────────────────────────────
#
# Aesthetic: "Cadastre" — land-registry editorial. Warm ivory ground, precise
# ink-dark typography, terracotta/sienna accent. All Gradio-internal colors are
# handled via the theme object so dropdowns, inputs and labels stay readable.
# CSS is reserved for the bespoke header, tab bar polish, typography, and chart
# container refinement — things the theme API cannot reach.

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&family=Source+Sans+3:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ── Root tokens (referenced in CSS only — Gradio theme handles component colors) */
:root {
  --ivory:       #faf8f4;
  --parchment:   #f2efe8;
  --ink:         #1c1917;
  --ink-soft:    #44403c;
  --ink-muted:   #78716c;
  --rule:        #d6d3cd;
  --rule-strong: #a8a29e;
  --sienna:      #b45309;
  --sienna-glow: rgba(180, 83, 9, 0.10);
  --navy:        #1e293b;
  --font-display: 'Playfair Display', Georgia, serif;
  --font-body:    'Source Sans 3', system-ui, sans-serif;
  --font-mono:    'IBM Plex Mono', monospace;
}

/* ── Page base ───────────────────────────────────────── */
body {
  background: var(--ivory) !important;
}
.gradio-container {
  max-width: 1200px !important;
}

/* ── Header banner ───────────────────────────────────── */
.app-header {
  background: var(--navy);
  padding: 1.8rem 2.2rem 1.5rem;
  position: relative;
  overflow: hidden;
}
.app-header::after {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, rgba(180,83,9,0.12) 0%, transparent 50%);
  pointer-events: none;
}
.header-eyebrow {
  font-family: var(--font-mono);
  font-size: 0.6rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: #fbbf24;
  margin-bottom: 0.4rem;
  position: relative;
  z-index: 1;
}
.header-title {
  font-family: var(--font-display);
  font-size: 1.85rem;
  font-weight: 700;
  color: #faf8f4;
  margin: 0 0 0.35rem;
  letter-spacing: -0.02em;
  line-height: 1.1;
  position: relative;
  z-index: 1;
}
.header-title em {
  color: #faf8f4;
  font-style: normal;
}
.header-chip {
  display: inline-block;
  background: rgba(217,119,6,0.2);
  border: 1px solid rgba(217,119,6,0.35);
  color: #fbbf24;
  font-family: var(--font-mono);
  font-size: 0.55rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  padding: 2px 8px;
  border-radius: 3px;
  vertical-align: middle;
  margin-left: 0.5rem;
  position: relative;
  top: -2px;
}
.header-meta {
  font-family: var(--font-body);
  font-size: 0.78rem;
  color: #fbbf24 !important;
  font-weight: 300;
  display: flex;
  align-items: center;
  gap: 1.2rem;
  flex-wrap: wrap;
  position: relative;
  z-index: 1;
}
.header-meta > span {
  color: #fbbf24 !important;
}
.header-meta a {
  color: #fbbf24 !important;
  text-decoration: none;
  font-weight: 400;
}
.header-sep {
  width: 3px;
  height: 3px;
  border-radius: 50%;
  background: #fbbf24;
  display: inline-block;
}

/* ── Tab bar — sits below navy header ────────────────── */
.tab-nav {
  background: var(--navy) !important;
  border: none !important;
  padding: 0 1.5rem !important;
  gap: 0 !important;
  border-bottom: 1px solid rgba(250,248,244,0.08) !important;
}
.tab-nav button {
  background: transparent !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  color: rgba(250,248,244,0.4) !important;
  font-family: var(--font-body) !important;
  font-size: 0.7rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  padding: 0.65rem 1rem !important;
  border-radius: 0 !important;
  margin-bottom: 0 !important;
  transition: color 0.15s ease, border-color 0.15s ease !important;
}
.tab-nav button:hover {
  color: rgba(250,248,244,0.75) !important;
  background: rgba(250,248,244,0.03) !important;
}
.tab-nav button.selected {
  color: #fbbf24 !important;
  border-bottom-color: #d97706 !important;
  background: transparent !important;
}

/* ── Section headings (h2 from gr.Markdown) ──────────── */
.prose h2, .markdown-text h2 {
  font-family: var(--font-display) !important;
  font-size: 1.35rem !important;
  font-weight: 700 !important;
  color: var(--ink) !important;
  letter-spacing: -0.01em !important;
  border-bottom: 2px solid var(--rule) !important;
  padding-bottom: 0.5rem !important;
  margin: 0 0 0.15rem !important;
}

/* ── Body text (p from gr.Markdown) ──────────────────── */
.prose p, .markdown-text p {
  font-family: var(--font-body) !important;
  color: var(--ink-soft) !important;
  font-size: 0.84rem !important;
  font-weight: 400 !important;
  line-height: 1.6 !important;
  margin: 0.25rem 0 1.1rem !important;
}

/* ── Prediction result styling ───────────────────────── */
.prose h3, .markdown-text h3 {
  font-family: var(--font-body) !important;
  font-size: 0.62rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.14em !important;
  text-transform: uppercase !important;
  color: var(--ink-muted) !important;
  border: none !important;
  margin: 0 0 0.5rem !important;
}
.prose strong, .markdown-text strong {
  font-family: var(--font-display) !important;
  color: var(--ink) !important;
}

/* ── Primary buttons ─────────────────────────────────── */
button.primary {
  background: var(--navy) !important;
  color: #faf8f4 !important;
  border: none !important;
  font-family: var(--font-body) !important;
  font-size: 0.72rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  padding: 0.6rem 1.5rem !important;
  border-radius: 5px !important;
  box-shadow: 0 1px 3px rgba(0,0,0,0.12) !important;
  transition: background 0.15s ease, box-shadow 0.15s ease, transform 0.1s ease !important;
}
button.primary:hover {
  background: #334155 !important;
  box-shadow: 0 3px 10px rgba(0,0,0,0.15) !important;
  transform: translateY(-1px) !important;
}
button.primary:active {
  transform: translateY(0) !important;
}

/* ── Checkbox label alignment fix ────────────────────── */
input[type="checkbox"] {
  accent-color: var(--sienna) !important;
}

/* ── Smooth scrollbar ────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--parchment); }
::-webkit-scrollbar-thumb { background: var(--rule-strong); border-radius: 3px; }
"""

# ── Gradio Theme ──────────────────────────────────────────────────────────────
#
# This controls ALL internal component colors — inputs, dropdowns, labels, etc.
# By setting these properly the readability problem is solved at the source.

THEME = gr.themes.Base(
    font=[gr.themes.GoogleFont("Source Sans 3"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"],
    primary_hue=gr.themes.Color(
        c50="#fef3c7", c100="#fde68a", c200="#fcd34d", c300="#fbbf24",
        c400="#f59e0b", c500="#d97706", c600="#b45309", c700="#92400e",
        c800="#78350f", c900="#451a03", c950="#2c0f02",
    ),
    secondary_hue=gr.themes.colors.stone,
    neutral_hue=gr.themes.colors.stone,
).set(
    # Page
    body_background_fill="#faf8f4",
    body_background_fill_dark="#faf8f4",
    body_text_color="#1c1917",
    body_text_color_dark="#1c1917",
    body_text_color_subdued="#78716c",
    body_text_color_subdued_dark="#78716c",

    # Blocks / panels
    block_background_fill="#ffffff",
    block_background_fill_dark="#ffffff",
    block_border_color="#d6d3cd",
    block_border_color_dark="#d6d3cd",
    block_border_width="1px",
    block_label_text_color="#44403c",
    block_label_text_color_dark="#44403c",
    block_label_text_size="sm",
    block_label_background_fill="#f2efe8",
    block_label_background_fill_dark="#f2efe8",
    block_radius="8px",
    block_shadow="0 1px 3px rgba(28,25,23,0.06)",

    # Inputs
    input_background_fill="#f5f3ee",
    input_background_fill_dark="#f5f3ee",
    input_border_color="#d6d3cd",
    input_border_color_dark="#d6d3cd",
    input_border_color_focus="#b45309",
    input_border_color_focus_dark="#b45309",
    input_placeholder_color="#a8a29e",
    input_placeholder_color_dark="#a8a29e",
    input_radius="5px",
    input_shadow="none",
    input_shadow_focus="0 0 0 2px rgba(180,83,9,0.12)",

    # Buttons
    button_primary_background_fill="#1e293b",
    button_primary_background_fill_dark="#1e293b",
    button_primary_text_color="#faf8f4",
    button_primary_text_color_dark="#faf8f4",
    button_primary_border_color="#1e293b",
    button_primary_border_color_dark="#1e293b",
    button_primary_background_fill_hover="#334155",
    button_primary_background_fill_hover_dark="#334155",
    button_secondary_background_fill="#f2efe8",
    button_secondary_background_fill_dark="#f2efe8",
    button_secondary_text_color="#1c1917",
    button_secondary_text_color_dark="#1c1917",
    button_secondary_border_color="#d6d3cd",
    button_secondary_border_color_dark="#d6d3cd",

    # Checkbox / radio
    checkbox_background_color="#f5f3ee",
    checkbox_background_color_dark="#f5f3ee",
    checkbox_border_color="#d6d3cd",
    checkbox_border_color_dark="#d6d3cd",
    checkbox_label_text_color="#44403c",
    checkbox_label_text_color_dark="#44403c",
    checkbox_label_background_fill="#faf8f4",
    checkbox_label_background_fill_dark="#faf8f4",

    # Table / borders
    border_color_primary="#d6d3cd",
    border_color_primary_dark="#d6d3cd",
    color_accent_soft="#fef3c7",
    color_accent_soft_dark="#fef3c7",
)

# ── Plotly chart tokens ───────────────────────────────────────────────────────

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#faf8f4",
    font=dict(family="Source Sans 3, sans-serif", color="#44403c", size=12),
    xaxis=dict(
        showgrid=True,
        gridcolor="#e7e5e0",
        gridwidth=1,
        zeroline=False,
        linecolor="#d6d3cd",
        tickfont=dict(family="IBM Plex Mono, monospace", size=10, color="#78716c"),
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="#e7e5e0",
        gridwidth=1,
        zeroline=False,
        linecolor="#d6d3cd",
        tickfont=dict(family="IBM Plex Mono, monospace", size=10, color="#78716c"),
    ),
    legend=dict(
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#d6d3cd",
        borderwidth=1,
        font=dict(size=11, family="Source Sans 3, sans-serif", color="#44403c"),
    ),
    margin=dict(l=60, r=24, t=60, b=50),
)

COLORS = dict(
    price="#1e293b",
    accent="#b45309",
    volume="#c4b5a0",
    forecast="#b45309",
    band="rgba(180,83,9,0.10)",
    cooling="rgba(185,28,28,0.55)",
)


# ── Tab 1: Property Valuation ─────────────────────────────────────────────────

def predict_price(  # noqa: PLR0913
    town, flat_type, floor_area, storey_range, lease_year, remaining_lease, flat_model
):
    try:
        result = predictor.predict(
            town=town,
            flat_type=flat_type,
            floor_area_sqm=float(floor_area),
            storey_range=storey_range,
            lease_commence_date=int(lease_year),
            remaining_lease=remaining_lease,
            flat_model=flat_model,
        )
        output = (
            f"### Estimated Resale Price\n\n"
            f"**${result.predicted_price:,.0f}**\n\n"
            f"80% range: ${result.price_low:,.0f} – ${result.price_high:,.0f}\n\n"
            f"**${result.price_per_sqm:,.0f} / sqm**"
        )
        return output
    except Exception as e:
        return f"Error: {e}"


def build_valuation_tab():
    with gr.Tab("Property Valuation"):
        gr.Markdown("## HDB Resale Price Estimator")
        gr.Markdown(
            "Enter flat details to receive a model-estimated resale price. "
            "Predictions are generated by a LightGBM model trained on "
            "228k+ transactions (2017 \u2013 Apr 2026)."
        )

        with gr.Row():
            with gr.Column():
                town = gr.Dropdown(choices=TOWNS, value="ANG MO KIO", label="Town")
                flat_type = gr.Dropdown(
                    choices=FLAT_TYPES, value="4 ROOM", label="Flat Type"
                )
                floor_area = gr.Number(
                    value=93, label="Floor Area (sqm)", minimum=20, maximum=400
                )
                storey_range = gr.Textbox(
                    value="07 TO 09", label="Storey Range (e.g. '07 TO 09')"
                )
                lease_year = gr.Number(
                    value=1985, label="Lease Commencement Year",
                    minimum=1960, maximum=2030,
                )
                remaining_lease = gr.Textbox(
                    value="61 years 04 months", label="Remaining Lease"
                )
                flat_model = gr.Dropdown(
                    choices=FLAT_MODELS, value="IMPROVED", label="Flat Model"
                )
                btn = gr.Button("Estimate Price", variant="primary")

            with gr.Column():
                output = gr.Markdown(label="Prediction Result")

        btn.click(
            predict_price,
            inputs=[
                town, flat_type, floor_area, storey_range,
                lease_year, remaining_lease, flat_model,
            ],
            outputs=output,
        )


# ── Tab 2: Market Trends ──────────────────────────────────────────────────────

def plot_trends(town, flat_type, show_volume):
    df = get_historical_trend(town, flat_type)
    if df.empty:
        return go.Figure().update_layout(title=f"No data for {town} / {flat_type}")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["month"], y=df["median_price"],
        mode="lines+markers",
        name="Median Price",
        line=dict(color=COLORS["price"], width=2.5),
        marker=dict(size=4, color=COLORS["price"]),
        hovertemplate="<b>%{x|%b %Y}</b><br>Median: $%{y:,.0f}<extra></extra>",
    ))

    if show_volume:
        fig.add_trace(go.Bar(
            x=df["month"], y=df["transaction_count"],
            name="Transactions",
            yaxis="y2",
            opacity=0.3,
            marker_color=COLORS["volume"],
            hovertemplate="<b>%{x|%b %Y}</b><br>Vol: %{y}<extra></extra>",
        ))
        fig.update_layout(
            yaxis2=dict(
                title="Transaction Volume",
                overlaying="y",
                side="right",
                showgrid=False,
                tickfont=dict(
                    family="IBM Plex Mono, monospace", size=10, color="#78716c"
                ),
            )
        )

    # Cooling measure annotations
    cooling_dates = ["2018-07-01", "2021-12-01", "2022-09-01", "2023-04-01"]
    for i, cd in enumerate(cooling_dates):
        fig.add_shape(
            type="line", x0=cd, x1=cd, y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color=COLORS["cooling"], dash="dot", width=1.5),
        )
        fig.add_annotation(
            x=cd, y=1.0, xref="x", yref="paper",
            text=f"CM{i + 1}", showarrow=False,
            font=dict(size=8, color="#b91c1c", family="IBM Plex Mono, monospace"),
            yanchor="bottom", xanchor="left", xshift=3,
        )

    layout = dict(
        **CHART_LAYOUT,
        title=dict(
            text=f"{town}  \u00b7  {flat_type}",
            font=dict(
                family="Playfair Display, Georgia, serif", size=16, color="#1c1917"
            ),
            x=0, xanchor="left",
        ),
        xaxis_title="Month",
        yaxis_title="Median Price (SGD)",
        hovermode="x unified",
        height=480,
        legend=dict(
            **CHART_LAYOUT["legend"],
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        ),
    )
    fig.update_layout(**layout)
    return fig


def build_trends_tab():
    with gr.Tab("Market Trends"):
        gr.Markdown("## Historical Price Trends")
        gr.Markdown(
            "Monthly median resale prices by town and flat type (2017 \u2013 Apr 2026). "
            "Dotted red lines mark Singapore cooling measure events (CM1\u2013CM4)."
        )
        with gr.Row():
            town = gr.Dropdown(choices=TOWNS, value="ANG MO KIO", label="Town")
            flat_type = gr.Dropdown(
                choices=FLAT_TYPES, value="4 ROOM", label="Flat Type"
            )
            show_vol = gr.Checkbox(value=True, label="Show transaction volume")

        plot = gr.Plot()
        btn = gr.Button("Show Trends", variant="primary")
        btn.click(plot_trends, inputs=[town, flat_type, show_vol], outputs=plot)


# ── Tab 3: Price Forecast ─────────────────────────────────────────────────────

def plot_forecast(town, flat_type):
    history = get_historical_trend(town, flat_type)
    fc = get_forecast(town, flat_type)

    if history.empty:
        return go.Figure().update_layout(title=f"No data for {town} / {flat_type}")

    fig = go.Figure()

    # Historical (last 24 months for context)
    hist_recent = history.tail(24)
    fig.add_trace(go.Scatter(
        x=hist_recent["month"], y=hist_recent["median_price"],
        mode="lines+markers",
        name="Historical Median",
        line=dict(color=COLORS["price"], width=2.5),
        marker=dict(size=4, color=COLORS["price"]),
        hovertemplate="<b>%{x|%b %Y}</b><br>$%{y:,.0f}<extra></extra>",
    ))

    if not fc.empty:
        # Confidence band
        fig.add_trace(go.Scatter(
            x=pd.concat([fc["ds"], fc["ds"].iloc[::-1]]),
            y=pd.concat([fc["yhat_upper"], fc["yhat_lower"].iloc[::-1]]),
            fill="toself",
            fillcolor=COLORS["band"],
            line=dict(color="rgba(255,255,255,0)"),
            name="80% Confidence Band",
            showlegend=True,
            hoverinfo="skip",
        ))
        # Forecast line
        fig.add_trace(go.Scatter(
            x=fc["ds"], y=fc["yhat"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color=COLORS["forecast"], width=2.5, dash="dash"),
            marker=dict(size=4, color=COLORS["forecast"]),
            hovertemplate=(
                "<b>%{x|%b %Y}</b><br>Forecast: $%{y:,.0f}<extra></extra>"
            ),
        ))

    layout = dict(
        **CHART_LAYOUT,
        title=dict(
            text=f"{town}  \u00b7  {flat_type}  \u2014  12-Month Forecast",
            font=dict(
                family="Playfair Display, Georgia, serif", size=16, color="#1c1917"
            ),
            x=0, xanchor="left",
        ),
        xaxis_title="Month",
        yaxis_title="Median Price (SGD)",
        hovermode="x unified",
        height=480,
        legend=dict(
            **CHART_LAYOUT["legend"],
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        ),
    )
    fig.update_layout(**layout)
    return fig


def build_forecast_tab():
    with gr.Tab("Price Forecast"):
        gr.Markdown("## 12-Month Price Forecast")
        gr.Markdown(
            "Prophet time-series model trained on 2017 \u2013 Apr 2026 HDB transactions. "
            "Incorporates Singapore cooling measure dates as structural changepoints. "
            "Shaded band represents the 80% confidence interval."
        )
        with gr.Row():
            town = gr.Dropdown(choices=TOWNS, value="ANG MO KIO", label="Town")
            flat_type = gr.Dropdown(
                choices=FLAT_TYPES, value="4 ROOM", label="Flat Type"
            )

        plot = gr.Plot()
        btn = gr.Button("Show Forecast", variant="primary")
        btn.click(plot_forecast, inputs=[town, flat_type], outputs=plot)


# ── App assembly ──────────────────────────────────────────────────────────────

HEADER_HTML = """
<div class="app-header">
  <div class="header-eyebrow">Singapore Real Estate Intelligence</div>
  <div class="header-title">
    PropVal <em>SG</em>
    <span class="header-chip">AVM v1.0</span>
  </div>
  <div class="header-meta">
    <span>Automated Valuation Model</span>
    <span class="header-sep"></span>
    <span>228k+ transactions &middot; 2017&ndash;Apr 2026</span>
    <span class="header-sep"></span>
    <span>LightGBM &middot; MAPE 6.4% &middot; R&sup2; 0.92</span>
    <span class="header-sep"></span>
    <span>Data: <a href="https://data.gov.sg" target="_blank">data.gov.sg</a></span>
  </div>
</div>
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="PropVal SG \u2014 HDB AVM") as demo:
        gr.HTML(HEADER_HTML)
        build_valuation_tab()
        build_trends_tab()
        build_forecast_tab()

    return demo


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 7860))
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        css=CUSTOM_CSS,
        theme=THEME,
        head='<script>document.documentElement.dataset.theme="light";</script>',
    )
