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
            f"Range: ${result.price_low:,.0f} – ${result.price_high:,.0f}\n\n"
            f"**${result.price_per_sqm:,.0f} / sqm**"
        )
        return output
    except Exception as e:
        return f"Error: {e}"


def build_valuation_tab():
    with gr.Tab("Property Valuation"):
        gr.Markdown("## HDB Resale Price Estimator")
        gr.Markdown("Enter flat details to get an estimated resale price from our AVM model.")

        with gr.Row():
            with gr.Column():
                town = gr.Dropdown(choices=TOWNS, value="ANG MO KIO", label="Town")
                flat_type = gr.Dropdown(
                    choices=FLAT_TYPES, value="4 ROOM", label="Flat Type"
                )
                floor_area = gr.Number(value=93, label="Floor Area (sqm)", minimum=20, maximum=400)
                storey_range = gr.Textbox(value="07 TO 09", label="Storey Range (e.g. '07 TO 09')")
                lease_year = gr.Number(
                    value=1985, label="Lease Commencement Year", minimum=1960, maximum=2030
                )
                remaining_lease = gr.Textbox(value="61 years 04 months", label="Remaining Lease")
                flat_model = gr.Dropdown(choices=FLAT_MODELS, value="IMPROVED", label="Flat Model")
                btn = gr.Button("Estimate Price", variant="primary")

            with gr.Column():
                output = gr.Markdown(label="Prediction Result")

        btn.click(
            predict_price,
            inputs=[town, flat_type, floor_area, storey_range, lease_year, remaining_lease, flat_model],  # noqa: E501
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
        line=dict(color="#2563eb", width=2),
        hovertemplate="<b>%{x|%b %Y}</b><br>Median: $%{y:,.0f}<extra></extra>",
    ))

    if show_volume:
        fig.add_trace(go.Bar(
            x=df["month"], y=df["transaction_count"],
            name="Transactions",
            yaxis="y2",
            opacity=0.3,
            marker_color="#94a3b8",
            hovertemplate="<b>%{x|%b %Y}</b><br>Vol: %{y}<extra></extra>",
        ))
        fig.update_layout(yaxis2=dict(title="Transaction Volume", overlaying="y", side="right"))

    # Shade cooling measure periods (add as shapes to avoid plotly vline string bug)
    cooling_dates = ["2018-07-01", "2021-12-01", "2022-09-01", "2023-04-01"]
    for cd in cooling_dates:
        fig.add_shape(type="line", x0=cd, x1=cd, y0=0, y1=1, xref="x", yref="paper",
                      line=dict(color="red", dash="dot", width=1), opacity=0.5)

    fig.update_layout(
        title=f"{town} · {flat_type} — Resale Price Trend (2017–2026)",
        xaxis_title="Month",
        yaxis_title="Median Price (SGD)",
        hovermode="x unified",
        height=500,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_trends_tab():
    with gr.Tab("Market Trends"):
        gr.Markdown("## Historical Price Trends")
        gr.Markdown(
            "Track monthly median resale prices by town and flat type. "
            "Red dotted lines mark Singapore cooling measure events."
        )
        with gr.Row():
            town = gr.Dropdown(choices=TOWNS, value="ANG MO KIO", label="Town")
            flat_type = gr.Dropdown(choices=FLAT_TYPES, value="4 ROOM", label="Flat Type")
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
        line=dict(color="#2563eb", width=2),
        hovertemplate="<b>%{x|%b %Y}</b><br>$%{y:,.0f}<extra></extra>",
    ))

    if not fc.empty:
        # Confidence band
        fig.add_trace(go.Scatter(
            x=pd.concat([fc["ds"], fc["ds"].iloc[::-1]]),
            y=pd.concat([fc["yhat_upper"], fc["yhat_lower"].iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(251,146,60,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="80% Confidence Band",
            showlegend=True,
        ))
        # Forecast line
        fig.add_trace(go.Scatter(
            x=fc["ds"], y=fc["yhat"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#f97316", width=2, dash="dash"),
            hovertemplate="<b>%{x|%b %Y}</b><br>Forecast: $%{y:,.0f}<extra></extra>",
        ))

    fig.update_layout(
        title=f"{town} · {flat_type} — 12-Month Price Forecast",
        xaxis_title="Month",
        yaxis_title="Median Price (SGD)",
        hovermode="x unified",
        height=500,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_forecast_tab():
    with gr.Tab("Price Forecast"):
        gr.Markdown("## 12-Month Price Forecast")
        gr.Markdown(
            "Prophet time-series model trained on 2017–2026 HDB transactions. "
            "Incorporates Singapore cooling measure dates as structural changepoints."
        )
        with gr.Row():
            town = gr.Dropdown(choices=TOWNS, value="ANG MO KIO", label="Town")
            flat_type = gr.Dropdown(choices=FLAT_TYPES, value="4 ROOM", label="Flat Type")

        plot = gr.Plot()
        btn = gr.Button("Show Forecast", variant="primary")
        btn.click(plot_forecast, inputs=[town, flat_type], outputs=plot)


# ── App assembly ──────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="Singapore HDB AVM",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# Singapore HDB Resale AVM\n"
            "**Automated Valuation Model** trained on 228k+ HDB transactions (2017–2026). "
            "Data: [data.gov.sg](https://data.gov.sg)."
        )
        build_valuation_tab()
        build_trends_tab()
        build_forecast_tab()

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
