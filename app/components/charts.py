"""
Chart Components
================
Reusable, beautifully styled chart components for the dashboard.
Features professional trading terminal aesthetics with high readability.
"""

import contextlib

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# =============================================================================
# CHART THEME CONSTANTS
# =============================================================================

# Color palette - inspired by professional trading terminals
CHART_COLORS = {
    # Candlestick colors
    "candle_up": "#00DC82",        # Vivid green for bullish
    "candle_down": "#FF5252",      # Vivid red for bearish
    "candle_up_fill": "#00DC82",   # Solid fill for up candles
    "candle_down_fill": "#FF5252", # Solid fill for down candles

    # Line colors
    "primary": "#00A3E0",          # Electric blue
    "secondary": "#8B5CF6",        # Purple
    "tertiary": "#F59E0B",         # Amber
    "accent": "#EC4899",           # Pink

    # Moving averages
    "ma_fast": "#FFB020",          # Gold for fast MA
    "ma_slow": "#A855F7",          # Purple for slow MA
    "ma_long": "#06B6D4",          # Cyan for long MA

    # Backgrounds and grids
    "bg_primary": "rgba(10, 15, 30, 0.95)",
    "bg_secondary": "rgba(15, 23, 42, 0.8)",
    "grid": "rgba(51, 65, 85, 0.4)",
    "grid_light": "rgba(71, 85, 105, 0.25)",

    # Text
    "text_primary": "#E2E8F0",
    "text_secondary": "#94A3B8",

    # Profit/Loss
    "profit": "#00DC82",
    "loss": "#FF5252",
}

# Common layout settings for all charts
BASE_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": CHART_COLORS["bg_secondary"],
    "font": {
        "family": "'IBM Plex Mono', 'SF Mono', monospace",
        "size": 12,
        "color": CHART_COLORS["text_primary"],
    },
    "xaxis": {
        "gridcolor": CHART_COLORS["grid"],
        "gridwidth": 1,
        "showgrid": True,
        "zeroline": False,
        "tickfont": {"size": 11, "color": CHART_COLORS["text_secondary"]},
        "title_font": {"size": 12, "color": CHART_COLORS["text_secondary"]},
    },
    "yaxis": {
        "gridcolor": CHART_COLORS["grid"],
        "gridwidth": 1,
        "showgrid": True,
        "zeroline": False,
        "tickfont": {"size": 11, "color": CHART_COLORS["text_secondary"]},
        "title_font": {"size": 12, "color": CHART_COLORS["text_secondary"]},
        "tickformat": "$.2f",
        "side": "right",  # Price scale on right side (trader convention)
    },
    "legend": {
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "right",
        "x": 1,
        "bgcolor": "rgba(0,0,0,0)",
        "font": {"size": 11, "color": CHART_COLORS["text_secondary"]},
    },
    "margin": {"l": 10, "r": 60, "t": 40, "b": 40},
    "hovermode": "x unified",
    "hoverlabel": {
        "bgcolor": CHART_COLORS["bg_primary"],
        "bordercolor": CHART_COLORS["grid"],
        "font": {"family": "'IBM Plex Mono', monospace", "size": 12},
    },
}


def apply_base_layout(fig: go.Figure, height: int = 400, **kwargs) -> go.Figure:
    """Apply consistent base layout to any chart."""
    layout = {**BASE_LAYOUT, "height": height, **kwargs}
    fig.update_layout(**layout)
    return fig


# =============================================================================
# CANDLESTICK CHART
# =============================================================================

def create_candlestick_chart(
    data: pd.DataFrame,
    title: str = "",
    height: int = 450,
    show_volume: bool = False,
    show_ma: bool = True,
    ma_periods: list[int] = None,
) -> go.Figure:
    """
    Create a professional candlestick chart with solid green/red fills.

    Args:
        data: DataFrame with OHLCV data (PX_OPEN, PX_HIGH, PX_LOW, PX_LAST)
        title: Chart title
        height: Chart height in pixels
        show_volume: Include volume subplot
        show_ma: Show moving averages
        ma_periods: List of MA periods to display

    Returns:
        Plotly figure with professional styling
    """
    # Create figure with secondary y-axis if volume is shown
    if ma_periods is None:
        ma_periods = [20, 50]
    if show_volume and 'PX_VOLUME' in data.columns:
        from plotly.subplots import make_subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.8, 0.2],
        )
    else:
        fig = go.Figure()

    # Candlestick trace with solid fills
    candlestick = go.Candlestick(
        x=data.index,
        open=data['PX_OPEN'],
        high=data['PX_HIGH'],
        low=data['PX_LOW'],
        close=data['PX_LAST'],
        name='Price',
        increasing={
            "line": {"color": CHART_COLORS["candle_up"], "width": 1},
            "fillcolor": CHART_COLORS["candle_up_fill"],
        },
        decreasing={
            "line": {"color": CHART_COLORS["candle_down"], "width": 1},
            "fillcolor": CHART_COLORS["candle_down_fill"],
        },
        whiskerwidth=0.8,
        hoverinfo='x+y',
    )

    if show_volume and 'PX_VOLUME' in data.columns:
        fig.add_trace(candlestick, row=1, col=1)
    else:
        fig.add_trace(candlestick)

    # Add moving averages
    if show_ma:
        ma_colors = [CHART_COLORS["ma_fast"], CHART_COLORS["ma_slow"], CHART_COLORS["ma_long"]]

        for i, period in enumerate(ma_periods):
            if len(data) >= period:
                ma = data['PX_LAST'].rolling(period).mean()
                ma_trace = go.Scatter(
                    x=data.index,
                    y=ma,
                    name=f'MA {period}',
                    line={"color": ma_colors[i % len(ma_colors)], "width": 1.5},
                    hovertemplate=f'MA{period}: $%{{y:.2f}}<extra></extra>',
                )

                if show_volume and 'PX_VOLUME' in data.columns:
                    fig.add_trace(ma_trace, row=1, col=1)
                else:
                    fig.add_trace(ma_trace)

    # Add volume bars
    if show_volume and 'PX_VOLUME' in data.columns:
        colors = [CHART_COLORS["candle_up"] if c >= o else CHART_COLORS["candle_down"]
                  for c, o in zip(data['PX_LAST'], data['PX_OPEN'])]

        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['PX_VOLUME'],
                marker_color=colors,
                marker_opacity=0.6,
                name='Volume',
                hovertemplate='Vol: %{y:,.0f}<extra></extra>',
            ),
            row=2, col=1
        )

        # Update axes for subplot
        fig.update_yaxes(title_text="Price", row=1, col=1, tickformat="$.2f", side="right")
        fig.update_yaxes(title_text="Volume", row=2, col=1, tickformat=".2s", side="right")

    # Apply professional layout
    fig = apply_base_layout(
        fig,
        height=height,
        title={
            "text": title,
            "font": {"size": 14, "color": CHART_COLORS["text_primary"]},
            "x": 0,
            "xanchor": 'left',
        } if title else None,
        xaxis_rangeslider_visible=False,
    )

    return fig


# =============================================================================
# FUTURES CURVE CHART
# =============================================================================

def create_futures_curve_chart(
    curve_data: pd.DataFrame,
    secondary_curve: pd.DataFrame | None = None,
    title: str = "Futures Curve",
    height: int = 350,
    show_spread: bool = False,
) -> go.Figure:
    """
    Create an elegant futures curve line chart with proper y-axis scaling.

    Args:
        curve_data: Primary curve DataFrame with 'contract_month' (or 'month') and 'price' columns
        secondary_curve: Optional second curve for comparison
        title: Chart title
        height: Chart height in pixels
        show_spread: Show spread between curves

    Returns:
        Plotly figure with professional styling
    """
    fig = go.Figure()

    def _prepare_curve_df(df: pd.DataFrame):
        """Sort curve chronologically and determine plotting columns."""
        if df is None or df.empty:
            return None, None, [], []

        sorted_df = df.copy()
        x_col = 'contract_month' if 'contract_month' in sorted_df.columns else 'month'

        if 'contract_date' in sorted_df.columns:
            sorted_df = sorted_df.sort_values('contract_date')
            order_values = pd.to_datetime(sorted_df['contract_date']).tolist()
        elif x_col == 'month':
            sorted_df = sorted_df.sort_values('month')
            order_values = sorted_df['month'].tolist()
        else:
            parsed_dates = pd.to_datetime(sorted_df[x_col], format="%b-%y", errors='coerce')
            sorted_df = sorted_df.assign(_order_key=parsed_dates).sort_values('_order_key')
            order_values = sorted_df['_order_key'].tolist()
            sorted_df = sorted_df.drop(columns='_order_key')

        sorted_df = sorted_df.reset_index(drop=True)
        return sorted_df, x_col, sorted_df[x_col].tolist(), order_values

    def _build_category_array(primary_labels, primary_orders, secondary_labels, secondary_orders):
        """Build chronological category order for axis labels."""
        pairs = []
        for label, value in zip(primary_labels, primary_orders):
            if label is not None:
                pairs.append((label, value))
        for label, value in zip(secondary_labels, secondary_orders):
            if label is not None:
                pairs.append((label, value))

        if not pairs:
            return []

        def _normalize(value):
            if isinstance(value, pd.Timestamp):
                return value.value
            if value is None:
                return float('inf')
            with contextlib.suppress(TypeError):
                if pd.isna(value):
                    return float('inf')
            with contextlib.suppress(TypeError, ValueError):
                return float(value)
            return float('inf')

        order_map: dict[str, float] = {}
        for label, value in pairs:
            normalized = _normalize(value)
            existing = order_map.get(label)
            if existing is None or normalized < existing:
                order_map[label] = normalized

        ordered = sorted(order_map.items(), key=lambda item: item[1])
        return [label for label, _ in ordered]

    curve_data, primary_x_column, primary_labels, primary_orders = _prepare_curve_df(curve_data)
    if curve_data is None or curve_data.empty:
        return fig

    secondary_labels: list = []
    secondary_orders: list = []
    secondary_x_column = None
    if secondary_curve is not None and not secondary_curve.empty:
        secondary_curve, secondary_x_column, secondary_labels, secondary_orders = _prepare_curve_df(secondary_curve)
    else:
        secondary_curve = None

    # Determine y-axis range with padding for readability
    all_prices = list(curve_data['price'])
    if secondary_curve is not None:
        all_prices.extend(list(secondary_curve['price']))

    price_min = min(all_prices)
    price_max = max(all_prices)
    price_range = price_max - price_min

    # Add 15% padding for better visualization
    y_min = price_min - (price_range * 0.15)
    y_max = price_max + (price_range * 0.15)

    # If range is very small, create reasonable bounds
    if price_range < 2:
        mid = (price_min + price_max) / 2
        y_min = mid - 2
        y_max = mid + 2

    # Primary curve with area fill for depth
    fig.add_trace(go.Scatter(
        x=curve_data[primary_x_column],
        y=curve_data['price'],
        name='Brent',
        mode='lines+markers',
        line={"color": CHART_COLORS["primary"], "width": 3, "shape": 'spline'},
        marker={
            "size": 8,
            "color": CHART_COLORS["primary"],
            "line": {"width": 2, "color": 'white'},
            "symbol": 'circle',
        },
        fill='tozeroy',
        fillcolor='rgba(0, 163, 224, 0.1)',
        hovertemplate='%{x}<br>$%{y:.2f}<extra>Brent</extra>',
    ))

    # Secondary curve if provided
    if secondary_curve is not None:
        fig.add_trace(go.Scatter(
            x=secondary_curve[secondary_x_column],
            y=secondary_curve['price'],
            name='WTI',
            mode='lines+markers',
            line={"color": CHART_COLORS["tertiary"], "width": 3, "shape": 'spline'},
            marker={
                "size": 8,
                "color": CHART_COLORS["tertiary"],
                "line": {"width": 2, "color": 'white'},
                "symbol": 'diamond',
            },
            hovertemplate='%{x}<br>$%{y:.2f}<extra>WTI</extra>',
        ))

    categoryarray = _build_category_array(
        primary_labels,
        primary_orders,
        secondary_labels,
        secondary_orders,
    )
    extra_layout = {
        "yaxis": dict(
            **BASE_LAYOUT["yaxis"],
            range=[y_min, y_max],
            title_text="Price ($/bbl)",
            dtick=max(1, round(price_range / 5)),
        ),
        "xaxis": dict(
            **BASE_LAYOUT["xaxis"],
            title_text="Contract Month",
        ),
        "xaxis_rangeslider_visible": False,
    }
    if categoryarray and isinstance(categoryarray[0], str):
        extra_layout["xaxis"].update({
            "categoryorder": "array",
            "categoryarray": categoryarray,
        })
        if len(categoryarray) > 12:
            extra_layout["xaxis"]["tickangle"] = -45
    elif len(curve_data) > 12:
        extra_layout["xaxis"]["tickangle"] = -45

    # Apply professional layout with adjusted axes
    fig = apply_base_layout(
        fig,
        height=height,
        title={
            "text": title,
            "font": {"size": 14, "color": CHART_COLORS["text_primary"]},
            "x": 0,
            "xanchor": 'left',
        } if title else None,
        **extra_layout,
    )

    return fig


# =============================================================================
# P&L CHART
# =============================================================================

def create_pnl_chart(
    pnl_data: pd.Series,
    cumulative: bool = True,
    height: int = 300,
    title: str = "",
) -> go.Figure:
    """
    Create a beautiful P&L chart with gradient fills.

    Args:
        pnl_data: P&L data series with timestamp index
        cumulative: Show cumulative P&L
        height: Chart height
        title: Chart title

    Returns:
        Plotly figure with professional styling
    """
    fig = go.Figure()

    y_data = pnl_data.cumsum() if cumulative else pnl_data
    current_pnl = y_data.iloc[-1] if len(y_data) > 0 else 0

    # Determine colors based on P&L
    if current_pnl >= 0:
        line_color = CHART_COLORS["profit"]
        fill_color = "rgba(0, 220, 130, 0.15)"
    else:
        line_color = CHART_COLORS["loss"]
        fill_color = "rgba(255, 82, 82, 0.15)"

    fig.add_trace(go.Scatter(
        x=pnl_data.index,
        y=y_data,
        fill='tozeroy',
        fillcolor=fill_color,
        line={"color": line_color, "width": 2.5},
        name='P&L',
        hovertemplate='$%{y:,.0f}<extra></extra>',
    ))

    # Zero line
    fig.add_hline(y=0, line_dash='solid', line_color='rgba(255,255,255,0.3)', line_width=1)

    fig = apply_base_layout(
        fig,
        height=height,
        title={
            "text": title,
            "font": {"size": 14, "color": CHART_COLORS["text_primary"]},
            "x": 0,
            "xanchor": 'left',
        } if title else None,
        yaxis=dict(
            **BASE_LAYOUT["yaxis"],
            title_text="P&L ($)",
            tickformat="$,.0f",
        ),
    )

    return fig


# =============================================================================
# GAUGE CHART
# =============================================================================

def create_gauge_chart(
    value: float,
    title: str,
    min_val: float = 0,
    max_val: float = 100,
    warning_threshold: float = 75,
    critical_threshold: float = 90,
    height: int = 220,
) -> go.Figure:
    """
    Create a modern gauge chart for utilization metrics.

    Args:
        value: Current value
        title: Chart title
        min_val: Minimum value
        max_val: Maximum value
        warning_threshold: Warning threshold
        critical_threshold: Critical threshold
        height: Chart height

    Returns:
        Plotly figure with professional styling
    """
    # Determine bar color based on value
    if value >= critical_threshold:
        bar_color = CHART_COLORS["loss"]
    elif value >= warning_threshold:
        bar_color = CHART_COLORS["tertiary"]
    else:
        bar_color = CHART_COLORS["primary"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 14, 'color': CHART_COLORS["text_secondary"]}},
        number={'font': {'size': 36, 'color': CHART_COLORS["text_primary"]}, 'suffix': '%'},
        gauge={
            'axis': {
                'range': [min_val, max_val],
                'tickwidth': 1,
                'tickcolor': CHART_COLORS["text_secondary"],
                'tickfont': {'size': 10, 'color': CHART_COLORS["text_secondary"]},
            },
            'bar': {'color': bar_color, 'thickness': 0.8},
            'bgcolor': CHART_COLORS["bg_secondary"],
            'borderwidth': 0,
            'steps': [
                {'range': [min_val, warning_threshold], 'color': 'rgba(0, 220, 130, 0.15)'},
                {'range': [warning_threshold, critical_threshold], 'color': 'rgba(245, 158, 11, 0.15)'},
                {'range': [critical_threshold, max_val], 'color': 'rgba(255, 82, 82, 0.15)'}
            ],
            'threshold': {
                'line': {'color': CHART_COLORS["loss"], 'width': 3},
                'thickness': 0.8,
                'value': critical_threshold
            }
        }
    ))

    fig.update_layout(
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': CHART_COLORS["text_primary"]},
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )

    return fig


# =============================================================================
# HEATMAP
# =============================================================================

def create_heatmap(
    data: pd.DataFrame,
    title: str = "",
    colorscale: str = "RdBu_r",
    height: int = 400,
) -> go.Figure:
    """
    Create a professional correlation heatmap.

    Args:
        data: DataFrame for heatmap
        title: Chart title
        colorscale: Color scale name
        height: Chart height

    Returns:
        Plotly figure
    """
    fig = px.imshow(
        data,
        text_auto='.2f',
        color_continuous_scale=colorscale,
        zmin=-1 if colorscale == "RdBu_r" else None,
        zmax=1 if colorscale == "RdBu_r" else None,
    )

    fig = apply_base_layout(
        fig,
        height=height,
        title={
            "text": title,
            "font": {"size": 14, "color": CHART_COLORS["text_primary"]},
            "x": 0,
            "xanchor": 'left',
        } if title else None,
    )

    # Update text styling
    fig.update_traces(
        textfont={"size": 11, "color": 'white'},
    )

    return fig


# =============================================================================
# BAR CHART
# =============================================================================

def create_bar_chart(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    height: int = 300,
    color_by_value: bool = True,
    orientation: str = 'v',
) -> go.Figure:
    """
    Create a styled bar chart with optional value-based coloring.

    Args:
        data: DataFrame with data
        x: Column name for x-axis
        y: Column name for y-axis
        title: Chart title
        height: Chart height
        color_by_value: Color bars based on positive/negative values
        orientation: 'v' for vertical, 'h' for horizontal

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    if color_by_value:
        colors = [CHART_COLORS["profit"] if v >= 0 else CHART_COLORS["loss"] for v in data[y]]
    else:
        colors = CHART_COLORS["primary"]

    fig.add_trace(go.Bar(
        x=data[x] if orientation == 'v' else data[y],
        y=data[y] if orientation == 'v' else data[x],
        marker_color=colors,
        marker_line_width=0,
        orientation=orientation,
        text=[f"${v:,.0f}" if isinstance(v, (int, float)) else str(v) for v in data[y]],
        textposition='outside',
        textfont={"size": 11, "color": CHART_COLORS["text_primary"]},
        hovertemplate='%{x}<br>$%{y:,.0f}<extra></extra>',
    ))

    fig = apply_base_layout(
        fig,
        height=height,
        title={
            "text": title,
            "font": {"size": 14, "color": CHART_COLORS["text_primary"]},
            "x": 0,
            "xanchor": 'left',
        } if title else None,
    )

    return fig


# =============================================================================
# VOLUME BAR CHART (for standalone volume)
# =============================================================================

def create_volume_chart(
    data: pd.DataFrame,
    height: int = 120,
) -> go.Figure:
    """
    Create a standalone volume bar chart.

    Args:
        data: DataFrame with PX_VOLUME, PX_OPEN, PX_LAST columns
        height: Chart height

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    colors = [CHART_COLORS["candle_up"] if c >= o else CHART_COLORS["candle_down"]
              for c, o in zip(data['PX_LAST'], data['PX_OPEN'])]

    fig.add_trace(go.Bar(
        x=data.index,
        y=data['PX_VOLUME'],
        marker_color=colors,
        marker_opacity=0.7,
        name='Volume',
        hovertemplate='%{y:,.0f}<extra></extra>',
    ))

    volume_yaxis = dict(BASE_LAYOUT["yaxis"])
    volume_yaxis.update({"tickformat": ".2s", "title_text": ""})

    fig = apply_base_layout(
        fig,
        height=height,
        yaxis=volume_yaxis,
        margin={"l": 10, "r": 60, "t": 5, "b": 30},
    )

    return fig


# =============================================================================
# OPEN INTEREST CHART
# =============================================================================

def create_open_interest_chart(
    data: pd.DataFrame,
    height: int = 120,
) -> go.Figure:
    """
    Create a standalone open interest area chart.

    Args:
        data: DataFrame with OPEN_INT column
        height: Chart height

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['OPEN_INT'],
        mode='lines',
        line={"color": CHART_COLORS["secondary"], "width": 1.5},
        fill='tozeroy',
        fillcolor='rgba(139, 92, 246, 0.15)',
        name='Open Interest',
        hovertemplate='%{y:,.0f}<extra></extra>',
    ))

    oi_yaxis = dict(BASE_LAYOUT["yaxis"])
    oi_yaxis.update({"tickformat": ".2s", "title_text": ""})

    fig = apply_base_layout(
        fig,
        height=height,
        yaxis=oi_yaxis,
        margin={"l": 10, "r": 60, "t": 5, "b": 30},
    )

    return fig


# =============================================================================
# LINE CHART
# =============================================================================

def create_line_chart(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    height: int = 300,
    fill: bool = False,
    show_markers: bool = True,
) -> go.Figure:
    """
    Create a styled line chart.

    Args:
        data: DataFrame with data
        x: Column name for x-axis
        y: Column name for y-axis
        title: Chart title
        height: Chart height
        fill: Fill area under line
        show_markers: Show data point markers

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data[x],
        y=data[y],
        mode='lines+markers' if show_markers else 'lines',
        line={"color": CHART_COLORS["primary"], "width": 2.5},
        marker={
            "size": 8,
            "color": CHART_COLORS["primary"],
            "line": {"width": 2, "color": 'white'},
        } if show_markers else None,
        fill='tozeroy' if fill else None,
        fillcolor='rgba(0, 163, 224, 0.1)' if fill else None,
        hovertemplate='%{x}<br>$%{y:.2f}<extra></extra>',
    ))

    fig = apply_base_layout(
        fig,
        height=height,
        title={
            "text": title,
            "font": {"size": 14, "color": CHART_COLORS["text_primary"]},
            "x": 0,
            "xanchor": 'left',
        } if title else None,
    )

    return fig
