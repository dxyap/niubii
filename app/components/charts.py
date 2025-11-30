"""
Chart Components
================
Reusable chart components for the dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def create_candlestick_chart(
    data: pd.DataFrame,
    title: str = "",
    show_volume: bool = True,
    show_ma: bool = True
) -> go.Figure:
    """
    Create a candlestick chart with optional volume and moving averages.
    
    Args:
        data: DataFrame with OHLCV data
        title: Chart title
        show_volume: Show volume subplot
        show_ma: Show moving averages
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['PX_OPEN'],
        high=data['PX_HIGH'],
        low=data['PX_LOW'],
        close=data['PX_LAST'],
        name='Price'
    ))
    
    if show_ma:
        # 20-day MA
        ma20 = data['PX_LAST'].rolling(20).mean()
        fig.add_trace(go.Scatter(
            x=data.index, y=ma20,
            name='MA 20',
            line=dict(color='orange', width=1)
        ))
        
        # 50-day MA
        ma50 = data['PX_LAST'].rolling(50).mean()
        fig.add_trace(go.Scatter(
            x=data.index, y=ma50,
            name='MA 50',
            line=dict(color='purple', width=1)
        ))
    
    fig.update_layout(
        title=title,
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=30 if title else 0, b=0),
    )
    
    return fig


def create_curve_chart(
    wti_curve: pd.DataFrame,
    brent_curve: Optional[pd.DataFrame] = None,
    title: str = "Futures Curve"
) -> go.Figure:
    """
    Create a futures curve chart.
    
    Args:
        wti_curve: WTI curve data
        brent_curve: Optional Brent curve data
        title: Chart title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=wti_curve['month'],
        y=wti_curve['price'],
        name='WTI',
        mode='lines+markers',
        line=dict(color='#00D26A', width=2)
    ))
    
    if brent_curve is not None:
        fig.add_trace(go.Scatter(
            x=brent_curve['month'],
            y=brent_curve['price'],
            name='Brent',
            mode='lines+markers',
            line=dict(color='#00A3E0', width=2)
        ))
    
    fig.update_layout(
        title=title,
        template='plotly_dark',
        xaxis_title='Month',
        yaxis_title='Price ($/bbl)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    
    return fig


def create_pnl_chart(
    pnl_data: pd.DataFrame,
    cumulative: bool = True
) -> go.Figure:
    """
    Create a P&L chart.
    
    Args:
        pnl_data: P&L data with timestamp index
        cumulative: Show cumulative P&L
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    y_data = pnl_data.cumsum() if cumulative else pnl_data
    
    colors = ['#00D26A' if v >= 0 else '#FF4B4B' for v in y_data]
    
    fig.add_trace(go.Scatter(
        x=pnl_data.index,
        y=y_data,
        fill='tozeroy',
        fillcolor='rgba(0, 210, 106, 0.3)' if y_data.iloc[-1] >= 0 else 'rgba(255, 75, 75, 0.3)',
        line=dict(color='#00D26A' if y_data.iloc[-1] >= 0 else '#FF4B4B', width=2),
        name='P&L'
    ))
    
    fig.add_hline(y=0, line_dash='solid', line_color='white', line_width=1)
    
    fig.update_layout(
        template='plotly_dark',
        yaxis_title='P&L ($)',
        margin=dict(l=0, r=0, t=10, b=0),
    )
    
    return fig


def create_gauge_chart(
    value: float,
    title: str,
    min_val: float = 0,
    max_val: float = 100,
    warning_threshold: float = 75,
    critical_threshold: float = 90
) -> go.Figure:
    """
    Create a gauge chart for utilization metrics.
    
    Args:
        value: Current value
        title: Chart title
        min_val: Minimum value
        max_val: Maximum value
        warning_threshold: Warning threshold
        critical_threshold: Critical threshold
        
    Returns:
        Plotly figure
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 14}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1},
            'bar': {'color': "#00A3E0"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [min_val, warning_threshold], 'color': 'rgba(0, 210, 106, 0.3)'},
                {'range': [warning_threshold, critical_threshold], 'color': 'rgba(255, 165, 0, 0.3)'},
                {'range': [critical_threshold, max_val], 'color': 'rgba(255, 75, 75, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': critical_threshold
            }
        }
    ))
    
    fig.update_layout(
        template='plotly_dark',
        height=200,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    
    return fig


def create_heatmap(
    data: pd.DataFrame,
    title: str = "",
    colorscale: str = "RdBu_r"
) -> go.Figure:
    """
    Create a heatmap chart.
    
    Args:
        data: DataFrame for heatmap
        title: Chart title
        colorscale: Color scale name
        
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
    
    fig.update_layout(
        title=title,
        template='plotly_dark',
        margin=dict(l=0, r=0, t=30 if title else 0, b=0),
    )
    
    return fig
