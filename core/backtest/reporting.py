"""
Backtest Reporting
==================
Generate comprehensive backtest reports and visualizations.

Provides:
- Performance summary reports
- Equity curve visualization
- Trade analysis
- Drawdown analysis
- Monthly returns heatmap
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import io

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from .engine import BacktestResult
from .metrics import PerformanceMetrics, calculate_monthly_returns, calculate_drawdown_series


# Chart colors (matching dashboard theme)
COLORS = {
    "primary": "#00A3E0",
    "secondary": "#8B5CF6",
    "profit": "#00DC82",
    "loss": "#FF5252",
    "background": "rgba(15, 23, 42, 0.8)",
    "grid": "rgba(51, 65, 85, 0.4)",
    "text": "#E2E8F0",
    "text_secondary": "#94A3B8",
}


def generate_summary_report(result: BacktestResult) -> str:
    """
    Generate text summary report of backtest results.
    
    Args:
        result: BacktestResult to report on
        
    Returns:
        Formatted text report
    """
    m = result.metrics
    
    report = []
    report.append("=" * 60)
    report.append(f"BACKTEST REPORT: {result.strategy_name}")
    report.append("=" * 60)
    report.append("")
    
    # Period
    report.append("PERIOD")
    report.append("-" * 40)
    report.append(f"Start Date:      {m.start_date}")
    report.append(f"End Date:        {m.end_date}")
    report.append(f"Trading Days:    {m.trading_days}")
    report.append("")
    
    # Returns
    report.append("RETURNS")
    report.append("-" * 40)
    report.append(f"Total Return:    ${m.total_return:,.2f} ({m.total_return_pct:+.2f}%)")
    report.append(f"Annualized:      {m.annualized_return:+.2f}%")
    report.append(f"CAGR:            {m.cagr:+.2f}%")
    report.append("")
    
    # Risk
    report.append("RISK")
    report.append("-" * 40)
    report.append(f"Volatility:      {m.annualized_volatility:.2f}%")
    report.append(f"Max Drawdown:    {m.max_drawdown:.2f}%")
    report.append(f"Max DD Duration: {m.max_drawdown_duration} days")
    report.append(f"VaR (95%):       {m.var_95:.2f}%")
    report.append(f"CVaR (95%):      {m.cvar_95:.2f}%")
    report.append("")
    
    # Risk-Adjusted Returns
    report.append("RISK-ADJUSTED RETURNS")
    report.append("-" * 40)
    report.append(f"Sharpe Ratio:    {m.sharpe_ratio:.2f}")
    report.append(f"Sortino Ratio:   {m.sortino_ratio:.2f}")
    report.append(f"Calmar Ratio:    {m.calmar_ratio:.2f}")
    report.append("")
    
    # Trading
    report.append("TRADING")
    report.append("-" * 40)
    report.append(f"Total Trades:    {m.total_trades}")
    report.append(f"Winning Trades:  {m.winning_trades}")
    report.append(f"Losing Trades:   {m.losing_trades}")
    report.append(f"Win Rate:        {m.win_rate:.1f}%")
    report.append(f"Profit Factor:   {m.profit_factor:.2f}")
    report.append(f"Avg Win:         ${m.avg_win:,.2f}")
    report.append(f"Avg Loss:        ${m.avg_loss:,.2f}")
    report.append(f"Largest Win:     ${m.largest_win:,.2f}")
    report.append(f"Largest Loss:    ${m.largest_loss:,.2f}")
    report.append(f"Expectancy:      ${m.expectancy:,.2f}")
    report.append("")
    
    report.append("=" * 60)
    
    return "\n".join(report)


def generate_html_report(result: BacktestResult) -> str:
    """
    Generate HTML report with embedded charts.
    
    Args:
        result: BacktestResult to report on
        
    Returns:
        HTML string
    """
    m = result.metrics
    
    html = []
    html.append("<!DOCTYPE html>")
    html.append("<html>")
    html.append("<head>")
    html.append("<title>Backtest Report - {}</title>".format(result.strategy_name))
    html.append("""
    <style>
        body { 
            font-family: 'Inter', -apple-system, sans-serif; 
            background: #0f172a; 
            color: #e2e8f0;
            margin: 0;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #00A3E0; border-bottom: 2px solid #00A3E0; padding-bottom: 10px; }
        h2 { color: #8B5CF6; margin-top: 30px; }
        .metrics-grid { 
            display: grid; 
            grid-template-columns: repeat(4, 1fr); 
            gap: 15px; 
            margin: 20px 0;
        }
        .metric-card {
            background: rgba(30, 41, 59, 0.8);
            border-radius: 8px;
            padding: 15px;
            border-left: 4px solid #00A3E0;
        }
        .metric-label { color: #94A3B8; font-size: 12px; text-transform: uppercase; }
        .metric-value { color: #e2e8f0; font-size: 24px; font-weight: bold; font-family: monospace; }
        .positive { color: #00DC82; }
        .negative { color: #FF5252; }
        table { 
            width: 100%; 
            border-collapse: collapse; 
            margin: 20px 0;
            background: rgba(30, 41, 59, 0.5);
        }
        th, td { 
            padding: 12px; 
            text-align: left; 
            border-bottom: 1px solid rgba(51, 65, 85, 0.5);
        }
        th { background: rgba(15, 23, 42, 0.8); color: #94A3B8; }
        tr:hover { background: rgba(51, 65, 85, 0.3); }
        .chart-container { margin: 30px 0; }
    </style>
    """)
    html.append("</head>")
    html.append("<body>")
    html.append("<div class='container'>")
    
    # Header
    html.append(f"<h1>📊 Backtest Report: {result.strategy_name}</h1>")
    html.append(f"<p>Period: {m.start_date} to {m.end_date} ({m.trading_days} days)</p>")
    
    # Key Metrics
    html.append("<h2>Key Metrics</h2>")
    html.append("<div class='metrics-grid'>")
    
    metrics = [
        ("Total Return", f"{m.total_return_pct:+.2f}%", m.total_return_pct >= 0),
        ("Sharpe Ratio", f"{m.sharpe_ratio:.2f}", m.sharpe_ratio >= 0),
        ("Max Drawdown", f"{m.max_drawdown:.2f}%", False),
        ("Win Rate", f"{m.win_rate:.1f}%", m.win_rate >= 50),
        ("CAGR", f"{m.cagr:+.2f}%", m.cagr >= 0),
        ("Sortino Ratio", f"{m.sortino_ratio:.2f}", m.sortino_ratio >= 0),
        ("Profit Factor", f"{m.profit_factor:.2f}", m.profit_factor >= 1),
        ("Total Trades", f"{m.total_trades}", True),
    ]
    
    for label, value, is_positive in metrics:
        color_class = "positive" if is_positive else "negative"
        html.append(f"""
        <div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value {color_class}'>{value}</div>
        </div>
        """)
    
    html.append("</div>")
    
    # Trade Statistics Table
    html.append("<h2>Trade Statistics</h2>")
    html.append("<table>")
    html.append("<tr><th>Metric</th><th>Value</th></tr>")
    
    trade_stats = [
        ("Winning Trades", m.winning_trades),
        ("Losing Trades", m.losing_trades),
        ("Average Win", f"${m.avg_win:,.2f}"),
        ("Average Loss", f"${m.avg_loss:,.2f}"),
        ("Largest Win", f"${m.largest_win:,.2f}"),
        ("Largest Loss", f"${m.largest_loss:,.2f}"),
        ("Expectancy", f"${m.expectancy:,.2f}"),
    ]
    
    for label, value in trade_stats:
        html.append(f"<tr><td>{label}</td><td>{value}</td></tr>")
    
    html.append("</table>")
    
    # Risk Metrics Table
    html.append("<h2>Risk Metrics</h2>")
    html.append("<table>")
    html.append("<tr><th>Metric</th><th>Value</th></tr>")
    
    risk_stats = [
        ("Annualized Volatility", f"{m.annualized_volatility:.2f}%"),
        ("Downside Volatility", f"{m.downside_volatility:.2f}%"),
        ("Max Drawdown Duration", f"{m.max_drawdown_duration} days"),
        ("Calmar Ratio", f"{m.calmar_ratio:.2f}"),
        ("VaR (95%)", f"{m.var_95:.2f}%"),
        ("CVaR (95%)", f"{m.cvar_95:.2f}%"),
    ]
    
    for label, value in risk_stats:
        html.append(f"<tr><td>{label}</td><td>{value}</td></tr>")
    
    html.append("</table>")
    
    html.append("</div>")
    html.append("</body>")
    html.append("</html>")
    
    return "\n".join(html)


def create_equity_chart(result: BacktestResult, height: int = 400) -> Optional[Any]:
    """
    Create equity curve chart.
    
    Args:
        result: BacktestResult
        height: Chart height
        
    Returns:
        Plotly figure or None if plotly not available
    """
    if not HAS_PLOTLY:
        return None
    
    equity = result.equity_curve
    
    fig = go.Figure()
    
    # Equity curve
    fig.add_trace(go.Scatter(
        x=equity.index,
        y=equity.values,
        name="Equity",
        line=dict(color=COLORS["primary"], width=2),
        fill="tozeroy",
        fillcolor="rgba(0, 163, 224, 0.1)",
    ))
    
    # Initial capital line
    initial = result.config.initial_capital if result.config else equity.iloc[0]
    fig.add_hline(
        y=initial,
        line_dash="dash",
        line_color=COLORS["text_secondary"],
        annotation_text="Initial Capital"
    )
    
    fig.update_layout(
        title="Equity Curve",
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS["background"],
        xaxis=dict(gridcolor=COLORS["grid"]),
        yaxis=dict(
            title="Portfolio Value ($)",
            gridcolor=COLORS["grid"],
            tickformat="$,.0f",
        ),
        hovermode="x unified",
    )
    
    return fig


def create_drawdown_chart(result: BacktestResult, height: int = 250) -> Optional[Any]:
    """
    Create drawdown chart.
    
    Args:
        result: BacktestResult
        height: Chart height
        
    Returns:
        Plotly figure or None
    """
    if not HAS_PLOTLY:
        return None
    
    dd = result.drawdown
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dd.index,
        y=dd.values,
        name="Drawdown",
        line=dict(color=COLORS["loss"], width=1.5),
        fill="tozeroy",
        fillcolor="rgba(255, 82, 82, 0.2)",
    ))
    
    fig.update_layout(
        title="Drawdown",
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS["background"],
        xaxis=dict(gridcolor=COLORS["grid"]),
        yaxis=dict(
            title="Drawdown (%)",
            gridcolor=COLORS["grid"],
            tickformat=".1f",
        ),
    )
    
    return fig


def create_returns_distribution(result: BacktestResult, height: int = 300) -> Optional[Any]:
    """
    Create returns distribution histogram.
    
    Args:
        result: BacktestResult
        height: Chart height
        
    Returns:
        Plotly figure or None
    """
    if not HAS_PLOTLY:
        return None
    
    returns = result.returns * 100  # Convert to percentage
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name="Daily Returns",
        marker_color=COLORS["primary"],
        opacity=0.7,
    ))
    
    # Add vertical line at mean
    mean_ret = returns.mean()
    fig.add_vline(
        x=mean_ret,
        line_dash="dash",
        line_color=COLORS["profit"],
        annotation_text=f"Mean: {mean_ret:.2f}%"
    )
    
    fig.update_layout(
        title="Returns Distribution",
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS["background"],
        xaxis=dict(title="Daily Return (%)", gridcolor=COLORS["grid"]),
        yaxis=dict(title="Frequency", gridcolor=COLORS["grid"]),
    )
    
    return fig


def create_monthly_heatmap(result: BacktestResult, height: int = 350) -> Optional[Any]:
    """
    Create monthly returns heatmap.
    
    Args:
        result: BacktestResult
        height: Chart height
        
    Returns:
        Plotly figure or None
    """
    if not HAS_PLOTLY:
        return None
    
    monthly = calculate_monthly_returns(result.equity_curve)
    
    if monthly.empty:
        return None
    
    # Prepare data for heatmap
    months = [col for col in monthly.columns if col != "Year"]
    z_data = monthly[months].values
    
    # Create color scale
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=months,
        y=monthly.index.astype(str),
        colorscale=[
            [0, COLORS["loss"]],
            [0.5, COLORS["background"]],
            [1, COLORS["profit"]],
        ],
        zmid=0,
        text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in z_data],
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>",
    ))
    
    fig.update_layout(
        title="Monthly Returns (%)",
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed"),
    )
    
    return fig


def create_trade_analysis_chart(result: BacktestResult, height: int = 300) -> Optional[Any]:
    """
    Create trade analysis chart (cumulative P&L per trade).
    
    Args:
        result: BacktestResult
        height: Chart height
        
    Returns:
        Plotly figure or None
    """
    if not HAS_PLOTLY or result.trades.empty:
        return None
    
    if "pnl" not in result.trades.columns:
        return None
    
    trades = result.trades.copy()
    trades["cumulative_pnl"] = trades["pnl"].cumsum()
    trades["trade_num"] = range(1, len(trades) + 1)
    
    fig = go.Figure()
    
    # Cumulative P&L line
    fig.add_trace(go.Scatter(
        x=trades["trade_num"],
        y=trades["cumulative_pnl"],
        name="Cumulative P&L",
        line=dict(color=COLORS["primary"], width=2),
    ))
    
    # Individual trade P&L bars
    colors = [COLORS["profit"] if pnl >= 0 else COLORS["loss"] for pnl in trades["pnl"]]
    
    fig.add_trace(go.Bar(
        x=trades["trade_num"],
        y=trades["pnl"],
        name="Trade P&L",
        marker_color=colors,
        opacity=0.6,
    ))
    
    fig.update_layout(
        title="Trade Analysis",
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS["background"],
        xaxis=dict(title="Trade #", gridcolor=COLORS["grid"]),
        yaxis=dict(title="P&L ($)", gridcolor=COLORS["grid"], tickformat="$,.0f"),
        barmode="overlay",
    )
    
    return fig


def create_full_report(result: BacktestResult) -> Dict[str, Any]:
    """
    Create full report with all charts.
    
    Args:
        result: BacktestResult
        
    Returns:
        Dictionary with text report and all charts
    """
    report = {
        "text": generate_summary_report(result),
        "html": generate_html_report(result),
        "metrics": result.metrics.to_dict(),
        "charts": {},
    }
    
    if HAS_PLOTLY:
        report["charts"]["equity"] = create_equity_chart(result)
        report["charts"]["drawdown"] = create_drawdown_chart(result)
        report["charts"]["returns_dist"] = create_returns_distribution(result)
        report["charts"]["monthly"] = create_monthly_heatmap(result)
        report["charts"]["trades"] = create_trade_analysis_chart(result)
    
    return report


def compare_results_chart(
    results: Dict[str, BacktestResult],
    height: int = 400
) -> Optional[Any]:
    """
    Create comparison chart for multiple backtest results.
    
    Args:
        results: Dictionary mapping names to BacktestResults
        height: Chart height
        
    Returns:
        Plotly figure or None
    """
    if not HAS_PLOTLY:
        return None
    
    fig = go.Figure()
    
    colors = [COLORS["primary"], COLORS["secondary"], COLORS["profit"], 
              COLORS["loss"], "#F59E0B", "#EC4899"]
    
    for i, (name, result) in enumerate(results.items()):
        # Normalize to percentage returns
        equity = result.equity_curve
        normalized = (equity / equity.iloc[0] - 1) * 100
        
        fig.add_trace(go.Scatter(
            x=normalized.index,
            y=normalized.values,
            name=name,
            line=dict(color=colors[i % len(colors)], width=2),
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["text_secondary"])
    
    fig.update_layout(
        title="Strategy Comparison (Cumulative Returns %)",
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS["background"],
        xaxis=dict(gridcolor=COLORS["grid"]),
        yaxis=dict(
            title="Return (%)",
            gridcolor=COLORS["grid"],
            tickformat=".1f",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hovermode="x unified",
    )
    
    return fig
