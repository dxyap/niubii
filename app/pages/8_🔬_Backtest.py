"""
Backtest Page
=============
Strategy backtesting and analysis interface.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv()

from app import shared_state
from app.components.theme import apply_theme, get_chart_config

# Import backtest modules
from core.backtest import (
    BacktestConfig,
    BacktestEngine,
    BollingerBandStrategy,
    BuyAndHoldStrategy,
    CostModelConfig,
    MACrossoverStrategy,
    MomentumStrategy,
    RSIMeanReversionStrategy,
    SimpleCostModel,
    StrategyConfig,
    create_drawdown_chart,
    create_equity_chart,
    create_monthly_heatmap,
    create_returns_distribution,
    create_trade_analysis_chart,
    generate_summary_report,
)

st.set_page_config(
    page_title="Backtest | Oil Trading",
    page_icon="🔬",
    layout="wide"
)

apply_theme(st)

# Get data loader
context = shared_state.get_dashboard_context(lookback_days=365)
data_loader = context.data_loader

# Check connection
connection_status = data_loader.get_connection_status()
data_mode = connection_status.get("data_mode", "disconnected")

st.title("🔬 Strategy Backtesting")

if data_mode == "disconnected":
    st.error("🔴 Bloomberg Terminal not connected. Live data required.")
    st.info(f"Connection error: {connection_status.get('connection_error', 'Unknown')}")
    st.stop()

# Sidebar configuration
with st.sidebar:
    st.header("Backtest Configuration")

    # Strategy selection
    st.subheader("Strategy")
    strategy_type = st.selectbox(
        "Strategy Type",
        ["MA Crossover", "RSI Mean Reversion", "Bollinger Bands", "Momentum", "Buy & Hold"]
    )

    # Strategy parameters based on type
    if strategy_type == "MA Crossover":
        fast_period = st.slider("Fast MA Period", 5, 50, 10)
        slow_period = st.slider("Slow MA Period", 20, 200, 30)
        strategy_params = {"fast_period": fast_period, "slow_period": slow_period}
    elif strategy_type == "RSI Mean Reversion":
        rsi_period = st.slider("RSI Period", 5, 30, 14)
        oversold = st.slider("Oversold Level", 10, 40, 30)
        overbought = st.slider("Overbought Level", 60, 90, 70)
        strategy_params = {"rsi_period": rsi_period, "oversold_level": oversold, "overbought_level": overbought}
    elif strategy_type == "Bollinger Bands":
        bb_period = st.slider("BB Period", 10, 50, 20)
        num_std = st.slider("Num Std Dev", 1.0, 3.0, 2.0, 0.5)
        strategy_params = {"period": bb_period, "num_std": num_std}
    elif strategy_type == "Momentum":
        lookback = st.slider("Lookback Period", 5, 60, 20)
        strategy_params = {"lookback": lookback}
    else:
        strategy_params = {}

    st.divider()

    # Instrument
    st.subheader("Instrument")
    instrument = st.selectbox(
        "Instrument",
        ["Brent Crude (CO1)", "WTI Crude (CL1)", "RBOB Gasoline (XB1)", "Heating Oil (HO1)"]
    )

    ticker_map = {
        "Brent Crude (CO1)": "CO1 Comdty",
        "WTI Crude (CL1)": "CL1 Comdty",
        "RBOB Gasoline (XB1)": "XB1 Comdty",
        "Heating Oil (HO1)": "HO1 Comdty",
    }
    ticker = ticker_map[instrument]

    st.divider()

    # Time period
    st.subheader("Time Period")
    lookback_days = st.slider("Lookback (days)", 90, 730, 365)

    st.divider()

    # Capital & Costs
    st.subheader("Capital & Costs")
    initial_capital = st.number_input("Initial Capital ($)", 100000, 10000000, 1000000, 100000)
    position_size = st.slider("Position Size (contracts)", 1, 20, 5)
    commission = st.number_input("Commission ($/contract)", 0.0, 10.0, 2.50, 0.50)
    slippage = st.number_input("Slippage (bps)", 0.0, 10.0, 1.0, 0.5)

# Main content
run_backtest = st.button("🚀 Run Backtest", type="primary", use_container_width=True)

if run_backtest or "backtest_result" in st.session_state:
    if run_backtest:
        with st.spinner("Running backtest..."):
            try:
                # Fetch historical data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=lookback_days)

                hist_data = data_loader.get_historical(
                    ticker,
                    start_date=start_date,
                    end_date=end_date
                )

                if hist_data is None or hist_data.empty:
                    st.error("Failed to fetch historical data. Please try again.")
                    st.stop()

                # Create strategy
                config = StrategyConfig(position_size=position_size, max_position=position_size * 2)

                if strategy_type == "MA Crossover":
                    strategy = MACrossoverStrategy(**strategy_params, config=config)
                elif strategy_type == "RSI Mean Reversion":
                    strategy = RSIMeanReversionStrategy(**strategy_params, config=config)
                elif strategy_type == "Bollinger Bands":
                    strategy = BollingerBandStrategy(**strategy_params, config=config)
                elif strategy_type == "Momentum":
                    strategy = MomentumStrategy(**strategy_params, config=config)
                else:
                    strategy = BuyAndHoldStrategy(config=config)

                # Create cost model
                cost_config = CostModelConfig(
                    commission_per_contract=commission,
                    slippage_ticks=slippage,
                    tick_size=0.01,
                    contract_multiplier=1000,
                )
                cost_model = SimpleCostModel(cost_config)

                # Create engine and run
                backtest_config = BacktestConfig(
                    initial_capital=initial_capital,
                    commission_per_contract=commission,
                    slippage_pct=slippage / 100,
                    max_position_size=position_size * 2,
                )

                engine = BacktestEngine(backtest_config, cost_model)
                result = engine.run(strategy, hist_data, ticker.replace(" Comdty", ""))

                # Store in session state
                st.session_state.backtest_result = result
                st.session_state.backtest_data = hist_data

            except Exception as e:
                st.error(f"Backtest failed: {e}")
                st.stop()

    # Display results
    result = st.session_state.get("backtest_result")

    if result:
        st.success(f"✅ Backtest completed: {result.strategy_name}")

        # Key Metrics Row
        st.subheader("Performance Summary")

        m = result.metrics

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            delta_color = "normal" if m.total_return_pct >= 0 else "inverse"
            st.metric("Total Return", f"{m.total_return_pct:+.2f}%", delta_color=delta_color)

        with col2:
            st.metric("Sharpe Ratio", f"{m.sharpe_ratio:.2f}")

        with col3:
            st.metric("Max Drawdown", f"{m.max_drawdown:.2f}%")

        with col4:
            st.metric("Win Rate", f"{m.win_rate:.1f}%")

        with col5:
            st.metric("Total Trades", f"{m.total_trades}")

        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Equity Curve",
            "📊 Trade Analysis",
            "📅 Monthly Returns",
            "📋 Full Report"
        ])

        with tab1:
            col1, col2 = st.columns([2, 1])

            with col1:
                # Equity curve
                fig = create_equity_chart(result, height=400)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, config=get_chart_config())

                # Drawdown chart
                fig = create_drawdown_chart(result, height=200)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, config=get_chart_config())

            with col2:
                st.markdown("**Risk-Adjusted Returns**")
                st.metric("Sortino Ratio", f"{m.sortino_ratio:.2f}")
                st.metric("Calmar Ratio", f"{m.calmar_ratio:.2f}")

                st.divider()

                st.markdown("**Risk Metrics**")
                st.metric("Volatility (Ann.)", f"{m.annualized_volatility:.2f}%")
                st.metric("VaR (95%)", f"{m.var_95:.2f}%")
                st.metric("CVaR (95%)", f"{m.cvar_95:.2f}%")

                st.divider()

                st.markdown("**Return Metrics**")
                st.metric("CAGR", f"{m.cagr:+.2f}%")
                st.metric("Annualized Return", f"{m.annualized_return:+.2f}%")

        with tab2:
            col1, col2 = st.columns([2, 1])

            with col1:
                # Trade analysis chart
                fig = create_trade_analysis_chart(result, height=350)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, config=get_chart_config())

                # Returns distribution
                fig = create_returns_distribution(result, height=250)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, config=get_chart_config())

            with col2:
                st.markdown("**Trade Statistics**")
                st.metric("Winning Trades", f"{m.winning_trades}")
                st.metric("Losing Trades", f"{m.losing_trades}")
                st.metric("Profit Factor", f"{m.profit_factor:.2f}")

                st.divider()

                st.markdown("**P&L Statistics**")
                st.metric("Average Win", f"${m.avg_win:,.2f}")
                st.metric("Average Loss", f"${m.avg_loss:,.2f}")
                st.metric("Largest Win", f"${m.largest_win:,.2f}")
                st.metric("Largest Loss", f"${m.largest_loss:,.2f}")
                st.metric("Expectancy", f"${m.expectancy:,.2f}")

            # Trade table
            st.markdown("**Trade History**")
            if not result.trades.empty:
                st.dataframe(
                    result.trades.tail(20),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "pnl": st.column_config.NumberColumn("P&L", format="$%.2f"),
                        "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                    }
                )
            else:
                st.info("No trades executed during backtest period.")

        with tab3:
            # Monthly returns heatmap
            fig = create_monthly_heatmap(result, height=400)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config=get_chart_config())

            # Monthly summary table
            st.markdown("**Monthly Statistics**")

            if not result.equity_curve.empty:
                monthly = result.equity_curve.resample("M").last().pct_change().dropna() * 100

                monthly_stats = pd.DataFrame({
                    "Month": monthly.index.strftime("%Y-%m"),
                    "Return (%)": monthly.values.round(2),
                })

                # Add color coding
                st.dataframe(
                    monthly_stats.tail(12),
                    use_container_width=True,
                    hide_index=True,
                )

        with tab4:
            # Full text report
            report_text = generate_summary_report(result)

            st.code(report_text, language="text")

            # Download buttons
            col1, col2 = st.columns(2)

            with col1:
                st.download_button(
                    "📄 Download Report (TXT)",
                    report_text,
                    file_name=f"backtest_{result.strategy_name}_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain",
                )

            with col2:
                # Export equity curve
                if not result.equity_curve.empty:
                    csv = result.equity_curve.to_csv()
                    st.download_button(
                        "📊 Download Equity Curve (CSV)",
                        csv,
                        file_name=f"equity_{result.strategy_name}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                    )

else:
    # Show instructions
    st.info("""
    ### How to Use the Backtester

    1. **Select a Strategy**: Choose from built-in strategies in the sidebar
    2. **Configure Parameters**: Adjust strategy-specific parameters
    3. **Set Capital & Costs**: Define initial capital and transaction costs
    4. **Run Backtest**: Click the button above to execute

    ### Available Strategies

    | Strategy | Description |
    |----------|-------------|
    | **MA Crossover** | Buy when fast MA crosses above slow MA, sell when crosses below |
    | **RSI Mean Reversion** | Buy when oversold (RSI < 30), sell when overbought (RSI > 70) |
    | **Bollinger Bands** | Buy at lower band, sell at upper band |
    | **Momentum** | Buy on breakout above recent high, sell on breakdown |
    | **Buy & Hold** | Simple buy and hold benchmark |

    ### Metrics Explained

    - **Sharpe Ratio**: Risk-adjusted return (higher is better, > 1 is good)
    - **Sortino Ratio**: Like Sharpe but only penalizes downside volatility
    - **Calmar Ratio**: CAGR divided by max drawdown
    - **Profit Factor**: Gross profits / Gross losses (> 1 is profitable)
    - **Win Rate**: Percentage of winning trades
    """)

# Footer
st.divider()
st.caption(
    "⚠️ Past performance does not guarantee future results. "
    "Backtest results are hypothetical and may not reflect actual trading performance."
)
