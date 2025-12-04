"""
Automation Page
===============
Execution automation and paper trading interface.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from app import shared_state
from app.components.theme import apply_theme, COLORS, get_chart_config

# Import execution modules
from core.execution import (
    OrderManager, Order, OrderStatus, OrderSide, OrderType,
    PositionSizer, SizingConfig, SizingMethod, SizingResult,
    TWAPAlgorithm, VWAPAlgorithm, AlgorithmConfig, AlgorithmType,
    PaperTradingEngine, PaperTradingConfig,
    AutomationEngine, AutomationRule, RuleConfig, RuleCondition, 
    RuleAction, RuleStatus, ConditionType, ActionType,
    create_signal_rule,
)

st.set_page_config(
    page_title="Automation | Oil Trading",
    page_icon="🤖",
    layout="wide"
)

apply_theme(st)

# Initialize session state
if "paper_trading_engine" not in st.session_state:
    config = PaperTradingConfig(initial_capital=1_000_000)
    st.session_state.paper_trading_engine = PaperTradingEngine(config)
    st.session_state.paper_trading_active = False

if "automation_engine" not in st.session_state:
    st.session_state.automation_engine = AutomationEngine()

# Get data loader
context = shared_state.get_dashboard_context(lookback_days=30)
data_loader = context.data_loader

st.title("🤖 Execution & Automation")

# Connection status
connection_status = data_loader.get_connection_status()
data_mode = connection_status.get("data_mode", "disconnected")

if data_mode == "disconnected":
    st.warning("🟡 Running in disconnected mode. Using simulated prices.")

# Tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Paper Trading",
    "📐 Position Sizing",
    "⚡ Execution Algos",
    "🔧 Automation Rules",
    "📋 Order Management"
])

# =============================================================================
# TAB 1: PAPER TRADING
# =============================================================================
with tab1:
    st.subheader("Paper Trading Dashboard")
    
    paper_engine = st.session_state.paper_trading_engine
    
    # Control buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if not st.session_state.paper_trading_active:
            if st.button("▶️ Start Session", type="primary"):
                paper_engine.start_session()
                st.session_state.paper_trading_active = True
                st.rerun()
        else:
            if st.button("⏹️ Stop Session"):
                session_result = paper_engine.stop_session()
                st.session_state.paper_trading_active = False
                st.session_state.last_session = session_result
                st.rerun()
    
    with col2:
        session_status = "🟢 Active" if st.session_state.paper_trading_active else "⚪ Inactive"
        st.metric("Session Status", session_status)
    
    with col3:
        account = paper_engine.get_account_info()
        st.metric("NAV", f"${account['nav']:,.0f}")
    
    with col4:
        pnl = account.get('realized_pnl', 0) + account.get('unrealized_pnl', 0)
        st.metric("Total P&L", f"${pnl:,.0f}", delta=f"{pnl/paper_engine.config.initial_capital*100:.2f}%")
    
    st.divider()
    
    # Order entry
    if st.session_state.paper_trading_active:
        st.markdown("### Quick Order Entry")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            symbol = st.selectbox("Symbol", ["CL1", "CO1", "XB1", "HO1"], key="pt_symbol")
        
        with col2:
            side = st.selectbox("Side", ["BUY", "SELL"], key="pt_side")
        
        with col3:
            quantity = st.number_input("Quantity", min_value=1, max_value=50, value=5, key="pt_qty")
        
        with col4:
            order_type = st.selectbox("Order Type", ["MARKET", "LIMIT"], key="pt_type")
        
        with col5:
            if order_type == "LIMIT":
                # Get current price for default
                current_price = 75.0
                limit_price = st.number_input("Limit Price", value=current_price, key="pt_limit")
            else:
                limit_price = None
                st.write("")
        
        # Update prices (simulated)
        prices = {
            "CL1": 72.50 + np.random.uniform(-0.5, 0.5),
            "CO1": 77.80 + np.random.uniform(-0.5, 0.5),
            "XB1": 2.35 + np.random.uniform(-0.02, 0.02),
            "HO1": 2.45 + np.random.uniform(-0.02, 0.02),
        }
        paper_engine.update_prices(prices)
        
        if st.button("Submit Order", type="primary"):
            order = paper_engine.submit_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price if order_type == "LIMIT" else None,
                strategy="Manual",
            )
            if order:
                st.success(f"Order submitted: {order.order_id}")
            else:
                st.error("Order rejected")
        
        st.divider()
    
    # Positions and P&L
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Current Positions")
        positions = paper_engine.get_positions()
        
        if positions:
            pos_data = []
            for symbol, pos in positions.items():
                pos_data.append({
                    "Symbol": symbol,
                    "Qty": pos["quantity"],
                    "Avg Price": f"${pos['avg_price']:.2f}",
                    "Market": f"${pos['market_price']:.2f}",
                    "Unrealized P&L": f"${pos['unrealized_pnl']:,.0f}",
                    "Realized P&L": f"${pos['realized_pnl']:,.0f}",
                })
            st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)
        else:
            st.info("No open positions")
    
    with col2:
        st.markdown("### P&L Summary")
        pnl_summary = paper_engine.get_pnl_summary()
        
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            st.metric("Initial Capital", f"${pnl_summary['initial_capital']:,.0f}")
            st.metric("Realized P&L", f"${pnl_summary['realized_pnl']:,.0f}")
            st.metric("Total Commission", f"${pnl_summary['total_commission']:,.0f}")
        
        with metrics_col2:
            st.metric("Unrealized P&L", f"${pnl_summary['unrealized_pnl']:,.0f}")
            st.metric("Return %", f"{pnl_summary['return_pct']:.2f}%")
            st.metric("Max Drawdown", f"{pnl_summary['max_drawdown']:.2f}%")
    
    # Recent fills
    st.markdown("### Recent Fills")
    fills = paper_engine.get_fills()
    if fills:
        fills_df = pd.DataFrame(fills[-10:])
        st.dataframe(fills_df, use_container_width=True, hide_index=True)
    else:
        st.info("No fills yet")

# =============================================================================
# TAB 2: POSITION SIZING
# =============================================================================
with tab2:
    st.subheader("Position Sizing Calculator")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Input Parameters")
        
        sizing_method = st.selectbox(
            "Sizing Method",
            ["Volatility Targeting", "Fixed Fractional", "Kelly Criterion", "ATR-Based", "VaR-Based"],
            key="sizing_method"
        )
        
        method_map = {
            "Volatility Targeting": SizingMethod.VOLATILITY_TARGET,
            "Fixed Fractional": SizingMethod.FIXED_FRACTIONAL,
            "Kelly Criterion": SizingMethod.KELLY,
            "ATR-Based": SizingMethod.ATR_BASED,
            "VaR-Based": SizingMethod.VAR_BASED,
        }
        
        account_value = st.number_input("Account Value ($)", 100000, 10000000, 1000000, 100000)
        current_price = st.number_input("Current Price ($)", 10.0, 200.0, 75.0, 1.0)
        volatility = st.slider("Annualized Volatility (%)", 10, 60, 25) / 100
        
        st.divider()
        
        if sizing_method == "Fixed Fractional":
            risk_per_trade = st.slider("Risk Per Trade (%)", 0.5, 5.0, 2.0, 0.5) / 100
            stop_loss = st.slider("Stop Loss (%)", 0.5, 5.0, 2.0, 0.5) / 100
        elif sizing_method == "Kelly Criterion":
            win_rate = st.slider("Win Rate (%)", 40, 70, 55) / 100
            win_loss_ratio = st.slider("Win/Loss Ratio", 1.0, 3.0, 1.5, 0.1)
            kelly_fraction = st.slider("Kelly Fraction (%)", 10, 50, 25) / 100
        elif sizing_method == "Volatility Targeting":
            target_vol = st.slider("Target Volatility (%)", 5, 30, 15) / 100
        else:
            risk_per_trade = st.slider("Risk Per Trade (%)", 0.5, 5.0, 2.0, 0.5) / 100
    
    with col2:
        st.markdown("### Sizing Result")
        
        # Create sizing config
        config = SizingConfig(
            method=method_map[sizing_method],
            account_value=account_value,
            max_position_pct=0.25,
            max_position_contracts=50,
        )
        
        if sizing_method == "Fixed Fractional":
            config.risk_per_trade_pct = risk_per_trade
            from core.execution.sizing import FixedFractional
            sizer = FixedFractional(config)
            result = sizer.calculate_size(price=current_price, volatility=volatility, stop_loss_pct=stop_loss)
        elif sizing_method == "Kelly Criterion":
            config.kelly_fraction = kelly_fraction
            from core.execution.sizing import KellyCriterion
            sizer = KellyCriterion(config)
            result = sizer.calculate_size(
                price=current_price, 
                volatility=volatility,
                win_rate=win_rate,
                avg_win_loss_ratio=win_loss_ratio,
            )
        elif sizing_method == "Volatility Targeting":
            config.target_volatility = target_vol
            from core.execution.sizing import VolatilityTargeting
            sizer = VolatilityTargeting(config)
            result = sizer.calculate_size(price=current_price, volatility=volatility)
        else:
            config.risk_per_trade_pct = risk_per_trade
            from core.execution.sizing import VolatilityTargeting
            sizer = VolatilityTargeting(config)
            result = sizer.calculate_size(price=current_price, volatility=volatility)
        
        # Display results
        st.metric("Recommended Contracts", result.contracts)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Notional Value", f"${result.notional_value:,.0f}")
            st.metric("Risk Amount", f"${result.risk_amount:,.0f}")
        with col_b:
            st.metric("Position %", f"{result.position_pct:.1f}%")
            st.metric("Method", result.method.value)
        
        st.markdown("**Rationale:**")
        st.info(result.rationale)
        
        if result.adjustments:
            st.warning("**Adjustments Applied:**")
            for adj in result.adjustments:
                st.write(f"- {adj}")

# =============================================================================
# TAB 3: EXECUTION ALGORITHMS
# =============================================================================
with tab3:
    st.subheader("Execution Algorithms")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Algorithm Configuration")
        
        algo_type = st.selectbox(
            "Algorithm Type",
            ["TWAP", "VWAP", "POV", "Implementation Shortfall"],
            key="algo_type"
        )
        
        st.divider()
        
        total_quantity = st.number_input("Total Quantity", 10, 100, 20)
        duration_minutes = st.slider("Duration (minutes)", 15, 240, 60)
        num_slices = st.slider("Number of Slices", 4, 24, 12)
        
        st.divider()
        
        if algo_type == "POV":
            participation = st.slider("Participation Rate (%)", 5, 25, 10) / 100
        elif algo_type == "Implementation Shortfall":
            urgency = st.slider("Urgency", 0.0, 1.0, 0.5, 0.1)
        
        randomize = st.checkbox("Randomize Timing & Size", value=True)
    
    with col2:
        st.markdown("### Execution Schedule Preview")
        
        # Create mock order
        from core.execution.oms import Order
        mock_order = Order(
            order_id="PREVIEW-001",
            symbol="CL1",
            side=OrderSide.BUY,
            quantity=total_quantity,
        )
        
        # Create algorithm config
        algo_config = AlgorithmConfig(
            algo_type=AlgorithmType[algo_type.replace(" ", "_").upper().replace("IMPLEMENTATION_SHORTFALL", "IS")],
            duration_minutes=duration_minutes,
            num_slices=num_slices,
            randomize_timing=randomize,
            randomize_size=randomize,
        )
        
        if algo_type == "POV":
            algo_config.participation_rate = participation
        elif algo_type == "Implementation Shortfall":
            algo_config.urgency = urgency
        
        # Generate schedule
        from core.execution.algorithms import get_execution_algorithm
        algo = get_execution_algorithm(algo_config)
        slices = algo.generate_schedule(mock_order, current_price=75.0)
        
        # Display schedule
        schedule_data = []
        for s in slices:
            schedule_data.append({
                "Slice": s.sequence,
                "Scheduled Time": s.scheduled_time.strftime("%H:%M:%S"),
                "Quantity": s.quantity,
                "% of Total": f"{s.quantity/total_quantity*100:.1f}%",
            })
        
        st.dataframe(pd.DataFrame(schedule_data), use_container_width=True, hide_index=True)
        
        # Visualization
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        times = [s.scheduled_time for s in slices]
        quantities = [s.quantity for s in slices]
        cumulative = np.cumsum(quantities)
        
        fig.add_trace(go.Bar(
            x=list(range(1, len(slices)+1)),
            y=quantities,
            name="Slice Quantity",
            marker_color=COLORS.get("bullish", "#10b981"),
        ))
        
        fig.add_trace(go.Scatter(
            x=list(range(1, len(slices)+1)),
            y=cumulative,
            name="Cumulative",
            yaxis="y2",
            line=dict(color=COLORS.get("accent", "#f59e0b")),
        ))
        
        fig.update_layout(
            title=f"{algo_type} Execution Schedule",
            xaxis_title="Slice Number",
            yaxis_title="Quantity",
            yaxis2=dict(
                title="Cumulative",
                overlaying="y",
                side="right",
            ),
            height=350,
            template="plotly_dark",
        )
        
        st.plotly_chart(fig, use_container_width=True, config=get_chart_config())

# =============================================================================
# TAB 4: AUTOMATION RULES
# =============================================================================
with tab4:
    st.subheader("Automation Rules Engine")
    
    automation_engine = st.session_state.automation_engine
    
    # Statistics
    stats = automation_engine.get_statistics()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rules", stats["total_rules"])
    with col2:
        st.metric("Active Rules", stats["active_rules"])
    with col3:
        st.metric("Total Triggers", stats["total_triggers"])
    with col4:
        st.metric("Recent Executions", stats["recent_executions"])
    
    st.divider()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Create New Rule")
        
        rule_name = st.text_input("Rule Name", "Long on High Confidence Signal")
        rule_symbol = st.selectbox("Symbol", ["CL1", "CO1", "XB1", "HO1"], key="rule_symbol")
        rule_direction = st.selectbox("Signal Direction", ["LONG", "SHORT"])
        min_confidence = st.slider("Minimum Confidence (%)", 50, 90, 65)
        
        st.markdown("**Sizing**")
        rule_sizing = st.selectbox(
            "Sizing Method",
            ["Volatility Targeting", "Fixed Quantity", "Risk Parity"],
            key="rule_sizing"
        )
        
        if rule_sizing == "Fixed Quantity":
            fixed_qty = st.number_input("Quantity", 1, 50, 5)
        else:
            risk_pct = st.slider("Risk %", 0.5, 5.0, 2.0, 0.5) / 100
        
        if st.button("Add Rule", type="primary"):
            rule_config = create_signal_rule(
                name=rule_name,
                symbol=rule_symbol,
                direction=rule_direction,
                min_confidence=min_confidence,
                sizing_method=SizingMethod.VOLATILITY_TARGET if rule_sizing == "Volatility Targeting" else SizingMethod.FIXED,
                risk_pct=risk_pct if rule_sizing != "Fixed Quantity" else 0.02,
            )
            automation_engine.add_rule(rule_config)
            st.success(f"Rule added: {rule_name}")
            st.rerun()
    
    with col2:
        st.markdown("### Active Rules")
        
        rules = automation_engine.get_rules()
        
        if rules:
            for rule in rules:
                with st.expander(f"📌 {rule.config.name}", expanded=False):
                    st.write(f"**ID:** {rule.config.rule_id}")
                    st.write(f"**Status:** {rule.config.status.value}")
                    st.write(f"**Conditions:** {len(rule.config.conditions)}")
                    st.write(f"**Triggers:** {rule.config.trigger_count}")
                    
                    if rule.config.last_triggered:
                        st.write(f"**Last Triggered:** {rule.config.last_triggered.strftime('%Y-%m-%d %H:%M')}")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if rule.is_active:
                            if st.button("Pause", key=f"pause_{rule.config.rule_id}"):
                                automation_engine.update_rule_status(rule.config.rule_id, RuleStatus.PAUSED)
                                st.rerun()
                        else:
                            if st.button("Activate", key=f"activate_{rule.config.rule_id}"):
                                automation_engine.update_rule_status(rule.config.rule_id, RuleStatus.ACTIVE)
                                st.rerun()
                    
                    with col_b:
                        if st.button("Delete", key=f"delete_{rule.config.rule_id}"):
                            automation_engine.remove_rule(rule.config.rule_id)
                            st.rerun()
        else:
            st.info("No automation rules configured")
    
    st.divider()
    
    # Execution history
    st.markdown("### Execution History")
    history = automation_engine.get_execution_history(limit=10)
    
    if history:
        hist_df = pd.DataFrame(history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
    else:
        st.info("No executions yet")

# =============================================================================
# TAB 5: ORDER MANAGEMENT
# =============================================================================
with tab5:
    st.subheader("Order Management System")
    
    oms = OrderManager(db_path="data/orders/orders.db")
    
    # Statistics
    order_stats = oms.get_statistics()
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Orders", order_stats.get("total_orders", 0))
    with col2:
        st.metric("Filled", order_stats.get("filled", 0))
    with col3:
        st.metric("Active", order_stats.get("active", 0))
    with col4:
        st.metric("Cancelled", order_stats.get("cancelled", 0))
    with col5:
        st.metric("Fill Rate", f"{order_stats.get('fill_rate', 0):.1f}%")
    
    st.divider()
    
    # Active orders
    st.markdown("### Active Orders")
    active_orders = oms.get_active_orders()
    
    if active_orders:
        order_data = []
        for order in active_orders:
            order_data.append({
                "Order ID": order.order_id,
                "Symbol": order.symbol,
                "Side": order.side.value,
                "Qty": order.quantity,
                "Type": order.order_type.value,
                "Status": order.status.value,
                "Filled": order.filled_quantity,
                "Avg Price": f"${order.avg_fill_price:.2f}" if order.avg_fill_price else "-",
                "Created": order.created_at.strftime("%H:%M:%S"),
            })
        
        st.dataframe(pd.DataFrame(order_data), use_container_width=True, hide_index=True)
        
        # Cancel buttons
        if st.button("Cancel All Orders"):
            oms.cancel_all_orders()
            st.success("All orders cancelled")
            st.rerun()
    else:
        st.info("No active orders")
    
    st.divider()
    
    # Order history
    st.markdown("### Order History")
    all_orders = oms.get_orders(limit=20)
    
    if all_orders:
        hist_data = []
        for order in all_orders:
            hist_data.append({
                "Order ID": order.order_id,
                "Symbol": order.symbol,
                "Side": order.side.value,
                "Qty": order.quantity,
                "Type": order.order_type.value,
                "Status": order.status.value,
                "Filled": order.filled_quantity,
                "Avg Price": f"${order.avg_fill_price:.2f}" if order.avg_fill_price > 0 else "-",
                "Strategy": order.strategy or "-",
                "Commission": f"${order.commission:.2f}",
            })
        
        st.dataframe(pd.DataFrame(hist_data), use_container_width=True, hide_index=True)
    else:
        st.info("No order history")

# Footer
st.divider()
st.caption(
    "⚠️ Paper trading mode uses simulated execution. "
    "Results may differ from live trading due to slippage and market conditions."
)
