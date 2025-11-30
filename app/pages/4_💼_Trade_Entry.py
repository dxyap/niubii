"""
Trade Entry Page
================
Manual trade entry with pre-trade risk checks.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.trading import TradeBlotter, PositionManager
from core.risk import RiskLimits

st.set_page_config(page_title="Trade Entry | Oil Trading", page_icon="💼", layout="wide")

# Initialize components
@st.cache_resource
def get_components():
    blotter = TradeBlotter(db_path=str(project_root / "data" / "trades" / "trades.db"))
    position_mgr = PositionManager(db_path=str(project_root / "data" / "trades" / "trades.db"))
    risk_limits = RiskLimits(config_path=str(project_root / "config" / "risk_limits.yaml"))
    return blotter, position_mgr, risk_limits

blotter, position_mgr, risk_limits = get_components()

st.title("💼 Trade Entry")
st.caption("Enter trades manually after execution | Pre-trade risk checks included")

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 New Trade Entry")
    
    with st.form("trade_entry_form"):
        # Trade details
        form_col1, form_col2 = st.columns(2)
        
        with form_col1:
            trade_date = st.date_input("Trade Date", value=datetime.now().date())
            
            instrument = st.selectbox(
                "Instrument",
                options=[
                    "CLF5 Comdty - WTI Jan 2025",
                    "CLG5 Comdty - WTI Feb 2025",
                    "CLH5 Comdty - WTI Mar 2025",
                    "COF5 Comdty - Brent Jan 2025",
                    "COG5 Comdty - Brent Feb 2025",
                    "XBF5 Comdty - RBOB Jan 2025",
                    "HOF5 Comdty - Heating Oil Jan 2025",
                ],
                index=0
            )
            
            side = st.radio("Side", options=["BUY", "SELL"], horizontal=True)
        
        with form_col2:
            trade_time = st.time_input("Trade Time", value=datetime.now().time())
            
            account = st.selectbox(
                "Account",
                options=["MAIN", "HEDGE", "SPEC"],
                index=0
            )
            
            quantity = st.number_input("Quantity (Contracts)", min_value=1, max_value=100, value=10)
        
        # Price and fees
        price_col1, price_col2 = st.columns(2)
        
        with price_col1:
            price = st.number_input("Execution Price", min_value=0.01, value=72.50, format="%.2f")
        
        with price_col2:
            commission = st.number_input("Commission/Fees ($)", min_value=0.0, value=25.0, format="%.2f")
        
        # Strategy and notes
        strategy = st.selectbox(
            "Strategy Tag",
            options=["Momentum", "Mean Reversion", "Term Structure", "Spread", "Signal-Based", "Discretionary", "Other"],
            index=0
        )
        
        signal_ref = st.text_input("Signal Reference (optional)", placeholder="SIG-2024-1130-001")
        
        notes = st.text_area("Notes", placeholder="Trade rationale, market conditions, etc.")
        
        # Submit button
        submitted = st.form_submit_button("💾 Save Trade", use_container_width=True)
        
        if submitted:
            # Extract ticker from selection
            ticker = instrument.split(" - ")[0]
            
            # Add trade to blotter
            trade_id = blotter.add_trade(
                instrument=ticker,
                side=side,
                quantity=quantity,
                price=price,
                trade_date=trade_date,
                trade_time=trade_time.strftime("%H:%M:%S"),
                commission=commission,
                strategy=strategy,
                signal_ref=signal_ref if signal_ref else None,
                notes=notes if notes else None,
                account=account
            )
            
            st.success(f"✅ Trade saved successfully! Trade ID: {trade_id}")
            st.balloons()

with col2:
    st.subheader("🔍 Pre-Trade Risk Check")
    
    # Current position
    ticker = st.session_state.get('selected_ticker', 'CL1 Comdty')[:2]
    
    # Mock current position
    current_position = 55 if ticker == "CL" else 15
    proposed_quantity = 10
    
    # Position limit check
    position_check = risk_limits.check_position_limit(
        ticker=f"{ticker}1 Comdty",
        current_quantity=current_position,
        proposed_quantity=proposed_quantity,
        price=72.50
    )
    
    st.markdown("**Position Limits**")
    
    if position_check['approved']:
        st.success(f"✅ Position: {position_check['new_quantity']} / {position_check['max_contracts']} contracts")
    else:
        st.error(f"❌ Position limit breach: {position_check['new_quantity']} > {position_check['max_contracts']}")
    
    st.progress(
        min(position_check['contract_utilization_pct'] / 100, 1.0),
        text=f"Contract Utilization: {position_check['contract_utilization_pct']:.0f}%"
    )
    
    st.divider()
    
    st.markdown("**Notional Exposure**")
    
    st.progress(
        min(position_check['notional_utilization_pct'] / 100, 1.0),
        text=f"Notional: ${position_check['new_notional']:,.0f} ({position_check['notional_utilization_pct']:.0f}%)"
    )
    
    st.divider()
    
    st.markdown("**VaR Impact**")
    
    # Mock VaR calculation
    current_var = 245000
    var_impact = proposed_quantity * 72.50 * 1000 * 0.02 * 1.65
    new_var = current_var + var_impact
    var_limit = 375000
    var_util = new_var / var_limit * 100
    
    if var_util < 90:
        st.success(f"✅ VaR: ${new_var:,.0f} ({var_util:.0f}% of limit)")
    elif var_util < 100:
        st.warning(f"⚠️ VaR: ${new_var:,.0f} ({var_util:.0f}% of limit)")
    else:
        st.error(f"❌ VaR breach: ${new_var:,.0f} > ${var_limit:,.0f}")
    
    st.progress(min(var_util / 100, 1.0))
    
    st.divider()
    
    st.markdown("**Concentration Check**")
    
    concentration = 52  # Mock value
    
    if concentration <= 40:
        st.success(f"✅ WTI concentration: {concentration}% (limit: 40%)")
    else:
        st.warning(f"⚠️ WTI concentration: {concentration}% (above 40% limit)")
    
    st.divider()
    
    # Override option
    st.markdown("**Risk Override**")
    
    override = st.checkbox("Override risk warnings")
    
    if override:
        override_reason = st.text_area(
            "Override Reason (required)",
            placeholder="Enter reason for overriding risk limits..."
        )

# Recent trades section
st.divider()
st.subheader("📋 Today's Trades")

todays_trades = blotter.get_todays_trades()

if not todays_trades.empty:
    display_trades = todays_trades[['trade_id', 'trade_time', 'instrument', 'side', 'quantity', 'price', 'strategy']].copy()
    
    st.dataframe(
        display_trades,
        use_container_width=True,
        hide_index=True,
        column_config={
            'trade_id': 'Trade ID',
            'trade_time': 'Time',
            'instrument': 'Symbol',
            'side': 'Side',
            'quantity': st.column_config.NumberColumn('Qty'),
            'price': st.column_config.NumberColumn('Price', format='$%.2f'),
            'strategy': 'Strategy',
        }
    )
    
    # Summary
    buy_trades = len(todays_trades[todays_trades['side'] == 'BUY'])
    sell_trades = len(todays_trades[todays_trades['side'] == 'SELL'])
    total_volume = todays_trades['quantity'].sum()
    total_commission = todays_trades['commission'].sum()
    
    sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
    
    with sum_col1:
        st.metric("Total Trades", len(todays_trades))
    with sum_col2:
        st.metric("Buy / Sell", f"{buy_trades} / {sell_trades}")
    with sum_col3:
        st.metric("Total Volume", f"{total_volume} contracts")
    with sum_col4:
        st.metric("Commissions", f"${total_commission:.2f}")
else:
    st.info("No trades entered today")

# Quick entry shortcuts
st.divider()
st.subheader("⚡ Quick Trade Entry")

quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)

with quick_col1:
    if st.button("Buy 5 WTI", use_container_width=True):
        trade_id = blotter.add_trade(
            instrument="CL1 Comdty",
            side="BUY",
            quantity=5,
            price=72.50,
            strategy="Quick Entry"
        )
        st.success(f"Trade {trade_id} saved!")
        st.rerun()

with quick_col2:
    if st.button("Sell 5 WTI", use_container_width=True):
        trade_id = blotter.add_trade(
            instrument="CL1 Comdty",
            side="SELL",
            quantity=5,
            price=72.50,
            strategy="Quick Entry"
        )
        st.success(f"Trade {trade_id} saved!")
        st.rerun()

with quick_col3:
    if st.button("Buy 5 Brent", use_container_width=True):
        trade_id = blotter.add_trade(
            instrument="CO1 Comdty",
            side="BUY",
            quantity=5,
            price=77.50,
            strategy="Quick Entry"
        )
        st.success(f"Trade {trade_id} saved!")
        st.rerun()

with quick_col4:
    if st.button("Sell 5 Brent", use_container_width=True):
        trade_id = blotter.add_trade(
            instrument="CO1 Comdty",
            side="SELL",
            quantity=5,
            price=77.50,
            strategy="Quick Entry"
        )
        st.success(f"Trade {trade_id} saved!")
        st.rerun()
