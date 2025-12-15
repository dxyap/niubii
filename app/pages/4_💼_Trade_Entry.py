"""
Trade Entry Page
================
Manual trade entry with pre-trade risk checks.
Enhanced with visual pre-trade feedback.

Performance optimizations:
- Trading components cached via @st.cache_resource
- Lazy imports after page initialization
"""

from datetime import datetime
from pathlib import Path

import streamlit as st

# Initialize page first (before heavy imports)
from app.page_utils import init_page

ctx = init_page(
    title="💼 Trade Entry",
    page_title="Trade Entry | Oil Trading",
    icon="💼",
)

st.caption("Enter trades manually after execution | Pre-trade risk checks included")

# Lazy imports after page init
from app.components.ui_components import render_compact_stats, render_progress_ring
from core.constants import REFERENCE_PRICES

# Initialize trading components (cached as resource)
project_root = Path(__file__).parent.parent.parent


@st.cache_resource(show_spinner=False)
def get_trading_components():
    """
    Initialize trading components (cached as resource).
    
    Database connections are pooled, so caching is safe.
    """
    from core.risk import RiskLimits
    from core.trading import PositionManager, TradeBlotter
    blotter = TradeBlotter(db_path=str(project_root / "data" / "trades" / "trades.db"))
    position_mgr = PositionManager(db_path=str(project_root / "data" / "trades" / "trades.db"))
    risk_limits = RiskLimits(config_path=str(project_root / "config" / "risk_limits.yaml"))
    return blotter, position_mgr, risk_limits


blotter, position_mgr, risk_limits = get_trading_components()
portfolio_summary = ctx.portfolio.summary
positions_df = ctx.portfolio.positions_dataframe

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 New Trade Entry")

    with st.form("trade_entry_form"):
        form_col1, form_col2 = st.columns(2)

        with form_col1:
            trade_date = st.date_input("Trade Date", value=datetime.now().date())

            instrument = st.selectbox(
                "Instrument",
                options=[
                    "CLF5 Comdty - WTI Jan 2025",
                    "CLG5 Comdty - WTI Feb 2025",
                    "COF5 Comdty - Brent Jan 2025",
                    "COG5 Comdty - Brent Feb 2025",
                    "XBF5 Comdty - RBOB Jan 2025",
                    "HOF5 Comdty - Heating Oil Jan 2025",
                ],
                index=0
            )
            st.session_state.selected_instrument = instrument

            side = st.radio("Side", options=["BUY", "SELL"], horizontal=True)

        with form_col2:
            trade_time = st.time_input("Trade Time", value=datetime.now().time())

            account = st.selectbox("Account", options=["MAIN", "HEDGE", "SPEC"], index=0)

            quantity = st.number_input("Quantity (Contracts)", min_value=1, max_value=100, value=10)
            st.session_state.trade_entry_quantity = quantity

        price_col1, price_col2 = st.columns(2)

        with price_col1:
            price = st.number_input("Execution Price", min_value=0.01, value=72.50, format="%.2f")
            st.session_state.trade_entry_price = price

        with price_col2:
            commission = st.number_input("Commission/Fees ($)", min_value=0.0, value=25.0, format="%.2f")

        strategy = st.selectbox(
            "Strategy Tag",
            options=["Momentum", "Mean Reversion", "Term Structure", "Spread", "Signal-Based", "Discretionary"],
            index=0
        )

        notes = st.text_area("Notes", placeholder="Trade rationale...")

        submitted = st.form_submit_button("💾 Save Trade", width="stretch")

        if submitted:
            ticker = instrument.split(" - ")[0]
            trade_id = blotter.add_trade(
                instrument=ticker,
                side=side,
                quantity=quantity,
                price=price,
                trade_date=trade_date,
                trade_time=trade_time.strftime("%H:%M:%S"),
                commission=commission,
                strategy=strategy,
                notes=notes if notes else None,
                account=account
            )
            st.success(f"✅ Trade saved! Trade ID: {trade_id}")
            st.balloons()

with col2:
    st.subheader("🔍 Pre-Trade Risk Check")

    selected_instrument = st.session_state.get("selected_instrument", "CLF5 Comdty - WTI Jan 2025")
    selected_ticker = selected_instrument.split(" - ")[0]
    ticker_prefix = selected_ticker[:2]
    current_position = 0
    if not positions_df.empty:
        mask = positions_df["ticker"].str.startswith(ticker_prefix)
        current_position = int(positions_df.loc[mask, "qty"].sum())
    proposed_quantity = int(st.session_state.get("trade_entry_quantity", 10))
    current_price = ctx.price_cache.get(selected_ticker)
    default_price = REFERENCE_PRICES.get(ticker_prefix, 72.5)
    manual_price = st.session_state.get("trade_entry_price", default_price)
    price_for_checks = current_price if current_price is not None else manual_price

    position_check = risk_limits.check_position_limit(
        ticker=selected_ticker,
        current_quantity=current_position,
        proposed_quantity=proposed_quantity,
        price=price_for_checks
    )

    # Position Limit Visual Check
    pos_util = position_check['contract_utilization_pct']
    pos_color = "#00DC82" if pos_util < 75 else "#f59e0b" if pos_util < 90 else "#ef4444"
    pos_status = "✅ Approved" if position_check['approved'] else "❌ Breach"
    
    st.markdown(f"""
    <div style="
        background: {'rgba(0, 220, 130, 0.1)' if position_check['approved'] else 'rgba(239, 68, 68, 0.1)'};
        border: 1px solid {'rgba(0, 220, 130, 0.3)' if position_check['approved'] else 'rgba(239, 68, 68, 0.3)'};
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Position Limits</span>
            <span style="font-size: 12px; color: {'#00DC82' if position_check['approved'] else '#ef4444'}; font-weight: 600;">{pos_status}</span>
        </div>
        <div style="font-size: 18px; font-weight: 600; color: #f1f5f9; font-family: 'IBM Plex Mono', monospace;">
            {position_check['new_quantity']} / {position_check['max_contracts']} contracts
        </div>
        <div style="font-size: 11px; color: #64748b; margin-top: 4px;">
            {pos_util:.0f}% utilized
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.progress(min(pos_util / 100, 1.0))

    st.markdown("")  # Spacing

    # VaR Impact Visual Check
    multiplier = ctx.data_loader.get_multiplier(selected_ticker)
    current_var = portfolio_summary['var_estimate']
    var_limit = portfolio_summary['var_limit']
    var_impact = proposed_quantity * price_for_checks * multiplier * 0.02 * 1.65
    new_var = current_var + var_impact
    var_util = new_var / var_limit * 100
    
    var_approved = var_util < 90
    var_status = "✅ Approved" if var_approved else "⚠️ Warning"
    
    st.markdown(f"""
    <div style="
        background: {'rgba(0, 220, 130, 0.1)' if var_approved else 'rgba(245, 158, 11, 0.1)'};
        border: 1px solid {'rgba(0, 220, 130, 0.3)' if var_approved else 'rgba(245, 158, 11, 0.3)'};
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">VaR Impact</span>
            <span style="font-size: 12px; color: {'#00DC82' if var_approved else '#f59e0b'}; font-weight: 600;">{var_status}</span>
        </div>
        <div style="font-size: 18px; font-weight: 600; color: #f1f5f9; font-family: 'IBM Plex Mono', monospace;">
            ${new_var:,.0f}
        </div>
        <div style="font-size: 11px; color: #64748b; margin-top: 4px;">
            {var_util:.0f}% of ${var_limit:,} limit (+${var_impact:,.0f} impact)
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.progress(min(var_util / 100, 1.0))
    
    # Overall approval status
    overall_approved = position_check['approved'] and var_approved
    if overall_approved:
        st.success("✅ Trade passes all risk checks")
    else:
        st.warning("⚠️ Trade requires risk review")

# Recent trades
st.divider()
st.subheader("📋 Today's Trades")

todays_trades = blotter.get_todays_trades()

if not todays_trades.empty:
    display_trades = todays_trades[['trade_id', 'trade_time', 'instrument', 'side', 'quantity', 'price', 'strategy']].copy()

    st.dataframe(
        display_trades,
        width="stretch",
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
else:
    st.info("No trades entered today")

# Quick entry
st.divider()
st.subheader("⚡ Quick Trade Entry")

st.markdown("""
<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px;">
""", unsafe_allow_html=True)

quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)

# Get live prices
wti_price = round(ctx.price_cache.get("CL1 Comdty") or 72.5, 2)
brent_price = round(ctx.price_cache.get("CO1 Comdty") or 77.2, 2)

with quick_col1:
    st.markdown(f"""
    <div style="text-align: center; font-size: 11px; color: #64748b; margin-bottom: 4px;">
        WTI @ ${wti_price:.2f}
    </div>
    """, unsafe_allow_html=True)
    if st.button("📈 Buy 5 WTI", width="stretch", type="primary"):
        trade_id = blotter.add_trade(instrument="CL1 Comdty", side="BUY", quantity=5, price=wti_price, strategy="Quick Entry")
        st.success(f"Trade {trade_id} saved!")
        st.rerun()

with quick_col2:
    st.markdown(f"""
    <div style="text-align: center; font-size: 11px; color: #64748b; margin-bottom: 4px;">
        WTI @ ${wti_price:.2f}
    </div>
    """, unsafe_allow_html=True)
    if st.button("📉 Sell 5 WTI", width="stretch"):
        trade_id = blotter.add_trade(instrument="CL1 Comdty", side="SELL", quantity=5, price=wti_price, strategy="Quick Entry")
        st.success(f"Trade {trade_id} saved!")
        st.rerun()

with quick_col3:
    st.markdown(f"""
    <div style="text-align: center; font-size: 11px; color: #64748b; margin-bottom: 4px;">
        Brent @ ${brent_price:.2f}
    </div>
    """, unsafe_allow_html=True)
    if st.button("📈 Buy 5 Brent", width="stretch", type="primary"):
        trade_id = blotter.add_trade(instrument="CO1 Comdty", side="BUY", quantity=5, price=brent_price, strategy="Quick Entry")
        st.success(f"Trade {trade_id} saved!")
        st.rerun()

with quick_col4:
    st.markdown(f"""
    <div style="text-align: center; font-size: 11px; color: #64748b; margin-bottom: 4px;">
        Brent @ ${brent_price:.2f}
    </div>
    """, unsafe_allow_html=True)
    if st.button("📉 Sell 5 Brent", width="stretch"):
        trade_id = blotter.add_trade(instrument="CO1 Comdty", side="SELL", quantity=5, price=brent_price, strategy="Quick Entry")
        st.success(f"Trade {trade_id} saved!")
        st.rerun()
