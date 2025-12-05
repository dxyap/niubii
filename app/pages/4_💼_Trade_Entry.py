"""
Trade Entry Page
================
Manual trade entry with pre-trade risk checks.
"""

from datetime import datetime
from pathlib import Path

import streamlit as st

from app.page_utils import init_page
from core.risk import RiskLimits
from core.trading import PositionManager, TradeBlotter

# Initialize page
ctx = init_page(
    title="💼 Trade Entry",
    page_title="Trade Entry | Oil Trading",
    icon="💼",
)

st.caption("Enter trades manually after execution | Pre-trade risk checks included")

# Initialize trading components (cached)
project_root = Path(__file__).parent.parent.parent

@st.cache_resource
def get_trading_components():
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

        with price_col2:
            commission = st.number_input("Commission/Fees ($)", min_value=0.0, value=25.0, format="%.2f")

        strategy = st.selectbox(
            "Strategy Tag",
            options=["Momentum", "Mean Reversion", "Term Structure", "Spread", "Signal-Based", "Discretionary"],
            index=0
        )

        notes = st.text_area("Notes", placeholder="Trade rationale...")

        submitted = st.form_submit_button("💾 Save Trade", use_container_width=True)

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

    position_check = risk_limits.check_position_limit(
        ticker=selected_ticker,
        current_quantity=current_position,
        proposed_quantity=proposed_quantity,
        price=current_price
    )

    st.markdown("**Position Limits**")

    if position_check['approved']:
        st.success(f"✅ Position: {position_check['new_quantity']} / {position_check['max_contracts']} contracts")
    else:
        st.error("❌ Position limit breach")

    st.progress(min(position_check['contract_utilization_pct'] / 100, 1.0))

    st.divider()

    st.markdown("**VaR Impact**")

    multiplier = ctx.data_loader.get_multiplier(selected_ticker)
    current_var = portfolio_summary['var_estimate']
    var_limit = portfolio_summary['var_limit']
    var_impact = proposed_quantity * (current_price or 72.5) * multiplier * 0.02 * 1.65
    new_var = current_var + var_impact
    var_util = new_var / var_limit * 100

    if var_util < 90:
        st.success(f"✅ VaR: ${new_var:,.0f} ({var_util:.0f}% of limit)")
    else:
        st.warning(f"⚠️ VaR: ${new_var:,.0f} ({var_util:.0f}% of limit)")

    st.progress(min(var_util / 100, 1.0))

# Recent trades
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
else:
    st.info("No trades entered today")

# Quick entry
st.divider()
st.subheader("⚡ Quick Trade Entry")

quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)

with quick_col1:
    if st.button("Buy 5 WTI", use_container_width=True):
        price = round(ctx.price_cache.get("CL1 Comdty") or 72.5, 2)
        trade_id = blotter.add_trade(instrument="CL1 Comdty", side="BUY", quantity=5, price=price, strategy="Quick Entry")
        st.success(f"Trade {trade_id} saved!")
        st.rerun()

with quick_col2:
    if st.button("Sell 5 WTI", use_container_width=True):
        price = round(ctx.price_cache.get("CL1 Comdty") or 72.5, 2)
        trade_id = blotter.add_trade(instrument="CL1 Comdty", side="SELL", quantity=5, price=price, strategy="Quick Entry")
        st.success(f"Trade {trade_id} saved!")
        st.rerun()

with quick_col3:
    if st.button("Buy 5 Brent", use_container_width=True):
        price = round(ctx.price_cache.get("CO1 Comdty") or 77.2, 2)
        trade_id = blotter.add_trade(instrument="CO1 Comdty", side="BUY", quantity=5, price=price, strategy="Quick Entry")
        st.success(f"Trade {trade_id} saved!")
        st.rerun()

with quick_col4:
    if st.button("Sell 5 Brent", use_container_width=True):
        price = round(ctx.price_cache.get("CO1 Comdty") or 77.2, 2)
        trade_id = blotter.add_trade(instrument="CO1 Comdty", side="SELL", quantity=5, price=price, strategy="Quick Entry")
        st.success(f"Trade {trade_id} saved!")
        st.rerun()
