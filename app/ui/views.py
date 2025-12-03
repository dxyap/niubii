"""
Reusable view/controller classes for the Streamlit dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Optional, Sequence

import streamlit as st

from app.components.charts import create_candlestick_chart, create_futures_curve_chart
from app.components.theme import get_chart_config
from app.dashboard_core import DashboardContext, PortfolioAnalytics


def _render_price_metric(label: str, values: Optional[Dict[str, float]]) -> None:
    """Render a Streamlit metric for price-oriented data."""
    if not values or values.get("current") is None:
        st.metric(label, "N/A", "No data")
        return

    current = float(values.get("current", 0.0))
    change = float(values.get("change", 0.0))
    change_pct = values.get("change_pct")
    pct_suffix = f" ({change_pct:+.2f}%)" if change_pct is not None else ""
    delta = f"{change:+.2f}{pct_suffix}"
    delta_color = "normal" if change >= 0 else "inverse"
    st.metric(label, f"${current:.2f}", delta, delta_color=delta_color)


def _render_spread_metric(label: str, data: Optional[Dict[str, float]]) -> None:
    """Render a Streamlit metric for spread data."""
    if not data:
        st.metric(label, "N/A", "No data")
        return

    spread = float(data.get("spread", 0.0) or 0.0)
    change = float(data.get("change", 0.0) or 0.0)
    st.metric(label, f"${spread:.2f}", f"{change:+0.2f}")


class RefreshController:
    """Session-level helper that tracks refresh state and manages auto-refresh."""

    def __init__(self, interval: int):
        self.interval = max(int(interval), 1)
        now = datetime.now()
        session = st.session_state
        session.setdefault("last_refresh", now)
        session.setdefault("auto_refresh", True)
        session.setdefault("refresh_count", 0)

    @property
    def auto_refresh_enabled(self) -> bool:
        return bool(st.session_state.auto_refresh)

    def update_auto_refresh(self, enabled: bool) -> None:
        st.session_state.auto_refresh = bool(enabled)

    @property
    def last_refresh(self) -> datetime:
        return st.session_state.last_refresh

    def format_last_refresh(self, fmt: str = "%H:%M:%S") -> str:
        return self.last_refresh.strftime(fmt)

    def refresh_now(self) -> None:
        st.session_state.last_refresh = datetime.now()
        st.session_state.refresh_count += 1
        st.rerun()

    def schedule_auto_refresh(self, on_cycle: Optional[Callable[[], None]] = None) -> None:
        """
        Schedule non-blocking auto-refresh. Clears caches via callback when tick fires.
        """
        if not self.auto_refresh_enabled:
            return

        try:
            from streamlit_autorefresh import st_autorefresh
        except ImportError:
            refresh_js = f"""
            <script>
                setTimeout(function() {{
                    window.parent.document.querySelector('button[kind="secondary"]')?.click() ||
                    window.parent.location.reload();
                }}, {self.interval * 1000});
            </script>
            """
            st.markdown(refresh_js, unsafe_allow_html=True)
            return

        count = st_autorefresh(
            interval=self.interval * 1000,
            limit=None,
            key="auto_refresh_counter",
        )
        if count > st.session_state.refresh_count:
            st.session_state.refresh_count = count
            st.session_state.last_refresh = datetime.now()
            if on_cycle:
                on_cycle()


@dataclass(frozen=True)
class ConnectionIndicator:
    """Simple value object for sidebar connection status."""

    color: str
    text: str

    @classmethod
    def from_mode(cls, data_mode: str) -> "ConnectionIndicator":
        palette = {
            "live": ("#00D26A", "Live Data"),
            "mock": ("#FFA500", "Mock Data (Dev)"),
            "disconnected": ("#FF4B4B", "Disconnected"),
        }
        color, text = palette.get(data_mode, palette["disconnected"])
        return cls(color=color, text=text)


class SidebarView:
    """Render sidebar content including connection status and quick prices."""

    def __init__(
        self,
        context: DashboardContext,
        refresh_controller: RefreshController,
        nav_links: Sequence[tuple[str, str]] = None,  # Kept for backward compatibility but not used
    ):
        self.context = context
        self.refresh_controller = refresh_controller
        # nav_links no longer used - Streamlit auto-generates page navigation

    def render(self) -> None:
        data_bundle = self.context.data
        connection_status = data_bundle.connection_status
        indicator = ConnectionIndicator.from_mode(connection_status.get("data_mode", "disconnected"))
        connection_error = connection_status.get("connection_error")

        with st.sidebar:
            st.image("https://img.icons8.com/fluency/96/000000/oil-barrel.png", width=60)
            st.title("Oil Trading")
            st.caption("Quantitative Dashboard")
            st.divider()

            st.markdown(
                f'<span class="live-indicator" style="background:{indicator.color};"></span>'
                f"<span style='color:{indicator.color};'>{indicator.text}</span>",
                unsafe_allow_html=True,
            )

            if indicator.text == "Disconnected" and connection_error:
                st.error(f"Connection error: {connection_error[:120]}")

            st.caption(f"Last Update: {self.refresh_controller.format_last_refresh('%H:%M:%S')}")

            toggle_label = f"Auto Refresh ({self.refresh_controller.interval}s)"
            auto_refresh = st.toggle(toggle_label, value=self.refresh_controller.auto_refresh_enabled)
            self.refresh_controller.update_auto_refresh(auto_refresh)

            if st.button("Refresh Now", use_container_width=True):
                self.refresh_controller.refresh_now()

            st.divider()
            self._render_quick_prices()

    def _render_quick_prices(self) -> None:
        prices = self.context.data.oil_prices
        st.subheader("Quick View")

        if prices is None:
            st.warning("Price data unavailable.")
            return

        col1, col2 = st.columns(2)
        with col1:
            _render_price_metric("WTI", prices.get("WTI"))
        with col2:
            _render_price_metric("Brent", prices.get("Brent"))

        spread = self.context.data.wti_brent_spread
        _render_spread_metric("WTI-Brent", spread)


class KeyMetricsView:
    """Render the top-level metrics row."""

    def __init__(self, context: DashboardContext):
        self.context = context

    def render(self) -> None:
        prices = self.context.data.oil_prices or {}
        spread = self.context.data.wti_brent_spread
        crack = self.context.data.crack_spread
        summary = self.context.portfolio.summary

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            _render_price_metric("WTI Crude", prices.get("WTI"))
        with col2:
            _render_price_metric("Brent Crude", prices.get("Brent"))
        with col3:
            _render_spread_metric("WTI-Brent", spread)
        with col4:
            if crack:
                st.metric("3-2-1 Crack", f"${crack.get('crack', 0):.2f}", f"{crack.get('change', 0):+0.2f}")
            else:
                st.metric("3-2-1 Crack", "N/A", "No data")
        with col5:
            total_pnl = summary["total_pnl"]
            pnl_return_pct = summary["pnl_return_pct"]
            st.metric(
                "Day P&L",
                PortfolioAnalytics.format_pnl(total_pnl),
                f"{pnl_return_pct:+.2f}%",
                delta_color="normal" if total_pnl >= 0 else "inverse",
            )

        st.divider()


class MarketOverviewView:
    """Render charts for prices and the futures curve."""

    def __init__(
        self,
        context: DashboardContext,
        historical_loader: Callable[[str, int], Optional["pd.DataFrame"]],
        curve_loader: Callable[[str, int], Optional["pd.DataFrame"]],
    ):
        self.context = context
        self._historical_loader = historical_loader
        self._curve_loader = curve_loader

    def render_price_chart(self) -> None:
        st.subheader("Brent Crude Oil Price")
        hist_data = self._historical_loader("CO1 Comdty", 90)

        if hist_data is None or hist_data.empty:
            st.info("Historical data unavailable.")
            return

        fig = create_candlestick_chart(
            data=hist_data,
            title="",
            height=380,
            show_volume=False,
            show_ma=True,
            ma_periods=[20, 50],
        )
        st.plotly_chart(fig, use_container_width=True, config=get_chart_config())

        prices = self.context.data.oil_prices or {}
        live_brent = prices.get("Brent", {}).get("current")
        current = live_brent if live_brent is not None else hist_data["PX_LAST"].iloc[-1]
        stat_cols = st.columns(4)
        with stat_cols[0]:
            st.metric("Current", f"${current:.2f}")
        with stat_cols[1]:
            st.metric("90D High", f"${hist_data['PX_HIGH'].max():.2f}")
        with stat_cols[2]:
            st.metric("90D Low", f"${hist_data['PX_LOW'].min():.2f}")
        with stat_cols[3]:
            st.metric("90D Avg", f"${hist_data['PX_LAST'].mean():.2f}")

    def render_curve_section(self) -> str:
        st.subheader("Brent Futures Curve")
        curve = self._curve_loader("brent", 18)

        if curve is None or curve.empty:
            st.info("Futures curve unavailable.")
            return "Unknown"

        fig = create_futures_curve_chart(curve_data=curve, title="", height=280)
        st.plotly_chart(fig, use_container_width=True, config=get_chart_config())

        prices = curve["price"]
        slope = (prices.iloc[-1] - prices.iloc[0]) / max(len(prices) - 1, 1)
        if slope > 0.05:
            structure = "Contango"
        elif slope < -0.05:
            structure = "Backwardation"
        else:
            structure = "Flat"

        def spread(idx: int) -> float:
            if len(prices) > idx:
                return float(prices.iloc[0] - prices.iloc[idx])
            return 0.0

        cols = st.columns(4)
        cols[0].metric("M1-M2 Spread", f"${spread(1):.2f}")
        cols[1].metric("M1-M6 Spread", f"${spread(5):.2f}")
        cols[2].metric("M1-M12 Spread", f"${spread(11):.2f}")
        cols[3].metric("Structure", structure)
        return structure


class PortfolioView:
    """Portfolio and risk rendering logic."""

    def __init__(self, context: DashboardContext):
        self.context = context

    def render_positions(self) -> None:
        st.subheader("Position Summary")
        table = self.context.portfolio.formatted_table()

        if table.empty:
            st.info("No active positions.")
            return

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Symbol": st.column_config.TextColumn("Symbol"),
                "Qty": st.column_config.NumberColumn("Qty", format="%d"),
                "Entry": st.column_config.NumberColumn("Entry", format="$%.2f"),
                "Current": st.column_config.NumberColumn("Current", format="$%.2f"),
                "P&L": st.column_config.TextColumn("P&L"),
            },
        )

        total_pnl = self.context.portfolio.summary["total_pnl"]
        color = "#22c55e" if total_pnl >= 0 else "#ef4444"
        st.markdown(
            f"<h3 style='color:{color}; font-family: IBM Plex Mono, monospace;'>"
            f"Total P&L: {PortfolioAnalytics.format_pnl(total_pnl)}</h3>",
            unsafe_allow_html=True,
        )

    def render_risk(self, curve_structure: str) -> None:
        st.divider()
        st.subheader("Risk Summary")
        summary = self.context.portfolio.summary
        st.progress(
            summary["var_utilization"] / 100,
            text=f"VaR Utilization: {summary['var_utilization']:.0f}%",
        )

        risk_metrics = {
            "VaR (95%, 1-day)": f"${summary['var_estimate']:,.0f}",
            "VaR Limit": f"${summary['var_limit']:,.0f}",
            "Gross Exposure": f"${summary['gross_exposure'] / 1e6:.1f}M",
            "Net Position": f"{summary['net_contracts']:.0f} contracts",
        }
        for label, value in risk_metrics.items():
            st.text(f"{label}: {value}")

        alerts = self.context.generate_alerts(curve_structure, summary["var_utilization"])
        st.divider()
        st.subheader("Active Alerts")
        if not alerts:
            st.success("No active alerts.")
            return

        for alert in alerts:
            severity = alert["severity"]
            message = alert["message"]
            if severity == "warning":
                st.warning(message)
            elif severity == "critical":
                st.error(message)
            else:
                st.info(message)


class ActivityFeedView:
    """Render the lower grid with trades and signals."""

    def __init__(self, context: DashboardContext):
        self.context = context

    def render(self) -> None:
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            self._render_recent_trades()
        with col2:
            self._render_signals()

    def _render_recent_trades(self) -> None:
        st.subheader("Recent Trades")
        trades = self.context.generate_recent_trades()
        if trades.empty:
            st.info("No recent trades.")
            return
        st.dataframe(trades, use_container_width=True, hide_index=True)

    def _render_signals(self) -> None:
        st.subheader("Active Signals")
        signals = self.context.generate_signals()
        if signals.empty:
            st.info("No active signals.")
            return
        st.dataframe(signals, use_container_width=True, hide_index=True)
