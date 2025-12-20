"""
Alerts Page
===========
Multi-channel alert system for trading signals, risk, and events.

Performance optimizations:
- Page initialized before heavy imports
- Alert engine cached in session state
- Lazy module imports
"""

import streamlit as st

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="Alerts - Oil Trading Dashboard",
    page_icon="🔔",
    layout="wide",
    initial_sidebar_state="expanded",
)

import sys
from datetime import datetime, time, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Apply theme first
from app.components.theme import COLORS, apply_theme, get_chart_config

apply_theme(st)

# Lazy imports after theme
import pandas as pd

from app import shared_state
from app.main import get_connection_status_cached

# Import alerts modules (lightweight enum/config imports)
from core.alerts import (
    AlertCategory,
    AlertEngine,
    AlertEngineConfig,
    AlertSeverity,
)
from core.alerts.history import AlertHistory
from core.alerts.rules import (
    create_pnl_alert,
    create_price_alert,
    create_risk_alert,
    create_signal_alert,
)
from core.alerts.scheduler import ReportConfig, ReportFrequency, ReportScheduler, ReportType

# Initialize session state
if "alert_engine" not in st.session_state:
    config = AlertEngineConfig(storage_path="data/alerts")
    st.session_state.alert_engine = AlertEngine(config=config)

if "alert_history" not in st.session_state:
    st.session_state.alert_history = AlertHistory()

if "report_scheduler" not in st.session_state:
    st.session_state.report_scheduler = ReportScheduler()

# Get data loader
context = shared_state.get_dashboard_context(lookback_days=30)
data_loader = context.data_loader

st.title("🔔 Alerts & Notifications")

# Connection status (using cached version to reduce API calls)
connection_status = get_connection_status_cached()
data_mode = connection_status.get("data_mode", "disconnected")

if data_mode == "disconnected":
    st.warning("🟡 Running in disconnected mode. Using simulated data.")

# Global controls
col1, col2, col3, col4 = st.columns(4)

alert_engine = st.session_state.alert_engine

with col1:
    if alert_engine.config.enabled:
        if st.button("🔴 Disable Alerts"):
            alert_engine.disable()
            st.rerun()
    else:
        if st.button("🟢 Enable Alerts", type="primary"):
            alert_engine.enable()
            st.rerun()

with col2:
    if alert_engine.config.muted:
        mute_text = "Muted"
        if alert_engine.config.mute_until:
            mute_text += f" until {alert_engine.config.mute_until.strftime('%H:%M')}"
        st.warning(f"🔇 {mute_text}")
        if st.button("Unmute"):
            alert_engine.unmute()
            st.rerun()
    else:
        mute_duration = st.selectbox(
            "Mute for",
            ["Select...", "15 min", "30 min", "1 hour", "2 hours"],
            key="mute_duration",
            label_visibility="collapsed"
        )
        if mute_duration != "Select...":
            durations = {"15 min": 15, "30 min": 30, "1 hour": 60, "2 hours": 120}
            alert_engine.mute(durations[mute_duration])
            st.rerun()

with col3:
    status = "🟢 Active" if alert_engine.config.enabled else "🔴 Disabled"
    if alert_engine.config.muted:
        status = "🔇 Muted"
    st.metric("Status", status)

with col4:
    stats = alert_engine.get_statistics()
    st.metric("Active Alerts", stats["active_alerts"])

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Active Alerts",
    "⚙️ Alert Rules",
    "📊 History & Analytics",
    "📬 Channels",
    "📅 Scheduled Reports"
])

# =============================================================================
# TAB 1: ACTIVE ALERTS
# =============================================================================
with tab1:
    st.subheader("Active Alerts")

    # Filter controls
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        severity_filter = st.multiselect(
            "Filter by Severity",
            ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"],
            default=["CRITICAL", "HIGH", "MEDIUM"],
            key="sev_filter"
        )

    with col2:
        category_filter = st.multiselect(
            "Filter by Category",
            ["SIGNAL", "RISK", "PRICE", "PNL", "EXECUTION", "POSITION", "SYSTEM"],
            key="cat_filter"
        )

    with col3:
        if st.button("🔄 Refresh"):
            st.rerun()

    # Get active alerts
    active_alerts = alert_engine.get_active_alerts()

    # Apply filters
    if severity_filter:
        active_alerts = [
            a for a in active_alerts
            if a.trigger.severity.value in severity_filter
        ]

    if category_filter:
        active_alerts = [
            a for a in active_alerts
            if a.trigger.category.value in category_filter
        ]

    if active_alerts:
        # Display alerts
        for event in active_alerts:
            trigger = event.trigger

            # Color based on severity
            severity_colors = {
                "INFO": "blue",
                "LOW": "green",
                "MEDIUM": "orange",
                "HIGH": "red",
                "CRITICAL": "red",
            }
            color = severity_colors.get(trigger.severity.value, "gray")

            # Status icon
            status_icon = "🚨" if trigger.severity.value == "CRITICAL" else "⚠️" if trigger.severity.value in ["HIGH", "MEDIUM"] else "ℹ️"

            with st.expander(f"{status_icon} {trigger.title}", expanded=trigger.severity.value in ["CRITICAL", "HIGH"]):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"**{trigger.message}**")
                    st.caption(f"Category: {trigger.category.value} | Time: {trigger.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                    st.caption(f"Alert ID: `{trigger.trigger_id}`")

                    if event.escalated:
                        st.error("⬆️ This alert has been escalated")

                with col2:
                    st.markdown(f"**Severity:** {trigger.severity.value}")

                    if not trigger.acknowledged:
                        if st.button("✓ Acknowledge", key=f"ack_{trigger.trigger_id}"):
                            alert_engine.acknowledge(trigger.trigger_id, "user")
                            st.success("Alert acknowledged")
                            st.rerun()
                    else:
                        st.success(f"✓ Acknowledged by {trigger.acknowledged_by}")

                    if st.button("✕ Resolve", key=f"resolve_{trigger.trigger_id}"):
                        alert_engine.resolve(trigger.trigger_id)
                        st.success("Alert resolved")
                        st.rerun()
    else:
        st.success("✅ No active alerts")

    # Summary statistics
    st.divider()
    st.markdown("### Alert Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Rules", stats["total_rules"])
    with col2:
        st.metric("Active Rules", stats["active_rules"])
    with col3:
        st.metric("Alerts Today", stats["alerts_today"])
    with col4:
        ack_rate = stats.get("acknowledgment_rate", 0)
        st.metric("Ack Rate", f"{ack_rate*100:.0f}%")

# =============================================================================
# TAB 2: ALERT RULES
# =============================================================================
with tab2:
    st.subheader("Alert Rules Configuration")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Create New Rule")

        rule_type = st.selectbox(
            "Rule Type",
            ["Price Alert", "Signal Alert", "Risk Alert", "P&L Alert", "Custom"],
            key="rule_type"
        )

        rule_name = st.text_input("Rule Name", f"New {rule_type}")

        if rule_type == "Price Alert":
            price_symbol = st.selectbox("Symbol", ["WTI", "Brent", "RBOB", "Heating Oil"], key="price_symbol")
            price_level = st.number_input("Price Level ($)", 10.0, 200.0, 75.0)
            price_direction = st.selectbox("Direction", ["above", "below"])
            price_severity = st.selectbox("Severity", ["MEDIUM", "HIGH", "LOW", "CRITICAL"], key="price_sev")

            if st.button("Create Price Alert", type="primary"):
                config = create_price_alert(
                    name=rule_name,
                    symbol=price_symbol,
                    price_level=price_level,
                    direction=price_direction,
                    severity=AlertSeverity(price_severity),
                )
                alert_engine.add_rule(config)
                st.success(f"Created alert: {rule_name}")
                st.rerun()

        elif rule_type == "Signal Alert":
            signal_direction = st.selectbox("Signal Direction", ["LONG", "SHORT"])
            signal_confidence = st.slider("Min Confidence (%)", 50, 90, 70)
            signal_severity = st.selectbox("Severity", ["MEDIUM", "HIGH", "LOW"], key="signal_sev")

            if st.button("Create Signal Alert", type="primary"):
                config = create_signal_alert(
                    name=rule_name,
                    direction=signal_direction,
                    min_confidence=signal_confidence,
                    severity=AlertSeverity(signal_severity),
                )
                alert_engine.add_rule(config)
                st.success(f"Created alert: {rule_name}")
                st.rerun()

        elif rule_type == "Risk Alert":
            risk_type = st.selectbox("Risk Type", ["var", "drawdown", "exposure", "concentration"])
            risk_threshold = st.number_input(
                "Threshold",
                value=0.05 if risk_type in ["drawdown", "concentration"] else 500000.0
            )
            risk_severity = st.selectbox("Severity", ["HIGH", "CRITICAL", "MEDIUM"], key="risk_sev")

            if st.button("Create Risk Alert", type="primary"):
                config = create_risk_alert(
                    name=rule_name,
                    risk_type=risk_type,
                    threshold=risk_threshold,
                    severity=AlertSeverity(risk_severity),
                )
                alert_engine.add_rule(config)
                st.success(f"Created alert: {rule_name}")
                st.rerun()

        elif rule_type == "P&L Alert":
            pnl_type = st.selectbox("Alert Type", ["loss", "profit"])
            pnl_threshold = st.number_input("Threshold ($)", 1000, 1000000, 50000)
            pnl_severity = st.selectbox("Severity", ["HIGH", "MEDIUM", "CRITICAL"], key="pnl_sev")

            if st.button("Create P&L Alert", type="primary"):
                config = create_pnl_alert(
                    name=rule_name,
                    threshold=pnl_threshold,
                    alert_type=pnl_type,
                    severity=AlertSeverity(pnl_severity),
                )
                alert_engine.add_rule(config)
                st.success(f"Created alert: {rule_name}")
                st.rerun()

    with col2:
        st.markdown("### Existing Rules")

        rules = alert_engine.get_rules()

        if rules:
            for rule in rules:
                config = rule.config
                status_icon = "✅" if config.enabled else "⏸️"

                with st.expander(f"{status_icon} {config.name}", expanded=False):
                    st.markdown(f"**ID:** `{config.rule_id}`")
                    st.markdown(f"**Category:** {config.category.value}")
                    st.markdown(f"**Severity:** {config.severity.value}")
                    st.markdown(f"**Channels:** {', '.join(config.channels)}")
                    st.markdown(f"**Cooldown:** {config.cooldown_minutes} min")
                    st.markdown(f"**Triggers:** {config.trigger_count}")

                    if config.last_triggered:
                        st.markdown(f"**Last Triggered:** {config.last_triggered.strftime('%Y-%m-%d %H:%M')}")

                    col_a, col_b, col_c = st.columns(3)

                    with col_a:
                        if config.enabled:
                            if st.button("Disable", key=f"dis_{config.rule_id}"):
                                alert_engine.disable_rule(config.rule_id)
                                st.rerun()
                        else:
                            if st.button("Enable", key=f"en_{config.rule_id}"):
                                alert_engine.enable_rule(config.rule_id)
                                st.rerun()

                    with col_b:
                        new_channels = st.multiselect(
                            "Channels",
                            ["email", "telegram", "slack", "sms"],
                            default=config.channels,
                            key=f"ch_{config.rule_id}"
                        )
                        if new_channels != config.channels:
                            alert_engine.update_rule(config.rule_id, {"channels": new_channels})

                    with col_c:
                        if st.button("🗑️ Delete", key=f"del_{config.rule_id}"):
                            alert_engine.remove_rule(config.rule_id)
                            st.rerun()
        else:
            st.info("No alert rules configured. Create one on the left.")

# =============================================================================
# TAB 3: HISTORY & ANALYTICS
# =============================================================================
with tab3:
    st.subheader("Alert History & Analytics")

    alert_history = st.session_state.alert_history

    # Date range
    col1, col2 = st.columns(2)

    with col1:
        days_back = st.slider("Days of History", 1, 90, 30)

    with col2:
        hist_category = st.selectbox(
            "Category Filter",
            ["All", "SIGNAL", "RISK", "PRICE", "PNL", "EXECUTION"],
            key="hist_cat"
        )

    since = datetime.now() - timedelta(days=days_back)

    # Get statistics
    history_stats = alert_history.get_statistics(since=since)

    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Alerts", history_stats["total"])
    with col2:
        st.metric("Acknowledged", history_stats["acknowledged"])
    with col3:
        st.metric("Resolved", history_stats["resolved"])
    with col4:
        st.metric("Ack Rate", f"{history_stats['acknowledgment_rate']*100:.0f}%")
    with col5:
        st.metric("Avg Response", f"{history_stats['avg_response_minutes']:.0f} min")

    st.divider()

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Alerts by Severity")
        severity_data = history_stats.get("by_severity", {})
        if severity_data:
            import plotly.graph_objects as go

            severities = list(severity_data.keys())
            counts = list(severity_data.values())

            colors = {
                "INFO": "#17a2b8",
                "LOW": "#28a745",
                "MEDIUM": "#ffc107",
                "HIGH": "#fd7e14",
                "CRITICAL": "#dc3545",
            }

            fig = go.Figure(data=[
                go.Pie(
                    labels=severities,
                    values=counts,
                    marker_colors=[colors.get(s, "#6c757d") for s in severities],
                    hole=0.4,
                )
            ])

            fig.update_layout(
                height=300,
                template="plotly_dark",
                showlegend=True,
            )

            st.plotly_chart(fig, width="stretch", config=get_chart_config())
        else:
            st.info("No alert data for this period")

    with col2:
        st.markdown("### Alerts by Category")
        category_data = history_stats.get("by_category", {})
        if category_data:
            import plotly.graph_objects as go

            categories = list(category_data.keys())
            counts = list(category_data.values())

            fig = go.Figure(data=[
                go.Bar(
                    x=categories,
                    y=counts,
                    marker_color=COLORS.get("accent", "#f59e0b"),
                )
            ])

            fig.update_layout(
                height=300,
                template="plotly_dark",
                xaxis_title="Category",
                yaxis_title="Count",
            )

            st.plotly_chart(fig, width="stretch", config=get_chart_config())
        else:
            st.info("No alert data for this period")

    # Alert history table
    st.markdown("### Recent Alerts")

    category = AlertCategory(hist_category) if hist_category != "All" else None
    records = alert_history.query(
        category=category,
        since=since,
        limit=50,
    )

    if records:
        table_data = []
        for record in records:
            table_data.append({
                "Time": record.created_at.strftime("%Y-%m-%d %H:%M"),
                "Severity": record.severity.value,
                "Category": record.category.value,
                "Title": record.title[:50],
                "Acknowledged": "✓" if record.acknowledged else "✗",
                "Resolved": "✓" if record.resolved else "✗",
            })

        st.dataframe(pd.DataFrame(table_data), width="stretch", hide_index=True)
    else:
        st.info("No alert history for the selected filters")

# =============================================================================
# TAB 4: NOTIFICATION CHANNELS
# =============================================================================
with tab4:
    st.subheader("Notification Channels")

    st.markdown("""
    Configure notification channels for alert delivery. Each channel can be independently
    enabled and configured with different severity filters.
    """)

    # Email
    with st.expander("📧 Email", expanded=True):
        col1, col2 = st.columns([2, 1])

        with col1:
            email_enabled = st.checkbox("Enable Email Notifications", key="email_enabled")

            if email_enabled:
                email_smtp = st.text_input("SMTP Server", placeholder="smtp.gmail.com")
                email_port = st.number_input("SMTP Port", value=587)
                email_user = st.text_input("Username", placeholder="alerts@company.com")
                email_pass = st.text_input("Password", type="password")
                email_recipients = st.text_area("Recipients (one per line)", placeholder="trader@company.com\nrisk@company.com")

        with col2:
            if email_enabled:
                email_severity = st.selectbox("Minimum Severity", ["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"], index=1, key="email_sev")

                if st.button("Test Email", key="test_email"):
                    st.info("Email test would be sent (requires configuration)")

    # Telegram
    with st.expander("📱 Telegram", expanded=False):
        col1, col2 = st.columns([2, 1])

        with col1:
            telegram_enabled = st.checkbox("Enable Telegram Notifications", key="telegram_enabled")

            if telegram_enabled:
                telegram_token = st.text_input("Bot Token", type="password", placeholder="123456:ABC-DEF...")
                telegram_chat = st.text_input("Chat ID", placeholder="-1001234567890")

        with col2:
            if telegram_enabled:
                telegram_severity = st.selectbox("Minimum Severity", ["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"], index=1, key="tg_sev")

                if st.button("Test Telegram", key="test_telegram"):
                    st.info("Telegram test would be sent (requires configuration)")

    # Slack
    with st.expander("💬 Slack", expanded=False):
        col1, col2 = st.columns([2, 1])

        with col1:
            slack_enabled = st.checkbox("Enable Slack Notifications", key="slack_enabled")

            if slack_enabled:
                slack_webhook = st.text_input("Webhook URL", type="password", placeholder="https://hooks.slack.com/services/...")
                slack_channel = st.text_input("Channel", placeholder="#trading-alerts")

        with col2:
            if slack_enabled:
                slack_severity = st.selectbox("Minimum Severity", ["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"], index=1, key="slack_sev")

                if st.button("Test Slack", key="test_slack"):
                    st.info("Slack test would be sent (requires configuration)")

    # SMS
    with st.expander("📲 SMS (Twilio)", expanded=False):
        col1, col2 = st.columns([2, 1])

        with col1:
            sms_enabled = st.checkbox("Enable SMS Notifications", key="sms_enabled")

            if sms_enabled:
                sms_sid = st.text_input("Twilio Account SID", type="password")
                sms_token = st.text_input("Twilio Auth Token", type="password")
                sms_from = st.text_input("From Number", placeholder="+1234567890")
                sms_to = st.text_area("To Numbers (one per line)", placeholder="+1234567890")

        with col2:
            if sms_enabled:
                sms_severity = st.selectbox("Minimum Severity", ["HIGH", "CRITICAL"], index=0, key="sms_sev")
                st.caption("SMS is recommended only for critical alerts")

                if st.button("Test SMS", key="test_sms"):
                    st.info("SMS test would be sent (requires configuration)")

    # Save button
    st.divider()
    if st.button("💾 Save Channel Configuration", type="primary"):
        st.success("Channel configuration saved (would persist to config/alerts.yaml)")

# =============================================================================
# TAB 5: SCHEDULED REPORTS
# =============================================================================
with tab5:
    st.subheader("Scheduled Reports")

    report_scheduler = st.session_state.report_scheduler

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Create Scheduled Report")

        report_type = st.selectbox(
            "Report Type",
            ["Daily P&L", "Daily Risk", "Weekly Performance", "Position Summary", "Market Overview"],
            key="report_type"
        )

        report_name = st.text_input("Report Name", f"My {report_type}")

        report_frequency = st.selectbox(
            "Frequency",
            ["Daily", "Weekly", "Monthly"],
            key="report_freq"
        )

        report_time = st.time_input("Send Time", value=time(18, 0))

        if report_frequency == "Daily":
            report_days = st.multiselect(
                "Days",
                ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                default=["Mon", "Tue", "Wed", "Thu", "Fri"],
                key="report_days"
            )
        elif report_frequency == "Weekly":
            report_day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"], key="week_day")
        else:
            report_day_month = st.number_input("Day of Month", 1, 28, 1)

        report_channels = st.multiselect(
            "Delivery Channels",
            ["email", "slack"],
            default=["email"],
            key="report_channels"
        )

        include_charts = st.checkbox("Include Charts", value=True)

        if st.button("Create Report Schedule", type="primary"):
            # Map frequency
            freq_map = {"Daily": ReportFrequency.DAILY, "Weekly": ReportFrequency.WEEKLY, "Monthly": ReportFrequency.MONTHLY}
            type_map = {
                "Daily P&L": ReportType.DAILY_PNL,
                "Daily Risk": ReportType.DAILY_RISK,
                "Weekly Performance": ReportType.WEEKLY_PNL,
                "Position Summary": ReportType.POSITION_SUMMARY,
                "Market Overview": ReportType.MARKET_OVERVIEW,
            }

            # Map days
            day_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}

            config = ReportConfig(
                report_id=f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                name=report_name,
                report_type=type_map.get(report_type, ReportType.CUSTOM),
                frequency=freq_map[report_frequency],
                schedule_time=report_time,
                schedule_days=[day_map.get(d, 0) for d in report_days] if report_frequency == "Daily" else [0],
                channels=report_channels,
                include_charts=include_charts,
            )

            report_scheduler.add_report(config)
            st.success(f"Created report schedule: {report_name}")
            st.rerun()

    with col2:
        st.markdown("### Scheduled Reports")

        reports = report_scheduler.get_reports()

        if reports:
            for report in reports:
                config = report.config
                status_icon = "✅" if config.enabled else "⏸️"

                with st.expander(f"{status_icon} {config.name}", expanded=False):
                    st.markdown(f"**Type:** {config.report_type.value}")
                    st.markdown(f"**Frequency:** {config.frequency.value}")
                    st.markdown(f"**Time:** {config.schedule_time.strftime('%H:%M')}")
                    st.markdown(f"**Channels:** {', '.join(config.channels)}")

                    if config.next_run:
                        st.markdown(f"**Next Run:** {config.next_run.strftime('%Y-%m-%d %H:%M')}")

                    if config.last_run:
                        st.markdown(f"**Last Run:** {config.last_run.strftime('%Y-%m-%d %H:%M')}")

                    st.markdown(f"**Run Count:** {config.run_count}")

                    col_a, col_b, col_c = st.columns(3)

                    with col_a:
                        if st.button("Run Now", key=f"run_{config.report_id}"):
                            if report_scheduler.run_report_now(config.report_id):
                                st.success("Report generated")
                            else:
                                st.error("Failed to generate report")

                    with col_b:
                        if config.enabled:
                            if st.button("Disable", key=f"dis_rpt_{config.report_id}"):
                                report_scheduler.update_report(config.report_id, {"enabled": False})
                                st.rerun()
                        else:
                            if st.button("Enable", key=f"en_rpt_{config.report_id}"):
                                report_scheduler.update_report(config.report_id, {"enabled": True})
                                st.rerun()

                    with col_c:
                        if st.button("Delete", key=f"del_rpt_{config.report_id}"):
                            report_scheduler.remove_report(config.report_id)
                            st.rerun()
        else:
            st.info("No scheduled reports. Create one on the left.")

        # Scheduler stats
        st.divider()
        st.markdown("### Scheduler Status")

        scheduler_stats = report_scheduler.get_statistics()

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Total Reports", scheduler_stats["total_reports"])
        with col_b:
            st.metric("Enabled", scheduler_stats["enabled_reports"])

# Footer
st.divider()
st.caption(
    "💡 Tip: Configure notification channels in the Channels tab, then create alert rules to start receiving notifications. "
    "Use scheduled reports for regular P&L and risk summaries."
)
