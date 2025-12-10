"""
UI Components
=============
Reusable UI components for enhanced dashboard presentation.
"""

import streamlit as st
from typing import Literal


def render_card(
    title: str,
    content: str = "",
    icon: str = "",
    footer: str = "",
    variant: Literal["default", "success", "warning", "error", "info"] = "default",
) -> None:
    """
    Render a styled card component.
    
    Args:
        title: Card title
        content: Card content (supports HTML)
        icon: Optional emoji/icon for the title
        footer: Optional footer text
        variant: Card style variant
    """
    border_colors = {
        "default": "#334155",
        "success": "rgba(0, 220, 130, 0.5)",
        "warning": "rgba(245, 158, 11, 0.5)",
        "error": "rgba(239, 68, 68, 0.5)",
        "info": "rgba(14, 165, 233, 0.5)",
    }
    
    border_color = border_colors.get(variant, border_colors["default"])
    
    st.markdown(f"""
    <div class="dashboard-card" style="border-color: {border_color};">
        <div class="dashboard-card-header">
            <span class="dashboard-card-title">{icon} {title}</span>
        </div>
        <div style="color: #cbd5e1; font-size: 14px; line-height: 1.6;">
            {content}
        </div>
        {f'<div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(51, 65, 85, 0.3); font-size: 12px; color: #64748b;">{footer}</div>' if footer else ''}
    </div>
    """, unsafe_allow_html=True)


def render_badge(
    text: str,
    variant: Literal["success", "warning", "error", "info", "neutral"] = "neutral",
) -> str:
    """
    Return HTML for a badge component.
    
    Args:
        text: Badge text
        variant: Badge style variant
        
    Returns:
        HTML string for the badge
    """
    class_name = f"badge badge-{variant}" if variant != "neutral" else "badge"
    
    if variant == "neutral":
        return f'<span class="badge" style="background: rgba(100, 116, 139, 0.2); color: #94a3b8; border: 1px solid rgba(100, 116, 139, 0.3);">{text}</span>'
    
    return f'<span class="{class_name}">{text}</span>'


def render_section_header(
    title: str,
    icon: str = "",
    subtitle: str = "",
) -> None:
    """
    Render a styled section header.
    
    Args:
        title: Section title
        icon: Optional emoji/icon
        subtitle: Optional subtitle/description
    """
    st.markdown(f"""
    <div class="section-header">
        <span class="section-header-icon">{icon}</span>
        <span class="section-header-title">{title}</span>
        {f'<span class="section-header-subtitle">{subtitle}</span>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


def render_news_item(
    title: str,
    source: str,
    time: str,
    summary: str,
    sentiment: Literal["bullish", "bearish", "neutral"] = "neutral",
    impact: Literal["HIGH", "MEDIUM", "LOW"] = "MEDIUM",
) -> None:
    """
    Render a styled news feed item.
    
    Args:
        title: News headline
        source: News source
        time: Time ago string
        summary: News summary
        sentiment: Sentiment indicator
        impact: Impact level
    """
    sentiment_colors = {
        "bullish": "#00DC82",
        "bearish": "#ef4444",
        "neutral": "#94a3b8",
    }
    
    sentiment_icons = {
        "bullish": "📈",
        "bearish": "📉",
        "neutral": "➡️",
    }
    
    impact_badges = {
        "HIGH": '<span class="badge badge-error">HIGH</span>',
        "MEDIUM": '<span class="badge badge-warning">MED</span>',
        "LOW": '<span class="badge badge-info">LOW</span>',
    }
    
    sentiment_color = sentiment_colors.get(sentiment, sentiment_colors["neutral"])
    sentiment_icon = sentiment_icons.get(sentiment, "➡️")
    impact_badge = impact_badges.get(impact, impact_badges["MEDIUM"])
    
    st.markdown(f"""
    <div class="news-item">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
            <span class="news-item-title">{title}</span>
            <div style="display: flex; gap: 8px; align-items: center; flex-shrink: 0; margin-left: 12px;">
                <span style="color: {sentiment_color}; font-size: 16px;">{sentiment_icon}</span>
                {impact_badge}
            </div>
        </div>
        <div class="news-item-meta">
            <span style="color: var(--accent-blue);">{source}</span> • {time}
        </div>
        <div class="news-item-summary">{summary}</div>
    </div>
    """, unsafe_allow_html=True)


def render_trending_topic(
    topic: str,
    trend: Literal["up", "down", "stable"] = "stable",
    sentiment: Literal["bullish", "bearish", "neutral"] = "neutral",
    volume: str = "medium",
) -> None:
    """
    Render a trending topic card.
    
    Args:
        topic: Topic name
        trend: Trend direction
        sentiment: Overall sentiment
        volume: Discussion volume
    """
    trend_icons = {"up": "📈", "down": "📉", "stable": "➡️"}
    sentiment_colors = {
        "bullish": "#00DC82",
        "bearish": "#ef4444", 
        "neutral": "#94a3b8",
    }
    
    icon = trend_icons.get(trend, "➡️")
    color = sentiment_colors.get(sentiment, "#94a3b8")
    
    st.markdown(f"""
    <div class="trending-topic">
        <div class="trending-icon">{icon}</div>
        <div class="trending-name">{topic}</div>
        <div class="trending-sentiment" style="color: {color};">{sentiment.upper()}</div>
    </div>
    """, unsafe_allow_html=True)


def render_stat_card(
    label: str,
    value: str,
    delta: str = "",
    delta_color: Literal["green", "red", "gray"] = "gray",
    icon: str = "",
) -> None:
    """
    Render a compact stat card.
    
    Args:
        label: Stat label
        value: Main value
        delta: Change value
        delta_color: Color for delta
        icon: Optional icon
    """
    colors = {"green": "#00DC82", "red": "#ef4444", "gray": "#94a3b8"}
    color = colors.get(delta_color, colors["gray"])
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid rgba(51, 65, 85, 0.5);
        text-align: center;
    ">
        {f'<div style="font-size: 20px; margin-bottom: 8px;">{icon}</div>' if icon else ''}
        <div style="font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">
            {label}
        </div>
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 20px; font-weight: 600; color: #f1f5f9;">
            {value}
        </div>
        {f'<div style="font-size: 12px; color: {color}; margin-top: 4px;">{delta}</div>' if delta else ''}
    </div>
    """, unsafe_allow_html=True)


def render_loading_placeholder(height: int = 100) -> None:
    """
    Render a loading shimmer placeholder.
    
    Args:
        height: Height in pixels
    """
    st.markdown(f"""
    <div class="loading-shimmer" style="height: {height}px;"></div>
    """, unsafe_allow_html=True)


def render_empty_state(
    title: str,
    description: str = "",
    icon: str = "📭",
) -> None:
    """
    Render an empty state placeholder.
    
    Args:
        title: Empty state title
        description: Optional description
        icon: Icon to display
    """
    st.markdown(f"""
    <div style="
        text-align: center;
        padding: 40px 20px;
        background: rgba(30, 41, 59, 0.3);
        border-radius: 12px;
        border: 1px dashed rgba(51, 65, 85, 0.5);
    ">
        <div style="font-size: 48px; margin-bottom: 16px; opacity: 0.5;">{icon}</div>
        <div style="font-size: 16px; font-weight: 600; color: #e2e8f0; margin-bottom: 8px;">
            {title}
        </div>
        {f'<div style="font-size: 14px; color: #64748b;">{description}</div>' if description else ''}
    </div>
    """, unsafe_allow_html=True)


def render_info_banner(
    message: str,
    variant: Literal["info", "success", "warning", "error"] = "info",
    icon: str = "",
) -> None:
    """
    Render an info banner.
    
    Args:
        message: Banner message
        variant: Banner style
        icon: Optional icon
    """
    colors = {
        "info": ("#0ea5e9", "rgba(14, 165, 233, 0.1)", "rgba(14, 165, 233, 0.3)"),
        "success": ("#00DC82", "rgba(0, 220, 130, 0.1)", "rgba(0, 220, 130, 0.3)"),
        "warning": ("#f59e0b", "rgba(245, 158, 11, 0.1)", "rgba(245, 158, 11, 0.3)"),
        "error": ("#ef4444", "rgba(239, 68, 68, 0.1)", "rgba(239, 68, 68, 0.3)"),
    }
    
    icons = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
    }
    
    text_color, bg_color, border_color = colors.get(variant, colors["info"])
    display_icon = icon or icons.get(variant, "ℹ️")
    
    st.markdown(f"""
    <div style="
        background: {bg_color};
        border: 1px solid {border_color};
        border-radius: 10px;
        padding: 12px 16px;
        display: flex;
        align-items: center;
        gap: 12px;
    ">
        <span style="font-size: 18px;">{display_icon}</span>
        <span style="color: {text_color}; font-size: 14px;">{message}</span>
    </div>
    """, unsafe_allow_html=True)


def render_kpi_row(kpis: list[dict]) -> None:
    """
    Render a row of KPI cards.
    
    Args:
        kpis: List of KPI dicts with keys: label, value, delta, delta_color, icon
    """
    cols = st.columns(len(kpis))
    for col, kpi in zip(cols, kpis):
        with col:
            render_stat_card(
                label=kpi.get("label", ""),
                value=kpi.get("value", ""),
                delta=kpi.get("delta", ""),
                delta_color=kpi.get("delta_color", "gray"),
                icon=kpi.get("icon", ""),
            )


def render_progress_ring(
    value: float,
    max_value: float = 100,
    label: str = "",
    color: str = "#0ea5e9",
    size: int = 120,
) -> None:
    """
    Render a circular progress indicator.
    
    Args:
        value: Current value
        max_value: Maximum value
        label: Label text
        color: Progress color
        size: Size in pixels
    """
    percentage = min(value / max_value * 100, 100)
    circumference = 2 * 3.14159 * 45
    dash_offset = circumference * (1 - percentage / 100)
    
    st.markdown(f"""
    <div style="text-align: center;">
        <svg width="{size}" height="{size}" viewBox="0 0 120 120">
            <circle cx="60" cy="60" r="45" fill="none" stroke="rgba(51, 65, 85, 0.3)" stroke-width="8"/>
            <circle cx="60" cy="60" r="45" fill="none" stroke="{color}" stroke-width="8"
                    stroke-linecap="round"
                    stroke-dasharray="{circumference}"
                    stroke-dashoffset="{dash_offset}"
                    transform="rotate(-90 60 60)"
                    style="transition: stroke-dashoffset 0.5s ease;"/>
            <text x="60" y="55" text-anchor="middle" fill="#f1f5f9" font-size="24" font-weight="600" font-family="IBM Plex Mono, monospace">
                {value:.0f}%
            </text>
            <text x="60" y="75" text-anchor="middle" fill="#64748b" font-size="10" font-family="Inter, sans-serif">
                {label}
            </text>
        </svg>
    </div>
    """, unsafe_allow_html=True)
