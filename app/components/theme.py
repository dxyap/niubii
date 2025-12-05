"""
Dashboard Theme
===============
Shared styling for all dashboard pages.
Professional trading terminal aesthetics with enhanced chart support.
"""

DASHBOARD_THEME_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Outfit:wght@400;500;600;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0a0f1a 0%, #111827 50%, #0f172a 100%);
    }

    [data-testid="stMetricValue"] {
        font-size: 26px;
        font-weight: 600;
        font-family: 'IBM Plex Mono', 'SF Mono', monospace;
        color: #e2e8f0 !important;
    }

    [data-testid="stMetricDelta"] {
        font-size: 13px;
        font-family: 'IBM Plex Mono', monospace;
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-family: 'Outfit', sans-serif;
        font-weight: 500;
        text-transform: uppercase;
        font-size: 11px;
        letter-spacing: 0.05em;
    }

    h1 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        color: #f1f5f9 !important;
        letter-spacing: -0.02em;
    }

    h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        color: #e2e8f0 !important;
    }

    p, span, div { color: #cbd5e1; }

    .stMarkdown { color: #cbd5e1; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid #334155;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f8fafc !important;
    }

    [data-testid="stSidebar"] .stMarkdown { color: #94a3b8; }

    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-family: 'Outfit', sans-serif;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.25);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #38bdf8 0%, #0ea5e9 100%);
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.4);
        transform: translateY(-1px);
    }

    .live-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background-color: #22c55e;
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 0 8px rgba(34, 197, 94, 0.6);
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { opacity: 1; box-shadow: 0 0 8px rgba(34, 197, 94, 0.6); }
        50% { opacity: 0.7; box-shadow: 0 0 16px rgba(34, 197, 94, 0.4); }
        100% { opacity: 1; box-shadow: 0 0 8px rgba(34, 197, 94, 0.6); }
    }

    .profit { color: #22c55e !important; }
    .loss { color: #ef4444 !important; }

    .dataframe {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
        color: #e2e8f0;
    }

    [data-testid="stDataFrame"] {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 8px;
        border: 1px solid #334155;
    }

    .stProgress > div > div {
        background: linear-gradient(90deg, #0ea5e9 0%, #22c55e 100%);
    }

    .stDivider {
        border-color: #334155;
    }

    [data-testid="stExpander"] {
        background: rgba(30, 41, 59, 0.3);
        border: 1px solid #334155;
        border-radius: 8px;
    }

    .status-ok { color: #22c55e; }
    .status-warning { color: #f59e0b; }
    .status-critical { color: #ef4444; }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* Enhanced chart container styling */
    [data-testid="stVegaLiteChart"],
    .stPlotlyChart {
        background: rgba(15, 23, 42, 0.4);
        border-radius: 12px;
        padding: 12px;
        border: 1px solid #334155;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    /* Chart card styling */
    .chart-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid rgba(51, 65, 85, 0.5);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.4);
        padding: 4px;
        border-radius: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        color: #94a3b8;
        padding: 8px 16px;
        font-family: 'Outfit', sans-serif;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(30, 41, 59, 0.6);
        color: #e2e8f0;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
        color: white !important;
    }

    /* Select box styling */
    [data-testid="stSelectbox"] > div > div {
        background-color: rgba(30, 41, 59, 0.8);
        border-color: #334155;
        border-radius: 8px;
    }

    /* Input styling */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background-color: rgba(30, 41, 59, 0.8);
        border-color: #334155;
        color: #e2e8f0;
        border-radius: 8px;
    }

    /* Slider styling */
    .stSlider > div > div > div {
        background-color: #0ea5e9;
    }

    /* Alert styling */
    .stAlert {
        border-radius: 8px;
    }

    /* Metric card enhancement */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.3);
        padding: 12px 16px;
        border-radius: 10px;
        border: 1px solid rgba(51, 65, 85, 0.3);
    }
</style>
"""


def apply_theme(st):
    """Apply the dashboard theme to the current page."""
    st.markdown(DASHBOARD_THEME_CSS, unsafe_allow_html=True)


# Color palette for consistent use
COLORS = {
    # Primary colors
    "primary": "#0ea5e9",
    "primary_light": "#38bdf8",
    "primary_dark": "#0284c7",

    # Secondary colors
    "secondary": "#8b5cf6",
    "secondary_light": "#a78bfa",

    # Semantic colors
    "success": "#00DC82",
    "warning": "#f59e0b",
    "error": "#FF5252",
    "info": "#00A3E0",

    # Candlestick colors (solid fills)
    "candle_up": "#00DC82",
    "candle_down": "#FF5252",

    # Text colors
    "text": "#e2e8f0",
    "text_muted": "#94a3b8",
    "text_bright": "#f8fafc",

    # Background colors
    "background": "#0f172a",
    "surface": "#1e293b",
    "surface_light": "#334155",
    "border": "#334155",

    # Chart colors
    "chart_bg": "rgba(15, 23, 42, 0.8)",
    "chart_grid": "rgba(51, 65, 85, 0.4)",

    # Moving average colors
    "ma_fast": "#FFB020",
    "ma_slow": "#A855F7",
    "ma_long": "#06B6D4",
}


# Plotly theme configuration - enhanced for professional trading charts
PLOTLY_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(15, 23, 42, 0.6)",
    "font": {
        "family": "'IBM Plex Mono', 'SF Mono', monospace",
        "color": "#e2e8f0",
        "size": 12,
    },
    "xaxis": {
        "gridcolor": "rgba(51, 65, 85, 0.4)",
        "gridwidth": 1,
        "zeroline": False,
        "tickfont": {"size": 11, "color": "#94a3b8"},
    },
    "yaxis": {
        "gridcolor": "rgba(51, 65, 85, 0.4)",
        "gridwidth": 1,
        "zeroline": False,
        "tickfont": {"size": 11, "color": "#94a3b8"},
        "side": "right",
    },
    "legend": {
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "right",
        "x": 1,
        "bgcolor": "rgba(0,0,0,0)",
        "font": {"size": 11, "color": "#94a3b8"},
    },
    "margin": {"l": 10, "r": 60, "t": 40, "b": 40},
    "hovermode": "x unified",
    "hoverlabel": {
        "bgcolor": "rgba(10, 15, 30, 0.95)",
        "bordercolor": "#334155",
        "font": {"family": "'IBM Plex Mono', monospace", "size": 12},
    },
}


# Chart-specific configurations
CANDLESTICK_CONFIG = {
    "increasing": {
        "line": {"color": "#00DC82", "width": 1},
        "fillcolor": "#00DC82",
    },
    "decreasing": {
        "line": {"color": "#FF5252", "width": 1},
        "fillcolor": "#FF5252",
    },
}


def get_chart_config():
    """Get chart display configuration for Streamlit."""
    return {
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "autoScale2d",
            "lasso2d",
            "select2d",
        ],
        "modeBarButtonsToAdd": [
            "drawline",
            "drawopenpath",
            "eraseshape",
        ],
    }
