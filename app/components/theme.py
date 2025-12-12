"""
Dashboard Theme
===============
Shared styling for all dashboard pages.
Professional trading terminal aesthetics with enhanced chart support.
Optimized for at-a-glance insights for oil traders.
"""

DASHBOARD_THEME_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

    /* ========== ROOT VARIABLES ========== */
    :root {
        --bg-primary: #0a0f1a;
        --bg-secondary: #111827;
        --bg-tertiary: #1e293b;
        --bg-card: rgba(30, 41, 59, 0.6);
        --bg-card-hover: rgba(30, 41, 59, 0.8);
        
        --border-primary: #334155;
        --border-light: rgba(51, 65, 85, 0.5);
        
        --text-primary: #f1f5f9;
        --text-secondary: #e2e8f0;
        --text-muted: #94a3b8;
        --text-dim: #64748b;
        
        --accent-blue: #0ea5e9;
        --accent-blue-light: #38bdf8;
        --accent-purple: #8b5cf6;
        --accent-cyan: #06b6d4;
        --accent-gold: #fbbf24;
        
        --success: #00DC82;
        --success-bg: rgba(0, 220, 130, 0.1);
        --success-gradient: linear-gradient(135deg, rgba(0, 220, 130, 0.2) 0%, rgba(0, 220, 130, 0.05) 100%);
        --warning: #f59e0b;
        --warning-bg: rgba(245, 158, 11, 0.1);
        --warning-gradient: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(245, 158, 11, 0.05) 100%);
        --error: #ef4444;
        --error-bg: rgba(239, 68, 68, 0.1);
        --error-gradient: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(239, 68, 68, 0.05) 100%);
        
        /* P&L specific colors */
        --pnl-profit: #00DC82;
        --pnl-loss: #FF5252;
        --pnl-neutral: #94a3b8;
        
        --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.2);
        --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.3);
        --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.4);
        --shadow-glow-blue: 0 0 20px rgba(14, 165, 233, 0.3);
        --shadow-glow-green: 0 0 20px rgba(0, 220, 130, 0.3);
        --shadow-glow-red: 0 0 20px rgba(255, 82, 82, 0.3);
        
        --radius-sm: 6px;
        --radius-md: 10px;
        --radius-lg: 14px;
        --radius-xl: 18px;
        
        --transition-fast: 0.15s ease;
        --transition-normal: 0.25s ease;
        --transition-slow: 0.4s ease;
    }

    /* ========== BASE STYLES ========== */
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 50%, #0f172a 100%);
    }

    * {
        scrollbar-width: thin;
        scrollbar-color: var(--border-primary) transparent;
    }

    *::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }

    *::-webkit-scrollbar-track {
        background: transparent;
    }

    *::-webkit-scrollbar-thumb {
        background: var(--border-primary);
        border-radius: 3px;
    }

    /* ========== TYPOGRAPHY ========== */
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: var(--text-primary) !important;
        letter-spacing: -0.03em;
        line-height: 1.2;
    }

    h2, h3, h4 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: var(--text-secondary) !important;
        letter-spacing: -0.02em;
    }

    p, span, div, label { 
        color: var(--text-muted); 
        font-family: 'Inter', sans-serif;
    }

    .stMarkdown { 
        color: var(--text-muted); 
        line-height: 1.6;
    }

    /* ========== METRICS ========== */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 600;
        font-family: 'IBM Plex Mono', 'SF Mono', monospace;
        color: var(--text-primary) !important;
        letter-spacing: -0.02em;
    }

    [data-testid="stMetricDelta"] {
        font-size: 13px;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 500;
    }

    [data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 11px;
        letter-spacing: 0.08em;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(135deg, var(--bg-card) 0%, rgba(15, 23, 42, 0.7) 100%);
        padding: 16px 20px;
        border-radius: var(--radius-lg);
        border: 1px solid var(--border-light);
        transition: all var(--transition-normal);
        backdrop-filter: blur(10px);
    }

    [data-testid="stMetric"]:hover {
        border-color: var(--accent-blue);
        box-shadow: var(--shadow-glow-blue);
        transform: translateY(-2px);
    }

    /* ========== SIDEBAR ========== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.98) 0%, rgba(30, 41, 59, 0.95) 100%);
        border-right: 1px solid var(--border-primary);
        backdrop-filter: blur(20px);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--text-primary) !important;
    }

    [data-testid="stSidebar"] .stMarkdown { 
        color: var(--text-muted); 
    }

    /* ========== BUTTONS ========== */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-blue) 0%, #0284c7 100%);
        color: white;
        border: none;
        border-radius: var(--radius-md);
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        padding: 0.6rem 1.2rem;
        transition: all var(--transition-normal);
        box-shadow: var(--shadow-sm), 0 0 20px rgba(14, 165, 233, 0.2);
        text-transform: uppercase;
        font-size: 12px;
        letter-spacing: 0.05em;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, var(--accent-blue-light) 0%, var(--accent-blue) 100%);
        box-shadow: var(--shadow-md), var(--shadow-glow-blue);
        transform: translateY(-2px);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* ========== LIVE INDICATOR ========== */
    .live-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background-color: var(--success);
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 0 12px rgba(0, 220, 130, 0.6);
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { 
            opacity: 1; 
            box-shadow: 0 0 12px rgba(0, 220, 130, 0.6);
            transform: scale(1);
        }
        50% { 
            opacity: 0.7; 
            box-shadow: 0 0 24px rgba(0, 220, 130, 0.4);
            transform: scale(1.1);
        }
    }

    /* ========== STATUS COLORS ========== */
    .profit { color: var(--success) !important; }
    .loss { color: var(--error) !important; }
    .status-ok { color: var(--success); }
    .status-warning { color: var(--warning); }
    .status-critical { color: var(--error); }

    /* ========== DATA TABLES ========== */
    .dataframe {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
        color: var(--text-secondary);
    }

    [data-testid="stDataFrame"] {
        background: var(--bg-card);
        border-radius: var(--radius-lg);
        border: 1px solid var(--border-light);
        overflow: hidden;
        backdrop-filter: blur(10px);
    }

    /* ========== PROGRESS BARS ========== */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--accent-blue) 0%, var(--success) 100%);
        border-radius: 4px;
    }

    /* ========== DIVIDERS ========== */
    .stDivider, hr {
        border-color: var(--border-light) !important;
        opacity: 0.5;
    }

    /* ========== EXPANDERS ========== */
    [data-testid="stExpander"] {
        background: var(--bg-card);
        border: 1px solid var(--border-light);
        border-radius: var(--radius-lg);
        overflow: hidden;
        transition: all var(--transition-normal);
    }

    [data-testid="stExpander"]:hover {
        border-color: var(--border-primary);
    }

    /* ========== HIDE STREAMLIT BRANDING (keep header for sidebar toggle) ========== */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { 
        visibility: visible; 
        background: transparent; 
    }

    /* ========== CHARTS ========== */
    [data-testid="stVegaLiteChart"],
    .stPlotlyChart {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.6) 0%, rgba(10, 15, 26, 0.8) 100%);
        border-radius: var(--radius-lg);
        padding: 16px;
        border: 1px solid var(--border-light);
        box-shadow: var(--shadow-md);
        backdrop-filter: blur(10px);
    }

    .chart-container {
        background: linear-gradient(135deg, var(--bg-card) 0%, rgba(15, 23, 42, 0.9) 100%);
        border-radius: var(--radius-lg);
        padding: 20px;
        border: 1px solid var(--border-light);
        box-shadow: var(--shadow-md);
    }

    /* ========== TABS ========== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: var(--radius-lg);
        border: 1px solid var(--border-light);
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: var(--radius-md);
        color: var(--text-muted);
        padding: 10px 20px;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 13px;
        transition: all var(--transition-fast);
        border: 1px solid transparent;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: var(--bg-card-hover);
        color: var(--text-secondary);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--accent-blue) 0%, #0284c7 100%) !important;
        color: white !important;
        box-shadow: var(--shadow-sm), 0 0 15px rgba(14, 165, 233, 0.3);
    }

    /* ========== FORM INPUTS ========== */
    [data-testid="stSelectbox"] > div > div {
        background-color: var(--bg-card);
        border-color: var(--border-primary);
        border-radius: var(--radius-md);
        transition: all var(--transition-fast);
    }

    [data-testid="stSelectbox"] > div > div:hover {
        border-color: var(--accent-blue);
    }

    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: var(--bg-card);
        border-color: var(--border-primary);
        color: var(--text-secondary);
        border-radius: var(--radius-md);
        padding: 10px 14px;
        transition: all var(--transition-fast);
    }

    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--accent-blue);
        box-shadow: 0 0 0 2px rgba(14, 165, 233, 0.2);
    }

    /* ========== SLIDER ========== */
    .stSlider > div > div > div {
        background-color: var(--accent-blue);
    }

    /* ========== ALERTS ========== */
    .stAlert {
        border-radius: var(--radius-md);
        border: none;
        backdrop-filter: blur(10px);
    }

    /* ========== CUSTOM CARDS ========== */
    .dashboard-card {
        background: linear-gradient(135deg, var(--bg-card) 0%, rgba(15, 23, 42, 0.8) 100%);
        border-radius: var(--radius-lg);
        padding: 20px;
        border: 1px solid var(--border-light);
        box-shadow: var(--shadow-sm);
        transition: all var(--transition-normal);
        backdrop-filter: blur(10px);
    }

    .dashboard-card:hover {
        border-color: var(--accent-blue);
        box-shadow: var(--shadow-md), var(--shadow-glow-blue);
        transform: translateY(-2px);
    }

    .dashboard-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid var(--border-light);
    }

    .dashboard-card-title {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 14px;
        color: var(--text-primary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* ========== NEWS FEED ITEMS ========== */
    .news-item {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.7) 100%);
        border-radius: var(--radius-md);
        padding: 16px;
        margin-bottom: 12px;
        border: 1px solid var(--border-light);
        transition: all var(--transition-normal);
        cursor: pointer;
    }

    .news-item:hover {
        border-color: var(--accent-blue);
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        transform: translateX(4px);
    }

    .news-item-title {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 14px;
        color: var(--text-primary);
        margin-bottom: 6px;
        line-height: 1.4;
    }

    .news-item-meta {
        font-size: 11px;
        color: var(--text-dim);
        margin-bottom: 8px;
    }

    .news-item-summary {
        font-size: 13px;
        color: var(--text-muted);
        line-height: 1.5;
    }

    /* ========== BADGES ========== */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    .badge-success {
        background: var(--success-bg);
        color: var(--success);
        border: 1px solid rgba(0, 220, 130, 0.3);
    }

    .badge-warning {
        background: var(--warning-bg);
        color: var(--warning);
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .badge-error {
        background: var(--error-bg);
        color: var(--error);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    .badge-info {
        background: rgba(14, 165, 233, 0.1);
        color: var(--accent-blue);
        border: 1px solid rgba(14, 165, 233, 0.3);
    }

    /* ========== TRENDING TOPICS ========== */
    .trending-topic {
        background: linear-gradient(135deg, var(--bg-card) 0%, rgba(15, 23, 42, 0.8) 100%);
        border-radius: var(--radius-md);
        padding: 14px;
        text-align: center;
        border: 1px solid var(--border-light);
        transition: all var(--transition-normal);
    }

    .trending-topic:hover {
        border-color: var(--accent-purple);
        box-shadow: 0 0 15px rgba(139, 92, 246, 0.2);
        transform: scale(1.02);
    }

    .trending-icon {
        font-size: 24px;
        margin-bottom: 8px;
    }

    .trending-name {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 11px;
        color: var(--text-primary);
        margin-bottom: 4px;
    }

    .trending-sentiment {
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* ========== WORD CLOUD CONTAINER ========== */
    .wordcloud-container {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(10, 15, 26, 0.95) 100%);
        border-radius: var(--radius-lg);
        padding: 16px;
        border: 1px solid var(--border-light);
        text-align: center;
    }

    /* ========== SENTIMENT GAUGE ========== */
    .sentiment-gauge-container {
        background: var(--bg-card);
        border-radius: var(--radius-lg);
        padding: 20px;
        border: 1px solid var(--border-light);
    }

    /* ========== LOADING ANIMATION ========== */
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    .loading-shimmer {
        background: linear-gradient(90deg, 
            var(--bg-card) 0%, 
            rgba(51, 65, 85, 0.3) 50%, 
            var(--bg-card) 100%);
        background-size: 200% 100%;
        animation: shimmer 1.5s infinite;
        border-radius: var(--radius-md);
    }

    /* ========== SECTION HEADERS ========== */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 16px;
    }

    .section-header-icon {
        font-size: 20px;
    }

    .section-header-title {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 16px;
        color: var(--text-primary);
    }

    .section-header-subtitle {
        font-size: 12px;
        color: var(--text-dim);
        margin-left: auto;
    }

    /* ========== ANIMATIONS ========== */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    .animate-fade-in {
        animation: fadeIn 0.5s ease forwards;
    }

    .animate-fade-in-up {
        animation: fadeInUp 0.5s ease forwards;
    }

    /* ========== RESPONSIVE ADJUSTMENTS ========== */
    @media (max-width: 768px) {
        [data-testid="stMetricValue"] {
            font-size: 22px;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 8px 12px;
            font-size: 12px;
        }
    }

    /* ========== TRADER QUICK-GLANCE COMPONENTS ========== */
    
    /* Market Pulse Header - Shows market state at a glance */
    .market-pulse {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 16px 24px;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 100%);
        border-radius: var(--radius-lg);
        border: 1px solid var(--border-light);
        margin-bottom: 20px;
        box-shadow: var(--shadow-md);
    }
    
    .market-pulse-item {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 4px;
    }
    
    .market-pulse-label {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-dim);
        font-weight: 600;
    }
    
    .market-pulse-value {
        font-size: 20px;
        font-weight: 700;
        font-family: 'IBM Plex Mono', monospace;
        color: var(--text-primary);
    }
    
    .market-pulse-change {
        font-size: 12px;
        font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
    }
    
    /* P&L Display - Large format for quick visibility */
    .pnl-display {
        text-align: center;
        padding: 24px;
        border-radius: var(--radius-lg);
        backdrop-filter: blur(10px);
    }
    
    .pnl-display.profit {
        background: var(--success-gradient);
        border: 1px solid rgba(0, 220, 130, 0.3);
        box-shadow: var(--shadow-glow-green);
    }
    
    .pnl-display.loss {
        background: var(--error-gradient);
        border: 1px solid rgba(255, 82, 82, 0.3);
        box-shadow: var(--shadow-glow-red);
    }
    
    .pnl-display.neutral {
        background: var(--bg-card);
        border: 1px solid var(--border-light);
    }
    
    .pnl-value {
        font-size: 42px;
        font-weight: 700;
        font-family: 'IBM Plex Mono', monospace;
        line-height: 1.1;
    }
    
    .pnl-value.profit { color: var(--pnl-profit); }
    .pnl-value.loss { color: var(--pnl-loss); }
    .pnl-value.neutral { color: var(--pnl-neutral); }
    
    .pnl-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        margin-top: 8px;
    }
    
    /* Position Heat Strip - Visual position status */
    .position-heat-strip {
        display: flex;
        gap: 8px;
        padding: 12px;
        background: var(--bg-card);
        border-radius: var(--radius-md);
        border: 1px solid var(--border-light);
        overflow-x: auto;
    }
    
    .position-chip {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 10px 16px;
        border-radius: var(--radius-md);
        min-width: 90px;
        transition: all var(--transition-fast);
    }
    
    .position-chip:hover {
        transform: translateY(-2px);
    }
    
    .position-chip.long {
        background: linear-gradient(135deg, rgba(0, 220, 130, 0.2) 0%, rgba(0, 220, 130, 0.05) 100%);
        border: 1px solid rgba(0, 220, 130, 0.3);
    }
    
    .position-chip.short {
        background: linear-gradient(135deg, rgba(255, 82, 82, 0.2) 0%, rgba(255, 82, 82, 0.05) 100%);
        border: 1px solid rgba(255, 82, 82, 0.3);
    }
    
    .position-chip-symbol {
        font-size: 11px;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: 0.03em;
    }
    
    .position-chip-qty {
        font-size: 14px;
        font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
    }
    
    .position-chip.long .position-chip-qty { color: var(--pnl-profit); }
    .position-chip.short .position-chip-qty { color: var(--pnl-loss); }
    
    .position-chip-pnl {
        font-size: 10px;
        font-family: 'IBM Plex Mono', monospace;
        margin-top: 2px;
    }
    
    /* Signal Indicator - Bold signal display */
    .signal-indicator {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 20px 24px;
        border-radius: var(--radius-lg);
        backdrop-filter: blur(10px);
    }
    
    .signal-indicator.long {
        background: linear-gradient(135deg, rgba(0, 220, 130, 0.15) 0%, rgba(0, 220, 130, 0.03) 100%);
        border-left: 4px solid var(--pnl-profit);
    }
    
    .signal-indicator.short {
        background: linear-gradient(135deg, rgba(255, 82, 82, 0.15) 0%, rgba(255, 82, 82, 0.03) 100%);
        border-left: 4px solid var(--pnl-loss);
    }
    
    .signal-indicator.neutral {
        background: var(--bg-card);
        border-left: 4px solid var(--text-muted);
    }
    
    .signal-direction {
        font-size: 28px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .signal-direction.long { color: var(--pnl-profit); }
    .signal-direction.short { color: var(--pnl-loss); }
    .signal-direction.neutral { color: var(--text-muted); }
    
    .signal-confidence {
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    
    .signal-confidence-value {
        font-size: 24px;
        font-weight: 700;
        font-family: 'IBM Plex Mono', monospace;
        color: var(--text-primary);
    }
    
    .signal-confidence-label {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-dim);
    }
    
    /* Risk Traffic Light */
    .risk-traffic-light {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 16px 20px;
        background: var(--bg-card);
        border-radius: var(--radius-lg);
        border: 1px solid var(--border-light);
    }
    
    .traffic-light-dot {
        width: 16px;
        height: 16px;
        border-radius: 50%;
        transition: all var(--transition-normal);
    }
    
    .traffic-light-dot.green {
        background: var(--success);
        box-shadow: 0 0 12px rgba(0, 220, 130, 0.6);
    }
    
    .traffic-light-dot.yellow {
        background: var(--warning);
        box-shadow: 0 0 12px rgba(245, 158, 11, 0.6);
    }
    
    .traffic-light-dot.red {
        background: var(--error);
        box-shadow: 0 0 12px rgba(239, 68, 68, 0.6);
        animation: pulse-red 1.5s infinite;
    }
    
    .traffic-light-dot.inactive {
        background: var(--text-dim);
        opacity: 0.3;
        box-shadow: none;
    }
    
    @keyframes pulse-red {
        0%, 100% { 
            box-shadow: 0 0 12px rgba(239, 68, 68, 0.6);
        }
        50% { 
            box-shadow: 0 0 24px rgba(239, 68, 68, 0.9);
        }
    }
    
    .traffic-light-text {
        font-size: 14px;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    /* Quick Action Buttons */
    .quick-action-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
    }
    
    .quick-action-btn {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 16px;
        border-radius: var(--radius-md);
        border: 1px solid var(--border-light);
        background: var(--bg-card);
        cursor: pointer;
        transition: all var(--transition-fast);
    }
    
    .quick-action-btn:hover {
        border-color: var(--accent-blue);
        transform: translateY(-2px);
        box-shadow: var(--shadow-glow-blue);
    }
    
    .quick-action-btn.buy {
        border-color: rgba(0, 220, 130, 0.3);
    }
    
    .quick-action-btn.buy:hover {
        border-color: var(--pnl-profit);
        box-shadow: var(--shadow-glow-green);
    }
    
    .quick-action-btn.sell {
        border-color: rgba(255, 82, 82, 0.3);
    }
    
    .quick-action-btn.sell:hover {
        border-color: var(--pnl-loss);
        box-shadow: var(--shadow-glow-red);
    }
    
    /* Mini Sparkline Container */
    .sparkline-container {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 4px 8px;
        background: rgba(15, 23, 42, 0.6);
        border-radius: var(--radius-sm);
    }
    
    /* Data Freshness Indicator */
    .data-freshness {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 11px;
        color: var(--text-dim);
    }
    
    .data-freshness-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
    }
    
    .data-freshness-dot.fresh {
        background: var(--success);
    }
    
    .data-freshness-dot.stale {
        background: var(--warning);
    }
    
    .data-freshness-dot.old {
        background: var(--error);
    }
    
    /* Compact Stats Row */
    .compact-stats-row {
        display: flex;
        gap: 16px;
        padding: 12px 16px;
        background: var(--bg-card);
        border-radius: var(--radius-md);
        border: 1px solid var(--border-light);
        flex-wrap: wrap;
    }
    
    .compact-stat {
        display: flex;
        flex-direction: column;
        gap: 2px;
    }
    
    .compact-stat-label {
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-dim);
    }
    
    .compact-stat-value {
        font-size: 14px;
        font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
        color: var(--text-primary);
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

    # P&L specific colors
    "profit": "#00DC82",
    "loss": "#FF5252",
    "neutral": "#94a3b8",
    
    # Position colors
    "long": "#00DC82",
    "short": "#FF5252",
    "flat": "#64748b",

    # Candlestick colors (solid fills)
    "candle_up": "#00DC82",
    "candle_down": "#FF5252",

    # Text colors
    "text": "#e2e8f0",
    "text_muted": "#94a3b8",
    "text_bright": "#f8fafc",
    "text_dim": "#64748b",

    # Background colors
    "background": "#0f172a",
    "surface": "#1e293b",
    "surface_light": "#334155",
    "border": "#334155",
    "card": "rgba(30, 41, 59, 0.6)",

    # Chart colors
    "chart_bg": "rgba(15, 23, 42, 0.8)",
    "chart_grid": "rgba(51, 65, 85, 0.4)",

    # Moving average colors
    "ma_fast": "#FFB020",
    "ma_slow": "#A855F7",
    "ma_long": "#06B6D4",
    
    # Signal colors
    "signal_long": "#00DC82",
    "signal_short": "#FF5252",
    "signal_neutral": "#94a3b8",
    
    # Risk traffic light
    "risk_green": "#00DC82",
    "risk_yellow": "#f59e0b",
    "risk_red": "#ef4444",
    
    # Accent
    "gold": "#fbbf24",
    "cyan": "#06b6d4",
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
