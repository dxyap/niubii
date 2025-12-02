"""
ML Signals Page
===============
Machine learning-powered trading signals and model analytics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from app import shared_state
from app.components.theme import apply_theme, COLORS, PLOTLY_LAYOUT, get_chart_config
from app.components.charts import CHART_COLORS, BASE_LAYOUT

st.set_page_config(page_title="ML Signals | Oil Trading", page_icon="🤖", layout="wide")
apply_theme(st)

# Initialize context
context = shared_state.get_dashboard_context(lookback_days=365)
data_loader = context.data_loader

st.title("🤖 ML-Powered Signals")

# Check data mode
connection_status = data_loader.get_connection_status()
data_mode = connection_status.get("data_mode", "disconnected")

if data_mode == "mock":
    st.caption("⚠️ Simulated data mode — Bloomberg not connected")
elif data_mode == "live":
    st.caption("🟢 Live market data from Bloomberg")

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Current Signals",
    "🧠 Model Performance", 
    "📈 Feature Analysis",
    "⚙️ Model Training"
])

# Try to load ML modules
try:
    from core.ml import FeatureEngineer, FeatureConfig, PredictionService, ModelMonitor
    from core.ml.models import GradientBoostModel, ModelConfig, EnsembleModel
    from core.ml.training import ModelTrainer, TrainingConfig
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    ML_ERROR = str(e)

with tab1:
    st.subheader("Current ML Signals")
    
    if not ML_AVAILABLE:
        st.warning(f"ML modules not fully loaded: {ML_ERROR}")
        st.info("Install ML dependencies: `pip install scikit-learn xgboost lightgbm`")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Get historical data for feature generation
            ticker = st.selectbox(
                "Select Instrument",
                ["CO1 Comdty (Brent)", "CL1 Comdty (WTI)"],
                key="ml_ticker"
            )
            ticker_code = ticker.split(" ")[0] + " Comdty"
            
            hist_data = data_loader.get_historical(
                ticker_code,
                start_date=datetime.now() - timedelta(days=365)
            )
            
            if hist_data is not None and not hist_data.empty:
                # Create features
                feature_engineer = FeatureEngineer(FeatureConfig(target_horizon=5))
                features = feature_engineer.create_features(hist_data, include_target=False)
                
                if not features.empty:
                    # Display latest features summary
                    st.markdown("**Latest Feature Snapshot**")
                    
                    latest = features.iloc[-1]
                    
                    # Key indicators in metrics
                    metric_cols = st.columns(4)
                    
                    with metric_cols[0]:
                        rsi = latest.get('rsi_14', 50)
                        st.metric(
                            "RSI (14)",
                            f"{rsi:.1f}",
                            delta="Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                        )
                    
                    with metric_cols[1]:
                        vol = latest.get('volatility_21d', 0) * 100
                        st.metric("21D Volatility", f"{vol:.1f}%")
                    
                    with metric_cols[2]:
                        momentum = latest.get('roc_10d', 0) * 100
                        st.metric(
                            "10D Momentum",
                            f"{momentum:+.2f}%",
                            delta="Bullish" if momentum > 0 else "Bearish"
                        )
                    
                    with metric_cols[3]:
                        ma_ratio = latest.get('price_ma_20_ratio', 1)
                        st.metric(
                            "Price/MA20",
                            f"{ma_ratio:.3f}",
                            delta="Above" if ma_ratio > 1 else "Below"
                        )
                    
                    st.divider()
                    
                    # Feature categories summary
                    st.markdown("**Feature Summary by Category**")
                    
                    categories = feature_engineer.get_feature_importance_template()
                    
                    cat_cols = st.columns(3)
                    
                    # Momentum features
                    with cat_cols[0]:
                        st.markdown("**📈 Momentum**")
                        momentum_features = {k: latest[k] for k in categories if categories.get(k) == 'Momentum' and k in latest.index}
                        for name, value in list(momentum_features.items())[:5]:
                            if 'rsi' in name.lower():
                                st.text(f"{name}: {value:.1f}")
                            else:
                                st.text(f"{name}: {value:.4f}")
                    
                    # Volatility features
                    with cat_cols[1]:
                        st.markdown("**📊 Volatility**")
                        vol_features = {k: latest[k] for k in categories if categories.get(k) == 'Volatility' and k in latest.index}
                        for name, value in list(vol_features.items())[:5]:
                            if 'volatility' in name.lower():
                                st.text(f"{name}: {value*100:.1f}%")
                            else:
                                st.text(f"{name}: {value:.4f}")
                    
                    # Volume features
                    with cat_cols[2]:
                        st.markdown("**📉 Volume**")
                        vol_features = {k: latest[k] for k in categories if categories.get(k) == 'Volume' and k in latest.index}
                        for name, value in list(vol_features.items())[:5]:
                            st.text(f"{name}: {value:.3f}")
                else:
                    st.warning("Insufficient data for feature generation")
            else:
                st.warning("No historical data available")
        
        with col2:
            st.markdown("**ML Signal Status**")
            
            # Check for saved model
            model_dir = Path("models")
            model_files = list(model_dir.glob("*.pkl")) if model_dir.exists() else []
            
            if model_files:
                st.success(f"✅ {len(model_files)} trained model(s) available")
                
                # Try to generate signal
                try:
                    latest_model = sorted(model_files)[-1]
                    prediction_service = PredictionService(latest_model)
                    
                    if prediction_service.is_ready and hist_data is not None:
                        signal = prediction_service.predict(hist_data)
                        
                        # Display signal
                        signal_type = signal.get('signal', 'UNKNOWN')
                        confidence = signal.get('confidence', 0)
                        
                        if signal_type == 'BULLISH':
                            st.success(f"🟢 **{signal_type}**")
                        elif signal_type == 'BEARISH':
                            st.error(f"🔴 **{signal_type}**")
                        else:
                            st.info(f"⚪ **{signal_type}**")
                        
                        st.metric("Confidence", f"{confidence*100:.1f}%")
                        
                        if 'probability_up' in signal:
                            st.metric("P(Up)", f"{signal['probability_up']*100:.1f}%")
                        
                        st.caption(f"Horizon: {signal.get('horizon', 5)} days")
                except Exception as e:
                    st.warning(f"Could not generate signal: {e}")
            else:
                st.info("📊 No trained models found")
                st.caption("Train a model in the 'Model Training' tab")
            
            st.divider()
            
            # Quick stats
            st.markdown("**Market Context**")
            if hist_data is not None and not hist_data.empty:
                current_price = hist_data['PX_LAST'].iloc[-1]
                change_1d = hist_data['PX_LAST'].pct_change().iloc[-1] * 100
                change_5d = hist_data['PX_LAST'].pct_change(5).iloc[-1] * 100
                
                st.metric("Current Price", f"${current_price:.2f}")
                st.metric("1D Change", f"{change_1d:+.2f}%")
                st.metric("5D Change", f"{change_5d:+.2f}%")

with tab2:
    st.subheader("Model Performance Analytics")
    
    if not ML_AVAILABLE:
        st.warning("ML modules not available")
    else:
        # Check for model files
        model_dir = Path("models")
        model_files = list(model_dir.glob("*.pkl")) if model_dir.exists() else []
        
        if not model_files:
            st.info("No trained models found. Train a model first.")
        else:
            selected_model = st.selectbox(
                "Select Model",
                [f.name for f in model_files],
                key="perf_model"
            )
            
            if selected_model:
                model_path = model_dir / selected_model
                
                # Try to load model and show info
                try:
                    # Try loading as ensemble first
                    try:
                        model = EnsembleModel.load(model_path)
                        model_type = "Ensemble"
                    except:
                        model = GradientBoostModel.load(model_path)
                        model_type = model.config.model_type.capitalize()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Model Information**")
                        st.text(f"Type: {model_type}")
                        st.text(f"Features: {len(model.feature_names)}")
                        
                        if hasattr(model, 'metadata') and model.metadata:
                            st.text(f"Trained: {model.metadata.get('trained_at', 'Unknown')[:10]}")
                            st.text(f"Train samples: {model.metadata.get('n_train_samples', 'N/A')}")
                    
                    with col2:
                        st.markdown("**Feature Importance**")
                        importance_df = model.get_feature_importance(10)
                        
                        if not importance_df.empty:
                            fig = go.Figure(go.Bar(
                                x=importance_df['importance'],
                                y=importance_df['feature'],
                                orientation='h',
                                marker_color=CHART_COLORS['primary'],
                            ))
                            
                            fig.update_layout(
                                **BASE_LAYOUT,
                                height=300,
                                yaxis=dict(autorange='reversed'),
                                xaxis_title='Importance',
                                margin=dict(l=150, r=20, t=20, b=40),
                            )
                            
                            st.plotly_chart(fig, use_container_width=True, config=get_chart_config())
                    
                    # Load training results if available
                    results_path = model_path.with_suffix('.json')
                    if results_path.exists():
                        import json
                        with open(results_path, 'r') as f:
                            results = json.load(f)
                        
                        st.divider()
                        st.markdown("**Training Metrics**")
                        
                        metric_cols = st.columns(4)
                        
                        test_metrics = results.get('test_metrics', {})
                        
                        with metric_cols[0]:
                            st.metric("Accuracy", f"{test_metrics.get('accuracy', 0)*100:.1f}%")
                        with metric_cols[1]:
                            st.metric("Precision", f"{test_metrics.get('precision', 0)*100:.1f}%")
                        with metric_cols[2]:
                            st.metric("Recall", f"{test_metrics.get('recall', 0)*100:.1f}%")
                        with metric_cols[3]:
                            st.metric("F1 Score", f"{test_metrics.get('f1', 0)*100:.1f}%")
                        
                        if 'roc_auc' in test_metrics:
                            st.metric("ROC AUC", f"{test_metrics['roc_auc']:.4f}")
                
                except Exception as e:
                    st.error(f"Error loading model: {e}")

with tab3:
    st.subheader("Feature Analysis")
    
    if not ML_AVAILABLE:
        st.warning("ML modules not available")
    else:
        # Get data and create features
        ticker = "CO1 Comdty"
        hist_data = data_loader.get_historical(
            ticker,
            start_date=datetime.now() - timedelta(days=365)
        )
        
        if hist_data is not None and not hist_data.empty:
            feature_engineer = FeatureEngineer(FeatureConfig())
            features = feature_engineer.create_features(hist_data, include_target=True)
            
            if not features.empty:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**Feature Correlation with Target**")
                    
                    # Calculate correlation with target
                    if 'target_direction' in features.columns:
                        correlations = features.drop(columns=['target_direction', 'target_return'], errors='ignore').corrwith(
                            features['target_direction']
                        ).abs().sort_values(ascending=False)
                        
                        top_corr = correlations.head(15)
                        
                        fig = go.Figure(go.Bar(
                            x=top_corr.values,
                            y=top_corr.index,
                            orientation='h',
                            marker_color=CHART_COLORS['secondary'],
                        ))
                        
                        fig.update_layout(
                            **BASE_LAYOUT,
                            height=400,
                            yaxis=dict(autorange='reversed'),
                            xaxis_title='Absolute Correlation',
                            margin=dict(l=180, r=20, t=20, b=40),
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, config=get_chart_config())
                
                with col2:
                    st.markdown("**Feature Statistics**")
                    
                    # Show feature categories
                    categories = feature_engineer.get_feature_importance_template()
                    category_counts = pd.Series(categories).value_counts()
                    
                    st.text(f"Total Features: {len(feature_engineer.feature_names)}")
                    st.divider()
                    
                    for cat, count in category_counts.items():
                        st.text(f"{cat}: {count}")
                
                st.divider()
                
                # Feature time series
                st.markdown("**Feature Time Series**")
                
                feature_to_plot = st.selectbox(
                    "Select Feature",
                    [f for f in feature_engineer.feature_names if 'target' not in f][:50],
                    key="feature_plot"
                )
                
                if feature_to_plot:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=features.index,
                        y=features[feature_to_plot],
                        mode='lines',
                        line=dict(color=CHART_COLORS['primary'], width=1.5),
                        name=feature_to_plot,
                    ))
                    
                    fig.update_layout(
                        **BASE_LAYOUT,
                        height=300,
                        yaxis_title=feature_to_plot,
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, config=get_chart_config())
        else:
            st.warning("No data available for feature analysis")

with tab4:
    st.subheader("Model Training")
    
    if not ML_AVAILABLE:
        st.warning("ML modules not available. Install dependencies:")
        st.code("pip install scikit-learn xgboost lightgbm")
    else:
        st.markdown("""
        Train a new ML model for price direction prediction.
        The model uses technical and derived features from historical price data.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Training Configuration**")
            
            ticker = st.selectbox(
                "Training Data",
                ["CO1 Comdty (Brent)", "CL1 Comdty (WTI)"],
                key="train_ticker"
            )
            
            lookback = st.slider("Lookback Period (days)", 180, 730, 365)
            
            target_horizon = st.selectbox(
                "Prediction Horizon",
                [1, 3, 5, 10, 21],
                index=2,
                format_func=lambda x: f"{x} day{'s' if x > 1 else ''}"
            )
            
            use_ensemble = st.checkbox("Use Ensemble Model", value=True)
            use_walk_forward = st.checkbox("Walk-Forward Validation", value=True)
        
        with col2:
            st.markdown("**Model Parameters**")
            
            n_estimators = st.slider("Number of Trees", 50, 500, 100)
            max_depth = st.slider("Max Depth", 3, 10, 6)
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)
        
        st.divider()
        
        if st.button("🚀 Train Model", type="primary"):
            ticker_code = ticker.split(" ")[0] + " Comdty"
            
            with st.spinner("Training model... This may take a few minutes."):
                try:
                    # Get historical data
                    hist_data = data_loader.get_historical(
                        ticker_code,
                        start_date=datetime.now() - timedelta(days=lookback)
                    )
                    
                    if hist_data is None or hist_data.empty:
                        st.error("No historical data available for training")
                    else:
                        # Configure training
                        feature_config = FeatureConfig(target_horizon=target_horizon)
                        model_config = ModelConfig(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                        )
                        training_config = TrainingConfig(
                            feature_config=feature_config,
                            model_config=model_config,
                            use_ensemble=use_ensemble,
                            use_walk_forward=use_walk_forward,
                        )
                        
                        # Train
                        trainer = ModelTrainer(training_config)
                        
                        if use_walk_forward:
                            results = trainer.walk_forward_train(hist_data)
                        else:
                            results = trainer.train(hist_data)
                        
                        # Save model
                        model_path = trainer.save_model()
                        
                        st.success(f"✅ Model trained and saved to {model_path}")
                        
                        # Display results
                        st.markdown("**Training Results**")
                        
                        if 'avg_metrics' in results:
                            metrics = results['avg_metrics']
                        elif 'test_metrics' in results:
                            metrics = results['test_metrics']
                        else:
                            metrics = {}
                        
                        metric_cols = st.columns(4)
                        
                        with metric_cols[0]:
                            st.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}%")
                        with metric_cols[1]:
                            st.metric("Precision", f"{metrics.get('precision', 0)*100:.1f}%")
                        with metric_cols[2]:
                            st.metric("Recall", f"{metrics.get('recall', 0)*100:.1f}%")
                        with metric_cols[3]:
                            st.metric("F1 Score", f"{metrics.get('f1', 0)*100:.1f}%")
                        
                        # Feature importance
                        if trainer.model:
                            st.markdown("**Top Features**")
                            importance_df = trainer.get_feature_importance(10)
                            st.dataframe(importance_df, hide_index=True)
                
                except Exception as e:
                    st.error(f"Training failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
