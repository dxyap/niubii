"""
ML Signals Page
===============
Machine learning-powered trading signals and model analytics.
"""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components.charts import BASE_LAYOUT, CHART_COLORS
from app.page_utils import get_chart_config, init_page

# Initialize page
ctx = init_page(
    title="🤖 ML-Powered Signals",
    page_title="ML Signals | Oil Trading",
    icon="🤖",
    lookback_days=365,
)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Current Signals",
    "🧠 Model Performance",
    "📈 Feature Analysis",
    "⚙️ Model Training"
])

# Try to load ML modules
try:
    from core.ml import FeatureConfig, FeatureEngineer
    from core.ml.models import EnsembleModel, GradientBoostModel, ModelConfig
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
            ticker = st.selectbox(
                "Select Instrument",
                ["CO1 Comdty (Brent)", "CL1 Comdty (WTI)"],
                key="ml_ticker"
            )
            ticker_code = ticker.split(" ")[0] + " Comdty"

            hist_data = ctx.data_loader.get_historical(
                ticker_code,
                start_date=datetime.now() - timedelta(days=365)
            )

            if hist_data is not None and not hist_data.empty:
                feature_engineer = FeatureEngineer(FeatureConfig(target_horizon=5))
                features = feature_engineer.create_features(hist_data, include_target=False)

                if not features.empty:
                    st.markdown("**Latest Feature Snapshot**")

                    latest = features.iloc[-1]

                    metric_cols = st.columns(4)

                    with metric_cols[0]:
                        rsi = latest.get('rsi_14', 50)
                        st.metric("RSI (14)", f"{rsi:.1f}")

                    with metric_cols[1]:
                        vol = latest.get('volatility_21d', 0) * 100
                        st.metric("21D Volatility", f"{vol:.1f}%")

                    with metric_cols[2]:
                        momentum = latest.get('roc_10d', 0) * 100
                        st.metric("10D Momentum", f"{momentum:+.2f}%")

                    with metric_cols[3]:
                        ma_ratio = latest.get('price_ma_20_ratio', 1)
                        st.metric("Price/MA20", f"{ma_ratio:.3f}")

        with col2:
            st.markdown("**ML Signal Status**")

            model_dir = Path("models")
            model_files = list(model_dir.glob("*.pkl")) if model_dir.exists() else []

            if model_files:
                st.success(f"✅ {len(model_files)} trained model(s) available")
            else:
                st.info("📊 No trained models found")
                st.caption("Train a model in the 'Model Training' tab")

with tab2:
    st.subheader("Model Performance Analytics")

    if not ML_AVAILABLE:
        st.warning("ML modules not available")
    else:
        model_dir = Path("models")
        model_files = list(model_dir.glob("*.pkl")) if model_dir.exists() else []

        if not model_files:
            st.info("No trained models found. Train a model first.")
        else:
            selected_model = st.selectbox("Select Model", [f.name for f in model_files], key="perf_model")

            if selected_model:
                try:
                    model_path = model_dir / selected_model

                    try:
                        model = EnsembleModel.load(model_path)
                        model_type = "Ensemble"
                    except Exception:
                        model = GradientBoostModel.load(model_path)
                        model_type = model.config.model_type.capitalize()

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Model Information**")
                        st.text(f"Type: {model_type}")
                        st.text(f"Features: {len(model.feature_names)}")

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
                                yaxis={"autorange": 'reversed'},
                                xaxis_title='Importance',
                                margin={"l": 150, "r": 20, "t": 20, "b": 40},
                            )

                            st.plotly_chart(fig, use_container_width=True, config=get_chart_config())
                except Exception as e:
                    st.error(f"Error loading model: {e}")

with tab3:
    st.subheader("Feature Analysis")

    if not ML_AVAILABLE:
        st.warning("ML modules not available")
    else:
        ticker = "CO1 Comdty"
        hist_data = ctx.data_loader.get_historical(
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
                            yaxis={"autorange": 'reversed'},
                            xaxis_title='Absolute Correlation',
                            margin={"l": 180, "r": 20, "t": 20, "b": 40},
                        )

                        st.plotly_chart(fig, use_container_width=True, config=get_chart_config())

                with col2:
                    st.markdown("**Feature Statistics**")
                    categories = feature_engineer.get_feature_importance_template()
                    category_counts = pd.Series(categories).value_counts()

                    st.text(f"Total Features: {len(feature_engineer.feature_names)}")
                    st.divider()

                    for cat, count in category_counts.items():
                        st.text(f"{cat}: {count}")

with tab4:
    st.subheader("Model Training")

    if not ML_AVAILABLE:
        st.warning("ML modules not available. Install dependencies:")
        st.code("pip install scikit-learn xgboost lightgbm")
    else:
        st.markdown("Train a new ML model for price direction prediction.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Training Configuration**")

            ticker = st.selectbox("Training Data", ["CO1 Comdty (Brent)", "CL1 Comdty (WTI)"], key="train_ticker")
            lookback = st.slider("Lookback Period (days)", 180, 730, 365)
            target_horizon = st.selectbox("Prediction Horizon", [1, 3, 5, 10], index=2, format_func=lambda x: f"{x} day{'s' if x > 1 else ''}")
            use_ensemble = st.checkbox("Use Ensemble Model", value=True)

        with col2:
            st.markdown("**Model Parameters**")

            n_estimators = st.slider("Number of Trees", 50, 500, 100)
            max_depth = st.slider("Max Depth", 3, 10, 6)
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)

        if st.button("🚀 Train Model", type="primary"):
            ticker_code = ticker.split(" ")[0] + " Comdty"

            with st.spinner("Training model..."):
                try:
                    hist_data = ctx.data_loader.get_historical(
                        ticker_code,
                        start_date=datetime.now() - timedelta(days=lookback)
                    )

                    if hist_data is None or hist_data.empty:
                        st.error("No historical data available for training")
                    else:
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
                        )

                        trainer = ModelTrainer(training_config)
                        results = trainer.train(hist_data)

                        model_path = trainer.save_model()

                        st.success(f"✅ Model trained and saved to {model_path}")

                        metrics = results.get('test_metrics', {})

                        metric_cols = st.columns(4)

                        with metric_cols[0]:
                            st.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}%")
                        with metric_cols[1]:
                            st.metric("Precision", f"{metrics.get('precision', 0)*100:.1f}%")
                        with metric_cols[2]:
                            st.metric("Recall", f"{metrics.get('recall', 0)*100:.1f}%")
                        with metric_cols[3]:
                            st.metric("F1 Score", f"{metrics.get('f1', 0)*100:.1f}%")

                except Exception as e:
                    st.error(f"Training failed: {e}")
