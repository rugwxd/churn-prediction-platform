"""Streamlit monitoring dashboard for the Churn Prediction Platform.

Displays model performance, feature drift alerts, prediction
distribution shifts, and business metrics.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
)


def load_training_results() -> dict | None:
    """Load training results if available."""
    path = PROJECT_ROOT / "data" / "processed" / "training_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_dataset() -> pd.DataFrame | None:
    """Load the generated dataset."""
    path = PROJECT_ROOT / "data" / "raw" / "churn_dataset.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def render_header():
    st.title("📊 Churn Prediction Platform")
    st.caption("Real-time model monitoring, drift detection, and business metrics")


def render_model_comparison(results: dict):
    """Render model comparison section."""
    st.header("Model Performance Comparison")

    models = results.get("models", {})
    if not models:
        st.warning("No model results available. Run training first.")
        return

    # Metrics table
    rows = []
    for name, metrics in models.items():
        rows.append(
            {
                "Model": name,
                "AUC-ROC": metrics["auc_roc"],
                "AUC-PR": metrics["auc_pr"],
                "F1 Score": metrics["f1"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "Train Time (s)": metrics["train_time_seconds"],
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.highlight_max(axis=0, subset=["AUC-ROC", "AUC-PR", "F1 Score"]),
        use_container_width=True,
    )

    # Bar chart comparison
    metric_cols = ["AUC-ROC", "AUC-PR", "F1 Score", "Precision", "Recall"]
    df_melted = df.melt(
        id_vars="Model", value_vars=metric_cols, var_name="Metric", value_name="Score"
    )

    fig = px.bar(
        df_melted,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        title="Model Metrics Comparison",
        color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"],
    )
    fig.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)


def render_feature_importance(results: dict):
    """Render SHAP feature importance."""
    st.header("Feature Importance (SHAP)")

    importance = results.get("feature_importance", {})
    if not importance:
        st.info("Feature importance not available.")
        return

    df = pd.DataFrame(list(importance.items()), columns=["Feature", "Importance"]).sort_values(
        "Importance", ascending=True
    )

    fig = px.bar(
        df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Features by Mean |SHAP Value|",
        color="Importance",
        color_continuous_scale="blues",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_churn_analysis(df: pd.DataFrame):
    """Render churn distribution analysis."""
    st.header("Churn Analysis")

    col1, col2, col3, col4 = st.columns(4)

    churn_rate = df["churned"].mean()
    total_users = len(df)
    churned_count = df["churned"].sum()

    if "mrr" in df.columns:
        at_risk_mrr = df[df["churned"] == 1]["mrr"].sum()
    else:
        at_risk_mrr = 0

    col1.metric("Total Users", f"{total_users:,}")
    col2.metric("Churn Rate", f"{churn_rate:.1%}")
    col3.metric("Churned Users", f"{churned_count:,}")
    col4.metric("Revenue at Risk", f"${at_risk_mrr:,.0f}/mo")

    # Churn by plan tier
    col_a, col_b = st.columns(2)

    with col_a:
        if "plan_tier" in df.columns:
            churn_by_plan = df.groupby("plan_tier")["churned"].mean().reset_index()
            churn_by_plan.columns = ["Plan Tier", "Churn Rate"]
            fig = px.bar(
                churn_by_plan,
                x="Plan Tier",
                y="Churn Rate",
                title="Churn Rate by Plan Tier",
                color="Churn Rate",
                color_continuous_scale="reds",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        if "signup_channel" in df.columns:
            churn_by_channel = df.groupby("signup_channel")["churned"].mean().reset_index()
            churn_by_channel.columns = ["Channel", "Churn Rate"]
            fig = px.bar(
                churn_by_channel,
                x="Channel",
                y="Churn Rate",
                title="Churn Rate by Signup Channel",
                color="Churn Rate",
                color_continuous_scale="reds",
            )
            st.plotly_chart(fig, use_container_width=True)


def render_feature_distributions(df: pd.DataFrame):
    """Render feature distribution plots split by churn status."""
    st.header("Feature Distributions")

    numerical_features = [
        "login_frequency",
        "avg_session_duration_min",
        "feature_usage_score",
        "days_since_last_login",
        "support_tickets_total",
        "monthly_active_days",
    ]

    available = [f for f in numerical_features if f in df.columns]

    cols_per_row = 3
    for i in range(0, len(available), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(available):
                break
            feat = available[idx]
            with col:
                fig = px.histogram(
                    df,
                    x=feat,
                    color="churned",
                    nbins=40,
                    barmode="overlay",
                    opacity=0.7,
                    title=feat.replace("_", " ").title(),
                    color_discrete_map={0: "#2ca02c", 1: "#d62728"},
                    labels={"churned": "Churned"},
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)


def render_correlation_heatmap(df: pd.DataFrame):
    """Render feature correlation heatmap."""
    st.header("Feature Correlations")

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numerical_cols) < 2:
        st.info("Not enough numerical features for correlation analysis.")
        return

    corr = df[numerical_cols].corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu_r",
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont={"size": 9},
        )
    )
    fig.update_layout(
        title="Pearson Correlation Matrix",
        height=600,
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Highlight strongest correlations with churn
    if "churned" in corr.columns:
        churn_corr = corr["churned"].drop("churned").abs().sort_values(ascending=False)
        st.subheader("Top Churn Correlates")
        st.dataframe(
            churn_corr.reset_index()
            .rename(columns={"index": "Feature", "churned": "|Correlation|"})
            .head(10),
            use_container_width=True,
        )


def render_drift_monitoring():
    """Render drift detection results."""
    st.header("Drift Monitoring")

    reports_dir = PROJECT_ROOT / "data" / "drift_reports"
    if not reports_dir.exists():
        st.info("No drift reports available. Run drift detection to generate reports.")
        return

    reports = sorted(reports_dir.glob("drift_results_*.json"), reverse=True)
    if not reports:
        st.info("No drift reports found.")
        return

    # Load latest report
    with open(reports[0]) as f:
        latest = json.load(f)

    drift_detected = latest.get("drift_detected", False)

    if drift_detected:
        st.error(f"⚠️ Drift detected in {len(latest.get('drifted_features', []))} features!")
        st.write("Drifted features:", latest.get("drifted_features", []))
    else:
        st.success("✅ No significant drift detected")

    # Feature-level drift details
    features = latest.get("features", {})
    if features:
        rows = []
        for feat, info in features.items():
            rows.append(
                {
                    "Feature": feat,
                    "Type": info.get("type", ""),
                    "Statistic": round(info.get("statistic", 0), 4),
                    "P-Value": round(info.get("p_value", 0), 4),
                    "Drift": "Yes" if info.get("drift_detected") else "No",
                }
            )

        drift_df = pd.DataFrame(rows)
        st.dataframe(drift_df, use_container_width=True)


def render_prediction_simulator(df: pd.DataFrame):
    """Simple prediction simulator using dataset statistics."""
    st.header("Prediction Simulator")
    st.caption("Adjust user features to see predicted churn risk")

    col1, col2, col3 = st.columns(3)
    with col1:
        login_freq = st.slider("Login Frequency", 0.0, 30.0, 10.0)
        session_dur = st.slider("Avg Session (min)", 0.0, 60.0, 15.0)
    with col2:
        feature_usage = st.slider("Feature Usage Score", 0.0, 100.0, 50.0)
        support_tickets = st.slider("Support Tickets", 0, 20, 2)
    with col3:
        days_inactive = st.slider("Days Since Login", 0.0, 60.0, 5.0)
        active_days = st.slider("Monthly Active Days", 0, 30, 15)

    # Simple heuristic risk score (placeholder for actual model)
    risk_score = (
        0.3 * (1 - login_freq / 30)
        + 0.2 * (1 - session_dur / 60)
        + 0.15 * (1 - feature_usage / 100)
        + 0.15 * (support_tickets / 20)
        + 0.1 * (days_inactive / 60)
        + 0.1 * (1 - active_days / 30)
    )
    risk_score = max(0, min(1, risk_score))

    # Display gauge
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_score * 100,
            title={"text": "Churn Risk Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 30], "color": "#d4edda"},
                    {"range": [30, 60], "color": "#fff3cd"},
                    {"range": [60, 80], "color": "#f8d7da"},
                    {"range": [80, 100], "color": "#721c24"},
                ],
            },
        )
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


def main():
    render_header()

    results = load_training_results()
    df = load_dataset()

    if results is None and df is None:
        st.warning(
            "No data available. Run `python scripts/train.py` first to generate "
            "data and train models."
        )
        return

    # Tabs for different views
    tabs = st.tabs(
        [
            "Model Performance",
            "Churn Analysis",
            "Feature Distributions",
            "Correlations",
            "Drift Monitoring",
            "Prediction Simulator",
        ]
    )

    with tabs[0]:
        if results:
            render_model_comparison(results)
            render_feature_importance(results)
        else:
            st.info("Run training to see model results.")

    with tabs[1]:
        if df is not None:
            render_churn_analysis(df)

    with tabs[2]:
        if df is not None:
            render_feature_distributions(df)

    with tabs[3]:
        if df is not None:
            render_correlation_heatmap(df)

    with tabs[4]:
        render_drift_monitoring()

    with tabs[5]:
        if df is not None:
            render_prediction_simulator(df)


if __name__ == "__main__":
    main()
