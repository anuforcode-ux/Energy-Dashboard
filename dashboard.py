# ============================================================
#   dashboard.py
#   Streamlit interactive dashboard (per Slide 5 module 7):
#   trend graphs, predicted vs actual, anomaly flags, cost projection.
#
#   Run with:  streamlit run dashboard.py
#   Requires:  pip install streamlit --break-system-packages
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

from data_pipeline import load_pipeline
from anomaly_detector import combined_anomalies
from alert_generator import generate_all_alerts

st.set_page_config(page_title="Household Energy Dashboard", layout="wide")


@st.cache_data
def get_data():
    return load_pipeline()


@st.cache_resource
def get_trained_model(_X_train, _y_train):
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(_X_train, _y_train)
    return model


@st.cache_data
def get_predictions_and_anomalies(_model, X_test, y_test, df_model, feature_cols,
                                   zscore_threshold, contamination):
    pred = pd.Series(_model.predict(X_test), index=y_test.index)
    anomalies = combined_anomalies(
        y_test, pred, df_model.loc[X_test.index], feature_cols,
        zscore_threshold=zscore_threshold, contamination=contamination
    )
    return pred, anomalies


# ---------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------
st.sidebar.title("⚡ Dashboard Controls")

data = get_data()
df_model = data['df_model']
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']
feature_cols = data['feature_cols']

min_date, max_date = y_test.index.min().date(), y_test.index.max().date()
date_range = st.sidebar.date_input(
    "Date range (test period)", value=(min_date, max_date),
    min_value=min_date, max_value=max_date
)

zscore_threshold = st.sidebar.slider("Z-score threshold", 1.0, 4.0, 2.0, 0.1)
contamination = st.sidebar.slider("Isolation Forest sensitivity", 0.005, 0.10, 0.02, 0.005)
cost_per_kwh = st.sidebar.number_input("Electricity cost (₹/kWh)", value=8.0, min_value=0.0, step=0.5)

# ---------------------------------------------------------------
# Compute (cached)
# ---------------------------------------------------------------
model = get_trained_model(X_train, y_train)
pred, anomalies = get_predictions_and_anomalies(
    model, X_test, y_test, df_model, feature_cols, zscore_threshold, contamination
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    mask = (anomalies.index.date >= start) & (anomalies.index.date <= end)
    view = anomalies.loc[mask]
else:
    view = anomalies

# ---------------------------------------------------------------
# Header + summary metrics
# ---------------------------------------------------------------
st.title("🏠 Household Energy Consumption Dashboard")
st.caption("CS5403 Machine Learning — PBL Project Review II")

col1, col2, col3, col4 = st.columns(4)
total_actual_kwh = view['actual'].sum()
n_anomalies = int(view['is_anomaly'].sum())
estimated_cost = total_actual_kwh * cost_per_kwh
rmse = np.sqrt(((view['actual'] - view['predicted']) ** 2).mean())

col1.metric("Total Usage (period)", f"{total_actual_kwh:,.1f} kWh")
col2.metric("Anomalies Flagged", f"{n_anomalies}", f"{n_anomalies/len(view)*100:.1f}% of hours" if len(view) else "0%")
col3.metric("Estimated Cost", f"₹{estimated_cost:,.0f}")
col4.metric("Model RMSE", f"{rmse:.3f} kWh")

# ---------------------------------------------------------------
# Trend graph: actual vs predicted, anomalies overlaid
# ---------------------------------------------------------------
st.subheader("Actual vs Predicted Usage")

fig = go.Figure()
fig.add_trace(go.Scatter(x=view.index, y=view['actual'], name='Actual',
                          line=dict(color='yellow', width=1.5)))
fig.add_trace(go.Scatter(x=view.index, y=view['predicted'], name='Predicted (Random Forest)',
                          line=dict(color='steelblue', width=1.2, dash='dash')))
flagged = view[view['is_anomaly']]
fig.add_trace(go.Scatter(x=flagged.index, y=flagged['actual'], mode='markers',
                          name='⚠️ Anomaly', marker=dict(color='red', size=8)))
fig.update_layout(xaxis_title='Time', yaxis_title='Power (kWh)', height=450,
                   legend=dict(orientation='h', yanchor='bottom', y=1.02))
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------
# Waste alerts
# ---------------------------------------------------------------
st.subheader("⚠️ Waste Alerts")
alerts = generate_all_alerts(view, top_n=15)
if alerts:
    for ts, msg in alerts:
        st.warning(f"**{ts.strftime('%Y-%m-%d %H:%M')}** — {msg}")
else:
    st.success("No anomalies in the selected range.")

# ---------------------------------------------------------------
# Cost projection
# ---------------------------------------------------------------
st.subheader("💰 Cost Projection")
daily_cost = view['actual'].resample('D').sum() * cost_per_kwh if len(view) else pd.Series(dtype=float)
if len(daily_cost):
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=daily_cost.index, y=daily_cost.values, marker_color='mediumseagreen'))
    fig2.update_layout(xaxis_title='Date', yaxis_title='Estimated Cost (₹)', height=350)
    st.plotly_chart(fig2, use_container_width=True)
    avg_daily = daily_cost.mean()
    st.info(f"At current usage patterns, projected monthly cost ≈ **₹{avg_daily * 30:,.0f}**.")

# ---------------------------------------------------------------
# Raw anomaly table
# ---------------------------------------------------------------
with st.expander("View raw anomaly data"):
    st.dataframe(view[['actual', 'predicted', 'residual', 'z_score', 'is_anomaly']])
