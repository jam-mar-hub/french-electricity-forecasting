import streamlit as st
import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os
import plotly.graph_objects as go

load_dotenv()

st.set_page_config(page_title="Electricity Forecasting", layout="wide")

@st.cache_data
def load_historical():
    conn = psycopg2.connect(
        host=os.getenv("SUPABASE_HOST"),
        database=os.getenv("SUPABASE_DB"),
        user=os.getenv("SUPABASE_USER"),
        password=os.getenv("SUPABASE_PASSWORD"),
        port=os.getenv("SUPABASE_PORT")
    )
    df = pd.read_sql("SELECT timestamp, value FROM historical_data ORDER BY timestamp ASC", conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df

@st.cache_data
def load_predictions():
    conn = psycopg2.connect(
        host=os.getenv("SUPABASE_HOST"),
        database=os.getenv("SUPABASE_DB"),
        user=os.getenv("SUPABASE_USER"),
        password=os.getenv("SUPABASE_PASSWORD"),
        port=os.getenv("SUPABASE_PORT")
    )
    df = pd.read_sql("SELECT timestamp, predicted_value, q10, q90, model_name, prediction_date FROM predictions ORDER BY timestamp ASC", conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['prediction_date'] = pd.to_datetime(df['prediction_date'], utc=True)
    return df

df_historical = load_historical()
df_predictions = load_predictions()

def calculate_mape(df_hist, df_pred):
    merged = df_pred.set_index('timestamp').join(
        df_hist.set_index('timestamp')['value'], how='inner'
    )
    if len(merged) == 0:
        return None
    mape = (abs(merged['predicted_value'] - merged['value']) / merged['value']).mean() * 100
    return round(mape, 2)

mape = calculate_mape(df_historical, df_predictions)

# --- HEADER ---
st.title("⚡ French Electricity Consumption Forecasting")
st.markdown("---")

# --- MÉTRIQUES ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Historical records", f"{len(df_historical):,} hours")
with col2:
    st.metric("Last update", df_historical['timestamp'].max().strftime("%Y-%m-%d %H:%M UTC"))
with col3:
    st.metric("Model MAPE", f"{mape}%" if mape else "N/A", help="Computed on past backtests vs ground truth")

st.markdown("---")

# --- HISTORIQUE ---
st.subheader("Historical Consumption")
period = st.selectbox("Period", ["7 days", "30 days", "90 days"], index=1)
days = int(period.split()[0])
df_period = df_historical.tail(24*days).set_index('timestamp')['value']
st.line_chart(df_period)

st.markdown("---")

# --- BACKTEST PAR PÉRIODE ---
st.subheader("Backtest — Model evaluation by period")

period_options = {
    "Jan 2025 – Jun 2025": ("2025-01-01", "2025-06-30"),
    "Jul 2025 – Dec 2025": ("2025-07-01", "2025-12-31"),
    "Jan 2026 – Today":    ("2026-01-01", "2026-04-19"),
}

selected_period = st.selectbox("Select a period", list(period_options.keys()))
start_str, end_str = period_options[selected_period]
start = pd.Timestamp(start_str, tz='UTC')
end = pd.Timestamp(end_str, tz='UTC')

df_batch = df_predictions[
    (df_predictions['timestamp'] >= start) &
    (df_predictions['timestamp'] <= end)
]
df_truth = df_historical[
    (df_historical['timestamp'] >= start) &
    (df_historical['timestamp'] <= end)
]

batch_mape = calculate_mape(df_truth, df_batch)

col1, col2 = st.columns(2)
with col1:
    st.metric("Period", selected_period)
with col2:
    st.metric("MAPE", f"{batch_mape}%" if batch_mape else "N/A")

# Contexte historique (168h avant le début de la période)
df_context = df_historical[
    (df_historical['timestamp'] >= start - pd.Timedelta(hours=168)) &
    (df_historical['timestamp'] < start)
]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_context['timestamp'],
    y=df_context['value'],
    name='Historical context',
    line=dict(color='#1f77b4')
))

fig.add_trace(go.Scatter(
    x=df_truth['timestamp'],
    y=df_truth['value'],
    name='Ground Truth',
    line=dict(color='green')
))

fig.add_trace(go.Scatter(
    x=df_batch['timestamp'],
    y=df_batch['predicted_value'],
    name='Forecast',
    line=dict(color='#ff7f0e')
))

fig.add_trace(go.Scatter(
    x=pd.concat([df_batch['timestamp'], df_batch['timestamp'][::-1]]),
    y=pd.concat([df_batch['q90'], df_batch['q10'][::-1]]),
    fill='toself',
    fillcolor='rgba(255,127,14,0.15)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Confidence interval (10%-90%)'
))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Consumption (MW)",
    hovermode='x unified',
    height=500
)

st.plotly_chart(fig, use_container_width=True)

if st.button("🔄 Refresh data"):
    st.cache_data.clear()
    st.rerun()