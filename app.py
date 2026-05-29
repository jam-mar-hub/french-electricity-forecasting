import streamlit as st
import pandas as pd
import psycopg2
import plotly.graph_objects as go
from datetime import date

st.set_page_config(page_title="Electricity Forecasting", layout="wide")

def get_connection():
    return psycopg2.connect(
        host=st.secrets["SUPABASE_HOST"],
        database=st.secrets["SUPABASE_DB"],
        user=st.secrets["SUPABASE_USER"],
        password=st.secrets["SUPABASE_PASSWORD"],
        port=st.secrets["SUPABASE_PORT"]
    )

@st.cache_data(ttl=3600)
def load_historical():
    conn = get_connection()
    df = pd.read_sql("SELECT timestamp, value FROM historical_data ORDER BY timestamp ASC", conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df

@st.cache_data(ttl=3600)
def load_predictions():
    conn = get_connection()
    # Nettoyage : q10 et q90 ont été retirés de la requête SQL
    df = pd.read_sql("SELECT timestamp, predicted_value, model_name, prediction_date FROM predictions ORDER BY timestamp ASC", conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['prediction_date'] = pd.to_datetime(df['prediction_date'], utc=True)
    return df

@st.cache_data(ttl=3600)
def load_rte_predictions():
    conn = get_connection()
    df = pd.read_sql("SELECT timestamp, predicted_value FROM predictions_rte WHERE horizon = 'H+48' ORDER BY timestamp ASC", conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df


@st.cache_data(ttl=3600)
def load_lora_predictions():
    conn = get_connection()
    df = pd.read_sql("SELECT timestamp, predicted_value, q10, q90 FROM predictions_lora ORDER BY timestamp ASC", conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df

df_historical = load_historical()
df_predictions = load_predictions()
df_rte = load_rte_predictions()
df_lora = load_lora_predictions()

def calculate_mape(df_hist, df_pred, value_col='predicted_value'):
    merged = df_pred.set_index('timestamp').join(
        df_hist.set_index('timestamp')['value'], how='inner'
    )
    if len(merged) == 0:
        return None
    mape = (abs(merged[value_col] - merged['value']) / merged['value']).mean() * 100
    return round(mape, 2)

mape = calculate_mape(df_historical, df_predictions)
mape_rte = calculate_mape(df_historical, df_rte)
mape_lora = calculate_mape(df_historical, df_lora)

# --- HEADER ---
st.title("⚡ French Electricity Consumption Forecasting")
st.markdown("---")

# --- MÉTRIQUES ---

col1, col2 = st.columns(2)
with col1:
    st.metric("Historical records", f"{len(df_historical):,} hours")
with col2:
    st.metric("Last update", df_historical['timestamp'].max().strftime("%d %b %Y, %H:%M UTC"))

st.markdown("**Overall MAPE — all available history**")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Chronos-2 Zero Shot", f"{mape}%" if mape else "N/A")
with col2:
    st.metric("RTE Benchmark", f"{mape_rte}%" if mape_rte else "N/A")
with col3:
    st.metric("Chronos-2 LoRA", f"{mape_lora}%" if mape_lora else "N/A")

st.markdown("---")

# --- HISTORIQUE ---
st.subheader("Historical Consumption")
period = st.selectbox("Period", ["7 days", "30 days", "90 days"], index=1)
days = int(period.split()[0])
df_period = df_historical.tail(24*days).set_index('timestamp')['value']
st.line_chart(df_period)

st.markdown("---")

# --- PRÉDICTIONS FUTURES ---
st.subheader("48h Forecast Comparison")

now = df_historical['timestamp'].max()

df_future = df_predictions[df_predictions['timestamp'] > now]
df_context = df_historical[df_historical['timestamp'] >= now - pd.Timedelta(hours=168)]
df_rte_future = df_rte[df_rte['timestamp'] > now]
df_lora_future = df_lora[df_lora['timestamp'] > now]

fig_future = go.Figure()

# Contexte historique
fig_future.add_trace(go.Scatter(
    x=df_context['timestamp'],
    y=df_context['value'],
    name='Historical context',
    line=dict(color='#1f77b4')
))

# Zero Shot
fig_future.add_trace(go.Scatter(
    x=df_future['timestamp'],
    y=df_future['predicted_value'],
    name='Chronos-2 Zero Shot Forecast',
    line=dict(color='#ff7f0e', width=3)
))

# Modèle RTE
fig_future.add_trace(go.Scatter(
    x=df_rte_future['timestamp'],
    y=df_rte_future['predicted_value'],
    name='RTE Benchmark (H+48)',
    line=dict(color='#8B5CF6', width=2)
))

#LoRA
fig_future.add_trace(go.Scatter(
    x=df_lora_future['timestamp'],
    y=df_lora_future['predicted_value'],
    name='Chronos-2 LoRA Forecast',
    line=dict(color='#10B981', width=2)
))


fig_future.update_layout(
    xaxis_title="Date",
    yaxis_title="Consumption (MW)",
    hovermode='x unified',
    height=450
)

st.plotly_chart(fig_future, use_container_width=True)

st.markdown("---")

# --- BACKTEST PAR PÉRIODE ---
st.subheader("Backtest — Model evaluation by period")

period_options = {
    "Jan 2025 – Jun 2025": ("2025-01-01", "2025-06-30"),
    "Jul 2025 – Dec 2025": ("2025-07-01", "2025-12-31"),
    "Jan 2026 – Today": ("2026-01-01", date.today().strftime("%Y-%m-%d")),
}

selected_period = st.selectbox("Select a period", list(period_options.keys()))
start_str, end_str = period_options[selected_period]
start = pd.Timestamp(start_str, tz='UTC')
end = pd.Timestamp(end_str, tz='UTC')

df_batch = df_predictions[(df_predictions['timestamp'] >= start) & (df_predictions['timestamp'] <= end)]
df_rte_batch = df_rte[(df_rte['timestamp'] >= start) & (df_rte['timestamp'] <= end)]
df_truth = df_historical[(df_historical['timestamp'] >= start) & (df_historical['timestamp'] <= end)]
df_lora_batch = df_lora[(df_lora['timestamp'] >= start) & (df_lora['timestamp'] <= end)]

batch_mape_lora = calculate_mape(df_truth, df_lora_batch)
batch_mape = calculate_mape(df_truth, df_batch)
batch_mape_rte = calculate_mape(df_truth, df_rte_batch)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Period", selected_period)
with col2:
    st.metric("Chronos-2 Zero Shot", f"{batch_mape}%" if batch_mape else "N/A")
with col3:
    st.metric("RTE Benchmark", f"{batch_mape_rte}%" if batch_mape_rte else "N/A")
with col4:
    st.metric("Chronos-2 LoRA", f"{batch_mape_lora}%" if batch_mape_lora else "N/A")

df_context = df_historical[(df_historical['timestamp'] >= start - pd.Timedelta(hours=168)) & (df_historical['timestamp'] < start)]

fig = go.Figure()

fig.add_trace(go.Scatter(x=df_context['timestamp'], y=df_context['value'], name='Historical context', line=dict(color='#1f77b4')))
fig.add_trace(go.Scatter(x=df_truth['timestamp'], y=df_truth['value'], name='Ground Truth', line=dict(color='green', width=2)))
fig.add_trace(go.Scatter(x=df_batch['timestamp'], y=df_batch['predicted_value'], name='Chronos-2 Zero Shot', line=dict(color='#ff7f0e')))
fig.add_trace(go.Scatter(x=df_rte_batch['timestamp'], y=df_rte_batch['predicted_value'], name='RTE Forecast', line=dict(color='#8B5CF6')))
fig.add_trace(go.Scatter(x=df_lora_batch['timestamp'], y=df_lora_batch['predicted_value'], name='Chronos-2 LoRA', line=dict(color="#10B981")))


fig.update_layout(xaxis_title="Date", yaxis_title="Consumption (MW)", hovermode='x unified', height=500)
st.plotly_chart(fig, use_container_width=True)

if st.button("🔄 Refresh data"):
    st.cache_data.clear()
    st.rerun()