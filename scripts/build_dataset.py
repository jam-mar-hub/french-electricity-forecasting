import pandas as pd
from datetime import datetime, timedelta
from src.ingestion.rte_client import fetch_realised
from src.processing.features import compute_hourly_avg

start_date = datetime(2020, 1, 1)
end_date = datetime.now().replace(microsecond=0)

all_df = []

current = start_date
while current < end_date:
    next_date = min(current + timedelta(days=180), end_date)
    print(f"Récupération {current.date()} -> {next_date.date()}")
    df_raw = fetch_realised(current, next_date)
    all_df.append(df_raw)
    current = next_date

df_concat = pd.concat(all_df, ignore_index=True)
df_final = compute_hourly_avg(df_concat)

# Remplissage des trous
full_range = pd.date_range(start=df_final['start_date'].min(),
                           end=df_final['start_date'].max(),
                           freq='h', tz='UTC')
df_full = pd.DataFrame({'start_date': full_range})
df_final = pd.merge(df_full, df_final, on='start_date', how='left')
df_final['avg_value_hourly'] = df_final['avg_value_hourly'].fillna(df_final['avg_value_hourly'].shift(168))
df_final['avg_value_hourly'] = df_final['avg_value_hourly'].interpolate(method='linear')

df_final.to_csv("data/consumption_data_cleaned.csv", index=False)
print(f"Terminé — {len(df_final)} lignes sauvegardées")