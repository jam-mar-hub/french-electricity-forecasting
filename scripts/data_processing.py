import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from dotenv import load_dotenv
import os

load_dotenv()

url_token = "https://digital.iservices.rte-france.com/token/oauth"
username = os.getenv("RTE_USERNAME")
password = os.getenv("RTE_PASSWORD")

response = requests.post(url_token, data={'grant_type': 'client_credentials'}, auth=(username, password))
token = response.json().get("access_token")

base_url = "https://digital.iservices.rte-france.com/open_api/consumption/v1/short_term"
headers = {"Authorization": f"Bearer {token}"}

def datearange(start_date, end_date, delta):
    current_date = start_date
    while current_date < end_date:
        next_date = min(current_date + delta, end_date)
        yield current_date, next_date
        current_date = next_date

start_date, end_date = datetime(2020, 1, 1), datetime.now().replace(microsecond=0)
all_values = []
all_start_date = []

for start, end in datearange(start_date, end_date, timedelta(days=180)):
    url = f"{base_url}?type=REALISED&start_date={start.isoformat()}%2B02:00&end_date={end.isoformat()}%2B02:00"
    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        data = res.json()
        if 'short_term' in data and len(data['short_term']) > 0:
            for entry in data['short_term'][0]['values']:
                all_start_date.append(entry['start_date'])
                all_values.append(entry['value'])
    else:
        print(f"Erreur sur la période {start.date()} - {end.date()}")

df = pd.DataFrame({'start_date': all_start_date, 'value': all_values})
df['start_date'] = pd.to_datetime(df['start_date'], utc=True)
df['date_column'] = df['start_date'].dt.date.astype(str)
df['hour_column'] = df['start_date'].dt.hour
df['min_column'] = df['start_date'].dt.minute

def consumption_avg(group):
    if len(group) < 4:
        return None
    return group['value'].mean()

avg_hourly = df.groupby(['date_column', 'hour_column']).apply(consumption_avg, include_groups=False).reset_index().rename(columns={0: 'avg_value_hourly'})
df_ = pd.merge(avg_hourly, df, on=['date_column', 'hour_column'], how='inner')
df_ = df_[df_['min_column'] == 0]

# On garde uniquement les colonnes utiles
df_final = df_[['start_date', 'avg_value_hourly']].copy()
df_final = df_final.drop_duplicates(subset='start_date', keep='first')

# Grille horaire complète pour combler les trous
full_range = pd.date_range(start=df_final['start_date'].min(),
                           end=df_final['start_date'].max(),
                           freq='h', tz='UTC')
df_full = pd.DataFrame({'start_date': full_range})
df_final = pd.merge(df_full, df_final, on='start_date', how='left')

# Remplissage J-7 puis interpolation linéaire
df_final['avg_value_hourly'] = df_final['avg_value_hourly'].fillna(df_final['avg_value_hourly'].shift(168))
df_final['avg_value_hourly'] = df_final['avg_value_hourly'].interpolate(method='linear')

df_final.to_csv('consumption_data_cleaned.csv', index=False)
print(f"Terminé ! {len(df_final)} lignes sauvegardées.")

df_final.tail(24*30).plot(x='start_date', y='avg_value_hourly', figsize=(12,6), color='blue')
plt.title("Consommation électrique horaire (Derniers 30 jours)")
plt.ylabel("MW")
plt.grid(True, alpha=0.3)
plt.show()