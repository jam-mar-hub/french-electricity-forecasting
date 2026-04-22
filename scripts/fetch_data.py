import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv
import requests
from psycopg2.extras import execute_values
from datetime import timedelta, datetime
import logging

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/fetch_data.log"),
        logging.StreamHandler()
    ]
)

logging.info("--- Début du pipeline ---")

url_token = "https://digital.iservices.rte-france.com/token/oauth"
username = os.getenv("RTE_USERNAME")
password = os.getenv("RTE_PASSWORD")

response = requests.post(url_token, data={'grant_type': 'client_credentials'}, auth=(username, password))
token = response.json().get("access_token")
logging.info("Token RTE récupéré")

base_url = "https://digital.iservices.rte-france.com/open_api/consumption/v1/short_term"
headers = {"Authorization": f"Bearer {token}"}

conn = psycopg2.connect(
    host=os.getenv("SUPABASE_HOST"),
    database=os.getenv("SUPABASE_DB"),
    user=os.getenv("SUPABASE_USER"),
    password=os.getenv("SUPABASE_PASSWORD"),
    port=os.getenv("SUPABASE_PORT")
)
cursor = conn.cursor()
logging.info("Connexion Supabase OK")

cursor.execute("SELECT timestamp FROM historical_data ORDER BY timestamp DESC LIMIT 1")
last_timestamp = cursor.fetchone()[0]
logging.info(f"Dernier timestamp en base : {last_timestamp}")

start_date = last_timestamp.replace(tzinfo=None).replace(hour=0, minute=0, second=0)
start_date = start_date - timedelta(days=1)
end_date = datetime.now().replace(microsecond=0)
logging.info(f"Période de collecte : {start_date} → {end_date}")

url = f"{base_url}?type=REALISED&start_date={start_date.isoformat()}%2B02:00&end_date={end_date.isoformat()}%2B02:00"
res = requests.get(url, headers=headers)

all_values = []
all_start_date = []

if res.status_code == 200:
    data = res.json()
    if 'short_term' in data and len(data['short_term']) > 0:
        for entry in data['short_term'][0]['values']:
            all_start_date.append(entry['start_date'])
            all_values.append(entry['value'])
    logging.info(f"Données récupérées : {len(all_values)} valeurs brutes")
else:
    logging.error(f"Erreur API : {res.json()}")
    cursor.close()
    conn.close()
    exit()

df = pd.DataFrame({'start_date': all_start_date, 'value': all_values})
df['start_date'] = pd.to_datetime(df['start_date'], utc=True)
df['date_column'] = df['start_date'].dt.date.astype(str)
df['hour_column'] = df['start_date'].dt.hour
df['min_column'] = df['start_date'].dt.minute

def consumption_avg(group):
    if len(group) < 4:
        return None
    return group['value'].mean()

# Calcul de la moyenne par heure
avg_hourly = df.groupby(['date_column', 'hour_column']).apply(consumption_avg, include_groups=False).reset_index(name='avg_value_hourly')
df_ = pd.merge(avg_hourly, df, on=['date_column', 'hour_column'], how='inner')
df_ = df_[df_['min_column'] == 0]
df_final = df_[['start_date', 'avg_value_hourly']].copy()

df_final = df_final.set_index('start_date').resample('h').mean()
df_final['avg_value_hourly'] = df_final['avg_value_hourly'].interpolate(method='linear')
df_final = df_final.reset_index()
df_final = df_final.dropna(subset=['avg_value_hourly'])
logging.info(f"Données nettoyées : {len(df_final)} lignes après traitement")

rows = [(row['start_date'], row['avg_value_hourly'], 'RTE') for _, row in df_final.iterrows()]

execute_values(cursor, """
    INSERT INTO historical_data (timestamp, value, source)
    VALUES %s
    ON CONFLICT (timestamp) DO NOTHING
""", rows)

conn.commit()
logging.info(f"{len(rows)} lignes traitées, insérées dans historical_data")
logging.info("--- Fin du pipeline ---")

cursor.close()
conn.close()