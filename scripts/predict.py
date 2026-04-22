import psycopg2
import pandas as pd
import os
from psycopg2.extras import execute_values
from datetime import datetime, timezone
from chronos import Chronos2Pipeline
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("--- Début de la génération de prédictions ---")

conn = psycopg2.connect(
    host=os.environ.get('SUPABASE_HOST'),
    database=os.environ.get('SUPABASE_DB'),
    user=os.environ.get('SUPABASE_USER'),
    password=os.environ.get('SUPABASE_PASSWORD'),
    port=os.environ.get('SUPABASE_PORT')
)

cursor = conn.cursor()
logging.info("Connexion Supabase OK")

cursor.execute("""
    SELECT timestamp, value FROM historical_data 
    ORDER BY timestamp DESC 
    LIMIT 168
""")
rows = cursor.fetchall()

context_df = pd.DataFrame(rows, columns=['start_date', 'avg_value_hourly'])
context_df = context_df.sort_values('start_date').reset_index(drop=True)
context_df['start_date'] = pd.to_datetime(context_df['start_date'], utc=True)
context_df['id_column'] = 'FR'
logging.info(f"Contexte récupéré : {len(context_df)} heures")

pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
logging.info("Modèle chargé")

pred_df = pipeline.predict_df(
    context_df,
    prediction_length=48,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column="id_column",
    timestamp_column="start_date",
    target="avg_value_hourly"
)

prediction_date = datetime.now(timezone.utc)

rows_to_insert = [
    (row['start_date'], row['predictions'], 'chronos-2', 'H+48', prediction_date)
    for _, row in pred_df.iterrows()
]

execute_values(cursor, """
    INSERT INTO predictions (timestamp, predicted_value, model_name, horizon, prediction_date)
    VALUES %s
    ON CONFLICT (timestamp) DO UPDATE SET
        predicted_value = EXCLUDED.predicted_value,
        prediction_date = EXCLUDED.prediction_date
""", rows_to_insert)

conn.commit()
logging.info(f"{len(rows_to_insert)} prédictions insérées")
logging.info("--- Fin ---")

cursor.close()
conn.close()