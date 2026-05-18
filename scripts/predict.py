import logging
import pandas as pd
from datetime import datetime, timezone
from psycopg2.extras import execute_values
from chronos import Chronos2Pipeline

from src.db.client import get_connection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/predict.log"),
        logging.StreamHandler()
    ]
)

def main():
    logging.info("--- Début des prédictions ---")

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT timestamp, value FROM historical_data 
        ORDER BY timestamp DESC 
        LIMIT 8192
    """)
    rows = cursor.fetchall()

    context_df = pd.DataFrame(rows, columns=['start_date', 'avg_value_hourly'])
    context_df = context_df.sort_values('start_date').reset_index(drop=True)
    context_df['start_date'] = pd.to_datetime(context_df['start_date'], utc=True)
    context_df['id_column'] = 'FR'
    logging.info(f"Contexte : {len(context_df)} heures")

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
    rows_to_insert = []
    for _, row in pred_df.iterrows():
        ts = row['start_date']
        predicted = row['predictions']
        mape_val = None
        rows_to_insert.append((
            ts, predicted, row['0.1'], row['0.9'], mape_val, 'chronos-2', 'H+48', prediction_date
        ))

    execute_values(cursor, """
        INSERT INTO predictions (timestamp, predicted_value, q10, q90, mape, model_name, horizon, prediction_date)
        VALUES %s
        ON CONFLICT (timestamp) DO UPDATE SET
            predicted_value = EXCLUDED.predicted_value,
            q10 = EXCLUDED.q10,
            q90 = EXCLUDED.q90,
            prediction_date = EXCLUDED.prediction_date
    """, rows_to_insert)

    conn.commit()
    logging.info(f"{len(rows_to_insert)} prédictions insérées")
    logging.info("--- Fin ---")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()