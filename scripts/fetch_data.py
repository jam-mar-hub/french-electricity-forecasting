import logging
from datetime import datetime, timedelta

from psycopg2.extras import execute_values

from src.db.client import get_connection
from src.ingestion.rte_client import fetch_realised
from src.processing.features import compute_hourly_avg

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/fetch_data.log"),
        logging.StreamHandler()
    ]
)

def main():
    logging.info("--- Début du pipeline ---")

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT timestamp FROM historical_data ORDER BY timestamp DESC LIMIT 1")
    last_timestamp = cursor.fetchone()[0]
    logging.info(f"Dernier timestamp en base : {last_timestamp}")

    start_date = last_timestamp.replace(tzinfo=None).replace(hour=0, minute=0, second=0) - timedelta(days=1)
    end_date = datetime.now().replace(microsecond=0)

    logging.info("Récupération du token RTE...")

    df_raw = fetch_realised(start_date, end_date)
    df_final = compute_hourly_avg(df_raw)
    logging.info(f"{len(df_final)} lignes après traitement")

    rows = [(row['start_date'], row['avg_value_hourly'], 'RTE') for _, row in df_final.iterrows()]
    execute_values(cursor, """
        INSERT INTO historical_data (timestamp, value, source)
        VALUES %s
        ON CONFLICT (timestamp) DO NOTHING
    """, rows)

    conn.commit()
    logging.info(f"{len(rows)} lignes insérées")
    logging.info("--- Fin du pipeline ---")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()