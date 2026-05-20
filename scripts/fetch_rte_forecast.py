import logging
from datetime import datetime, timedelta

from psycopg2.extras import execute_values

from src.db.client import get_connection
from src.ingestion.rte_client import fetch_forecast
from src.processing.features import compute_hourly_avg

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/fetch_rte_predictions.log"),
        logging.StreamHandler()
    ]
)

def main():
    logging.info("--- Début du pipeline (Historique Prédictions RTE) ---")

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT timestamp FROM predictions_rte ORDER BY timestamp DESC LIMIT 1")
    last_timestamp = cursor.fetchone()[0]
    logging.info(f"Dernier timestamp en base : {last_timestamp}")

    # 1. Configuration des dates (1er Janvier 2020 jusqu'à aujourd'hui)
    start_date = last_timestamp.replace(tzinfo=None).replace(hour=0, minute=0, second=0) - timedelta(days=1)
    global_end_date = (datetime.now() + timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0)

    logging.info(f"Lancement de la récupération du {start_date} au {global_end_date}")
    logging.info("Récupération du token RTE...")

    # Fenêtre maximale de 180 jours par requête pour respecter les limites de l'API RTE
    chunk_size = timedelta(days=180)
    current_start = start_date

    # Date d'exécution du script (utilisée pour 'prediction_date')
    execution_now = datetime.now().replace(microsecond=0)

    while current_start < global_end_date:
        current_end = min(current_start + chunk_size, global_end_date)
        
        logging.info(f"Requête RTE : de {current_start} à {current_end}...")
        
        try:
            # Appel API pour le modèle "D-2"
            df_raw = fetch_forecast(current_start, current_end, "D-2")
            
            if df_raw is not None and not df_raw.empty:
                df_final = compute_hourly_avg(df_raw)
                logging.info(f"{len(df_final)} lignes après traitement pour ce bloc")

                # Mapping des données pour correspondre à la table 'predictions_rte'
                # Ordre : timestamp, predicted_value, model_name, horizon, prediction_date
                rows = [
                    (
                        row['start_date'],          # -> timestamp
                        row['avg_value_hourly'],     # -> predicted_value
                        'RTE-D1',                   # -> model_name (Varchar)
                        'H+48',                      # -> horizon (Varchar)
                        execution_now               # -> prediction_date (TIMESTAMPTZ)
                    ) 
                    for _, row in df_final.iterrows()
                ]
                
                # Requête d'insertion adaptée à la structure Supabase
                # ON CONFLICT (timestamp) DO NOTHING évite les doublons si tu relances le script
                execute_values(cursor, """
                    INSERT INTO predictions_rte (timestamp, predicted_value, model_name, horizon, prediction_date)
                    VALUES %s
                    ON CONFLICT (timestamp) DO NOTHING
                """, rows)
                
                conn.commit()  # Sauvegarde sur Supabase pour ce bloc
                logging.info(f"{len(rows)} prédictions insérées avec succès.")
            else:
                logging.warning(f"Pas de données renvoyées pour la période {current_start} - {current_end}")
                
        except Exception as e:
            logging.error(f"Erreur lors du traitement du bloc {current_start} - {current_end} : {e}")
            conn.rollback()  # Annule la transaction du bloc en cours si Supabase rejette la requête

        # Passage au bloc de 180 jours suivant
        current_start = current_end

    logging.info("--- Fin du pipeline ---")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()