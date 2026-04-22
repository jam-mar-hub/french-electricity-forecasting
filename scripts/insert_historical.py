from psycopg2.extras import execute_values
import psycopg2
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

# Connexion à Supabase
conn = psycopg2.connect(
    host=os.getenv("SUPABASE_HOST"),
    database=os.getenv("SUPABASE_DB"),
    user=os.getenv("SUPABASE_USER"),
    password=os.getenv("SUPABASE_PASSWORD"),
    port=os.getenv("SUPABASE_PORT")
)
cursor = conn.cursor()

df = pd.read_csv("data/consumption_data_cleaned.csv")
df['start_date'] = pd.to_datetime(df['start_date'], utc=True)

rows = [
    (row['start_date'], row['avg_value_hourly'], 'RTE')
    for _, row in df.iterrows()
]

execute_values(cursor, """
    INSERT INTO historical_data (timestamp, value, source)
    VALUES %s
    ON CONFLICT (timestamp) DO NOTHING
""", rows)

conn.commit()
print(f"{cursor.rowcount} lignes insérées dans historical_data")

cursor.close()
conn.close()