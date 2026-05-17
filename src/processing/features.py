import pandas as pd

def compute_hourly_avg(df : pd.DataFrame) -> pd.DataFrame:
    df['date_column'] = df['start_date'].dt.date.astype(str)
    df['hour_column'] = df['start_date'].dt.hour
    df['min_column'] = df['start_date'].dt.minute

    avg_hourly = df.groupby(['date_column', 'hour_column'])['value'].mean().reset_index().rename(columns={'value': 'avg_value_hourly'})

    df_ = pd.merge(avg_hourly, df, on=['date_column', 'hour_column'], how='inner')
    df_ = df_[df_['min_column'] == 0]

    df_final = df_[['start_date', 'avg_value_hourly']].copy()
    df_final = df_final.drop_duplicates(subset='start_date', keep='first')
    df_final = df_final.sort_values('start_date').reset_index(drop=True)

    return df_final