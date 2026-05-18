import requests
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()


def get_token():
    url_token = "https://digital.iservices.rte-france.com/token/oauth"
    username = os.getenv("RTE_USERNAME")
    password = os.getenv("RTE_PASSWORD")
    response = requests.post(
        url_token,
        data={'grant_type': 'client_credentials'},
        auth=(username, password)
    )
    return response.json().get("access_token")


def fetch_realised(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"}
    base_url = "https://digital.iservices.rte-france.com/open_api/consumption/v1/short_term"

    url = f"{base_url}?type=REALISED&start_date={start_date.isoformat()}%2B02:00&end_date={end_date.isoformat()}%2B02:00"
    res = requests.get(url, headers=headers)

    if res.status_code != 200:
        raise ValueError(f"Erreur API RTE : {res.status_code} - {res.text}")

    data = res.json()
    entries = data['short_term'][0]['values']

    df = pd.DataFrame({
        'start_date': [e['start_date'] for e in entries],
        'value': [e['value'] for e in entries]
    })
    df['start_date'] = pd.to_datetime(df['start_date'], utc=True)
    return df

def fetch_forecast(start_date: datetime, end_date: datetime, forecast_type: str = "D-2") -> pd.DataFrame:
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"}
    base_url = "https://digital.iservices.rte-france.com/open_api/consumption/v1/short_term"
    
    url = f"{base_url}?type={forecast_type}&start_date={start_date.isoformat()}%2B02:00&end_date={end_date.isoformat()}%2B02:00"
    res = requests.get(url, headers=headers)

    if res.status_code != 200:
        raise ValueError(f"Erreur API RTE : {res.status_code} - {res.text}")

    data = res.json()
    entries = data['short_term'][0]['values']

    df = pd.DataFrame({
        'start_date': [e['start_date'] for e in entries],
        'value': [e['value'] for e in entries]
    })
    df['start_date'] = pd.to_datetime(df['start_date'], utc=True)
    return df