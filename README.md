# French Electricity Consumption Forecasting

End-to-end ML pipeline for forecasting French electricity consumption using the RTE API and Chronos-2, a zero-shot foundation model developed by Amazon.

**Live demo** : https://french-electricity-forecasting.streamlit.app/

---

## Results

| Model | MAPE |
|-------|------|
| Chronos-2 (zero-shot) | ~4.79% |

---

## Architecture

```
RTE API → fetch_data.py → Supabase (historical_data)
                               ↓
                          predict.py (Chronos-2) → Supabase (predictions)
                               ↓
                        Streamlit Cloud (dashboard)
```

**Automation**
- `fetch_data.py` runs daily at 00:00 UTC via GitHub Actions
- `predict.py` runs daily at 01:00 UTC via Kaggle (GPU T4)

---

## Project Structure

```
electricity-forecasting/
├── src/
│   ├── db/
│   │   └── client.py          # Supabase connection
│   ├── ingestion/
│   │   └── rte_client.py      # RTE API client
│   └── processing/
│       └── features.py        # Preprocessing logic
├── scripts/
│   ├── fetch_data.py          # Daily data ingestion
│   ├── predict.py             # Chronos-2 inference
│   ├── build_dataset.py       # Historical dataset builder
│   └── insert_historical.py   # One-time historical insert
├── app.py                     # Streamlit dashboard
├── .github/workflows/
│   └── pipeline.yml           # GitHub Actions CI/CD
└── pyproject.toml
```

---

## Stack

- **Data** : RTE Open Data API
- **Model** : [Chronos-2](https://github.com/amazon-science/chronos-forecasting) (zero-shot, no fine-tuning)
- **Database** : Supabase (PostgreSQL)
- **Dashboard** : Streamlit Cloud
- **Automation** : GitHub Actions + Kaggle Notebooks (GPU T4)
- **Code quality** : ruff, uv

---

## Local Setup

```bash
git clone https://github.com/jam-mar-hub/french-electricity-forecasting
cd french-electricity-forecasting
uv sync
```

Create a `.env` file :

```
RTE_USERNAME=your_rte_username
RTE_PASSWORD=your_rte_password
SUPABASE_HOST=your_host
SUPABASE_DB=postgres
SUPABASE_USER=your_user
SUPABASE_PASSWORD=your_password
SUPABASE_PORT=5432
```

Run the pipeline :

```bash
python -m scripts.fetch_data
python -m scripts.predict
streamlit run app.py
```

---

## Limitations & Next Steps

- Chronos-2 is used zero-shot — fine-tuning on French consumption data could improve MAPE
- External features (temperature, TEMPO signal, public holidays) could be added via a gradient boosting model (XGBoost/LightGBM)
- Price forecasting (EPEX Spot API) as a natural extension
