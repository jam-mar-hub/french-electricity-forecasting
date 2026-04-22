# ⚡ French Electricity Consumption Forecasting

This project provides an end-to-end pipeline for monitoring and predicting electricity consumption in France. It leverages the **RTE (Réseau de Transport d'Électricité) API** for real-time data and Amazon's **Chronos-2** (a generative time-series model) for high-accuracy forecasting.

## 🏗️ Project Architecture

The project is structured around a data pipeline that moves from raw API ingestion to a PostgreSQL database (Supabase), followed by AI-driven predictions and a Streamlit visualization dashboard.

1.  **Data Ingestion & Cleaning**: `data_processing.py`
2.  **Database Seeding**: `insert_historical.py`
3.  **Automated Updates**: `fetch_data.py` (via Cron)
4.  **Time-Series Forecasting**: `predict.py` (Amazon Chronos)
5.  **Visualization**: `streamlit/app.py`

---

## 🛠️ Components

### 1. Data Processing (`scripts/data_processing.py`)
This script handles the initial heavy lifting. It fetches historical consumption data from 2020 to the present day using the RTE API.
- **Cleaning**: It calculates hourly averages from 15-minute intervals.
- **Gap Filling**: It handles missing data points using a combination of J-7 (same time last week) shifting and linear interpolation.
- **Output**: Generates a cleaned CSV: `data/consumption_data_cleaned.csv`.

### 2. Historical Data Injection (`scripts/insert_historical.py`)
Used to seed the Supabase PostgreSQL database. It reads the cleaned CSV and performs a bulk insert into the `historical_data` table. It uses `ON CONFLICT (timestamp) DO NOTHING` to ensure data integrity without duplicates.

### 3. Automated Data Fetcher (`scripts/fetch_data.py`)
Designed to be run as a **Cron job**.
- It identifies the last recorded timestamp in the database.
- Fetches new "Realised" consumption data from RTE.
- Processes and appends new records to the database automatically.

### 4. Predictive Modeling (`scripts/predict.py`)
The core intelligence of the project. Instead of traditional regression, it uses **Amazon Chronos-2**, a state-of-the-art foundation model for time-series forecasting.
- **Context**: It retrieves the last 168 hours (1 week) of consumption data.
- **Forecast**: It generates a 48-hour forecast (H+48) including confidence intervals (10th and 90th percentiles).
- **Storage**: Predictions are stored in the `predictions` table for visualization and backtesting.

### 5. Dashboard (`streamlit/app.py`)
A comprehensive UI to explore the data:
- **Real-time Metrics**: Displays total records, last update time, and model accuracy (MAPE).
- **Historical Analysis**: Interactive charts for the last 7, 30, or 90 days.
- **Backtesting**: A dedicated section to compare model predictions against ground truth for specific historical periods, including confidence interval shading.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- A Supabase account (PostgreSQL)
- RTE API credentials (Username/Password from the RTE Digital Services portal)

### Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory with the following variables:
   ```env
   RTE_USERNAME=your_rte_username
   RTE_PASSWORD=your_rte_password
   SUPABASE_HOST=your_db_host
   SUPABASE_DB=postgres
   SUPABASE_USER=postgres
   SUPABASE_PASSWORD=your_db_password
   SUPABASE_PORT=5432
   ```

### Running the Project
1. **Initialize Data**:
   ```bash
   python scripts/data_processing.py
   python scripts/insert_historical.py
   ```
2. **Generate Predictions**:
   ```bash
   python scripts/predict.py
   ```
3. **Launch Dashboard**:
   ```bash
   streamlit run streamlit/app.py
   ```

## 📈 Model Performance
The system calculates the **Mean Absolute Percentage Error (MAPE)** by comparing past predictions stored in the database with the actual values later fetched by the automated pipeline. This allows for continuous monitoring of the Chronos model's performance on French grid data.

---
*Note: The `models/` directory contains legacy XGBoost experiments which are currently deprecated in favor of the Chronos transformer-based approach.*
