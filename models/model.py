import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import optuna
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from workalendar.europe import France

# ── 1. CHARGEMENT & FEATURES ─────────────────────────────────────────────────

df = pd.read_csv("consumption_data_cleaned.csv")
df['start_date'] = pd.to_datetime(df['start_date'], utc=True).dt.tz_convert('Europe/Paris')
df = df.sort_values('start_date').reset_index(drop=True)

cal = France()
holidays = set()
for year in range(2019, 2026):
    for date, _ in cal.holidays(year):
        holidays.add(date)

df.drop(columns=['hour_column'])

df['hour']       = df['start_date'].dt.hour
df['dayofweek']  = df['start_date'].dt.dayofweek
df['month']      = df['start_date'].dt.month
df['day_of_year']= df['start_date'].dt.dayofyear
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['is_holiday'] = df['start_date'].dt.date.apply(lambda d: d in holidays).astype(int)

df['hour_sin']  = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos']  = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['dow_sin']   = np.sin(2 * np.pi * df['dayofweek'] / 7)
df['dow_cos']   = np.cos(2 * np.pi * df['dayofweek'] / 7)

df['lag_24']  = df['avg_value_hourly'].shift(24)
df['lag_48']  = df['avg_value_hourly'].shift(48)
df['lag_72']  = df['avg_value_hourly'].shift(72)
df['lag_168'] = df['avg_value_hourly'].shift(168)
df['lag_336'] = df['avg_value_hourly'].shift(336)

df['rolling_24h']     = df['avg_value_hourly'].shift(1).rolling(24).mean()
df['rolling_7d']      = df['avg_value_hourly'].shift(1).rolling(168).mean()
df['rolling_24h_std'] = df['avg_value_hourly'].shift(1).rolling(24).std()

df['diff_24_48']  = df['lag_24'] - df['lag_48']
df['diff_24_168'] = df['lag_24'] - df['lag_168']

df = df.dropna()

FEATURES = [
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
    'day_of_year', 'is_weekend', 'is_holiday',
    'lag_24', 'lag_48', 'lag_72', 'lag_168', 'lag_336',
    'rolling_24h', 'rolling_7d', 'rolling_24h_std',
    'diff_24_48', 'diff_24_168',
]
TARGET = 'avg_value_hourly'

# ── 2. SPLIT ─────────────────────────────────────────────────────────────────

split_date = '2025-01-01'
train = df[df['start_date'] < split_date].reset_index(drop=True)
test  = df[df['start_date'] >= split_date].reset_index(drop=True)

X_train, y_train = train[FEATURES].values, train[TARGET].values
X_test,  y_test  = test[FEATURES].values,  test[TARGET].values
dates_test = test['start_date']  # Series avec index propre 0..N

tscv = TimeSeriesSplit(n_splits=5)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── 3. MODÈLE 1 : XGBoost baseline ───────────────────────────────────────────

print("Entraînement XGBoost baseline...")
model_xgb_base = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    objective='reg:squarederror',
    random_state=42,
    verbosity=0,
)
model_xgb_base.fit(X_train, y_train)
pred_xgb_base = model_xgb_base.predict(X_test)
mae_xgb_base  = mean_absolute_error(y_test, pred_xgb_base)
print(f"  MAE XGBoost baseline : {mae_xgb_base:.0f} MW")

model_xgb_base.save_model("model_xgb_baseline.ubj")


# ── 4. MODÈLE 2 : XGBoost tuné avec Optuna ───────────────────────────────────
"""
print("\nOptuna XGBoost (50 trials)...")

def objective_xgb(trial):
    params = {
        'n_estimators'     : trial.suggest_int('n_estimators', 300, 1500),
        'learning_rate'    : trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth'        : trial.suggest_int('max_depth', 3, 8),
        'subsample'        : trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight' : trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha'        : trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda'       : trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'objective'        : 'reg:squarederror',
        'random_state'     : 42,
        'verbosity'        : 0,
    }
    maes = []
    for tr_idx, val_idx in tscv.split(X_train):
        m = xgb.XGBRegressor(**params)
        m.fit(X_train[tr_idx], y_train[tr_idx],
              eval_set=[(X_train[val_idx], y_train[val_idx])],
              verbose=False)
        maes.append(mean_absolute_error(y_train[val_idx], m.predict(X_train[val_idx])))
    return np.mean(maes)

study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=50, show_progress_bar=True)

best_xgb = study_xgb.best_params
best_xgb.update({'objective': 'reg:squarederror', 'random_state': 42, 'verbosity': 0})

model_xgb_tuned = xgb.XGBRegressor(**best_xgb)
model_xgb_tuned.fit(X_train, y_train)
pred_xgb_tuned = model_xgb_tuned.predict(X_test)
mae_xgb_tuned  = mean_absolute_error(y_test, pred_xgb_tuned)
print(f"  MAE XGBoost tuned    : {mae_xgb_tuned:.0f} MW")

model_xgb_tuned.save_model("model_xgb_tuned.ubj")
with open("best_params_xgb.json", "w") as f:
    json.dump(study_xgb.best_params, f, indent=2)
"""

print("\nChargement XGBoost tuned...")
model_xgb_tuned = xgb.XGBRegressor()
model_xgb_tuned.load_model("model_xgb_tuned.ubj")
pred_xgb_tuned = model_xgb_tuned.predict(X_test)
mae_xgb_tuned  = mean_absolute_error(y_test, pred_xgb_tuned)
print(f"  MAE XGBoost tuned    : {mae_xgb_tuned:.0f} MW")

# ── 5. MODÈLE 3 : LightGBM tuné avec Optuna ──────────────────────────────────
# LightGBM utilise une approche différente de XGBoost :
#   - XGBoost : construit l'arbre niveau par niveau (level-wise)
#   - LightGBM : construit l'arbre feuille par feuille (leaf-wise)
# => LightGBM est souvent plus rapide et parfois plus précis sur données tabulaires.
# Les hyperparamètres sont similaires mais pas identiques (num_leaves remplace max_depth).

"""
print("\nOptuna LightGBM (50 trials)...")

X_train_df = train[FEATURES] 

def objective_lgb(trial):
    params = {
        'n_estimators'    : trial.suggest_int('n_estimators', 300, 1500),
        'learning_rate'   : trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves'      : trial.suggest_int('num_leaves', 20, 200),      # équivalent max_depth
        'max_depth'       : trial.suggest_int('max_depth', 3, 8),          # garde quand même un plafond
        'subsample'       : trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50), # équivalent min_child_weight
        'reg_alpha'       : trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda'      : trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'random_state'    : 42,
        'verbose'         : -1,
    }
    maes = []
    for tr_idx, val_idx in tscv.split(X_train_df):
        m = lgb.LGBMRegressor(**params)
        m.fit(X_train_df.iloc[tr_idx], y_train[tr_idx],
              eval_set=[(X_train_df.iloc[val_idx], y_train[val_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        maes.append(mean_absolute_error(y_train[val_idx], m.predict(X_train_df.iloc[val_idx])))
    return np.mean(maes)

study_lgb = optuna.create_study(direction='minimize')
study_lgb.optimize(objective_lgb, n_trials=50, show_progress_bar=True)

best_lgb = study_lgb.best_params
best_lgb.update({'random_state': 42, 'verbose': -1})

model_lgb = lgb.LGBMRegressor(**best_lgb)
model_lgb.fit(train[FEATURES], train[TARGET])
pred_lgb = model_lgb.predict(test[FEATURES]) 
mae_lgb  = mean_absolute_error(test[TARGET], pred_lgb)
print(f"  MAE LightGBM tuned   : {mae_lgb:.0f} MW")

model_lgb.booster_.save_model("model_lgb_tuned.txt")
with open("best_params_lgb.json", "w") as f:
    json.dump(study_lgb.best_params, f, indent=2)
"""

print("\nChargement LightGBM tuned...")
# 1. Charger le modèle LightGBM depuis le fichier .txt
model_lgb = lgb.Booster(model_file="model_lgb_tuned.txt")
    
 # 2. Prédire (le Booster de LightGBM utilise directement .predict)
pred_lgb = model_lgb.predict(X_test)
    
 # 3. Calculer la MAE
mae_lgb = mean_absolute_error(y_test, pred_lgb)
   
print(f"  MAE LightGBM tuned   : {mae_lgb:.0f} MW")


# ── 6. GRAPHIQUES ─────────────────────────────────────────────────────────────
# On génère deux figures :
#   Fig 1 — Prédictions vs réel sur une fenêtre lisible (30 derniers jours)
#   Fig 2 — Comparaison des MAE par modèle (bar chart)

results = {
    'XGBoost baseline' : (pred_xgb_base, mae_xgb_base, '#aec6e8'),
    'XGBoost tuned'    : (pred_xgb_tuned, mae_xgb_tuned, '#1f77b4'),
    'LightGBM tuned'   : (pred_lgb, mae_lgb, '#ff7f0e'),
}

# -- Figure 1 : courbes sur les 30 derniers jours du test --
fig, ax = plt.subplots(figsize=(16, 6))

# Filtre 30 derniers jours — iloc[-1] pour éviter le KeyError sur index non-réinitialisé
mask_30d = (dates_test >= (dates_test.iloc[-1] - np.timedelta64(30, 'D'))).values

ax.plot(dates_test.values[mask_30d], y_test[mask_30d],
        label='Réel', color='black', linewidth=1.5, zorder=5)

for label, (preds, mae, color) in results.items():
    ax.plot(dates_test.values[mask_30d], preds[mask_30d],
            label=f'{label}  (MAE = {mae:.0f} MW)',
            color=color, linewidth=1.2, linestyle='--', alpha=0.85)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
fig.autofmt_xdate()

ax.set_title('Prédictions vs Réel — 30 derniers jours du test', fontsize=13)
ax.set_ylabel('Consommation (MW)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('comparaison_courbes.png', dpi=150)
plt.show()
print("Graphique sauvegardé : comparaison_courbes.png")

# -- Figure 2 : bar chart MAE --
fig2, ax2 = plt.subplots(figsize=(7, 4))

labels = list(results.keys())
maes   = [v[1] for v in results.values()]
colors = [v[2] for v in results.values()]

bars = ax2.bar(labels, maes, color=colors, width=0.5, edgecolor='white')

# Afficher la valeur de MAE au dessus de chaque barre
for bar, mae in zip(bars, maes):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 5,
             f'{mae:.0f} MW',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_ylabel('MAE (MW)')
ax2.set_title('Comparaison des modèles — MAE sur le test set 2025')
ax2.set_ylim(0, max(maes) * 1.2)
ax2.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('comparaison_mae.png', dpi=150)
plt.show()
print("Graphique sauvegardé : comparaison_mae.png")

# ── 7. RÉSUMÉ FINAL ──────────────────────────────────────────────────────────

print(f"\n{'='*50}")
print(f"{'Modèle':<25} {'MAE (MW)':>10}  {'Gain vs baseline':>18}")
print(f"{'-'*50}")
for label, (_, mae, _) in results.items():
    gain = mae_xgb_base - mae
    gain_pct = gain / mae_xgb_base * 100
    gain_str = f"{gain:+.0f} MW ({gain_pct:+.1f}%)" if gain != 0 else "—"
    print(f"{label:<25} {mae:>10.0f}  {gain_str:>18}")
print(f"{'='*50}")


test_clean = test.dropna(subset=['lag_168', 'avg_value_hourly'])

# Calcul de la MAE de la baseline J-7 sur le jeu de test
mae_naive_test = mean_absolute_error(test_clean['avg_value_hourly'], test_clean['lag_168'])

print(f"MAE de la Baseline (J-7) : {mae_naive_test:.0f} MW")