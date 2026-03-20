"""
CA683 — Full pipeline: load → clean → fuse → features → train → evaluate → figures
Run from project root: python src/run_pipeline.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

FIGURES = Path('reports/figures')
FIGURES.mkdir(parents=True, exist_ok=True)
PROCESSED = Path('data/processed')

# ─────────────────────────────────────────────
# 1. LOAD CLEANED DATA
# ─────────────────────────────────────────────
print('='*65)
print(' CA683 — Dublin Traffic Congestion — Full Pipeline')
print('='*65)

print('\n[1/6] Loading cleaned datasets...')
traffic = pd.read_csv(PROCESSED / 'traffic_cleaned.csv')
weather = pd.read_csv(PROCESSED / 'weather_cleaned.csv', parse_dates=['datetime'])
events  = pd.read_csv(PROCESSED / 'events_cleaned.csv',
                      parse_dates=['start_datetime', 'end_datetime'])

print(f'  Traffic : {len(traffic):>10,} rows × {traffic.shape[1]} cols')
print(f'  Weather : {len(weather):>10,} rows × {weather.shape[1]} cols')
print(f'  Events  : {len(events):>10,} rows × {events.shape[1]} cols')

# ─────────────────────────────────────────────
# 2. PARSE & AGGREGATE TRAFFIC TO HOURLY
# ─────────────────────────────────────────────
print('\n[2/6] Parsing timestamps & aggregating to hourly...')

traffic['datetime'] = pd.to_datetime(traffic['End_Time'], errors='coerce')
traffic = traffic.dropna(subset=['datetime'])
traffic = traffic[traffic['Sum_Volume'] >= 0].copy()
traffic['hour'] = traffic['datetime'].dt.floor('h')

traffic_total = (
    traffic.groupby('hour')
    .agg(
        total_volume=('Sum_Volume', 'sum'),
        avg_volume=('Avg_Volume', 'mean'),
        n_detectors=('Detector', 'count'),
    )
    .reset_index()
)

print(f'  Hourly rows   : {len(traffic_total):,}')
print(f'  Date range    : {traffic_total["hour"].min()} → {traffic_total["hour"].max()}')

# ─────────────────────────────────────────────
# 3. MERGE WITH WEATHER
# ─────────────────────────────────────────────
print('\n[3/6] Merging traffic + weather...')

weather['hour'] = weather['datetime'].dt.floor('h')
weather_cols = [
    'hour', 'temp', 'feelslike', 'humidity', 'precip',
    'precipprob', 'windspeed', 'windgust', 'cloudcover',
    'visibility', 'sealevelpressure', 'uvindex',
]
weather_slim = weather[weather_cols].drop_duplicates(subset='hour')
merged = pd.merge(traffic_total, weather_slim, on='hour', how='inner')
print(f'  After merge   : {len(merged):,} rows')

# ─────────────────────────────────────────────
# 4. EVENTS FEATURES
# ─────────────────────────────────────────────
print('\n[4/6] Computing events features...')

def get_event_features(hour_ts):
    active = events[
        (events['start_datetime'] <= hour_ts) &
        (events['end_datetime']   >= hour_ts)
    ]
    n = len(active)
    return pd.Series({
        'is_event_hour'   : int(n > 0),
        'event_count'     : n,
        'max_impact_score': float(active['traffic_impact_score'].max()) if n > 0 else 0.0,
        'total_attendance': int(active['estimated_attendance'].sum())   if n > 0 else 0,
    })

event_feats = merged['hour'].apply(get_event_features)
merged = pd.concat([merged, event_feats], axis=1)
print(f'  Event hours   : {merged["is_event_hour"].sum()} / {len(merged)}')

# ─────────────────────────────────────────────
# 5. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print('\n[5/6] Feature engineering...')

df = merged.sort_values('hour').reset_index(drop=True).copy()

df['hour_of_day']    = df['hour'].dt.hour
df['day_of_week']    = df['hour'].dt.dayofweek
df['day_of_month']   = df['hour'].dt.day
df['month']          = df['hour'].dt.month
df['is_weekend']     = (df['day_of_week'] >= 5).astype(int)
df['is_rush_am']     = ((df['hour_of_day'] >= 7)  & (df['hour_of_day'] <= 9)).astype(int)
df['is_rush_pm']     = ((df['hour_of_day'] >= 16) & (df['hour_of_day'] <= 19)).astype(int)

df['hour_sin']       = np.sin(2 * np.pi * df['hour_of_day'] / 24)
df['hour_cos']       = np.cos(2 * np.pi * df['hour_of_day'] / 24)
df['dow_sin']        = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos']        = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['month_sin']      = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos']      = np.cos(2 * np.pi * df['month'] / 12)

df['is_raining']     = (df['precip'] > 0).astype(int)
df['heavy_rain']     = (df['precip'] > 5).astype(int)
df['is_cold']        = (df['temp'] < 5).astype(int)
df['low_visibility'] = (df['visibility'] < 5).astype(int)

df['lag_1h']         = df['total_volume'].shift(1)
df['lag_2h']         = df['total_volume'].shift(2)
df['lag_24h']        = df['total_volume'].shift(24)
df['rolling_3h']     = df['total_volume'].shift(1).rolling(3).mean()
df['rolling_6h']     = df['total_volume'].shift(1).rolling(6).mean()
df['rolling_24h']    = df['total_volume'].shift(1).rolling(24).mean()

df = df.dropna().reset_index(drop=True)
print(f'  Final dataset : {len(df):,} rows × {df.shape[1]} cols')

# ─────────────────────────────────────────────
# 6. TRAIN / TEST SPLIT + MODELS
# ─────────────────────────────────────────────
print('\n[6/6] Training models...')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

TARGET = 'total_volume'
FEATURE_COLS = [
    'hour_of_day', 'day_of_week', 'day_of_month', 'month',
    'is_weekend', 'is_rush_am', 'is_rush_pm',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    'temp', 'feelslike', 'humidity', 'precip', 'precipprob',
    'windspeed', 'windgust', 'cloudcover', 'visibility', 'sealevelpressure', 'uvindex',
    'is_raining', 'heavy_rain', 'is_cold', 'low_visibility',
    'is_event_hour', 'event_count', 'max_impact_score', 'total_attendance',
    'lag_1h', 'lag_2h', 'lag_24h', 'rolling_3h', 'rolling_6h', 'rolling_24h',
]

X = df[FEATURE_COLS]
y = df[TARGET]

split_date = pd.Timestamp('2023-06-01')
train_mask = df['hour'] < split_date
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

print(f'  Train : {len(X_train):,} rows  |  Test : {len(X_test):,} rows')

MODELS = {
    'Linear Regression' : LinearRegression(),
    'Ridge Regression'  : Ridge(alpha=1.0),
    'Random Forest'     : RandomForestRegressor(n_estimators=200, max_depth=12,
                                                 min_samples_leaf=2, random_state=42, n_jobs=-1),
    'Gradient Boosting' : GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                                      learning_rate=0.05, random_state=42),
    'XGBoost'           : XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                                        subsample=0.8, colsample_bytree=0.8,
                                        random_state=42, verbosity=0),
    'LightGBM'          : LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                                         num_leaves=63, random_state=42, verbose=-1),
}

results = {}
trained  = {}

for name, model in MODELS.items():
    print(f'  {name}...', end=' ', flush=True)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    results[name] = {'RMSE': rmse, 'MAE': mae, 'R²': r2, 'preds': preds}
    trained[name] = model
    print(f'RMSE={rmse:,.0f}  MAE={mae:,.0f}  R²={r2:.4f}')

# ─────────────────────────────────────────────
# RESULTS TABLE
# ─────────────────────────────────────────────
print('\n' + '='*65)
print(' RESULTS — TEST SET (June 2023)')
print('='*65)
res_df = pd.DataFrame({k: {m: v for m, v in v2.items() if m != 'preds'}
                       for k, v2 in results.items()}).T
res_df = res_df.astype(float).round(2).sort_values('R²', ascending=False)
print(res_df.to_string())
best_name = res_df.index[0]
print(f'\n  Best model: {best_name}  R²={res_df.loc[best_name, "R²"]:.4f}')
print('='*65)

# Save CSV for reference
res_df.to_csv(FIGURES / 'model_metrics.csv')

# ─────────────────────────────────────────────
# FIGURE 1 — MODEL COMPARISON
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics_plot = ['RMSE', 'MAE', 'R²']
colors       = ['#e74c3c', '#f39c12', '#27ae60']

for ax, metric, color in zip(axes, metrics_plot, colors):
    data = res_df[metric].sort_values(ascending=(metric != 'R²'))
    bars = ax.barh(data.index, data.values, color=color, alpha=0.85)
    ax.set_title(metric, fontsize=13, fontweight='bold')
    ax.bar_label(bars, fmt='%.0f' if metric != 'R²' else '%.3f', padding=4, fontsize=9)
    if metric == 'R²':
        ax.set_xlim(0, 1.1)

plt.suptitle('Model Comparison — Traffic Volume Prediction (June 2023 test)',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIGURES / 'model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('\n  ✓ model_comparison.png saved')

# ─────────────────────────────────────────────
# FIGURE 2 — FEATURE IMPORTANCE (best tree model)
# ─────────────────────────────────────────────
best_model = trained[best_name]
if hasattr(best_model, 'feature_importances_'):
    imp = pd.Series(best_model.feature_importances_, index=FEATURE_COLS)
    imp = imp.sort_values(ascending=True).tail(20)

    plt.figure(figsize=(10, 7))
    bars = plt.barh(imp.index, imp.values, color='steelblue', alpha=0.85)
    plt.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
    plt.title(f'Top 20 Feature Importances — {best_name}', fontsize=13, fontweight='bold')
    plt.xlabel('Importance (Gini / gain)')
    plt.tight_layout()
    plt.savefig(FIGURES / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ feature_importance.png saved')

    top10 = imp.tail(10)[::-1]
    print('\n  Top 10 features:')
    for feat, val in top10.items():
        print(f'    {feat:<22} {val:.4f}')

# ─────────────────────────────────────────────
# FIGURE 3 — ACTUAL vs PREDICTED (best model, first 7 days)
# ─────────────────────────────────────────────
preds_best = results[best_name]['preds']
test_hours = df.loc[~train_mask, 'hour'].values
n = min(168, len(test_hours))  # 7 days

fig, axes = plt.subplots(2, 1, figsize=(15, 10))

axes[0].plot(test_hours[:n], y_test.values[:n], label='Actual',    lw=1.5)
axes[0].plot(test_hours[:n], preds_best[:n],    label='Predicted', lw=1.5, ls='--', alpha=0.85)
axes[0].set_title(f'Actual vs Predicted — {best_name}  (first 7 days of June 2023)',
                  fontsize=12, fontweight='bold')
axes[0].set_ylabel('Total Volume (vehicles/h)')
axes[0].legend(); axes[0].grid(alpha=0.3)

lim = [min(y_test.min(), preds_best.min()), max(y_test.max(), preds_best.max())]
axes[1].scatter(y_test, preds_best, alpha=0.25, s=8, color='steelblue')
axes[1].plot(lim, lim, 'r--', lw=2, label='Perfect prediction')
axes[1].set_xlabel('Actual Volume');  axes[1].set_ylabel('Predicted Volume')
axes[1].set_title(f'Scatter — R²={res_df.loc[best_name, "R²"]:.4f}',
                  fontsize=12, fontweight='bold')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES / 'actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print('  ✓ actual_vs_predicted.png saved')

# ─────────────────────────────────────────────
# FIGURE 4 — TRAFFIC VOLUME BY HOUR OF DAY
# ─────────────────────────────────────────────
hourly_avg = df.groupby('hour_of_day')['total_volume'].mean()
rush_hours = [7, 8, 9, 16, 17, 18, 19]
colors_bar = ['#1a9de0' if h in rush_hours else '#2c3e6b' for h in hourly_avg.index]

plt.figure(figsize=(12, 5))
plt.bar(hourly_avg.index, hourly_avg.values, color=colors_bar, alpha=0.9, edgecolor='white')
plt.xlabel('Hour of Day'); plt.ylabel('Avg Total Volume (vehicles/h)')
plt.title('Average Traffic Volume by Hour of Day (Jan–Jun 2023)\nBlue = rush hours',
          fontsize=12, fontweight='bold')
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig(FIGURES / 'traffic_by_hour.png', dpi=150, bbox_inches='tight')
plt.close()
print('  ✓ traffic_by_hour.png saved')

# ─────────────────────────────────────────────
# FIGURE 5 — WEEKDAY vs WEEKEND PROFILE
# ─────────────────────────────────────────────
wd = df[df['is_weekend'] == 0].groupby('hour_of_day')['total_volume'].mean()
we = df[df['is_weekend'] == 1].groupby('hour_of_day')['total_volume'].mean()

plt.figure(figsize=(12, 5))
plt.plot(wd.index, wd.values, label='Weekday', lw=2, color='#1a9de0')
plt.plot(we.index, we.values, label='Weekend', lw=2, color='#e74c3c', ls='--')
plt.xlabel('Hour of Day'); plt.ylabel('Avg Total Volume (vehicles/h)')
plt.title('Weekday vs Weekend Traffic Profile', fontsize=12, fontweight='bold')
plt.xticks(range(0, 24)); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES / 'weekday_vs_weekend.png', dpi=150, bbox_inches='tight')
plt.close()
print('  ✓ weekday_vs_weekend.png saved')

# ─────────────────────────────────────────────
# FIGURE 6 — WEATHER IMPACT (rain vs no rain)
# ─────────────────────────────────────────────
rain_effect = df.groupby(['hour_of_day', 'is_raining'])['total_volume'].mean().unstack()
if 0 in rain_effect.columns and 1 in rain_effect.columns:
    plt.figure(figsize=(12, 5))
    plt.plot(rain_effect.index, rain_effect[0], label='No rain', lw=2, color='#27ae60')
    plt.plot(rain_effect.index, rain_effect[1], label='Rain',    lw=2, color='#3498db', ls='--')
    plt.xlabel('Hour of Day'); plt.ylabel('Avg Total Volume')
    plt.title('Traffic Volume: Rain vs No Rain', fontsize=12, fontweight='bold')
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / 'rain_effect.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ rain_effect.png saved')

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print('\n' + '='*65)
print(' SUMMARY FOR PAPER')
print('='*65)
print(f'\n  Authors       : Jules Francois & Quentin Chofflet')
print(f'  Module        : CA683 — Dublin City University')
print(f'\n  Dataset sizes (after cleaning):')
print(f'    Traffic     : {len(traffic):>10,} rows  (6 monthly SCATS files)')
print(f'    Weather     : {len(weather):>10,} rows  (hourly, Jan–Jun 2023)')
print(f'    Events      : {len(events):>10,} rows  (synthetic — 247 events)')
print(f'\n  After fusion  : {len(df):,} hourly observations')
print(f'  Features      : {len(FEATURE_COLS)} features')
print(f'  Train (Jan–May) : {train_mask.sum():,} rows')
print(f'  Test (June)   : {(~train_mask).sum():,} rows')
print(f'\n  Model Results:')
for idx, row in res_df.iterrows():
    marker = ' ← BEST' if idx == best_name else ''
    print(f'    {idx:<22} RMSE={row["RMSE"]:>8,.0f}  MAE={row["MAE"]:>8,.0f}  R²={row["R²"]:.4f}{marker}')
print(f'\n  Figures saved in: reports/figures/')
print('='*65)
