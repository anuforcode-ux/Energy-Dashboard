# ============================================================
#   HOUSEHOLD ENERGY CONSUMPTION FORECASTING - FULL PROGRAM
#   CS5403 - Machine Learning | PBL Project (100%)
#   Combines: EDA, Linear Regression, Random Forest, LSTM,
#             Prophet, Anomaly Detection + Alerts, Final Comparison
#
#   Base installs : pip install pandas numpy matplotlib seaborn scikit-learn
#   Extra installs: pip install tensorflow prophet --break-system-packages
#   (LSTM/Prophet sections auto-skip with a clear message if not installed)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  HOUSEHOLD ENERGY CONSUMPTION FORECASTING - FULL PROGRAM")
print("=" * 60)

# ============================================================
# STEP 1: LOAD DATASET
# ============================================================
print("\n[STEP 1] Loading Dataset...")

try:
    df = pd.read_csv(
        'household_power_consumption.txt', sep=';',
        low_memory=False, na_values=['?']
    )
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
    df.set_index('datetime', inplace=True)
    df.drop(columns=['Date', 'Time'], inplace=True)
    print(f"  Dataset loaded! Shape: {df.shape}")
except FileNotFoundError:
    print("  Dataset file not found! Generating sample data for demo...")
    np.random.seed(42)
    dates = pd.date_range(start='2007-01-01', end='2010-12-31', freq='h')
    hour_pattern = np.array([0.3,0.2,0.2,0.2,0.3,0.5,0.8,1.2,1.0,0.8,
                              0.7,0.8,1.0,0.9,0.8,0.9,1.1,1.5,1.8,1.6,
                              1.4,1.2,0.9,0.5])
    month_pattern = np.array([1.3,1.2,1.0,0.9,0.8,0.9])
    power = [max(0.1, hour_pattern[d.hour]*month_pattern[d.month-1] + np.random.normal(0,0.15))
             for d in dates]
    df = pd.DataFrame({'Global_active_power': power}, index=dates)
    print(f"  Sample data generated! Shape: {df.shape}")

# ============================================================
# STEP 2: DATA CLEANING & PREPROCESSING
# ============================================================
print("\n[STEP 2] Cleaning & Preprocessing Data...")

if 'Global_active_power' in df.columns:
    df = df[['Global_active_power']].copy()
else:
    df = df.iloc[:, :1].copy()
    df.columns = ['Global_active_power']

df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')

missing = df.isnull().sum().values[0]
total = len(df)
print(f"  Total rows     : {total:,}")
print(f"  Missing values : {missing:,} ({missing/total*100:.2f}%)")

df['Global_active_power'] = df['Global_active_power'].ffill().bfill()
print(f"  Missing values after cleaning: {df.isnull().sum().values[0]}")

df_hourly = df.resample('h').mean()
df_daily = df.resample('D').mean()
print(f"  Resampled to hourly: {df_hourly.shape[0]:,} hourly records")
print(f"  Resampled to daily : {df_daily.shape[0]:,} daily records")

# ============================================================
# STEP 3: EDA - EXPLORATORY DATA ANALYSIS (GRAPHS)
# ============================================================
print("\n[STEP 3] Drawing EDA Graphs...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Exploratory Data Analysis - Energy Consumption Patterns',
             fontsize=16, fontweight='bold', y=1.02)

axes[0, 0].plot(df_daily.index, df_daily['Global_active_power'],
                color='steelblue', linewidth=0.8, alpha=0.8)
axes[0, 0].set_title('Daily Energy Consumption Over Time', fontweight='bold')
axes[0, 0].set_xlabel('Date'); axes[0, 0].set_ylabel('Power (kWh)')
axes[0, 0].grid(True, alpha=0.3)

hourly_tmp = df_hourly.copy()
hourly_tmp['hour'] = hourly_tmp.index.hour
hourly_avg = hourly_tmp.groupby('hour')['Global_active_power'].mean()
axes[0, 1].bar(hourly_avg.index, hourly_avg.values, color='coral', edgecolor='darkred', alpha=0.8)
axes[0, 1].set_title('Average Usage by Hour of Day', fontweight='bold')
axes[0, 1].set_xlabel('Hour'); axes[0, 1].set_ylabel('Avg Power (kWh)')
axes[0, 1].grid(True, alpha=0.3, axis='y')

daily_tmp = df_daily.copy()
daily_tmp['month'] = daily_tmp.index.month
monthly_avg = daily_tmp.groupby('month')['Global_active_power'].mean()
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
axes[1, 0].bar(month_names, monthly_avg.values, color='mediumseagreen', edgecolor='darkgreen', alpha=0.8)
axes[1, 0].set_title('Average Usage by Month (Seasonal Pattern)', fontweight='bold')
axes[1, 0].set_xlabel('Month'); axes[1, 0].set_ylabel('Avg Power (kWh)')
axes[1, 0].grid(True, alpha=0.3, axis='y')

daily_tmp['dayofweek'] = daily_tmp.index.dayofweek
day_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
dayofweek_avg = daily_tmp.groupby('dayofweek')['Global_active_power'].mean()
colors = ['steelblue']*5 + ['salmon']*2
axes[1, 1].bar(day_names, dayofweek_avg.values, color=colors, edgecolor='black', alpha=0.8)
axes[1, 1].set_title('Average Usage by Day of Week', fontweight='bold')
axes[1, 1].set_xlabel('Day'); axes[1, 1].set_ylabel('Avg Power (kWh)')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('eda_graphs.png', dpi=150, bbox_inches='tight')
plt.show()
print("  EDA graphs saved as 'eda_graphs.png'")

# ============================================================
# STEP 4: FEATURE ENGINEERING
# ============================================================
print("\n[STEP 4] Feature Engineering...")

df_model = df_hourly.copy()
df_model['hour'] = df_model.index.hour
df_model['day_of_week'] = df_model.index.dayofweek
df_model['month'] = df_model.index.month
df_model['is_weekend'] = (df_model['day_of_week'] >= 5).astype(int)
df_model['is_morning'] = ((df_model['hour'] >= 6) & (df_model['hour'] <= 9)).astype(int)
df_model['is_evening'] = ((df_model['hour'] >= 18) & (df_model['hour'] <= 22)).astype(int)

def get_season(month):
    if month in [12, 1, 2]: return 0
    elif month in [3, 4, 5]: return 1
    elif month in [6, 7, 8]: return 2
    else: return 3

df_model['season'] = df_model['month'].apply(get_season)
df_model['lag_1h'] = df_model['Global_active_power'].shift(1)
df_model['lag_24h'] = df_model['Global_active_power'].shift(24)
df_model['lag_168h'] = df_model['Global_active_power'].shift(168)
df_model['rolling_24h_mean'] = df_model['Global_active_power'].rolling(24).mean()

np.random.seed(42)
month_temps = {1:8, 2:9, 3:13, 4:17, 5:21, 6:25, 7:28, 8:27, 9:23, 10:18, 11:13, 12:9}
df_model['temperature'] = df_model['month'].map(month_temps) + np.random.normal(0, 3, len(df_model))

df_model.dropna(inplace=True)

feature_cols = ['hour', 'day_of_week', 'month', 'is_weekend',
                'is_morning', 'is_evening', 'season',
                'lag_1h', 'lag_24h', 'lag_168h',
                'rolling_24h_mean', 'temperature']

print(f"  Features created: {feature_cols}")
print(f"  Dataset shape after feature engineering: {df_model.shape}")

# ============================================================
# STEP 5: TRAIN/TEST SPLIT (chronological — never shuffle time series)
# ============================================================
print("\n[STEP 5] Splitting Data into Train and Test...")

X = df_model[feature_cols]
y = df_model['Global_active_power']
split_idx = int(len(X) * 0.80)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
print(f"  Training set : {len(X_train):,} rows (first 80%)")
print(f"  Testing set  : {len(X_test):,} rows (last 20%)")

# ============================================================
# STEP 6: LINEAR REGRESSION (BASELINE MODEL)
# ============================================================
print("\n[STEP 6] Training Linear Regression (Baseline Model)...")

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_mae = mean_absolute_error(y_test, lr_pred)
print(f"  RMSE : {lr_rmse:.4f} kWh   MAE : {lr_mae:.4f} kWh")

# ============================================================
# STEP 7: RANDOM FOREST MODEL
# ============================================================
print("\n[STEP 7] Training Random Forest Model...")

rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_mae = mean_absolute_error(y_test, rf_pred)
print(f"  RMSE : {rf_rmse:.4f} kWh   MAE : {rf_mae:.4f} kWh")
print(f"  Improvement over Linear Regression: {((lr_rmse-rf_rmse)/lr_rmse)*100:.1f}%")

# ============================================================
# STEP 8: LSTM MODEL  (deep learning — needs tensorflow)
# ============================================================
print("\n[STEP 8] Training LSTM Model...")

lstm_ok = True
lstm_rmse = lstm_mae = None
WINDOW = 24

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler

    tf.random.set_seed(42)

    all_cols = feature_cols + ['Global_active_power']
    data_arr = df_model[all_cols].values.astype('float32')

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    features_scaled = feature_scaler.fit_transform(data_arr[:, :-1])
    target_scaled = target_scaler.fit_transform(data_arr[:, -1:]).ravel()

    def make_sequences(feat, targ, window=WINDOW):
        Xs, ys = [], []
        for i in range(window, len(feat)):
            Xs.append(feat[i-window:i])
            ys.append(targ[i])
        return np.array(Xs), np.array(ys)

    X_seq, y_seq = make_sequences(features_scaled, target_scaled)
    seq_split = split_idx - WINDOW
    X_train_seq, X_test_seq = X_seq[:seq_split], X_seq[seq_split:]
    y_train_seq, y_test_seq = y_seq[:seq_split], y_seq[seq_split:]

    lstm_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(WINDOW, len(feature_cols))),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1),
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    print("  Training... (this may take a few minutes)")
    lstm_model.fit(X_train_seq, y_train_seq, validation_split=0.1,
                   epochs=15, batch_size=64, callbacks=[early_stop], verbose=1)

    lstm_pred_scaled = lstm_model.predict(X_test_seq, verbose=0).ravel()
    lstm_y_test = target_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).ravel()
    lstm_pred = target_scaler.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).ravel()

    lstm_rmse = np.sqrt(mean_squared_error(lstm_y_test, lstm_pred))
    lstm_mae = mean_absolute_error(lstm_y_test, lstm_pred)
    print(f"  RMSE : {lstm_rmse:.4f} kWh   MAE : {lstm_mae:.4f} kWh")

except ImportError:
    lstm_ok = False
    print("  tensorflow not installed — skipping LSTM.")
    print("  Install with: pip install tensorflow --break-system-packages")

# ============================================================
# STEP 9: PROPHET MODEL  (seasonal forecasting — needs prophet)
# ============================================================
print("\n[STEP 9] Training Prophet Model...")

prophet_ok = True
prophet_rmse = prophet_mae = None

try:
    from prophet import Prophet

    prophet_df = df_daily.reset_index()[['datetime', 'Global_active_power']]
    prophet_df.columns = ['ds', 'y']
    prophet_df = prophet_df.dropna()

    p_split = int(len(prophet_df) * 0.80)
    p_train, p_test = prophet_df.iloc[:p_split], prophet_df.iloc[p_split:]

    prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                             daily_seasonality=False, interval_width=0.95)
    prophet_model.add_country_holidays(country_name='IN')  # swap to match your data's country if needed
    prophet_model.fit(p_train)

    future = prophet_model.make_future_dataframe(periods=len(p_test), freq='D')
    forecast = prophet_model.predict(future)
    forecast_test = forecast.iloc[p_split:].reset_index(drop=True)

    prophet_y_test = p_test['y'].values
    prophet_pred = forecast_test['yhat'].values[:len(prophet_y_test)]
    prophet_lower = forecast_test['yhat_lower'].values[:len(prophet_y_test)]
    prophet_upper = forecast_test['yhat_upper'].values[:len(prophet_y_test)]
    prophet_dates = p_test['ds'].values

    prophet_rmse = np.sqrt(mean_squared_error(prophet_y_test, prophet_pred))
    prophet_mae = mean_absolute_error(prophet_y_test, prophet_pred)
    print(f"  RMSE : {prophet_rmse:.4f} kWh   MAE : {prophet_mae:.4f} kWh  (daily scale)")

except ImportError:
    prophet_ok = False
    print("  prophet not installed — skipping Prophet.")
    print("  Install with: pip install prophet --break-system-packages")

# ============================================================
# STEP 10: FULL ANOMALY DETECTION (Isolation Forest + Z-Score)
# ============================================================
print("\n[STEP 10] Running Full Anomaly Detection...")

# Signal 1: Z-score on Random Forest residuals
residual = y_test - rf_pred
z_score = (residual - residual.mean()) / residual.std()
is_anomaly_z = z_score.abs() > 2.0

# Signal 2: Isolation Forest on the raw feature space
iso = IsolationForest(contamination=0.02, random_state=42, n_jobs=-1)
iso_input = df_model.loc[X_test.index, feature_cols + ['Global_active_power']]
iso_preds = iso.fit_predict(iso_input)
is_anomaly_iso = pd.Series(iso_preds == -1, index=X_test.index)

anomalies = pd.DataFrame({
    'actual': y_test, 'predicted': rf_pred, 'residual': residual,
    'z_score': z_score, 'is_anomaly_zscore': is_anomaly_z,
    'is_anomaly_isoforest': is_anomaly_iso,
}, index=y_test.index)
anomalies['is_anomaly'] = anomalies['is_anomaly_zscore'] | anomalies['is_anomaly_isoforest']

n_flagged = anomalies['is_anomaly'].sum()
print(f"  Flagged {n_flagged} / {len(anomalies)} hours ({n_flagged/len(anomalies)*100:.2f}%)")

# ── Plain-language alert generator ──
DAY_NAMES = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

def hour_label(h):
    if h == 0: return '12AM'
    if h < 12: return f'{h}AM'
    if h == 12: return '12PM'
    return f'{h-12}PM'

def likely_cause(row, hour):
    ratio = row['actual'] / row['predicted'] if row['predicted'] > 0 else float('inf')
    if 18 <= hour <= 22: return 'possible AC / heater left running'
    if 6 <= hour <= 9: return 'possible high-draw morning appliance use (kettle, heater, dryer)'
    if 0 <= hour <= 5: return 'unusual — most appliances should be idle at this hour, worth checking'
    if ratio > 3: return 'large unexplained spike — check for an appliance left on'
    return 'unusual usage pattern for this time of day'

def generate_alert(row, ts):
    day = DAY_NAMES[ts.dayofweek]
    hl = hour_label(ts.hour)
    ratio = row['actual'] / row['predicted'] if row['predicted'] > 0 else float('inf')
    cause = likely_cause(row, ts.hour)
    if ratio >= 1:
        return (f"Usage on {day} {hl} was {ratio:.1f}x higher than predicted "
                f"({row['actual']:.2f} kWh vs {row['predicted']:.2f} kWh expected) — {cause}.")
    drop_pct = (1 - ratio) * 100
    return (f"Usage on {day} {hl} was {drop_pct:.0f}% lower than predicted "
            f"({row['actual']:.2f} kWh vs {row['predicted']:.2f} kWh expected) — "
            f"possibly away from home or an appliance not running as scheduled.")

flagged = anomalies[anomalies['is_anomaly']].copy()
flagged['abs_residual'] = flagged['residual'].abs()
top_alerts = flagged.sort_values('abs_residual', ascending=False).head(10)

print(f"\n  Top {len(top_alerts)} waste alerts:")
for ts, row in top_alerts.iterrows():
    print(f"    [{ts}] {generate_alert(row, ts)}")

anomalies.to_csv('anomalies_output.csv')
print("\n  Full anomaly table saved to 'anomalies_output.csv'")

# Anomaly graph (last 30 days of test set)
fig, ax = plt.subplots(figsize=(14, 5))
sample_anomaly = anomalies.tail(720)
ax.plot(sample_anomaly.index, sample_anomaly['actual'], label='Actual', color='black', linewidth=1, alpha=0.8)
ax.plot(sample_anomaly.index, sample_anomaly['predicted'], label='Predicted (RF)', color='blue',
        linewidth=1, linestyle='--', alpha=0.7)
anomalies_in_sample = sample_anomaly[sample_anomaly['is_anomaly']]
ax.scatter(anomalies_in_sample.index, anomalies_in_sample['actual'], color='red', s=40,
           zorder=5, label='⚠️ Anomaly (Possible Waste)')
ax.set_title('Anomaly Detection — Red Dots = Unusual Energy Spikes', fontweight='bold', fontsize=13)
ax.set_xlabel('Date'); ax.set_ylabel('Power (kWh)')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('anomaly_detection.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Anomaly detection graph saved as 'anomaly_detection.png'")

# ============================================================
# STEP 11: FINAL MODEL COMPARISON GRAPH (all trained models)
# ============================================================
print("\n[STEP 11] Generating Final Model Comparison...")

models_list = ['Linear Regression', 'Random Forest']
rmses_list = [lr_rmse, rf_rmse]
maes_list = [lr_mae, rf_mae]
if lstm_ok:
    models_list.append('LSTM')
    rmses_list.append(lstm_rmse)
    maes_list.append(lstm_mae)

fig, ax = plt.subplots(figsize=(9, 5))
x_pos = np.arange(len(models_list))
width = 0.35
bars1 = ax.bar(x_pos - width/2, rmses_list, width, label='RMSE', color='tomato', edgecolor='black')
bars2 = ax.bar(x_pos + width/2, maes_list, width, label='MAE', color='steelblue', edgecolor='black')
for bars in (bars1, bars2):
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
ax.set_xticks(x_pos); ax.set_xticklabels(models_list)
ax.set_ylabel('Error (kWh)')
ax.set_title('Final Model Comparison — Hourly Forecasting', fontweight='bold', fontsize=13)
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('final_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved 'final_model_comparison.png'")

if prophet_ok:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(prophet_dates, prophet_y_test, label='Actual', color='black', linewidth=1.5)
    ax.plot(prophet_dates, prophet_pred, label='Prophet Forecast', color='purple',
            linewidth=1.2, linestyle='--')
    ax.fill_between(prophet_dates, prophet_lower, prophet_upper, color='purple', alpha=0.15,
                     label='95% Confidence Interval')
    ax.set_title(f"Prophet Forecast (Daily) — RMSE: {prophet_rmse:.4f} | MAE: {prophet_mae:.4f}",
                 fontweight='bold')
    ax.set_xlabel('Date'); ax.set_ylabel('Power (kWh)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('prophet_forecast.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved 'prophet_forecast.png'")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("  PROJECT SUMMARY - FULL PROGRAM (100%)")
print("=" * 60)
print(f"\n  Dataset         : UCI Household Power Consumption")
print(f"  Total Records   : {len(df_model):,} hourly readings")
print(f"  Features Used   : {len(feature_cols)}")
print(f"\n  MODEL RESULTS:")
for name, rmse, mae in zip(models_list, rmses_list, maes_list):
    print(f"    {name:<20} RMSE: {rmse:.4f}   MAE: {mae:.4f}")
if prophet_ok:
    print(f"    {'Prophet (daily)':<20} RMSE: {prophet_rmse:.4f}   MAE: {prophet_mae:.4f}")
print(f"\n  ANOMALIES DETECTED : {n_flagged} / {len(anomalies)} hours ({n_flagged/len(anomalies)*100:.2f}%)")
print(f"  ALERTS GENERATED   : {len(top_alerts)} (top by severity)")
print(f"\n  FILES SAVED:")
print(f"  eda_graphs.png            - 4 EDA analysis graphs")
print(f"  anomaly_detection.png     - Anomaly flags graph")
print(f"  final_model_comparison.png- RMSE/MAE across all models")
if prophet_ok:
    print(f"  prophet_forecast.png      - Prophet forecast with confidence band")
print(f"  anomalies_output.csv      - Full anomaly + alert data")
print(f"\n  DASHBOARD: run 'streamlit run dashboard.py' separately")
print(f"  (Streamlit apps can't run inside this script — see note below)")
print(f"\n{'=' * 60}")
print("  CODE EXECUTION COMPLETE!")
print("=" * 60)
