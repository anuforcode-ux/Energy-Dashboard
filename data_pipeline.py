# ============================================================
#   data_pipeline.py
#   Loads, cleans, resamples, and feature-engineers the
#   UCI Household Power Consumption dataset.
#   Used by energy_project_part2.py AND dashboard.py so both
#   see identical features (same code path = no train/serve skew).
# ============================================================

import pandas as pd
import numpy as np

DATA_FILE = 'household_power_consumption.txt'

FEATURE_COLS = [
    'hour', 'day_of_week', 'month', 'is_weekend',
    'is_morning', 'is_evening', 'season',
    'lag_1h', 'lag_24h', 'lag_168h',
    'rolling_24h_mean', 'temperature'
]


def load_raw(path=DATA_FILE):
    """Load the raw UCI txt file. Falls back to synthetic data if missing."""
    try:
        df = pd.read_csv(
            path, sep=';', low_memory=False, na_values=['?']
        )
        df['datetime'] = pd.to_datetime(
            df['Date'] + ' ' + df['Time'], dayfirst=True
        )
        df.set_index('datetime', inplace=True)
        df.drop(columns=['Date', 'Time'], inplace=True)
        return df
    except FileNotFoundError:
        np.random.seed(42)
        dates = pd.date_range('2007-01-01', '2010-12-31', freq='H')
        hour_pattern = np.array([0.3,0.2,0.2,0.2,0.3,0.5,0.8,1.2,1.0,0.8,
                                  0.7,0.8,1.0,0.9,0.8,0.9,1.1,1.5,1.8,1.6,
                                  1.4,1.2,0.9,0.5])
        month_pattern = np.array([1.3,1.2,1.0,0.9,0.8,0.9,1.1,1.2,0.9,0.8,1.0,1.3])
        power = [max(0.1, hour_pattern[d.hour]*month_pattern[d.month-1] + np.random.normal(0,0.15))
                 for d in dates]
        return pd.DataFrame({'Global_active_power': power}, index=dates)


def clean_and_resample(df):
    """Keep Global_active_power, coerce numeric, ffill/bfill, resample hourly + daily."""
    if 'Global_active_power' in df.columns:
        df = df[['Global_active_power']].copy()
    else:
        df = df.iloc[:, :1].copy()
        df.columns = ['Global_active_power']

    df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
    df['Global_active_power'] = df['Global_active_power'].ffill().bfill()

    df_hourly = df.resample('h').mean()
    df_daily = df.resample('D').mean()
    return df_hourly, df_daily


def _season(month):
    if month in (12, 1, 2):
        return 0
    if month in (3, 4, 5):
        return 1
    if month in (6, 7, 8):
        return 2
    return 3


def engineer_features(df_hourly):
    """Same feature set as Part 1 (Step 4), factored out for reuse."""
    df_model = df_hourly.copy()
    df_model['hour'] = df_model.index.hour
    df_model['day_of_week'] = df_model.index.dayofweek
    df_model['month'] = df_model.index.month
    df_model['is_weekend'] = (df_model['day_of_week'] >= 5).astype(int)
    df_model['is_morning'] = ((df_model['hour'] >= 6) & (df_model['hour'] <= 9)).astype(int)
    df_model['is_evening'] = ((df_model['hour'] >= 18) & (df_model['hour'] <= 22)).astype(int)
    df_model['season'] = df_model['month'].apply(_season)

    df_model['lag_1h'] = df_model['Global_active_power'].shift(1)
    df_model['lag_24h'] = df_model['Global_active_power'].shift(24)
    df_model['lag_168h'] = df_model['Global_active_power'].shift(168)
    df_model['rolling_24h_mean'] = df_model['Global_active_power'].rolling(24).mean()

    np.random.seed(42)
    month_temps = {1: 8, 2: 9, 3: 13, 4: 17, 5: 21, 6: 25, 7: 28, 8: 27,
                   9: 23, 10: 18, 11: 13, 12: 9}
    df_model['temperature'] = (df_model['month'].map(month_temps)
                                + np.random.normal(0, 3, len(df_model)))

    df_model.dropna(inplace=True)
    return df_model


def chronological_split(df_model, feature_cols=FEATURE_COLS, train_frac=0.80):
    """80/20 split preserving time order (never shuffle time series data)."""
    X = df_model[feature_cols]
    y = df_model['Global_active_power']
    split_idx = int(len(X) * train_frac)
    return (X.iloc[:split_idx], X.iloc[split_idx:],
            y.iloc[:split_idx], y.iloc[split_idx:], split_idx)


def load_pipeline(path=DATA_FILE):
    """One-call convenience wrapper: raw file -> ready-to-model dataframe."""
    raw = load_raw(path)
    df_hourly, df_daily = clean_and_resample(raw)
    df_model = engineer_features(df_hourly)
    X_train, X_test, y_train, y_test, split_idx = chronological_split(df_model)
    return {
        'df_hourly': df_hourly, 'df_daily': df_daily, 'df_model': df_model,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'split_idx': split_idx, 'feature_cols': FEATURE_COLS,
    }


if __name__ == '__main__':
    data = load_pipeline()
    print(f"df_model shape: {data['df_model'].shape}")
    print(f"Train: {len(data['X_train']):,}  Test: {len(data['X_test']):,}")
