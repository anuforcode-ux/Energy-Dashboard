# ============================================================
#   anomaly_detector.py
#   Combines two complementary signals per Slide 4/5:
#     1. Z-Score on (actual - predicted) residual  -> "how far off was
#        the model, relative to its usual error?"
#     2. Isolation Forest on the raw feature space -> "does this reading
#        look unusual on its own terms (time of day, season, etc.),
#        independent of any one model's prediction?"
#   Flagging BOTH catches more real waste and fewer one-off model quirks
#   than either signal alone.
# ============================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def zscore_anomalies(y_true, y_pred, index, threshold=2.0):
    """Anomaly = residual more than `threshold` std devs from the mean residual."""
    residual = y_true - y_pred
    z = (residual - residual.mean()) / residual.std()
    return pd.DataFrame({
        'actual': y_true, 'predicted': y_pred,
        'residual': residual, 'z_score': z,
        'is_anomaly_zscore': z.abs() > threshold,
    }, index=index)


def isolation_forest_anomalies(df_model, feature_cols, contamination=0.02):
    """
    Anomaly = flagged as an outlier by Isolation Forest over the full
    feature space (not just the target). contamination = expected
    fraction of anomalous rows (2% is a reasonable starting point;
    tune against known spike days per Slide 8's "threshold tuning" challenge).
    """
    iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    X = df_model[feature_cols + ['Global_active_power']]
    preds = iso.fit_predict(X)  # -1 = anomaly, 1 = normal
    scores = iso.decision_function(X)  # lower = more anomalous
    return pd.DataFrame({
        'is_anomaly_isoforest': preds == -1,
        'isoforest_score': scores,
    }, index=df_model.index)


def combined_anomalies(y_true, y_pred, df_model, feature_cols,
                        zscore_threshold=2.0, contamination=0.02,
                        require_both=False):
    """
    Merges both signals on their shared index.
    require_both=False (default): flag if EITHER method flags it (higher recall,
        catches more real waste, more false positives — good for a first pass).
    require_both=True: flag only if BOTH agree (higher precision, fewer alerts,
        better once you're tuning down false-alarm rate for a production dashboard).
    """
    z_df = zscore_anomalies(y_true, y_pred, y_true.index if hasattr(y_true, 'index') else None,
                             threshold=zscore_threshold)
    iso_df = isolation_forest_anomalies(df_model, feature_cols, contamination=contamination)

    merged = z_df.join(iso_df, how='inner')
    if require_both:
        merged['is_anomaly'] = merged['is_anomaly_zscore'] & merged['is_anomaly_isoforest']
    else:
        merged['is_anomaly'] = merged['is_anomaly_zscore'] | merged['is_anomaly_isoforest']
    return merged


if __name__ == '__main__':
    import sys, os
    sys.path.append(os.path.dirname(__file__))
    from data_pipeline import load_pipeline
    from sklearn.ensemble import RandomForestRegressor

    data = load_pipeline()
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(data['X_train'], data['y_train'])
    pred = pd.Series(rf.predict(data['X_test']), index=data['y_test'].index)

    result = combined_anomalies(data['y_test'], pred, data['df_model'].loc[data['X_test'].index],
                                 data['feature_cols'])
    print(f"Flagged {result['is_anomaly'].sum()} / {len(result)} hours "
          f"({result['is_anomaly'].mean()*100:.2f}%)")
    print(result[result['is_anomaly']].head())
