# ============================================================
#   alert_generator.py
#   Turns a flagged anomaly row into a plain-language waste alert,
#   e.g. "Usage on Tuesday 8PM was 2.8x higher than predicted -
#   possible AC left running" (the exact example from Slide 7).
#
#   This is intentionally simple rule-based logic, not NLP —
#   Slide 8 flags "rule-based NLP logic" as a challenge; this is
#   the rule-based half of that, a reasonable place to stop for
#   a PBL-scope project. Swap in a template-ranking model later
#   if you want to go further.
# ============================================================

import pandas as pd

DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


def _hour_label(hour):
    if hour == 0:
        return '12AM'
    if hour < 12:
        return f'{hour}AM'
    if hour == 12:
        return '12PM'
    return f'{hour - 12}PM'


def _likely_cause(row, hour):
    """Very rough heuristic classification of *why* a spike might have happened,
    based only on time-of-day — good enough for a first-pass alert, not a diagnosis."""
    ratio = row['actual'] / row['predicted'] if row['predicted'] > 0 else float('inf')
    if 18 <= hour <= 22:
        return 'possible AC / heater left running'
    if 6 <= hour <= 9:
        return 'possible high-draw morning appliance use (kettle, heater, dryer)'
    if 0 <= hour <= 5:
        return 'unusual — most appliances should be idle at this hour, worth checking'
    if ratio > 3:
        return 'large unexplained spike — check for an appliance left on'
    return 'unusual usage pattern for this time of day'


def generate_alert(row, timestamp):
    """
    row: a row from anomaly_detector's combined_anomalies() output
         (needs 'actual', 'predicted' columns).
    timestamp: the pandas Timestamp index for this row.
    Returns a human-readable alert string.
    """
    day = DAY_NAMES[timestamp.dayofweek]
    hour_label = _hour_label(timestamp.hour)
    ratio = row['actual'] / row['predicted'] if row['predicted'] > 0 else float('inf')
    cause = _likely_cause(row, timestamp.hour)

    if ratio >= 1:
        return (f"Usage on {day} {hour_label} was {ratio:.1f}x higher than predicted "
                f"({row['actual']:.2f} kWh vs {row['predicted']:.2f} kWh expected) — {cause}.")
    else:
        drop_pct = (1 - ratio) * 100
        return (f"Usage on {day} {hour_label} was {drop_pct:.0f}% lower than predicted "
                f"({row['actual']:.2f} kWh vs {row['predicted']:.2f} kWh expected) — "
                f"possibly away from home or an appliance not running as scheduled.")


def generate_all_alerts(anomaly_df, top_n=None):
    """
    anomaly_df: output of anomaly_detector.combined_anomalies(), filtered to
                is_anomaly == True rows (or pass the full df; this filters for you).
    Returns a list of (timestamp, alert_string) tuples, sorted by severity
    (largest absolute residual first).
    """
    flagged = anomaly_df[anomaly_df['is_anomaly']].copy()
    flagged['abs_residual'] = flagged['residual'].abs()
    flagged = flagged.sort_values('abs_residual', ascending=False)
    if top_n:
        flagged = flagged.head(top_n)

    alerts = []
    for ts, row in flagged.iterrows():
        alerts.append((ts, generate_alert(row, ts)))
    return alerts


if __name__ == '__main__':
    import sys, os
    sys.path.append(os.path.dirname(__file__))
    from data_pipeline import load_pipeline
    from anomaly_detector import combined_anomalies
    from sklearn.ensemble import RandomForestRegressor

    data = load_pipeline()
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(data['X_train'], data['y_train'])
    pred = pd.Series(rf.predict(data['X_test']), index=data['y_test'].index)

    anomalies = combined_anomalies(data['y_test'], pred,
                                    data['df_model'].loc[data['X_test'].index],
                                    data['feature_cols'])
    alerts = generate_all_alerts(anomalies, top_n=5)
    print(f"Top {len(alerts)} alerts:\n")
    for ts, msg in alerts:
        print(f"  [{ts}] {msg}")
