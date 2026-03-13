import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from datetime import timedelta

# ============================================================
# ML-①  ANOMALY DETECTION — Isolation Forest
# ============================================================
def detect_anomalies(all_ds):
    def build_feature_vector(ds):
        m  = ds['metrics']
        tf = ds['top_freqs']
        # pad top_freqs ถ้าน้อยกว่า 5
        freqs = [tf[i][0] if i < len(tf) else 0.0 for i in range(5)]
        mags  = [tf[i][1] if i < len(tf) else 0.0 for i in range(5)]
        return [
            m['rms'], m['peak'], m['p2p'], m['std'],
            m['crest'], m['kurt'], m['skew'],
            *freqs, *mags
        ]

    feat_names = (['RMS','Peak','P2P','Std','CF','Kurtosis','Skewness']
                  + [f'Freq{i+1}' for i in range(5)]
                  + [f'Mag{i+1}'  for i in range(5)])

    X_all    = np.array([build_feature_vector(ds) for ds in all_ds])
    labels_t = [ds['label'] for ds in all_ds]

    iso = IsolationForest(contamination=0.1, random_state=42, n_estimators=200)
    iso.fit(X_all)
    scores    = iso.decision_function(X_all)   # ยิ่งต่ำ = ยิ่งแปลก
    preds     = iso.predict(X_all)             # +1 = ปกติ, -1 = anomaly
    anomaly_labels = ['🔴 ANOMALY' if p == -1 else '✅ Normal' for p in preds]

    df_anomaly = pd.DataFrame({
        'Period':         labels_t,
        'Anomaly Score':  [round(s, 5) for s in scores],
        'Status':         anomaly_labels,
        'CF (clean)':     [round(ds['metrics']['crest'], 4) for ds in all_ds],
        'Kurtosis':       [round(ds['metrics']['kurt'],  4) for ds in all_ds],
        'RMS (G)':        [round(ds['metrics']['rms'],   5) for ds in all_ds],
        'Peak (G)':       [round(ds['metrics']['peak'],  5) for ds in all_ds],
    })

    return scores, preds, anomaly_labels, df_anomaly, iso, X_all, feat_names


# ============================================================
# ML-②  CF TREND FORECAST  —  Linear + Polynomial Regression
# ============================================================
def forecast_trend(all_ds, iso_zones_config):
    cf_vals  = np.array([ds['metrics']['crest'] for ds in all_ds], dtype=float)
    x_idx    = np.arange(len(all_ds), dtype=float)
    dt_objs  = [ds['dt_obj'] for ds in all_ds]

    if len(dt_objs) >= 2:
        total_days = (dt_objs[-1] - dt_objs[0]).days
        avg_interval_days = total_days / (len(dt_objs) - 1)
    else:
        avg_interval_days = 90

    lin_reg = LinearRegression()
    lin_reg.fit(x_idx.reshape(-1,1), cf_vals)
    slope = float(lin_reg.coef_[0])
    intercept = float(lin_reg.intercept_)

    poly_coeffs = np.polyfit(x_idx, cf_vals, deg=min(2, len(all_ds)-1))
    poly_fn     = np.poly1d(poly_coeffs)

    n_future  = 4
    x_future  = np.arange(len(all_ds), len(all_ds) + n_future, dtype=float)
    cf_lin_f  = lin_reg.predict(x_future.reshape(-1,1))
    cf_poly_f = poly_fn(x_future)

    future_dates = [dt_objs[-1] + timedelta(days=avg_interval_days*(i+1))
                    for i in range(n_future)]

    forecast_rows = []
    # historical
    for ds, cf_l, cf_p in zip(all_ds, lin_reg.predict(x_idx.reshape(-1,1)), poly_fn(x_idx)):
        forecast_rows.append({
            'Period': ds['label'], 'Type': 'Actual',
            'Date': ds['meta'].get('datetime','-'),
            'CF Actual': round(ds['metrics']['crest'],4),
            'CF Linear Fit': round(float(cf_l),4),
            'CF Poly Fit':   round(float(cf_p),4),
            'Zone (Linear)': next((z for z,info in iso_zones_config.items()
                                    if float(cf_l)<=info['max']),'D'),
        })
    # forecast
    for fd, cf_l, cf_p in zip(future_dates, cf_lin_f, cf_poly_f):
        z_l = next((z for z,info in iso_zones_config.items() if float(cf_l)<=info['max']),'D')
        forecast_rows.append({
            'Period': fd.strftime('%b-%y'), 'Type': 'Forecast',
            'Date': fd.strftime('%d-%b-%Y'),
            'CF Actual': None,
            'CF Linear Fit': round(float(cf_l),4),
            'CF Poly Fit':   round(float(cf_p),4),
            'Zone (Linear)': z_l,
        })
    df_forecast = pd.DataFrame(forecast_rows)
    return lin_reg, poly_fn, future_dates, cf_lin_f, cf_poly_f, df_forecast, avg_interval_days, slope, intercept


# ============================================================
# ML-③  ZONE CLASSIFICATION  —  Rule-based vs Random Forest
# ============================================================
def classify_zones(all_ds, iso_zones_config):
    np.random.seed(42)
    def gen_samples(cf_range, kurt_range, rms_range, n=80):
        cf   = np.random.uniform(*cf_range,  n)
        kurt = np.random.uniform(*kurt_range,n)
        rms  = np.random.uniform(*rms_range, n)
        peak = cf * rms * np.random.uniform(0.9,1.1,n)
        std  = rms * np.random.uniform(0.95,1.05,n)
        return np.column_stack([rms, peak, cf, kurt, std])

    X_train_A = gen_samples((1.0, 2.5), (2.8, 3.0), (0.05, 0.15))
    X_train_B = gen_samples((2.5, 3.0), (2.9, 3.2), (0.15, 0.25))
    X_train_C = gen_samples((3.0, 4.0), (3.1, 4.0), (0.25, 0.40))
    X_train_D = gen_samples((4.0, 7.0), (3.5, 8.0), (0.40, 1.00))

    X_train = np.vstack([X_train_A, X_train_B, X_train_C, X_train_D])
    y_train = np.array(['A']*80 + ['B']*80 + ['C']*80 + ['D']*80)

    rf_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
    ])
    rf_pipe.fit(X_train, y_train)

    X_test_list = []
    for ds in all_ds:
        m = ds['metrics']
        X_test_list.append([m['rms'], m['peak'], m['crest'], m['kurt'], m['std']])
    X_test = np.array(X_test_list)

    rf_preds  = rf_pipe.predict(X_test)
    rf_probas = rf_pipe.predict_proba(X_test)
    rf_classes = rf_pipe.classes_

    clf_rows = []
    for i, ds in enumerate(all_ds):
        rule_z = ds['iso_zone']
        rf_z   = rf_preds[i]
        row = {
            'Period':    ds['label'],
            'Rule Zone': rule_z,
            'RF Zone':   rf_z,
            'Match':     'Yes' if rule_z == rf_z else 'No',
        }
        for c_idx, c_name in enumerate(rf_classes):
            row[f'P(Zone {c_name})'] = round(float(rf_probas[i, c_idx]), 4)
        row['CF']       = round(ds['metrics']['crest'], 4)
        row['Kurtosis'] = round(ds['metrics']['kurt'], 4)
        row['RMS']      = round(ds['metrics']['rms'], 5)
        clf_rows.append(row)

    df_classify = pd.DataFrame(clf_rows)
    return rf_pipe, rf_preds, rf_probas, rf_classes, df_classify
