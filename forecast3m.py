# -*- coding: utf-8 -*-
"""
Forecast rolling 3 meses (desde ahora hasta fin de mes+2)
Salida: public/forecast_3m.json y public/forecast_3m.csv
Columnas: fecha, hora, llamadas_recibidas, tmo_pred_seg, ejecutivos_requeridos
"""

import os, json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib

# ---------------- Erlang-C ----------------
def erlang_c_prob_wait(N: int, A: float) -> float:
    if A <= 0: return 0.0
    if N <= 0: return 1.0
    if A >= N: return 1.0
    summ = 0.0
    term = 1.0
    for k in range(1, N):
        summ += term
        term *= A / k
    summ += term
    pn = term * (A / N) / (1.0 - (A / N))
    return float(pn / (summ + pn))

def service_level(N: int, A: float, AHT_sec: float, ASA_target_sec: float) -> float:
    if A <= 0: return 1.0
    if N <= 0: return 0.0
    P_w = 1.0 if A >= N else erlang_c_prob_wait(N, A)
    expo = - (N - A) * (ASA_target_sec / max(AHT_sec, 1e-9))
    return 1.0 - P_w * np.exp(expo)

def required_agents_erlang_c(calls_per_hour, AHT_sec, sla_target=0.90, asa_target_sec=22.0, max_occupancy=0.80):
    if calls_per_hour <= 0 or AHT_sec <= 0: return 0
    lam = calls_per_hour / 3600.0
    A = lam * AHT_sec
    N = max(1, int(np.ceil(A / max_occupancy)))
    while True:
        if service_level(N, A, AHT_sec, asa_target_sec) >= sla_target and (A / N) <= max_occupancy:
            return N
        N += 1
        if N > 10000: return N
# ------------------------------------------

# ===== Parámetros =====
TARGET_COL     = "recibidos"
TMO_COL_OPT    = "tmo (segundos)"
SLA_TARGET     = 0.90
ASA_TARGET_S   = 22.0
MAX_OCCUPANCY  = 0.80
SEED           = 42
np.random.seed(SEED)

PUBLIC_DIR   = "./public"
OUTPUT_JSON  = f"{PUBLIC_DIR}/forecast_3m.json"
OUTPUT_CSV   = f"{PUBLIC_DIR}/forecast_3m.csv"

# ===== Utilidades tolerantes a nombres =====
def find_first(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def find_model_path():
    p = find_first(["./model_lgb.pkl", "./model-lgb.pkl"])
    if not p:
        raise FileNotFoundError("No encontré el modelo (model_lgb.pkl / model-lgb.pkl).")
    print("🧠 Modelo:", p)
    return p

def find_features_path():
    p = find_first(["./features.json"])
    if not p: raise FileNotFoundError("Falta features.json")
    print("📑 Features:", p)
    return p

def find_metrics_path():
    return find_first(["./train_metrics.json", "./trimetrics.json"])

def find_data_file():
    p = find_first(["./data/Hosting ia.xlsx", "./Data/Hosting ia.xlsx", "./Hosting ia.xlsx"])
    if not p: raise FileNotFoundError("No se encontró el histórico (data/Hosting ia.xlsx).")
    print("📄 Histórico:", p)
    return p

def ensure_datetime(df):
    for c in ["datetime","datatime","fecha_hora","ts"]:
        if c in df.columns:
            df["datetime"] = pd.to_datetime(df[c], errors="coerce"); return df
    fcol = next((c for c in ["fecha","date","dia"] if c in df.columns), None)
    hcol = next((c for c in ["hora","time"] if c in df.columns), None)
    if fcol and hcol:
        f = pd.to_datetime(df[fcol], errors="coerce", dayfirst=True)
        h = pd.to_timedelta(df[hcol].astype(str), errors="coerce")
        mask_num = df[hcol].apply(lambda x: str(x).isdigit())
        if mask_num.any():
            h_num = pd.to_timedelta(pd.to_numeric(df.loc[mask_num, hcol], errors="coerce"), unit="h")
            h.loc[mask_num] = h_num.values
        df["datetime"] = f + h.fillna(pd.Timedelta(0)); return df
    if fcol:
        f = pd.to_datetime(df[fcol], errors="coerce", dayfirst=True)
        df["datetime"] = f; return df
    raise ValueError("No encontré columnas para construir 'datetime'.")

# ---------- NUEVO: agregación inteligente ----------
def aggregate_to_hour_smart(df, target_col, tmo_col=None):
    """
    Regla:
    - Si en la hora hay 1 fila -> se deja igual.
    - Si hay 2 filas y sus minutos no son 00 (ej: 00 y 30) -> asumimos medias horas -> SUMA.
    - Si hay >2 filas, o 2 filas con el mismo valor exacto (posible duplicado),
      o la varianza es ~0 -> usamos PROMEDIO para evitar duplicaciones.
    """
    df = df.copy()
    df["dt_floor"] = df["datetime"].dt.floor("h")
    df["minute"]   = df["datetime"].dt.minute

    def agg_group(g):
        n = len(g)
        vals = g[target_col].astype(float)
        # ¿hay minutos 00 y 30?
        mins = set(g["minute"].tolist())
        two_halfhours = (n == 2) and (mins <= {0,30}) and (0 in mins or 30 in mins) and not np.isclose(vals.iloc[0], vals.iloc[1])
        all_equal = vals.nunique(dropna=True) == 1
        if n == 1:
            target = vals.iloc[0]
        elif two_halfhours:
            target = vals.sum()
        elif n == 2 and all_equal:
            target = vals.mean()   # duplicado exacto
        elif n > 2:
            # si parece ya-horario repetido (mucha repetición) -> promedio
            if vals.std(ddof=0) < 1e-6:
                target = vals.mean()
            else:
                # si realmente son subfragmentos (15min, etc.), suma
                target = vals.sum()
        else:
            # caso genérico
            target = vals.sum()

        out = {"datetime": g["dt_floor"].iloc[0], target_col: target}
        if tmo_col and tmo_col in g.columns:
            # TMO ponderado por target
            w = (g[tmo_col].astype(float) * vals).sum()
            out[tmo_col] = (w / target) if target > 0 else 0.0
        return pd.Series(out)

    agg = df.groupby("dt_floor", as_index=False).apply(agg_group)
    return agg.sort_values("datetime").reset_index(drop=True)

def add_time_cols(df):
    df["hour"]  = df["datetime"].dt.hour
    df["dow"]   = df["datetime"].dt.dayofweek
    df["week"]  = df["datetime"].dt.isocalendar().week.astype(int)
    df["month"] = df["datetime"].dt.month
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24); df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    df["dow_sin"]  = np.sin(2*np.pi*df["dow"]/7);   df["dow_cos"]  = np.cos(2*np.pi*df["dow"]/7)
    return df

def build_feature_row(ts, hist_df, X_cols, use_tmo, target_col, tmo_col, window_hours=24*365):
    row = pd.DataFrame({"datetime":[ts]}); row = add_time_cols(row)
    h = hist_df.set_index("datetime")
    y_series = h["yhat"].fillna(h[target_col])
    for lag in range(1, 25):
        row[f"{target_col}_lag{lag}"] = y_series.reindex([ts - pd.Timedelta(hours=lag)]).values[0]
    w = y_series.loc[ts - pd.Timedelta(hours=window_hours):ts]
    row[f"{target_col}_roll3"]  = w.tail(3).mean()
    row[f"{target_col}_roll6"]  = w.tail(6).mean()
    row[f"{target_col}_roll24"] = w.tail(24).mean()
    if use_tmo and tmo_col in h.columns:
        tmo_series = h[tmo_col]
        row[f"{tmo_col}_lag1"]  = tmo_series.reindex([ts - pd.Timedelta(hours=1)]).values[0]
        row[f"{tmo_col}_roll3"] = tmo_series.loc[ts - pd.Timedelta(hours=window_hours):ts].tail(3).mean()
    return row.reindex(columns=set(["datetime"] + X_cols))

def end_of_month_plus2(ts):
    y, m = ts.year, ts.month
    m3 = m + 3
    y3 = y + (m3 - 1)//12
    m3 = ((m3 - 1) % 12) + 1
    first_next = pd.Timestamp(year=y3, month=m3, day=1, hour=0)
    return first_next - pd.Timedelta(hours=1)

def estimate_future_tmo(ts, hist_df, tmo_col, use_tmo=True):
    if use_tmo and tmo_col in hist_df.columns:
        s = hist_df.set_index("datetime")[tmo_col]
        last24 = s.loc[ts - pd.Timedelta(hours=24):ts]
        if len(last24) >= 1: return float(last24.mean())
        return float(s.dropna().iloc[-1]) if s.dropna().size else 300.0
    return 300.0

# ===== Carga artefactos y datos =====
MODEL_PATH   = find_model_path()
FEATS_PATH   = find_features_path()
METRICS_PATH = find_metrics_path()

model = joblib.load(MODEL_PATH)
with open(FEATS_PATH, "r", encoding="utf-8") as f:
    X_cols = json.load(f)["feature_columns"]

# Histórico para seed (forzamos motor openpyxl en Actions)
raw = pd.read_excel(find_data_file(), engine="openpyxl")
df  = ensure_datetime(raw)

use_tmo = (TMO_COL_OPT in df.columns)
keep = ["datetime", TARGET_COL] + ([TMO_COL_OPT] if use_tmo else [])
df  = df[keep].dropna(subset=["datetime"]).sort_values("datetime")

# ---------- usar agregación inteligente ----------
hist = aggregate_to_hour_smart(df, TARGET_COL, TMO_COL_OPT if use_tmo else None)

# Cálculo de percentil 99 histórico (para cap)
p99 = float(np.nanpercentile(hist[TARGET_COL].values, 99)) if len(hist) else 0.0
cap_value = 1.1 * max(p99, 1.0)   # margen 10%

hist = hist.sort_values("datetime").reset_index(drop=True)
hist["yhat"] = np.nan

# ===== Horizonte =====
now = pd.Timestamp.now(tz=None).floor("h")
start_gen = max(now + pd.Timedelta(hours=1), hist["datetime"].max() + pd.Timedelta(hours=1))
end_gen   = end_of_month_plus2(now)
future_hours = int((end_gen - start_gen) / pd.Timedelta(hours=1)) + 1
future_index = [start_gen + pd.Timedelta(hours=i) for i in range(max(0, future_hours))]

# ===== Predicción recursiva + cap + Erlang-C =====
rows = []
for ts in future_index:
    feat_row   = build_feature_row(ts, hist, X_cols, use_tmo, TARGET_COL, TMO_COL_OPT)
    x          = feat_row[X_cols].values
    yhat_calls = float(model.predict(x)[0])

    # ---------- CAP por P99 histórico ----------
    yhat_calls = max(0.0, min(yhat_calls, cap_value))

    tmo_hat    = estimate_future_tmo(ts, hist, TMO_COL_OPT, use_tmo)

    # alimentar histórico
    new_row = {"datetime": ts, TARGET_COL: np.nan}
    if use_tmo: new_row[TMO_COL_OPT] = tmo_hat
    new_row["yhat"] = yhat_calls
    hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)

    N_req = required_agents_erlang_c(
        calls_per_hour=yhat_calls,
        AHT_sec=max(tmo_hat, 1.0),
        sla_target=SLA_TARGET,
        asa_target_sec=ASA_TARGET_S,
        max_occupancy=MAX_OCCUPANCY
    )

    rows.append({
        "datetime": ts,
        "llamadas_recibidas": yhat_calls,
        "tmo_pred_seg": max(tmo_hat, 0.0),
        "ejecutivos_requeridos": int(N_req)
    })

forecast = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
forecast["fecha"] = forecast["datetime"].dt.strftime("%Y-%m-%d")
forecast["hora"]  = forecast["datetime"].dt.strftime("%H:%M")
forecast = forecast[["fecha","hora","llamadas_recibidas","tmo_pred_seg","ejecutivos_requeridos","datetime"]]

# ===== Guardar =====
os.makedirs(PUBLIC_DIR, exist_ok=True)
forecast.drop(columns=["datetime"]).to_json(OUTPUT_JSON, orient="records", force_ascii=False, indent=2)
forecast.drop(columns=["datetime"]).to_csv(OUTPUT_CSV, index=False)

print("✅ Forecast 3 meses guardado en:")
print(" -", OUTPUT_JSON)
print(" -", OUTPUT_CSV)
print("Cap P99 usado:", cap_value)
print(forecast.head(5))


