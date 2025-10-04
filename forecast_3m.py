# -*- coding: utf-8 -*-
"""
Genera forecast rolling de 3 meses (desde ahora hasta fin de mes+2),
graba JSON y CSV con: fecha, hora, llamadas_recibidas, tmo_pred_seg, ejecutivos_requeridos.
- Usa model_lgb.pkl + features.json (modelo de llamadas)
- Seed de lags desde data/Hosting ia.xlsx
- Dimensiona con Erlang C (SL=90% a 22s) y productividad 80% (max_occupancy)
"""
import os, json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib

from erlang_utils import required_agents_erlang_c

# ==========================
# Parámetros operativos
# ==========================
DATA_PATHS = ["./data/Hosting ia.xlsx", "./Hosting ia.xlsx"]
MODEL_PATH = "./model_lgb.pkl"
FEATS_PATH = "./features.json"
METRICS_PATH = "./train_metrics.json"   # opcional, para bandas
OUTPUT_JSON = "./forecast_3m.json"
OUTPUT_CSV  = "./forecast_3m.csv"

TARGET_COL        = "recibidos"      # objetivo usado al entrenar
TMO_COL_OPT       = "tmo (segundos)" # si existe, se usa para lags/rollings
SLA_TARGET        = 0.90             # 90%
ASA_TARGET_S      = 22.0             # 22 segundos
MAX_OCCUPANCY     = 0.80             # productividad 80%
SEED              = 42

np.random.seed(SEED)

# ==========================
# Utilidades
# ==========================
def find_data_file():
    for p in DATA_PATHS:
        if os.path.exists(p):
            print("📄 Histórico:", p)
            return p
    raise FileNotFoundError("No se encontró el histórico. Sube 'data/Hosting ia.xlsx'.")

def ensure_datetime(df):
    # intenta detectar columna datetime directa
    for c in ["datetime","datatime","fecha_hora","ts"]:
        if c in df.columns:
            df["datetime"] = pd.to_datetime(df[c], errors="coerce")
            return df
    # fecha + hora
    fcol = next((c for c in ["fecha","date","dia"] if c in df.columns), None)
    hcol = next((c for c in ["hora","time"] if c in df.columns), None)
    if fcol and hcol:
        f = pd.to_datetime(df[fcol], errors="coerce", dayfirst=True)
        h = pd.to_timedelta(df[hcol].astype(str), errors="coerce")
        # si la hora está como entero 0..23
        mask_num = df[hcol].apply(lambda x: str(x).isdigit())
        if mask_num.any():
            h_num = pd.to_timedelta(pd.to_numeric(df.loc[mask_num, hcol], errors="coerce"), unit="h")
            h.loc[mask_num] = h_num.values
        df["datetime"] = f + h.fillna(pd.Timedelta(0))
        return df
    if fcol:
        f = pd.to_datetime(df[fcol], errors="coerce", dayfirst=True)
        df["datetime"] = f
        return df
    raise ValueError("No encontré columnas para construir 'datetime'.")

def aggregate_to_hour(df, target_col, tmo_col=None):
    df = df.copy()
    df["datetime_h"] = df["datetime"].dt.floor("h")
    if tmo_col and tmo_col in df.columns:
        df["_tmo_w"] = df[tmo_col]*df[target_col]
        agg = (df.groupby("datetime_h",as_index=False)
                 .agg({target_col:"sum","_tmo_w":"sum"}))
        agg[tmo_col] = np.where(agg[target_col]>0, agg["_tmo_w"]/agg[target_col], 0.0)
        agg = agg.drop(columns=["_tmo_w"])
    else:
        agg = df.groupby("datetime_h",as_index=False).agg({target_col:"sum"})
    return agg.rename(columns={"datetime_h":"datetime"})

def add_time_cols(df):
    df["hour"]  = df["datetime"].dt.hour
    df["dow"]   = df["datetime"].dt.dayofweek
    df["week"]  = df["datetime"].dt.isocalendar().week.astype(int)
    df["month"] = df["datetime"].dt.month
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    df["dow_sin"]  = np.sin(2*np.pi*df["dow"]/7)
    df["dow_cos"]  = np.cos(2*np.pi*df["dow"]/7)
    return df

def build_feature_row(ts, hist_df, X_cols, use_tmo, target_col, tmo_col,
                      robust_window_hours=24*365):
    row = pd.DataFrame({"datetime":[ts]})
    row = add_time_cols(row)

    h = hist_df.set_index("datetime")
    # y_series: usa yhat si existe; si NaN, usa valor real
    y_series = h["yhat"].fillna(h[target_col])

    # lags target 1..24
    for lag in range(1, 25):
        row[f"{target_col}_lag{lag}"] = y_series.reindex([ts - pd.Timedelta(hours=lag)]).values[0]
    # rollings target
    hist_window = y_series.loc[ts - pd.Timedelta(hours=robust_window_hours):ts]
    row[f"{target_col}_roll3"]  = hist_window.tail(3).mean()
    row[f"{target_col}_roll6"]  = hist_window.tail(6).mean()
    row[f"{target_col}_roll24"] = hist_window.tail(24).mean()

    # TMO lags si corresponde
    if use_tmo and tmo_col in h.columns:
        tmo_series = h[tmo_col]
        row[f"{tmo_col}_lag1"]  = tmo_series.reindex([ts - pd.Timedelta(hours=1)]).values[0]
        row[f"{tmo_col}_roll3"] = tmo_series.loc[ts - pd.Timedelta(hours=robust_window_hours):ts].tail(3).mean()

    # mantener solo columnas que el modelo espera
    row = row.reindex(columns=set(["datetime"] + X_cols))
    return row

# ==========================
# Carga artefactos
# ==========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Falta model_lgb.pkl")
if not os.path.exists(FEATS_PATH):
    raise FileNotFoundError("Falta features.json")

model = joblib.load(MODEL_PATH)
with open(FEATS_PATH, "r", encoding="utf-8") as f:
    X_cols = json.load(f)["feature_columns"]

rmse_for_bands = None
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH,"r",encoding="utf-8") as f:
        mets = json.load(f)
        rmse_for_bands = float(mets.get("test",{}).get("RMSE", np.nan))

# ==========================
# Seed histórico para lags
# ==========================
raw = pd.read_excel(find_data_file())
df  = ensure_datetime(raw)

use_tmo = (TMO_COL_OPT in df.columns)
keep = ["datetime", TARGET_COL] + ([TMO_COL_OPT] if use_tmo else [])
df  = df[keep].dropna(subset=["datetime"]).sort_values("datetime")

hist = aggregate_to_hour(df, target_col=TARGET_COL, tmo_col=(TMO_COL_OPT if use_tmo else None))
hist = hist.sort_values("datetime").reset_index(drop=True)
hist["yhat"] = np.nan

# ==========================
# Horizonte: ahora -> fin de mes + 2
# ==========================
now = pd.Timestamp.now(tz=None).floor("h")
start_gen = max(now + pd.Timedelta(hours=1), hist["datetime"].max() + pd.Timedelta(hours=1))

def end_of_month_plus2(ts):
    # fin de mes+2 (23:00 de ese día)
    y, m = ts.year, ts.month
    # mes+2
    m2 = m + 2
    y2 = y + (m2 - 1)//12
    m2 = ((m2 - 1) % 12) + 1
    # primer día mes+3
    m3 = m + 3
    y3 = y + (m3 - 1)//12
    m3 = ((m3 - 1) % 12) + 1
    first_next = pd.Timestamp(year=y3, month=m3, day=1, hour=0)
    end_m2 = first_next - pd.Timedelta(hours=1)
    return end_m2

end_gen = end_of_month_plus2(now)

future_hours = int((end_gen - start_gen) / pd.Timedelta(hours=1)) + 1
future_index = [start_gen + pd.Timedelta(hours=i) for i in range(max(0, future_hours))]

# ==========================
# Predicción recursiva + TMO estimado
# ==========================
# Para TMO futuro: si no tenemos modelo de TMO, usamos heurística:
#   tmo_pred = rolling 24h del histórico (o último valor si no hay suficientes)
def estimate_future_tmo(ts, hist_df, tmo_col):
    if use_tmo and tmo_col in hist_df.columns:
        s = hist_df.set_index("datetime")[tmo_col]
        last24 = s.loc[ts - pd.Timedelta(hours=24):ts]
        if len(last24) >= 1:
            return float(last24.mean())
        # fallback: último valor conocido
        return float(s.dropna().iloc[-1]) if s.dropna().size else 300.0
    # si no hay tmo en histórico, usar 300s como valor neutro
    return 300.0

rows = []
for ts in future_index:
    feat_row = build_feature_row(ts, hist, X_cols, use_tmo, TARGET_COL, TMO_COL_OPT)
    x = feat_row[X_cols].values
    yhat_calls = float(model.predict(x)[0])

    # TMO futuro (segundos)
    tmo_hat = estimate_future_tmo(ts, hist, TMO_COL_OPT)

    # Guardar pred y alimentar histórico para siguientes lags
    new_row = {"datetime": ts, TARGET_COL: np.nan}
    if use_tmo:
        new_row[TMO_COL_OPT] = tmo_hat
    new_row["yhat"] = yhat_calls
    hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)

    # Dimensionamiento via Erlang C
    N_req = required_agents_erlang_c(
        calls_per_hour=max(yhat_calls, 0.0),
        AHT_sec=max(tmo_hat, 1.0),
        sla_target=SLA_TARGET,
        asa_target_sec=ASA_TARGET_S,
        max_occupancy=MAX_OCCUPANCY
    )

    rows.append({
        "datetime": ts,
        "llamadas_recibidas": max(yhat_calls, 0.0),
        "tmo_pred_seg": max(tmo_hat, 0.0),
        "ejecutivos_requeridos": int(N_req)
    })

forecast = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)

# Columnas separadas fecha/hora
forecast["fecha"] = forecast["datetime"].dt.strftime("%Y-%m-%d")
forecast["hora"]  = forecast["datetime"].dt.strftime("%H:%M")

# Orden final
forecast = forecast[["fecha","hora","llamadas_recibidas","tmo_pred_seg","ejecutivos_requeridos","datetime"]]

# ==========================
# Guardar JSON/CSV
# ==========================
# JSON pensado para el front: lista de objetos
out = forecast.drop(columns=["datetime"]).copy()
out.to_json(OUTPUT_JSON, orient="records", force_ascii=False, indent=2)
forecast.drop(columns=["datetime"]).to_csv(OUTPUT_CSV, index=False)

print("✅ Forecast 3 meses guardado:")
print(" -", OUTPUT_JSON)
print(" -", OUTPUT_CSV)
print(forecast.head(5))
