# forecast3m.py
# Inferencia conjunta Llamadas + TMO descargando modelos desde el último Release
import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from utils_release import download_asset_from_latest

# ---------- Parámetros ----------
OWNER = "Supervision-Inbound"           # <--- ajusta si tu org/usuario cambia
REPO  = "wf-Analytics-AI2.5"            # <--- nombre exacto del repo

ASSET_LLAMADAS = "modelo_llamadas.pkl"
ASSET_TMO      = "modelo_tmo.pkl"

MODELS_DIR = "models"
OUT_CSV    = "data_out/predicciones.csv"
OUT_JSON   = "public/predicciones.json"

HOURS_AHEAD = 24 * 90  # 90 días hacia adelante (3 meses)
FREQ        = "H"      # por hora

# Semillas (si no hay estado reciente disponible)
DEFAULT_LL  = 100.0    # base inicial llamadas
DEFAULT_TMO = 180.0    # base inicial tmo (segundos)

# --------------------------------

def add_time_features(df):
    df["dow"] = df["ts"].dt.dayofweek
    df["doy"] = df["ts"].dt.dayofyear
    df["week"] = df["ts"].dt.isocalendar().week.astype(int)
    df["month"] = df["ts"].dt.month
    df["hour"] = df["ts"].dt.hour
    df["sin_hour"] = np.sin(2*np.pi*df["hour"]/24)
    df["cos_hour"] = np.cos(2*np.pi*df["hour"]/24)
    df["sin_dow"] = np.sin(2*np.pi*df["dow"]/7)
    df["cos_dow"] = np.cos(2*np.pi*df["dow"]/7)
    return df

def build_feature_matrix(df, target_col):
    feats = [
        "sin_hour","cos_hour","sin_dow","cos_dow",
        "dow","month",
        f"{target_col}_lag1", f"{target_col}_lag24",
        f"{target_col}_ma24", f"{target_col}_ma168",
        f"{target_col}_samehour_7d"
    ]
    return df[feats].copy()

def ensure_dirs():
    os.makedirs("public", exist_ok=True)
    os.makedirs("data_out", exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

def main():
    ensure_dirs()

    # 1) Descargar modelos desde el último Release
    print("Descargando modelos desde Release…")
    p_ll = download_asset_from_latest(OWNER, REPO, ASSET_LLAMADAS, os.path.join(MODELS_DIR, ASSET_LLAMADAS))
    p_tm = download_asset_from_latest(OWNER, REPO, ASSET_TMO,      os.path.join(MODELS_DIR, ASSET_TMO))
    print("Modelos descargados:", p_ll, p_tm)

    # 2) Cargar modelos
    mdl_ll = joblib.load(p_ll)
    mdl_tmo = joblib.load(p_tm) if os.path.exists(p_tm) else None

    # 3) Construir timeline futuro
    start = pd.Timestamp.utcnow().floor("H")
    idx = pd.date_range(start, periods=HOURS_AHEAD, freq=FREQ, tz="UTC").tz_convert("America/Santiago").tz_localize(None)
    df = pd.DataFrame({"ts": idx})
    df = add_time_features(df)

    # 4) Semillas simples (si no mantienes estado histórico)
    df["seed_ll"]  = DEFAULT_LL
    df["seed_tmo"] = DEFAULT_TMO

    # Llamadas seeds → features
    df["llamadas_lag1"]       = df["seed_ll"].shift(1)
    df["llamadas_lag24"]      = df["seed_ll"].shift(24)
    df["llamadas_ma24"]       = df["seed_ll"].rolling(24, min_periods=1).mean()
    df["llamadas_ma168"]      = df["seed_ll"].rolling(24*7, min_periods=1).mean()
    df["llamadas_samehour_7d"]= df["seed_ll"].shift(24*7)

    # TMO seeds → features
    df["tmo_seg_lag1"]        = df["seed_tmo"].shift(1)
    df["tmo_seg_lag24"]       = df["seed_tmo"].shift(24)
    df["tmo_seg_ma24"]        = df["seed_tmo"].rolling(24, min_periods=1).mean()
    df["tmo_seg_ma168"]       = df["seed_tmo"].rolling(24*7, min_periods=1).mean()
    df["tmo_seg_samehour_7d"] = df["seed_tmo"].shift(24*7)

    # 5) Predicción
    X_ll = build_feature_matrix(df, "llamadas").fillna(method="bfill").fillna(method="ffill")
    pred_ll = mdl_ll.predict(X_ll)

    if mdl_tmo is not None:
        X_tmo = build_feature_matrix(df, "tmo_seg").fillna(method="bfill").fillna(method="ffill")
        pred_tmo = mdl_tmo.predict(X_tmo)
    else:
        pred_tmo = np.full(len(df), np.nan)

    out = pd.DataFrame({
        "ts": df["ts"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "pred_llamadas": np.maximum(0, pred_ll).round(2),
        "pred_tmo_seg": np.maximum(0, pred_tmo).round(2)
    })

    # 6) Guardar CSV y JSON
    out.to_csv(OUT_CSV, index=False)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    print(f"OK -> {OUT_CSV}")
    print(f"OK -> {OUT_JSON}")

if __name__ == "__main__":
    main()


