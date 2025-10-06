# -*- coding: utf-8 -*-
"""
forecast3m.py
Inferencia diaria y horaria con el modelo de llamadas (.keras, Keras 3).
- Descarga modelos desde el último Release (assets):
    - models/modelo_llamadas_nn.keras  (obligatorio)
    - models/modelo_tmo.pkl            (opcional, se ignora si no está)
- Genera:
    - data_out/predicciones.csv
    - data_out/predicciones.json
    - public/predicciones.json
    - data_out/erlang_forecast.json
    - public/erlang_forecast.json
    - public/last_update.json
"""

from __future__ import annotations
import os
import io
import json
import math
import time
import shutil
import zipfile
import datetime as dt
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests

# ====== Keras 3 / TF 2.17 ======
import keras
from keras.models import load_model as keras_load_model


# --------------------------------------------------------------------------------------
# Utilidades del repositorio
# --------------------------------------------------------------------------------------
DATA_OUT = "data_out"
PUBLIC = "public"
MODELS_DIR = "models"

LLAMADAS_MODEL_NAME = "modelo_llamadas_nn.keras"   # asset del release
TMO_MODEL_NAME = "modelo_tmo.pkl"                  # se ignora para este flujo

# Ventana usada por el modelo (la arquitectura espera (None, 28, 1))
SEQ_WINDOW = 28
# Horizonte: 90 días
HORIZON_DAYS = 90

os.makedirs(DATA_OUT, exist_ok=True)
os.makedirs(PUBLIC, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# --------------------------------------------------------------------------------------
# Descarga de modelos desde Releases
# --------------------------------------------------------------------------------------
def download_asset_from_latest_release(asset_name: str, dst_path: str) -> bool:
    """
    Descarga un asset por nombre desde el último Release del mismo repo.
    Requiere GITHUB_TOKEN y GITHUB_REPOSITORY (org/repo) en el entorno.
    """
    repo = os.environ.get("GITHUB_REPOSITORY")
    token = os.environ.get("GITHUB_TOKEN")
    if not repo:
        raise RuntimeError("GITHUB_REPOSITORY no está definido (org/repo).")
    if not token:
        raise RuntimeError("GITHUB_TOKEN no está definido en env.")

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}",
                           "Accept": "application/vnd.github+json"})

    # Último release
    rel_url = f"https://api.github.com/repos/{repo}/releases/latest"
    r = session.get(rel_url, timeout=60)
    r.raise_for_status()
    rel = r.json()

    # Buscar asset
    asset = None
    for a in rel.get("assets", []):
        if a.get("name") == asset_name:
            asset = a
            break
    if not asset:
        print(f"[download_asset] No se encontró asset '{asset_name}' en el último release.")
        return False

    # Descargar (URL de asset para browser)
    dl_url = asset["url"]
    # Para descargar binario del asset se usa este header
    headers = {"Accept": "application/octet-stream"}
    with session.get(dl_url, headers=headers, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        with open(dst_path, "wb") as f:
            shutil.copyfileobj(resp.raw, f)

    print(f"[download_asset] Descargado: {asset_name} -> {dst_path}")
    return True


def ensure_models_from_release() -> Tuple[str | None, str | None]:
    """
    Descarga a ./models los ficheros de modelos necesarios si no existen.
    Devuelve las rutas (llamadas_model_path, tmo_model_path).
    """
    ll_path = os.path.join(MODELS_DIR, LLAMADAS_MODEL_NAME)
    tmo_path = os.path.join(MODELS_DIR, TMO_MODEL_NAME)

    print("Descargando modelos desde Release…")
    # Siempre intentamos actualizar el de llamadas (para no quedar desfasados).
    ok_ll = download_asset_from_latest_release(LLAMADAS_MODEL_NAME, ll_path)
    # El de TMO es opcional en este flujo (se puede ignorar)
    ok_tmo = download_asset_from_latest_release(TMO_MODEL_NAME, tmo_path)
    if not ok_tmo:
        tmo_path = None

    print("Modelos descargados:",
          ll_path if ok_ll else "(no)",
          tmo_path if ok_tmo else "(no)")
    return (ll_path if ok_ll else None, tmo_path)


# --------------------------------------------------------------------------------------
# Carga robusta para Keras 3
# --------------------------------------------------------------------------------------
def load_keras3(path: str):
    """
    Carga robusta para .keras (Keras 3) en runners sin GPU.
    1) intento normal
    2) safe_mode=False
    3) registra inicializador Orthogonal en custom_objects
    """
    try:
        return keras_load_model(path, compile=False)
    except Exception as e1:
        print("[load_keras3] Retry safe_mode=False ::", e1)
        try:
            return keras_load_model(path, compile=False, safe_mode=False)
        except Exception as e2:
            print("[load_keras3] Retry with custom_objects ::", e2)
            return keras_load_model(
                path,
                compile=False,
                safe_mode=False,
                custom_objects={
                    "Orthogonal": keras.initializers.Orthogonal,
                },
            )


# --------------------------------------------------------------------------------------
# Features de calendario (ctx de 6 columnas)
# --------------------------------------------------------------------------------------
def season_from_month(m: int) -> int:
    # DJF=0, MAM=1, JJA=2, SON=3 (hemisferio sur ajusta estaciones si prefieres)
    if m in (12, 1, 2):
        return 0
    if m in (3, 4, 5):
        return 1
    if m in (6, 7, 8):
        return 2
    return 3


def build_ctx_for_date(d: dt.date) -> np.ndarray:
    """
    Devuelve vector de 6 elementos:
    [dow, month, sin_doy, cos_doy, is_weekend, season]
    """
    dow = d.weekday()              # 0..6
    month = d.month                # 1..12
    doy = d.timetuple().tm_yday    # 1..366
    sin_doy = math.sin(2 * math.pi * doy / 366.0)
    cos_doy = math.cos(2 * math.pi * doy / 366.0)
    is_weekend = 1 if dow >= 5 else 0
    seas = season_from_month(month)
    return np.array([dow, month, sin_doy, cos_doy, is_weekend, seas], dtype="float32")


# --------------------------------------------------------------------------------------
# Carga histórico de totales diarios (para semilla de la ventana)
# --------------------------------------------------------------------------------------
def load_history_totals() -> pd.Series:
    """
    Busca un histórico razonable para sembrar la ventana de 28 días.
    Prioridad:
      1) data_out/train_daily_samples.csv (si lo dejó el entrenamiento)
      2) public/predicciones.json (usar últimos días reales o predichos)
    Si no encuentra nada, genera una serie dummy constante (100).
    """
    # 1) CSV del entrenamiento
    csv_path = os.path.join(DATA_OUT, "train_daily_samples.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Se espera una columna 'date' y 'total' (formato del script de entrenamiento que te pasé).
        # Si tu CSV tiene otros nombres, ajusta aquí.
        if {"date", "total"}.issubset(df.columns):
            s = (
                pd.to_datetime(df["date"])
                .dt.date
            )
            totals = pd.Series(df["total"].astype(float).values, index=s)
            totals = totals.sort_index()
            if len(totals) >= SEQ_WINDOW:
                return totals

    # 2) JSON público anterior
    pub_json = os.path.join(PUBLIC, "predicciones.json")
    if os.path.exists(pub_json):
        try:
            data = json.load(open(pub_json, "r", encoding="utf-8"))
            # esperar [{"date":"YYYY-MM-DD","total":...}, ...]
            rows = [(dt.date.fromisoformat(r["date"]), float(r["total"])) for r in data]
            ser = pd.Series(dict(rows)).sort_index()
            if len(ser) >= SEQ_WINDOW:
                return ser
        except Exception:
            pass

    # fallback: constante
    dates = [dt.date.today() - dt.timedelta(days=i) for i in range(SEQ_WINDOW, 0, -1)]
    return pd.Series([100.0] * SEQ_WINDOW, index=dates)


# --------------------------------------------------------------------------------------
# Inferencia autoregresiva diaria + perfil horario (24h)
# --------------------------------------------------------------------------------------
@dataclass
class DayForecast:
    date: dt.date
    total: float
    profile: List[float]  # 24 valores que suman ~1


def forecast_days(mdl_ll, seed_totals: pd.Series, horizon: int = HORIZON_DAYS) -> List[DayForecast]:
    """
    Autoregresivo: usa los últimos 28 días (semilla + predichos) para el siguiente día.
    Entrada del modelo:
      - seq_totales: (1, 28, 1)
      - ctx: (1, 6)
    Salidas:
      - y_total: escalar
      - y_profile: vector 24 (probabilidades)
    """
    history = seed_totals.copy().sort_index()
    last_date = history.index[-1]
    results: List[DayForecast] = []

    for step in range(horizon):
        d = last_date + dt.timedelta(days=step + 1)

        # ventana de 28
        seq = history.iloc[-SEQ_WINDOW:].values.astype("float32").reshape(1, SEQ_WINDOW, 1)
        ctx = build_ctx_for_date(d).reshape(1, -1)

        # forward
        pred = mdl_ll.predict({"seq_totales": seq, "ctx": ctx}, verbose=0)

        # Compatibilidad: el modelo puede devolver dict o lista
        if isinstance(pred, dict):
            y_total = float(pred["y_total"].reshape(-1)[0])
            y_profile = pred["y_profile"].reshape(-1)
        else:
            # orden de salidas: [y_total, y_profile] según construcción
            y_total = float(pred[0].reshape(-1)[0])
            y_profile = pred[1].reshape(-1)

        # Sanitizar perfil
        y_profile = np.maximum(y_profile, 1e-9)
        y_profile = y_profile / y_profile.sum()

        # No negativos
        y_total = max(y_total, 0.0)

        # Append
        results.append(DayForecast(date=d, total=y_total, profile=y_profile.tolist()))
        # extender histórico (autoregresivo)
        history.loc[d] = y_total

    return results


# --------------------------------------------------------------------------------------
# Guardados
# --------------------------------------------------------------------------------------
def save_daily_json(results: List[DayForecast]):
    out = [{"date": r.date.isoformat(), "total": round(float(r.total), 3)} for r in results]
    # data_out
    with open(os.path.join(DATA_OUT, "predicciones.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    # public
    with open(os.path.join(PUBLIC, "predicciones.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def save_hourly_csv_and_json(results: List[DayForecast]):
    rows = []
    for r in results:
        for h in range(24):
            val = r.total * r.profile[h]
            rows.append({
                "date": r.date.isoformat(),
                "hour": h,
                "total": float(r.total),
                "profile": float(r.profile[h]),
                "value": float(val),
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_OUT, "predicciones.csv"), index=False)
    with open(os.path.join(DATA_OUT, "predicciones.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)


def save_erlang_stub(results: List[DayForecast]):
    """
    Deja un JSON espejo para 'erlang_forecast' (si tu front lo espera).
    """
    erows = []
    for r in results:
        erows.append({"date": r.date.isoformat(), "calls": float(r.total)})
    # data_out
    with open(os.path.join(DATA_OUT, "erlang_forecast.json"), "w", encoding="utf-8") as f:
        json.dump(erows, f, ensure_ascii=False, indent=2)
    # public
    with open(os.path.join(PUBLIC, "erlang_forecast.json"), "w", encoding="utf-8") as f:
        json.dump(erows, f, ensure_ascii=False, indent=2)


def save_last_update():
    info = {
        "ts_utc": dt.datetime.utcnow().isoformat() + "Z"
    }
    with open(os.path.join(PUBLIC, "last_update.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    # 1) Modelos
    p_ll, _p_tmo = ensure_models_from_release()
    if not p_ll or not os.path.exists(p_ll):
        raise RuntimeError("No se encontró el modelo de llamadas (.keras) en ./models")

    # 2) Cargar Keras 3 de forma robusta (soluciona tu error de 'Orthogonal')
    mdl_ll = load_keras3(p_ll)

    # 3) Semilla histórica
    hist = load_history_totals()

    # 4) Forecast
    results = forecast_days(mdl_ll, hist, horizon=HORIZON_DAYS)

    # 5) Guardar
    save_daily_json(results)
    save_hourly_csv_and_json(results)
    save_erlang_stub(results)
    save_last_update()

    print(f"[VAL] Total -> días generados: {len(results)}")
    print("Guardado outputs en:", DATA_OUT, "y", PUBLIC)


if __name__ == "__main__":
    main()

