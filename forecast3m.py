# -*- coding: utf-8 -*-
"""
forecast3m.py (v2)
- Publica diarios y horarios en /public
- Des-normaliza las predicciones si hay meta o env vars
"""

from __future__ import annotations
import os, json, math, shutil, datetime as dt
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests

import keras
from keras.models import load_model as keras_load_model

DATA_OUT = "data_out"
PUBLIC = "public"
MODELS_DIR = "models"

LLAMADAS_MODEL_NAME = "modelo_llamadas_nn.keras"
TMO_MODEL_NAME = "modelo_tmo.pkl"          # opcional
TRAIN_META_NAME = "training_meta.json"     # opcional (con y_mode)

SEQ_WINDOW = 28
HORIZON_DAYS = 90

os.makedirs(DATA_OUT, exist_ok=True)
os.makedirs(PUBLIC, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------- Release download ----------------
def download_asset_from_latest_release(asset_name: str, dst_path: str) -> bool:
    repo = os.environ.get("GITHUB_REPOSITORY")
    token = os.environ.get("GITHUB_TOKEN")
    if not repo or not token:
        raise RuntimeError("Faltan GITHUB_REPOSITORY o GITHUB_TOKEN.")
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {token}",
                      "Accept": "application/vnd.github+json"})
    rel = s.get(f"https://api.github.com/repos/{repo}/releases/latest", timeout=60).json()
    asset = next((a for a in rel.get("assets", []) if a.get("name")==asset_name), None)
    if not asset:
        print(f"[download_asset] No está '{asset_name}' en el último release.")
        return False
    with s.get(asset["url"], headers={"Accept":"application/octet-stream"}, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dst_path, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    print(f"[download_asset] Descargado: {asset_name} -> {dst_path}")
    return True

def ensure_models_from_release() -> Tuple[str|None, str|None, str|None]:
    ll = os.path.join(MODELS_DIR, LLAMADAS_MODEL_NAME)
    tmo = os.path.join(MODELS_DIR, TMO_MODEL_NAME)
    meta = os.path.join(MODELS_DIR, TRAIN_META_NAME)

    print("Descargando modelos desde Release…")
    ok_ll = download_asset_from_latest_release(LLAMADAS_MODEL_NAME, ll)
    ok_tmo = download_asset_from_latest_release(TMO_MODEL_NAME, tmo)
    ok_meta = download_asset_from_latest_release(TRAIN_META_NAME, meta)
    print("Modelos descargados:", ll if ok_ll else "(no)", tmo if ok_tmo else "(no)", meta if ok_meta else "(no)")
    return (ll if ok_ll else None, tmo if ok_tmo else None, meta if ok_meta else None)

# ---------------- Keras load robust ----------------
def load_keras3(path: str):
    try:
        return keras_load_model(path, compile=False)
    except Exception as e1:
        print("[load_keras3] safe_mode=False ::", e1)
        try:
            return keras_load_model(path, compile=False, safe_mode=False)
        except Exception as e2:
            print("[load_keras3] custom_objects ::", e2)
            return keras_load_model(
                path, compile=False, safe_mode=False,
                custom_objects={"Orthogonal": keras.initializers.Orthogonal},
            )

# ---------------- Post-proc (des-normalización) ----------------
@dataclass
class YPost:
    mode: str = "none"    # none | log1p | linear
    scale: float = 1.0    # para linear: y = y*scale + offset
    offset: float = 0.0

def load_y_post(meta_path: str|None) -> YPost:
    # 1) meta del release
    if meta_path and os.path.exists(meta_path):
        try:
            m = json.load(open(meta_path, "r", encoding="utf-8"))
            if m.get("y_mode") == "log1p":
                return YPost("log1p", 1.0, 0.0)
            if m.get("y_mode") == "linear":
                return YPost("linear", float(m.get("scale",1.0)), float(m.get("offset",0.0)))
        except Exception as e:
            print("[load_y_post] meta inválida:", e)
    # 2) env vars manuales (sobrescriben)
    mode = os.getenv("Y_MODE", "").lower()
    if mode in ("log1p","linear","none"):
        if mode=="linear":
            return YPost("linear", float(os.getenv("Y_SCALE","1.0")), float(os.getenv("Y_OFFSET","0.0")))
        if mode=="log1p":
            return YPost("log1p", 1.0, 0.0)
        return YPost("none", 1.0, 0.0)
    return YPost()  # none

def post_y(y: float, cfg: YPost) -> float:
    if cfg.mode == "log1p":
        # inversa de log1p
        return max(math.expm1(y), 0.0)
    if cfg.mode == "linear":
        return max(y * cfg.scale + cfg.offset, 0.0)
    return max(y, 0.0)

# ---------------- Calendario ----------------
def season_from_month(m: int) -> int:
    if m in (12,1,2): return 0
    if m in (3,4,5):  return 1
    if m in (6,7,8):  return 2
    return 3

def build_ctx_for_date(d: dt.date) -> np.ndarray:
    dow = d.weekday()
    month = d.month
    doy = d.timetuple().tm_yday
    sin_doy = math.sin(2*math.pi*doy/366.0)
    cos_doy = math.cos(2*math.pi*doy/366.0)
    is_weekend = 1 if dow>=5 else 0
    seas = season_from_month(month)
    return np.array([dow, month, sin_doy, cos_doy, is_weekend, seas], dtype="float32")

# ---------------- Historia para semilla ----------------
def load_history_totals() -> pd.Series:
    csv_path = os.path.join(DATA_OUT, "train_daily_samples.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if {"date","total"}.issubset(df.columns):
            s = pd.Series(df["total"].astype(float).values,
                          index=pd.to_datetime(df["date"]).dt.date).sort_index()
            if len(s) >= SEQ_WINDOW:
                return s
    pub_json = os.path.join(PUBLIC, "predicciones_daily.json")
    if os.path.exists(pub_json):
        try:
            rows = json.load(open(pub_json, "r", encoding="utf-8"))
            ser = pd.Series({dt.date.fromisoformat(r["date"]): float(r["total"]) for r in rows}).sort_index()
            if len(ser) >= SEQ_WINDOW:
                return ser
        except: pass
    dates = [dt.date.today() - dt.timedelta(days=i) for i in range(SEQ_WINDOW, 0, -1)]
    return pd.Series([100.0]*SEQ_WINDOW, index=dates)

# ---------------- Inferencia ----------------
@dataclass
class DayForecast:
    date: dt.date
    total: float
    profile: List[float]

def forecast_days(mdl_ll, seed: pd.Series, horizon: int, ypost: YPost) -> List[DayForecast]:
    hist = seed.copy().sort_index()
    last = hist.index[-1]
    out: List[DayForecast] = []

    for step in range(horizon):
        d = last + dt.timedelta(days=step+1)
        seq = hist.iloc[-SEQ_WINDOW:].values.astype("float32").reshape(1, SEQ_WINDOW, 1)
        ctx = build_ctx_for_date(d).reshape(1, -1)
        pred = mdl_ll.predict({"seq_totales": seq, "ctx": ctx}, verbose=0)
        if isinstance(pred, dict):
            y_total = float(pred["y_total"].reshape(-1)[0])
            y_profile = pred["y_profile"].reshape(-1)
        else:
            y_total = float(pred[0].reshape(-1)[0])
            y_profile = pred[1].reshape(-1)

        # des-normalizar + sanear
        y_total = post_y(y_total, ypost)
        y_profile = np.maximum(y_profile, 1e-9)
        y_profile = y_profile / y_profile.sum()

        out.append(DayForecast(d, y_total, y_profile.tolist()))
        hist.loc[d] = y_total
    return out

# ---------------- Guardados ----------------
def save_daily(results: List[DayForecast]):
    daily = [{"date": r.date.isoformat(), "total": round(float(r.total), 3)} for r in results]
    # compatibilidad
    json.dump(daily, open(os.path.join(PUBLIC, "predicciones.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    # nuevo nombre explícito
    json.dump(daily, open(os.path.join(PUBLIC, "predicciones_daily.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(daily, open(os.path.join(DATA_OUT, "predicciones_daily.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def save_hourly(results: List[DayForecast]):
    rows = []
    by_day = {}
    for r in results:
        values = []
        for h in range(24):
            v = float(r.total * r.profile[h])
            rows.append({"date": r.date.isoformat(), "hour": h,
                         "total": float(r.total), "profile": float(r.profile[h]), "value": v})
            values.append(v)
        by_day[r.date.isoformat()] = values

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_OUT, "predicciones.csv"), index=False)
    json.dump(rows, open(os.path.join(DATA_OUT, "predicciones.json"), "w", encoding="utf-8"), ensure_ascii=False)
    # publicar también en /public
    json.dump(rows, open(os.path.join(PUBLIC, "predicciones_hourly.json"), "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(by_day, open(os.path.join(PUBLIC, "predicciones_by_day.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def save_erlang_stub(results: List[DayForecast]):
    erows = [{"date": r.date.isoformat(), "calls": float(r.total)} for r in results]
    json.dump(erows, open(os.path.join(DATA_OUT, "erlang_forecast.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(erows, open(os.path.join(PUBLIC, "erlang_forecast.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def save_last_update():
    json.dump({"ts_utc": dt.datetime.utcnow().isoformat()+"Z"},
              open(os.path.join(PUBLIC, "last_update.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

# ---------------- main ----------------
def main():
    p_ll, _p_tmo, p_meta = ensure_models_from_release()
    if not p_ll or not os.path.exists(p_ll):
        raise RuntimeError("Falta modelo .keras")
    mdl_ll = load_keras3(p_ll)

    ypost = load_y_post(p_meta)  # des-normalización
    print(f"[y_post] mode={ypost.mode} scale={ypost.scale} offset={ypost.offset}")

    hist = load_history_totals()
    res = forecast_days(mdl_ll, hist, HORIZON_DAYS, ypost)

    save_daily(res)
    save_hourly(res)
    save_erlang_stub(res)
    save_last_update()
    print(f"[DONE] Días={len(res)} | Outputs en {DATA_OUT} y {PUBLIC}")

if __name__ == "__main__":
    main()

