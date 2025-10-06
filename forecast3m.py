# forecast3m.py
# Inferencia para modelo NN de llamadas (Keras 3) -> salida por HORA
# - Carga modelo: models/modelo_llamadas_nn.keras
# - Lee histórico horario (igual que training): Hosting ia.xlsx
# - Repite pipeline (features) y predice próximos días (horizonte configurable)
# - Exporta: fecha, hora, llamadas (CSV/JSON) + last_update.json
# ----------------------------------------------------------------------

import os
import io
import json
import math
import zipfile
import datetime as dt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

# Keras 3 / TF CPU
from keras.models import load_model

# Opcional: si sigues bajando modelos desde el Release
import requests

# -------------------- Parámetros editables --------------------
RUTA_DATA        = "Hosting ia.xlsx"  # mismo archivo del entrenamiento en el repo
HOJA_EXCEL       = 0
COL_FECHA        = "fecha"
COL_HORA         = "hora"
COL_LLAMADAS     = "recibidos"

# Horizonte de predicción en días (3 meses aprox.)
HORIZON_DAYS     = 90
SEQ_DIAS         = 28   # igual que en training

# Paths salida
DATAOUT_DIR      = "data_out"
PUBLIC_DIR       = "public"
CSV_OUT_PATH     = f"{DATAOUT_DIR}/predicciones.csv"
JSON_OUT_PATH    = f"{DATAOUT_DIR}/predicciones.json"
PUBLIC_JSON_PATH = f"{PUBLIC_DIR}/predicciones.json"
ERLANG_JSON_OUT  = f"{DATAOUT_DIR}/erlang_forecast.json"
ERLANG_JSON_PUB  = f"{PUBLIC_DIR}/erlang_forecast.json"
LAST_UPDATE_JSON = f"{PUBLIC_DIR}/last_update.json"

# Release (opcional si descargas modelo)
RELEASE_MODELS_URL = os.environ.get("RELEASE_MODELS_URL", "")  # si lo usas, pon la URL en el action
NEED_DOWNLOAD      = True  # si quieres descargar desde release; si ya está en repo, pon False
MODEL_DIR          = "models"
MODEL_FILE         = f"{MODEL_DIR}/modelo_llamadas_nn.keras"

os.makedirs(DATAOUT_DIR, exist_ok=True)
os.makedirs(PUBLIC_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------- Utils de features (mismo training) --------------------
def read_data(path, hoja=None):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path, sheet_name=hoja)
    if ext == ".csv":
        return pd.read_csv(path)
    raise ValueError("Formato no soportado (usa .xlsx/.xls o .csv)")

def ensure_datetime(df, col_fecha, col_hora):
    df["fecha_dt"] = pd.to_datetime(df[col_fecha], errors="coerce", dayfirst=True).dt.date
    df["hora_str"] = df[col_hora].astype(str).str.slice(0,5)
    df["ts"] = pd.to_datetime(df["fecha_dt"].astype(str)+" "+df["hora_str"], errors="coerce")
    return df.dropna(subset=["ts"]).copy()

def add_time_features(df):
    df["dow"] = df["ts"].dt.dayofweek
    df["month"] = df["ts"].dt.month
    df["hour"] = df["ts"].dt.hour
    return df

def robust_baseline_by_dow_hour(df, y):
    grp = df.groupby(["dow","hour"])[y].agg(["median"]).rename(columns={"median":"med"})
    def mad(x):
        m = np.median(x)
        return np.median(np.abs(x - m))
    grp["mad"] = df.groupby(["dow","hour"])[y].apply(mad).values
    return grp

def detect_peaks(df, y, mad_k=3.5, min_consec=1):
    base = robust_baseline_by_dow_hour(df, y)
    df = df.merge(base, left_on=["dow","hour"], right_index=True, how="left")
    df["upper_cap"] = df["med"] + mad_k * df["mad"].replace(0, df["mad"].median())
    df["is_peak"] = (df[y] > df["upper_cap"]).astype(int)
    if min_consec > 1:
        df = df.sort_values("ts")
        runs = (df["is_peak"].diff(1) != 0).cumsum()
        sizes = df.groupby(runs)["is_peak"].transform("sum")
        df["is_peak"] = np.where((df["is_peak"]==1) & (sizes>=min_consec), 1, 0)
    return df

def smooth_series(df, y, method="cap"):
    # (igual que training) – si hay pico, recorta a upper_cap
    df[y+"_smooth"] = np.where(df["is_peak"]==1,
                               df["upper_cap"] if method=="cap" else df["med"],
                               df[y])
    return df

def build_daily_matrices(df, target_col_smooth="recibidos_smooth"):
    # Pivot a 24 columnas por hora + contexto del día
    dfc = df.copy()
    dfc["date"] = dfc["ts"].dt.date
    piv = dfc.pivot_table(index="date", columns="hour", values=target_col_smooth, aggfunc="sum").fillna(0.0)
    piv = piv.reindex(columns=range(24), fill_value=0.0)
    piv.columns = [f"h{h:02d}" for h in piv.columns]

    first = dfc.sort_values("ts").groupby("date").first().reset_index()
    ctx = first[["date","dow","month"]].copy()
    ctx["sin_month"] = np.sin(2*np.pi*ctx["month"]/12); ctx["cos_month"] = np.cos(2*np.pi*ctx["month"]/12)
    ctx["sin_dow"]   = np.sin(2*np.pi*ctx["dow"]/7);    ctx["cos_dow"]   = np.cos(2*np.pi*ctx["dow"]/7)

    day = ctx.merge(piv.reset_index(), on="date", how="inner")
    h_cols = [c for c in day.columns if c.startswith("h")]
    day["total"] = day[h_cols].sum(axis=1)

    prof = day[h_cols].div(day["total"].replace(0, np.nan), axis=0).fillna(1.0/24.0)
    prof.columns = [c.replace("h","p") for c in prof.columns]

    day = pd.concat([day, prof], axis=1).sort_values("date").reset_index(drop=True)
    return day, {"h_cols":h_cols, "p_cols":[c for c in day.columns if c.startswith("p")]}

def make_sequences(day_df, seq_len=28):
    # Devuelve secuencia de totales por día (para último tramo histórico)
    day = day_df.copy().reset_index(drop=True)
    totals = day["total"].values.astype(np.float32)
    if len(totals) < seq_len:
        raise ValueError(f"Histórico insuficiente: necesito >= {seq_len} días y hay {len(totals)}")
    seq = totals[-seq_len:]  # última ventana
    return seq.reshape(1, seq_len, 1).astype(np.float32)

def ctx_for_date(d):
    dow = d.weekday()
    month = d.month
    sin_month = np.sin(2*np.pi*month/12); cos_month = np.cos(2*np.pi*month/12)
    sin_dow   = np.sin(2*np.pi*dow/7);    cos_dow   = np.cos(2*np.pi*dow/7)
    return np.array([[dow, month, sin_month, cos_month, sin_dow, cos_dow]], dtype=np.float32)

def hourly_from_heads(y_total, y_profile):
    # y_total ya debe venir en escala LINEAL (expm1 aplicado)
    return y_total * y_profile

# -------------------- Descarga opcional desde Release --------------------
def download_models_from_release():
    if not NEED_DOWNLOAD or not RELEASE_MODELS_URL:
        print("Saltando descarga de Release (NEED_DOWNLOAD=False o sin URL).")
        return
    print("Descargando modelos desde Release…")
    r = requests.get(RELEASE_MODELS_URL, timeout=60)
    r.raise_for_status()
    # Acepta: .keras directo o zip con modelos
    if RELEASE_MODELS_URL.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            zf.extractall(".")
    else:
        with open(MODEL_FILE, "wb") as f:
            f.write(r.content)
    print("Modelos descargados.")

# -------------------- Main --------------------
def main():
    download_models_from_release()

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"No encuentro el modelo NN en {MODEL_FILE}")

    print("Cargando modelo Keras 3:", MODEL_FILE)
    mdl = load_model(MODEL_FILE, compile=False)  # importante en Keras 3

    print("Leyendo histórico:", RUTA_DATA)
    df = read_data(RUTA_DATA, hoja=HOJA_EXCEL).copy()
    if COL_LLAMADAS not in df.columns:
        raise ValueError(f"Falta columna '{COL_LLAMADAS}'")

    # Mismo pipeline del entrenamiento
    df = ensure_datetime(df, COL_FECHA, COL_HORA)
    df = add_time_features(df).sort_values("ts").reset_index(drop=True)

    # Suavizado CAP (como training). Ajusta K y método si cambiaste en el entreno
    df_pk = detect_peaks(df.copy(), COL_LLAMADAS, mad_k=3.5, min_consec=1)
    df_pk = smooth_series(df_pk, COL_LLAMADAS, method="cap")
    df_pk = df_pk.rename(columns={COL_LLAMADAS+"_smooth": "recibidos_smooth"})

    # Día -> total + perfil
    day, meta = build_daily_matrices(df_pk, "recibidos_smooth")

    # Última ventana de 28 días (totales) para iniciar la proyección
    seq = make_sequences(day, seq_len=SEQ_DIAS)  # shape (1, 28, 1)

    # Predicción iterativa día a día
    start_date = day["date"].max() + dt.timedelta(days=1)
    all_rows = []

    cur_seq = seq.copy()  # vamos desplazando la ventana
    for d in range(HORIZON_DAYS):
        date_d = start_date + dt.timedelta(days=d)
        X_ctx  = ctx_for_date(date_d)   # shape (1, 6)

        # y_total_pred viene en escala LOG1P en el training,
        # así que aquí INVERSIÓN: expm1 para volver a llamadas reales
        y_total_log, y_profile = mdl.predict({"seq_totales":cur_seq, "ctx":X_ctx}, verbose=0)
        # Si tu modelo se guardó con total ya en lineal, comenta la línea de expm1:
        y_total = np.expm1(y_total_log).clip(min=0.0).reshape(-1)   # -> (1,)
        y_prof  = y_profile.reshape(-1)                             # -> (24,)

        # Asegurar perfil válido
        if not np.isfinite(y_prof).all() or y_prof.sum() <= 0:
            y_prof = np.ones(24, dtype=np.float32) / 24.0
        else:
            y_prof = y_prof / y_prof.sum()

        hourly = (y_total[0] * y_prof).astype(float)  # 24 horas

        # Guardar filas por hora
        for h in range(24):
            all_rows.append({
                "fecha": date_d.strftime("%Y-%m-%d"),
                "hora": f"{h:02d}:00",
                "llamadas": float(hourly[h])
            })

        # Desplazar ventana de totales (agregar y_total del día recién predicho)
        next_total = y_total[0]
        cur_seq = np.concatenate([cur_seq[:,1:,:], np.array([[[next_total]]], dtype=np.float32)], axis=1)

    # Exportar
    pred_df = pd.DataFrame(all_rows, columns=["fecha","hora","llamadas"])
    # Redondeo opcional (negocios suelen querer enteros)
    pred_df["llamadas"] = pred_df["llamadas"].clip(lower=0).round(0)

    pred_df.to_csv(CSV_OUT_PATH, index=False)
    pred_df.to_json(JSON_OUT_PATH, orient="records", force_ascii=False)

    # duplicado a public/
    pred_df.to_json(PUBLIC_JSON_PATH, orient="records", force_ascii=False)

    # erlang (si no lo usas ahora, igual dejamos un stub coherente)
    erlang_stub = {
        "version": "nn_v1",
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "note": "Erlang no calculado en esta versión; usando 0 por defecto."
    }
    with open(ERLANG_JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(erlang_stub, f, ensure_ascii=False)
    with open(ERLANG_JSON_PUB, "w", encoding="utf-8") as f:
        json.dump(erlang_stub, f, ensure_ascii=False)

    # last_update
    last_update = {"updated_at_utc": dt.datetime.utcnow().isoformat() + "Z"}
    with open(LAST_UPDATE_JSON, "w", encoding="utf-8") as f:
        json.dump(last_update, f, ensure_ascii=False)

    print(f"OK -> {CSV_OUT_PATH}  |  {PUBLIC_JSON_PATH}")

if __name__ == "__main__":
    main()

