# forecast3m.py
# Inferencia conjunta (recibidos + tmo) + dimensionamiento con Erlang-C
# Descarga modelos desde el último Release y publica JSON/CSV listos para front.

import os
import json
import joblib
import numpy as np
import pandas as pd
from utils_release import download_asset_from_latest

# ---------- Parámetros ----------
OWNER = "Supervision-Inbound"         # <- ajusta si cambia
REPO  = "wf-Analytics-AI2.5"          # <- exacto

ASSET_LLAMADAS = "modelo_llamadas.pkl"
ASSET_TMO      = "modelo_tmo.pkl"

MODELS_DIR = "models"

OUT_CSV            = "data_out/predicciones.csv"
OUT_JSON_PUBLIC    = "public/predicciones.json"
OUT_JSON_DATAOUT   = "data_out/predicciones.json"
OUT_JSON_ERLANG    = "public/erlang_forecast.json"   # <--- NUEVO
OUT_JSON_ERLANG_DO = "data_out/erlang_forecast.json" # <--- NUEVO
STAMP_JSON         = "public/last_update.json"

HOURS_AHEAD = 24 * 90  # 90 días
FREQ        = "H"

# Nombres como se entrenó
TARGET_LLAMADAS = "recibidos"
TARGET_TMO      = "tmo_seg"

# Semillas si no hay estado
DEFAULT_LL  = 100.0
DEFAULT_TMO = 180.0

# Parámetros de operación (puedes cambiar arriba si hace falta)
SLA_TARGET   = 0.90   # 90%
ASA_TARGET_S = 22     # 22 segundos
MAX_OCC      = 0.85   # 85% de ocupación máxima
SHRINKAGE    = 0.30   # 30% de ausentismo/pausas (valor legacy; ahora derivamos desde turnos)

# ===== NUEVO: Definición explícita de turno/productividad =====
# Turno de 10h, 1h colación, 2 breaks de 15m, y 15% de estados auxiliares.
SHIFT_HOURS = 10.0
LUNCH_HOURS = 1.0
BREAKS_MIN  = [15, 15]   # minutos
AUX_RATE    = 0.15       # 15% del tiempo productivo en estados auxiliares

def scheduled_productivity_factor(shift_hours: float,
                                  lunch_hours: float,
                                  breaks_min: list,
                                  aux_rate: float) -> float:
    """
    Retorna el factor de productividad programada (horas neto productivas / horas agendadas).
    - shift_hours: horas del turno (agendadas)
    - lunch_hours: horas de colación (no productivas)
    - breaks_min: lista de breaks en minutos
    - aux_rate: proporción de estados auxiliares sobre el tiempo productivo bruto
    """
    breaks_h = sum(breaks_min) / 60.0
    productive_hours = max(0.0, shift_hours - lunch_hours - breaks_h)  # 10 - 1 - 0.5 = 8.5
    net_productive_hours = productive_hours * (1.0 - aux_rate)         # 8.5 * 0.85 = 7.225
    return (net_productive_hours / shift_hours) if shift_hours > 0 else 0.0

# --------------------------------

# ===== Erlang-C utils =====
def erlang_c_prob_wait(N: int, A: float) -> float:
    """Probabilidad de espera (Erlang C)."""
    if A <= 0: return 0.0
    if N <= 0: return 1.0
    if A >= N: return 1.0
    # sumatoria estable
    summ = 0.0
    term = 1.0
    for k in range(N):
        if k > 0:
            term *= A / k
        summ += term
    pn = term * (A / N) / (1 - A / N)
    p_wait = pn / (summ + pn)
    return float(p_wait)

def service_level(A: float, N: int, AHT: float, T: float) -> float:
    """SL = P(espera <= T) con Erlang C y aproximación clásica."""
    if N <= 0 or A <= 0: return 0.0
    if A >= N: return 0.0
    pw = erlang_c_prob_wait(N, A)
    # Fórmula: 1 - Pw * exp(-(N-A)*(T/AHT))
    return 1.0 - pw * np.exp(-(N - A) * (T / AHT))

def required_agents(calls_h: float, aht_s: float,
                    sla_target: float, asa_s: float,
                    max_occ: float, shrinkage: float) -> dict:
    """
    Devuelve N (agentes productivos) y N_sched (agendados con shrinkage),
    junto con métricas intermedias.
    """
    calls_h = max(0.0, float(calls_h))
    aht_s   = max(1.0, float(aht_s))   # evitar división por 0

    lamb = calls_h / 3600.0            # llamadas por segundo (intervalo de 1 hora)
    A = lamb * aht_s                    # Erlangs

    # piso por ocupación
    N_occ_min = int(np.ceil(A / max_occ)) if max_occ > 0 else int(np.ceil(A + 1))
    # piso por estabilidad
    N = max(N_occ_min, int(np.ceil(A)) + 1)

    # subir hasta cumplir SLA
    for _ in range(1000):
        sl = service_level(A, N, aht_s, asa_s)
        if sl >= sla_target and (A / N) <= max_occ:
            break
        N += 1

    N_productive = int(N)
    N_scheduled  = int(np.ceil(N_productive / (1.0 - shrinkage))) if (1.0 - shrinkage) > 0 else N_productive

    return {
        "A_erlangs": float(A),
        "N_productive": N_productive,
        "N_scheduled": N_scheduled,
        "occupancy": float(A / N_productive) if N_productive > 0 else 0.0,
        "service_level": float(service_level(A, N_productive, aht_s, asa_s))
    }

# ===== Features utils =====
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

    # 3) Timeline (UTC -> America/Santiago -> naive)
    start = pd.Timestamp.utcnow().floor("H")
    idx = pd.date_range(start, periods=HOURS_AHEAD, freq=FREQ, tz="UTC").tz_convert("America/Santiago").tz_localize(None)
    df = pd.DataFrame({"ts": idx})
    df = add_time_features(df)

    # 4) Semillas
    df["seed_ll"]  = DEFAULT_LL
    df["seed_tmo"] = DEFAULT_TMO

    # ===== Llamadas (recibidos_*) =====
    df[f"{TARGET_LLAMADAS}_lag1"]        = df["seed_ll"].shift(1)
    df[f"{TARGET_LLAMADAS}_lag24"]       = df["seed_ll"].shift(24)
    df[f"{TARGET_LLAMADAS}_ma24"]        = df["seed_ll"].rolling(24, min_periods=1).mean()
    df[f"{TARGET_LLAMADAS}_ma168"]       = df["seed_ll"].rolling(24*7, min_periods=1).mean()
    df[f"{TARGET_LLAMADAS}_samehour_7d"] = df["seed_ll"].shift(24*7)

    # ===== TMO (tmo_seg_*) =====
    df[f"{TARGET_TMO}_lag1"]        = df["seed_tmo"].shift(1)
    df[f"{TARGET_TMO}_lag24"]       = df["seed_tmo"].shift(24)
    df[f"{TARGET_TMO}_ma24"]        = df["seed_tmo"].rolling(24, min_periods=1).mean()
    df[f"{TARGET_TMO}_ma168"]       = df["seed_tmo"].rolling(24*7, min_periods=1).mean()
    df[f"{TARGET_TMO}_samehour_7d"] = df["seed_tmo"].shift(24*7)

    # 5) Predicción modelos
    X_ll  = build_feature_matrix(df, TARGET_LLAMADAS).fillna(method="bfill").fillna(method="ffill")
    pred_ll = mdl_ll.predict(X_ll)

    if mdl_tmo is not None:
        X_tmo = build_feature_matrix(df, TARGET_TMO).fillna(method="bfill").fillna(method="ffill")
        pred_tmo = mdl_tmo.predict(X_tmo)
    else:
        pred_tmo = np.full(len(df), DEFAULT_TMO)

    # Redondeo a enteros (como pediste)
    pred_ll_int  = np.maximum(0, np.rint(pred_ll).astype(int))
    pred_tmo_int = np.maximum(0, np.rint(pred_tmo).astype(int))

    out = pd.DataFrame({
        "ts": df["ts"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "pred_llamadas": pred_ll_int,
        "pred_tmo_seg": pred_tmo_int
    })

    # 6) Guardar CSV y JSONs de predicciones
    out.to_csv(OUT_CSV, index=False)
    payload = out.to_dict(orient="records")
    with open(OUT_JSON_PUBLIC, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(OUT_JSON_DATAOUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # ===== NUEVO: calcular productividad programada y shrinkage derivado =====
    prod_factor = scheduled_productivity_factor(
        SHIFT_HOURS, LUNCH_HOURS, BREAKS_MIN, AUX_RATE
    )
    derived_shrinkage = max(0.0, min(1.0, 1.0 - prod_factor))
    # print opcional de control
    print(f"[Turno] factor_productividad={prod_factor:.4f} -> shrinkage_derivado={derived_shrinkage:.4f}")

    # 7) Calcular Erlang-C y agentes (enteros)
    erlang_rows = []
    for row in payload:
        calls_h = int(row["pred_llamadas"])
        aht_s   = int(row["pred_tmo_seg"])
        dims = required_agents(
            calls_h=calls_h,
            aht_s=aht_s,
            sla_target=SLA_TARGET,
            asa_s=ASA_TARGET_S,
            max_occ=MAX_OCC,
            shrinkage=derived_shrinkage   # <--- usar shrinkage derivado del turno real
        )
        erlang_rows.append({
            "ts": row["ts"],
            "llamadas": int(calls_h),
            "tmo_seg": int(aht_s),
            "erlangs": round(dims["A_erlangs"], 4),
            "agentes_productivos": int(dims["N_productive"]),
            "agentes_agendados": int(dims["N_scheduled"]),
            "occupancy": round(dims["occupancy"], 4),
            "service_level": round(dims["service_level"], 4),
            "params": {
                "SLA_TARGET": SLA_TARGET,
                "ASA_TARGET_S": ASA_TARGET_S,
                "MAX_OCC": MAX_OCC,
                "SHRINKAGE_INPUT": SHRINKAGE,            # valor legacy configurado
                "SHIFT_HOURS": SHIFT_HOURS,
                "LUNCH_HOURS": LUNCH_HOURS,
                "BREAKS_MIN": BREAKS_MIN,
                "AUX_RATE": AUX_RATE,
                "DERIVED_SHRINKAGE": round(derived_shrinkage, 4),
                "PRODUCTIVITY_FACTOR": round(prod_factor, 4)
            }
        })

    with open(OUT_JSON_ERLANG, "w", encoding="utf-8") as f:
        json.dump(erlang_rows, f, ensure_ascii=False, indent=2)
    with open(OUT_JSON_ERLANG_DO, "w", encoding="utf-8") as f:
        json.dump(erlang_rows, f, ensure_ascii=False, indent=2)

    # 8) Timestamp para forzar actualización
    stamp = {
        "generated_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "horizon_hours": HOURS_AHEAD,
        "records": len(out)
    }
    with open(STAMP_JSON, "w", encoding="utf-8") as f:
        json.dump(stamp, f, ensure_ascii=False, indent=2)

    print(f"OK -> {OUT_CSV}")
    print(f"OK -> {OUT_JSON_PUBLIC}")
    print(f"OK -> {OUT_JSON_DATAOUT}")
    print(f"OK -> {OUT_JSON_ERLANG}")
    print(f"OK -> {OUT_JSON_ERLANG_DO}")
    print(f"OK -> {STAMP_JSON}")

if __name__ == "__main__":
    main()

