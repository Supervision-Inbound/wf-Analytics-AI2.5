# forecast3m.py
# Inferencia conjunta (recibidos + tmo) + dimensionamiento con Erlang-C/A
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
OUT_JSON_ERLANG    = "public/erlang_forecast.json"
OUT_JSON_ERLANG_DO = "data_out/erlang_forecast.json"
STAMP_JSON         = "public/last_update.json"

# (HOURS_AHEAD ya no se usa; mantenido para compatibilidad si otros scripts lo leen)
HOURS_AHEAD = 24 * 90
FREQ        = "H"

# Nombres como se entrenó
TARGET_LLAMADAS = "recibidos"
TARGET_TMO      = "tmo_seg"

# Semillas si no hay estado
DEFAULT_LL  = 100.0
DEFAULT_TMO = 180.0

# Parámetros de operación
SLA_TARGET   = 0.90   # 90%
ASA_TARGET_S = 22     # 22 segundos
MAX_OCC      = 0.85   # 85% ocupación máxima
SHRINKAGE    = 0.30   # legacy (se mantiene visible en JSON)

# ===== Turno/productividad (tu realidad operativa) =====
# Turno 10h, 1h colación, 2 breaks de 15m, 15% auxiliares
SHIFT_HOURS = 10.0
LUNCH_HOURS = 1.0
BREAKS_MIN  = [15, 15]   # minutos
AUX_RATE    = 0.15

# ===== Erlang-A (abandono/paciencia) + restricciones adicionales =====
USE_ERLANG_A       = True      # si False -> Erlang-C tradicional
MEAN_PATIENCE_S    = 60.0      # paciencia media ~ exponencial
ABANDON_MAX        = 0.06      # abandono máximo permitido (<= 6%)
AWT_MAX_S          = 120.0     # tiempo medio de espera objetivo (segundos)
USE_STRICT_OCC_CAP = True      # respetar tope de ocupación

# ===== Pausa legal entre llamadas =====
INTERCALL_GAP_S    = 10.0

# ===== NUEVO: Absentismo mensual aplicado a AGENDADOS =====
ABSENTEEISM_RATE   = 0.23      # 23% mensual (colchón adicional de dotación)

# --------------------------------

# ===== Erlang-C utils =====
def erlang_c_prob_wait(N: int, A: float) -> float:
    """Probabilidad de espera (Erlang C)."""
    if A <= 0: return 0.0
    if N <= 0: return 1.0
    if A >= N: return 1.0
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
    """SL = P(espera <= T) con Erlang-C (clásico, sin abandono)."""
    if N <= 0 or A <= 0: return 0.0
    if A >= N: return 0.0
    pw = erlang_c_prob_wait(N, A)
    return 1.0 - pw * np.exp(-(N - A) * (T / AHT))

def erlang_c_awt(A: float, N: int, AHT: float) -> float:
    """
    AWT (espera media) aproximada para Erlang-C:
    E[W] = Pw * AHT / (N - A), con Pw = Erlang-C.
    """
    if N <= 0 or A <= 0 or N <= A:
        return float('inf')
    pw = erlang_c_prob_wait(N, A)
    return float(pw * (AHT / (N - A)))

# ===== Erlang-A (abandono con paciencia exponencial) =====
def erlang_a_metrics(A: float, N: int, aht_s: float, patience_s: float, T: float):
    """
    Aproximación práctica M/M/N+M (Erlang-A):
      μ = 1/AHT, θ = 1/paciencia, r = μ*(N - A) (si N>A)
    - SL_A (global) = (1 - Pw) + Pw * [ r/(r+θ) * (1 - exp(-(r+θ)*T)) ]
    - Abandono (global) ≈ Pw * [ θ/(r+θ) ]
    - AWT (medio) ≈ Pw * 1/(r+θ)
    """
    if N <= 0 or A <= 0:
        return 0.0, 1.0, float('inf')
    mu = 1.0 / max(aht_s, 1.0)
    theta = 1.0 / max(patience_s, 1.0)
    if N <= A:
        return 0.0, 1.0, float('inf')
    pw = erlang_c_prob_wait(N, A)  # % que espera (baseline C)
    r = mu * (N - A)
    rate = r + theta
    sl_a = (1.0 - pw) + pw * (r / rate) * (1.0 - np.exp(-rate * T))
    aban = pw * (theta / rate)
    awt  = pw * (1.0 / rate)
    # recortes numéricos
    sl_a = float(min(max(sl_a, 0.0), 1.0))
    aban = float(min(max(aban, 0.0), 1.0))
    return sl_a, aban, float(awt)

# ===== Turnos/productividad =====
def scheduled_productivity_factor(shift_hours: float,
                                  lunch_hours: float,
                                  breaks_min: list,
                                  aux_rate: float) -> float:
    """factor de productividad programada = horas neto productivas / horas agendadas."""
    breaks_h = sum(breaks_min) / 60.0
    productive_hours = max(0.0, shift_hours - lunch_hours - breaks_h)  # p.ej. 10 - 1 - 0.5 = 8.5
    net_productive_hours = productive_hours * (1.0 - aux_rate)         # 8.5 * 0.85 = 7.225
    return (net_productive_hours / shift_hours) if shift_hours > 0 else 0.0

# ===== Dimensionamiento =====
def required_agents(calls_h: float, aht_s: float,
                    sla_target: float, asa_s: float,
                    max_occ: float, shrinkage: float,
                    use_erlang_a: bool = USE_ERLANG_A,
                    patience_s: float = MEAN_PATIENCE_S,
                    abandon_max: float = ABANDON_MAX,
                    awt_max_s: float = AWT_MAX_S,
                    intercall_gap_s: float = INTERCALL_GAP_S,
                    use_strict_occ_cap: bool = USE_STRICT_OCC_CAP) -> dict:
    """
    Devuelve N_productive y N_scheduled con métricas:
      - A_erlangs, occupancy, service_level, abandon_rate, avg_wait_s
    Cumple simultáneamente: SLA, ocupación, abandono y AWT máximo.
    Suma la pausa legal entre llamadas al AHT efectivo.
    """
    calls_h = max(0.0, float(calls_h))
    # AHT efectivo = AHT modelo + pausa legal entre llamadas
    aht_eff = max(1.0, float(aht_s) + float(intercall_gap_s))

    lamb = calls_h / 3600.0            # λ (1h)
    A = lamb * aht_eff                  # Erlangs con AHT efectivo

    # piso por ocupación
    N_occ_min = int(np.ceil(A / max_occ)) if (use_strict_occ_cap and max_occ > 0) else int(np.ceil(A + 1))
    # piso por estabilidad
    N = max(N_occ_min, int(np.ceil(A)) + 1)

    service = 0.0
    aban = 1.0
    awt = float('inf')

    # subir hasta cumplir metas
    for _ in range(3000):
        if use_erlang_a:
            service, aban, awt = erlang_a_metrics(A, N, aht_eff, patience_s, asa_s)
        else:
            service = service_level(A, N, aht_eff, asa_s)
            aban = 0.0
            awt  = erlang_c_awt(A, N, aht_eff)
        occ = A / N if N > 0 else 1.0

        cond_sla  = (service >= sla_target)
        cond_occ  = (occ <= max_occ) if use_strict_occ_cap else True
        cond_aban = (aban <= abandon_max) if use_erlang_a else True
        cond_awt  = (awt <= awt_max_s)

        if cond_sla and cond_occ and cond_aban and cond_awt:
            break
        N += 1

    N_productive = int(N)
    N_scheduled  = int(np.ceil(N_productive / (1.0 - shrinkage))) if (1.0 - shrinkage) > 0 else N_productive

    return {
        "A_erlangs": float(A),
        "N_productive": N_productive,
        "N_scheduled": N_scheduled,
        "occupancy": float(A / N_productive) if N_productive > 0 else 0.0,
        "service_level": float(service),
        "abandon_rate": float(aban if use_erlang_a else 0.0),
        "avg_wait_s": float(awt),
        "model": "Erlang-A" if use_erlang_a else "Erlang-C"
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

def build_month_window_idx():
    """
    Devuelve un DatetimeIndex (naive, hora a hora) desde el 1 del mes anterior
    hasta el último día del mes siguiente a las 23:00, en horario local America/Santiago.
    """
    tz = "America/Santiago"
    now_local = pd.Timestamp.now(tz=tz).floor("H")

    current_period = now_local.to_period("M")          # mes actual
    prev_period    = current_period - 1                # mes anterior
    next_period    = current_period + 1                # mes siguiente

    # Inicio: 1 del mes anterior 00:00
    start = pd.Timestamp(year=prev_period.year, month=prev_period.month, day=1,
                         hour=0, minute=0, second=0, tz=tz)

    # Fin: último día del mes siguiente 23:00
    last_day_next = pd.Timestamp(year=next_period.year, month=next_period.month, day=1, tz=tz) + pd.offsets.MonthEnd(1)
    end = last_day_next.replace(hour=23, minute=0, second=0, microsecond=0)

    # Generar rango horario en local y devolver naive (sin tz) manteniendo las horas locales
    idx = pd.date_range(start=start, end=end, freq="H", tz=tz)
    return idx.tz_localize(None)

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

    # 3) Timeline: del 1 del mes anterior al fin del mes siguiente (horas locales)
    idx = build_month_window_idx()
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

    # ===== Productividad programada y shrinkage derivado =====
    prod_factor = scheduled_productivity_factor(
        SHIFT_HOURS, LUNCH_HOURS, BREAKS_MIN, AUX_RATE
    )
    derived_shrinkage = max(0.0, min(1.0, 1.0 - prod_factor))
    # ===== NUEVO: shrinkage efectivo con absentismo =====
    effective_shrinkage = 1.0 - ( (1.0 - derived_shrinkage) * (1.0 - ABSENTEEISM_RATE) )
    print(f"[Turno] productividad={prod_factor:.4f} -> shrinkage_derivado={derived_shrinkage:.4f} -> absentismo={ABSENTEEISM_RATE:.2f} -> shrinkage_efectivo={effective_shrinkage:.4f}")

    # 7) Calcular Erlang (C/A) y agentes
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
            shrinkage=effective_shrinkage,   # <<< usamos shrinkage EFECTIVO (turno + absentismo)
            use_erlang_a=USE_ERLANG_A,
            patience_s=MEAN_PATIENCE_S,
            abandon_max=ABANDON_MAX,
            awt_max_s=AWT_MAX_S,
            intercall_gap_s=INTERCALL_GAP_S,
            use_strict_occ_cap=USE_STRICT_OCC_CAP
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
            "abandon_rate": round(dims["abandon_rate"], 4),
            "avg_wait_s": round(dims["avg_wait_s"], 2),
            "model": dims["model"],
            "params": {
                "SLA_TARGET": SLA_TARGET,
                "ASA_TARGET_S": ASA_TARGET_S,
                "MAX_OCC": MAX_OCC,
                "SHIFT_HOURS": SHIFT_HOURS,
                "LUNCH_HOURS": LUNCH_HOURS,
                "BREAKS_MIN": BREAKS_MIN,
                "AUX_RATE": AUX_RATE,
                "DERIVED_SHRINKAGE": round(derived_shrinkage, 4),
                "PRODUCTIVITY_FACTOR": round(prod_factor, 4),
                "USE_ERLANG_A": USE_ERLANG_A,
                "MEAN_PATIENCE_S": MEAN_PATIENCE_S,
                "ABANDON_MAX": ABANDON_MAX,
                "AWT_MAX_S": AWT_MAX_S,
                "INTERCALL_GAP_S": INTERCALL_GAP_S,
                "ABSENTEEISM_RATE": ABSENTEEISM_RATE,
                "EFFECTIVE_SHRINKAGE": round(effective_shrinkage, 4)
            }
        })

    with open(OUT_JSON_ERLANG, "w", encoding="utf-8") as f:
        json.dump(erlang_rows, f, ensure_ascii=False, indent=2)
    with open(OUT_JSON_ERLANG_DO, "w", encoding="utf-8") as f:
        json.dump(erlang_rows, f, ensure_ascii=False, indent=2)

    # 8) Timestamp para forzar actualización
    horizon_hours = len(idx)  # horizonte real (del 1 mes anterior al fin del mes siguiente)
    stamp = {
        "generated_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "horizon_hours": int(horizon_hours),
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

