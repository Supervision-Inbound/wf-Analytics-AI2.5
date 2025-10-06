# forecast3m.py
# Inferencia conjunta (recibidos + tmo) + dimensionamiento con Erlang-C/A
# Descarga modelos desde el último Release y publica JSON/CSV/JSON históricos listos para front.
# Novedades (versión con DEBUG prints):
#  - Horizonte: 1er día del mes anterior -> último día del mes siguiente (hora a hora, America/Santiago)
#  - Distribución diaria→horaria: primero calcula total diario y luego reparte por pesos (perfil por DOW)
#  - JSON histórico: mueve registros fuera de la ventana actual a public/prediccion_historica.json
#  - Erlang-A/C con pausa legal, ocupación máxima, abandono y AWT objetivo
#  - Redondeo a enteros (llamadas, TMO, agentes)
#  - DEBUG: imprime muestras y métricas para verificar cambios

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

# Histórico acumulado
HIST_JSON_PUBLIC   = "public/prediccion_historica.json"

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

# ===== DEBUG =====
DEBUG = True   # Pon en False si no quieres ver los prints en consola

# --------------------------------
def ensure_dirs():
    os.makedirs("public", exist_ok=True)
    os.makedirs("data_out", exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

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
    """AWT ≈ Pw * AHT / (N - A)."""
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
    pw = erlang_c_prob_wait(N, A)  # baseline C
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

# ====== Pesos horarios por día de semana (distribución diaria->horaria) ======
def compute_hourly_weights_by_dow(df_ts_hours: pd.DataFrame, raw_hourly_calls: np.ndarray) -> pd.DataFrame:
    """
    Calcula pesos por hora y por día de semana (dow) a partir del patrón del propio modelo.
    1) Toma las predicciones horarias "crudas" (sin redistribución) como referencia histórica.
    2) Para cada dow, promedia por hora; normaliza a que las 24h sumen 1.
    Retorna un DataFrame con columnas: dow, hour, weight (0..1)
    """
    tmp = pd.DataFrame({
        "dow": df_ts_hours["dow"].values,
        "hour": df_ts_hours["hour"].values,
        "y": raw_hourly_calls.astype(float)
    })
    prof = tmp.groupby(["dow","hour"], as_index=False)["y"].mean()
    prof["weight"] = prof["y"]
    # normalizar por DOW
    prof["sum_dow"] = prof.groupby("dow")["weight"].transform("sum").replace(0, np.nan)
    prof["weight"] = prof["weight"] / prof["sum_dow"]
    prof["weight"] = prof["weight"].fillna(1.0/24.0)  # fallback uniforme si algo queda NaN
    return prof[["dow","hour","weight"]].copy()

def redistribute_daily_to_hourly(df_ts_hours: pd.DataFrame,
                                 raw_hourly_calls: np.ndarray,
                                 weights_by_dow: np.ndarray) -> np.ndarray:
    """
    Para cada día:
      - calcula el total diario a partir de las predicciones crudas (robusto a outliers diarios),
      - toma los pesos de su DOW y reparte total_diario * weight(hora).
    Devuelve array de llamadas por hora redistribuidas (float).
    """
    df = df_ts_hours.copy()
    df["raw_call"] = raw_hourly_calls.astype(float)
    df["date"] = df["ts"].dt.date

    # tabla de pesos (join por dow, hour)
    w = weights_by_dow.copy()
    df = df.merge(w, on=["dow","hour"], how="left")
    df["weight"] = df["weight"].fillna(1.0/24.0)

    # total diario desde crudo
    daily_totals = df.groupby("date", as_index=False)["raw_call"].sum().rename(columns={"raw_call":"daily_total"})
    df = df.merge(daily_totals, on="date", how="left")

    # repartir
    df["call_redistributed"] = df["daily_total"] * df["weight"]

    return df["call_redistributed"].values

def try_read_json(path: str):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def write_json(path: str, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def dedup_by_ts(records):
    """Elimina duplicados por 'ts', conservando el último."""
    seen = {}
    for r in records:
        seen[r["ts"]] = r
    out = [seen[k] for k in sorted(seen.keys())]
    return out

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

    # 5) Predicción modelos (crudo horario)
    X_ll  = build_feature_matrix(df, TARGET_LLAMADAS).fillna(method="bfill").fillna(method="ffill")
    pred_ll_raw = mdl_ll.predict(X_ll).astype(float)

    if mdl_tmo is not None:
        X_tmo = build_feature_matrix(df, TARGET_TMO).fillna(method="bfill").fillna(method="ffill")
        pred_tmo_raw = mdl_tmo.predict(X_tmo).astype(float)
    else:
        pred_tmo_raw = np.full(len(df), DEFAULT_TMO, dtype=float)

    # ====== Distribución diaria → horaria por pesos (perfil por DOW) ======
    weights_by_dow = compute_hourly_weights_by_dow(df, pred_ll_raw)
    pred_ll_redist = redistribute_daily_to_hourly(df, pred_ll_raw, weights_by_dow)

    # ---- DEBUG BLOCK: verificación de predicciones antes de guardar ----
    if DEBUG:
        print("\n===== DEBUG: Ventana temporal =====")
        print(f"Inicio ventana: {df['ts'].min()}  |  Fin ventana: {df['ts'].max()}")
        print(f"Horas en ventana: {len(df)}")

        # Preparar un preview de 12 filas (crudo vs redistribuido)
        debug_preview = pd.DataFrame({
            "ts": df["ts"].astype(str),
            "dow": df["dow"],
            "hour": df["hour"],
            "raw_ll": np.round(pred_ll_raw, 2),
            "redist_ll": np.round(pred_ll_redist, 2),
            "tmo_raw": np.round(pred_tmo_raw, 2),
        }).head(12)
        print("\n===== DEBUG: Predicciones (primeras 12 filas) =====")
        print(debug_preview.to_string(index=False))

        # Totales diarios (primer día de la ventana)
        first_day = df["ts"].dt.date.min()
        mask_day = df["ts"].dt.date == first_day
        total_raw_day1 = float(np.sum(pred_ll_raw[mask_day]))
        total_red_day1 = float(np.sum(pred_ll_redist[mask_day]))
        print("\n===== DEBUG: Totales día 1 (deben ser iguales) =====")
        print(f"Fecha: {first_day} | Total crudo: {round(total_raw_day1,2)} | Total redistribuido: {round(total_red_day1,2)}")

        # Métricas globales
        print("\n===== DEBUG: Métricas globales =====")
        print(f"Llamadas crudo -> min:{np.min(pred_ll_raw):.2f}  max:{np.max(pred_ll_raw):.2f}  mean:{np.mean(pred_ll_raw):.2f}")
        print(f"Llamadas redist-> min:{np.min(pred_ll_redist):.2f}  max:{np.max(pred_ll_redist):.2f}  mean:{np.mean(pred_ll_redist):.2f}")
        print(f"TMO seg       -> min:{np.min(pred_tmo_raw):.2f}  max:{np.max(pred_tmo_raw):.2f}  mean:{np.mean(pred_tmo_raw):.2f}")
        print("===== FIN DEBUG PRED =====\n")

    # Redondeo a enteros (como pediste)
    pred_ll_int  = np.maximum(0, np.rint(pred_ll_redist).astype(int))
    pred_tmo_int = np.maximum(0, np.rint(pred_tmo_raw).astype(int))

    # Salida base
    out = pd.DataFrame({
        "ts": df["ts"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "pred_llamadas": pred_ll_int,
        "pred_tmo_seg": pred_tmo_int
    })

    # 6) Guardar CSV y JSONs de predicciones (ventana actual)
    out.to_csv(OUT_CSV, index=False)
    payload_current = out.to_dict(orient="records")
    write_json(OUT_JSON_PUBLIC, payload_current)
    write_json(OUT_JSON_DATAOUT, payload_current)

    # ======= JSON Histórico: mover lo que sale de la ventana =======
    prev_public = try_read_json(OUT_JSON_PUBLIC)  # (después de escribir ya es el "nuevo")
    prev_dataout = try_read_json(OUT_JSON_DATAOUT)
    hist_records = try_read_json(HIST_JSON_PUBLIC)

    start_str = pd.to_datetime(idx.min()).strftime("%Y-%m-%d %H:%M:%S")
    prev_candidates = prev_dataout if len(prev_dataout) > len(prev_public) else prev_public
    prev_old = [r for r in prev_candidates if r.get("ts","") < start_str]

    if prev_old:
        hist_merged = dedup_by_ts(hist_records + prev_old)
        write_json(HIST_JSON_PUBLIC, hist_merged)
        if DEBUG:
            print(f"[Hist] Movidos {len(prev_old)} registros antiguos a {HIST_JSON_PUBLIC}")
    else:
        if not os.path.exists(HIST_JSON_PUBLIC):
            write_json(HIST_JSON_PUBLIC, [])
        if DEBUG:
            print("[Hist] Sin cambios en histórico")

    # ===== Productividad programada y shrinkage derivado =====
    prod_factor = scheduled_productivity_factor(
        SHIFT_HOURS, LUNCH_HOURS, BREAKS_MIN, AUX_RATE
    )
    derived_shrinkage = max(0.0, min(1.0, 1.0 - prod_factor))
    effective_shrinkage = 1.0 - ( (1.0 - derived_shrinkage) * (1.0 - ABSENTEEISM_RATE) )

    if DEBUG:
        print(f"[Turno] productividad={prod_factor:.4f} -> shrinkage_derivado={derived_shrinkage:.4f} "
              f"-> absentismo={ABSENTEEISM_RATE:.2f} -> shrinkage_efectivo={effective_shrinkage:.4f}")

    # 7) Calcular Erlang (C/A) y agentes
    erlang_rows = []
    for row in payload_current:
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

    write_json(OUT_JSON_ERLANG, erlang_rows)
    write_json(OUT_JSON_ERLANG_DO, erlang_rows)

    # 8) Timestamp para forzar actualización
    horizon_hours = len(idx)
    stamp = {
        "generated_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "horizon_hours": int(horizon_hours),
        "records": len(out)
    }
    write_json(STAMP_JSON, stamp)

    # ---- DEBUG: resumen final de escritura ----
    if DEBUG:
        print("\n===== DEBUG: Resumen de archivos escritos =====")
        print(f"CSV     -> {OUT_CSV}           (rows: {len(out)})")
        print(f"JSON    -> {OUT_JSON_PUBLIC}   (rows: {len(out)})")
        print(f"JSON cp -> {OUT_JSON_DATAOUT}  (rows: {len(out)})")
        print(f"ERLANG  -> {OUT_JSON_ERLANG}   (rows: {len(erlang_rows)})")
        print(f"ERLANG2 -> {OUT_JSON_ERLANG_DO}(rows: {len(erlang_rows)})")
        print(f"STAMP   -> {STAMP_JSON}")
        print(f"HIST    -> {HIST_JSON_PUBLIC}  (exists: {os.path.exists(HIST_JSON_PUBLIC)})")

    print("\nOK - Inferencia completada.")

if __name__ == "__main__":
    main()
