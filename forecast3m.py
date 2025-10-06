# forecast3m.py
# Inferencia con modelo NN (Keras 3) para llamadas:
#  - Predice TOTAL diario y PERFIL horario (24 pesos) por día.
#  - Distribuye las horas según el perfil previsto.
#  - Predice TMO por hora (modelo .pkl legado, opcional).
#  - Dimensiona con Erlang-A y publica JSON/CSV para front.

import os
import json
import joblib
import numpy as np
import pandas as pd

# 🔸 Keras 3 (no uses tensorflow.keras aquí)
import keras
from keras.models import load_model

from utils_release import download_asset_from_latest

# ---------- Parámetros ----------
OWNER = "Supervision-Inbound"
REPO  = "wf-Analytics-AI2.5"

ASSET_LLAMADAS = "modelo_llamadas_nn.keras"  # <-- NUEVO (Keras 3)
ASSET_TMO      = "modelo_tmo.pkl"            # opcional

MODELS_DIR = "models"

OUT_CSV            = "data_out/predicciones.csv"
OUT_JSON_PUBLIC    = "public/predicciones.json"
OUT_JSON_DATAOUT   = "data_out/predicciones.json"
OUT_JSON_ERLANG    = "public/erlang_forecast.json"
OUT_JSON_ERLANG_DO = "data_out/erlang_forecast.json"
STAMP_JSON         = "public/last_update.json"

FREQ_H = "H"

# Semillas si no hay estado
DEFAULT_DAILY    = 2400.0   # total llamadas/día semilla
DEFAULT_TMO      = 180.0    # seg
SEQ_LEN          = 28       # debe coincidir con el entrenado (ver logs)
TZ                = "America/Santiago"

# ===== Operación / Erlang =====
SLA_TARGET   = 0.90
ASA_TARGET_S = 22
MAX_OCC      = 0.85
SHRINKAGE    = 0.30   # legacy, pero lo sobrescribimos con productividad + absentismo
USE_ERLANG_A       = True
MEAN_PATIENCE_S    = 60.0
ABANDON_MAX        = 0.06
AWT_MAX_S          = 120.0
USE_STRICT_OCC_CAP = True
INTERCALL_GAP_S    = 10.0

# Turno/productividad
SHIFT_HOURS = 10.0
LUNCH_HOURS = 1.0
BREAKS_MIN  = [15, 15]
AUX_RATE    = 0.15
ABSENTEEISM_RATE = 0.23

# ------------- Utils calendario/ctx (debe calzar con entrenamiento) -------------
def add_time_cols(df):
    df["dow"]   = df["date"].dt.dayofweek
    df["doy"]   = df["date"].dt.dayofyear
    df["week"]  = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    # Señales estacionales suaves (anuales)
    df["sin_y"] = np.sin(2*np.pi*df["doy"]/366.0)
    df["cos_y"] = np.cos(2*np.pi*df["doy"]/366.0)
    # marca de fin/inicio de mes (binarias)
    dom = df["date"].dt.day
    last_dom = (df["date"] + pd.offsets.MonthEnd(0)).dt.day
    df["is_month_start"] = (dom == 1).astype(int)
    df["is_month_end"]   = (dom == last_dom).astype(int)
    return df

def build_ctx_matrix(cal_df):
    """
    Debe reproducir el orden/tamaño usado al entrenar.
    En el template de entrenamiento propuse 6 features:
      [dow, month, sin_y, cos_y, is_month_start, is_month_end]
    """
    ctx = cal_df[["dow","month","sin_y","cos_y","is_month_start","is_month_end"]].astype(float).copy()
    return ctx.values

# -------------------- Perfil horario → 24 pesos --------------------
def expand_daily_to_hourly(dates, totals, profiles):
    """
    dates: Serie de fechas (diarias, sin hora)
    totals: np.array shape [N_days] con totales predichos
    profiles: np.array shape [N_days, 24] con softmax de cada hora
    Devuelve DataFrame con ts (hora) y llamadas por hora (int).
    """
    rows = []
    for d, tot, prof in zip(dates, totals, profiles):
        prof = np.maximum(prof, 0.0)
        prof_sum = prof.sum()
        if prof_sum <= 0:
            # perfil degenerado -> uniforme
            prof = np.ones(24) / 24.0
        else:
            prof = prof / prof_sum
        # reparto entero con corrección por residuo:
        raw = tot * prof
        ints = np.floor(raw).astype(int)
        resid = int(round(tot - ints.sum()))
        if resid > 0:
            order = np.argsort(-(raw - ints))[:resid]
            ints[order] += 1
        elif resid < 0:
            order = np.argsort(raw - ints)[:abs(resid)]
            ints[order] -= 1
        for h in range(24):
            rows.append({
                "ts": pd.Timestamp(d).replace(hour=h, minute=0, second=0),
                "pred_llamadas": int(max(0, ints[h]))
            })
    return pd.DataFrame(rows)

# -------------------- Erlang (C / A) --------------------
def erlang_c_prob_wait(N: int, A: float) -> float:
    if A <= 0: return 0.0
    if N <= 0: return 1.0
    if A >= N: return 1.0
    summ = 0.0
    term = 1.0
    for k in range(N):
        if k > 0: term *= A / k
        summ += term
    pn = term * (A / N) / (1 - A / N)
    return float(pn / (summ + pn))

def service_level(A: float, N: int, AHT: float, T: float) -> float:
    if N <= 0 or A <= 0 or A >= N: return 0.0
    pw = erlang_c_prob_wait(N, A)
    return 1.0 - pw * np.exp(-(N - A) * (T / AHT))

def erlang_c_awt(A: float, N: int, AHT: float) -> float:
    if N <= 0 or A <= 0 or N <= A: return float('inf')
    pw = erlang_c_prob_wait(N, A)
    return float(pw * (AHT / (N - A)))

def erlang_a_metrics(A: float, N: int, aht_s: float, patience_s: float, T: float):
    if N <= 0 or A <= 0: return 0.0, 1.0, float('inf')
    mu = 1.0 / max(aht_s, 1.0)
    theta = 1.0 / max(patience_s, 1.0)
    if N <= A: return 0.0, 1.0, float('inf')
    pw = erlang_c_prob_wait(N, A)
    r = mu * (N - A)
    rate = r + theta
    sl_a = (1.0 - pw) + pw * (r / rate) * (1.0 - np.exp(-rate * T))
    aban = pw * (theta / rate)
    awt  = pw * (1.0 / rate)
    return float(np.clip(sl_a, 0.0, 1.0)), float(np.clip(aban, 0.0, 1.0)), float(awt)

def scheduled_productivity_factor(shift_hours, lunch_hours, breaks_min, aux_rate):
    breaks_h = sum(breaks_min) / 60.0
    productive_hours = max(0.0, shift_hours - lunch_hours - breaks_h)
    net_productive_hours = productive_hours * (1.0 - aux_rate)
    return (net_productive_hours / shift_hours) if shift_hours > 0 else 0.0

def required_agents(calls_h: float, aht_s: float,
                    sla_target: float, asa_s: float,
                    max_occ: float, shrinkage: float,
                    use_erlang_a: bool = USE_ERLANG_A,
                    patience_s: float = MEAN_PATIENCE_S,
                    abandon_max: float = ABANDON_MAX,
                    awt_max_s: float = AWT_MAX_S,
                    intercall_gap_s: float = INTERCALL_GAP_S,
                    use_strict_occ_cap: bool = USE_STRICT_OCC_CAP) -> dict:
    calls_h = max(0.0, float(calls_h))
    aht_eff = max(1.0, float(aht_s) + float(intercall_gap_s))
    lamb = calls_h / 3600.0
    A = lamb * aht_eff
    N_occ_min = int(np.ceil(A / max_occ)) if (use_strict_occ_cap and max_occ > 0) else int(np.ceil(A + 1))
    N = max(N_occ_min, int(np.ceil(A)) + 1)
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

# -------------------- Ventana temporal --------------------
def build_month_window_days():
    now_local = pd.Timestamp.now(tz=TZ).floor("D")
    current_period = now_local.to_period("M")
    prev_period    = current_period - 1
    next_period    = current_period + 1
    start = pd.Timestamp(year=prev_period.year, month=prev_period.month, day=1, tz=TZ).tz_localize(None)
    last_day_next = pd.Timestamp(year=next_period.year, month=next_period.month, day=1, tz=TZ) + pd.offsets.MonthEnd(1)
    end = last_day_next.tz_localize(None)
    days = pd.date_range(start=start, end=end, freq="D")
    return days

# -------------------- Main --------------------
def main():
    os.makedirs("public", exist_ok=True)
    os.makedirs("data_out", exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Descargando modelos desde Release…")
    p_ll = download_asset_from_latest(OWNER, REPO, ASSET_LLAMADAS, os.path.join(MODELS_DIR, ASSET_LLAMADAS))
    p_tm = download_asset_from_latest(OWNER, REPO, ASSET_TMO,      os.path.join(MODELS_DIR, ASSET_TMO))
    print("Modelos descargados:", p_ll, p_tm)

    # Cargar modelos
    mdl_ll = load_model(p_ll, compile=False)  # Keras 3
    mdl_tmo = joblib.load(p_tm) if os.path.exists(p_tm) else None

    # Calendario diario de la ventana
    days = build_month_window_days()
    cal = pd.DataFrame({"date": days})
    cal = add_time_cols(cal)
    ctx  = build_ctx_matrix(cal)  # shape [N_days, 6]

    # Semilla de 60 días previos (uniforme por simplicidad)
    seed_days = pd.date_range(end=cal["date"].min() - pd.Timedelta(days=1),
                              periods=max(SEQ_LEN, 60), freq="D")
    seed_tot = np.full(len(seed_days), DEFAULT_DAILY, dtype=float)

    # Predicción autoregresiva día-a-día
    totals_pred = []
    profiles_pred = []
    hist = list(seed_tot)  # historial de totales
    for i in range(len(cal)):
        seq = np.array(hist[-SEQ_LEN:], dtype=float).reshape(1, SEQ_LEN, 1)
        c   = ctx[i].reshape(1, -1)
        y_total, y_prof = mdl_ll.predict({"seq_totales": seq, "ctx": c}, verbose=0)
        total = float(max(0.0, y_total.squeeze()))
        prof  = y_prof.squeeze().astype(float)  # ya softmax
        totals_pred.append(total)
        profiles_pred.append(prof)
        hist.append(total)

    totals_pred = np.array(totals_pred)
    profiles_pred = np.vstack(profiles_pred)  # [N_days, 24]

    # Expandir a horas según perfil de cada día
    hourly_calls = expand_daily_to_hourly(cal["date"], totals_pred, profiles_pred)
    # Alinear a string ts
    hourly_calls["ts"] = hourly_calls["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # TMO por hora
    if mdl_tmo is not None:
        # Features simples de tiempo para TMO (como en inferencia clásica)
        df_tmo = pd.DataFrame({"ts": pd.to_datetime(hourly_calls["ts"])})
        df_tmo["dow"] = df_tmo["ts"].dt.dayofweek
        df_tmo["month"] = df_tmo["ts"].dt.month
        df_tmo["hour"] = df_tmo["ts"].dt.hour
        df_tmo["sin_hour"] = np.sin(2*np.pi*df_tmo["hour"]/24)
        df_tmo["cos_hour"] = np.cos(2*np.pi*df_tmo["hour"]/24)
        df_tmo["sin_dow"] = np.sin(2*np.pi*df_tmo["dow"]/7)
        df_tmo["cos_dow"] = np.cos(2*np.pi*df_tmo["dow"]/7)
        feats = ["sin_hour","cos_hour","sin_dow","cos_dow","dow","month"]
        tmo_pred = mdl_tmo.predict(df_tmo[feats])
        tmo_pred = np.maximum(0, np.rint(tmo_pred).astype(int))
    else:
        tmo_pred = np.full(len(hourly_calls), int(round(DEFAULT_TMO)))

    # Salida base
    out = pd.DataFrame({
        "ts": hourly_calls["ts"],
        "pred_llamadas": hourly_calls["pred_llamadas"].astype(int),
        "pred_tmo_seg": tmo_pred.astype(int)
    })

    # Guardar CSV/JSON
    out.to_csv(OUT_CSV, index=False)
    payload = out.to_dict(orient="records")
    with open(OUT_JSON_PUBLIC, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(OUT_JSON_DATAOUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Productividad y shrinkage efectivo
    prod_factor = scheduled_productivity_factor(SHIFT_HOURS, LUNCH_HOURS, BREAKS_MIN, AUX_RATE)
    derived_shrinkage = max(0.0, min(1.0, 1.0 - prod_factor))
    effective_shrinkage = 1.0 - ((1.0 - derived_shrinkage) * (1.0 - ABSENTEEISM_RATE))
    print(f"[Turno] productividad={prod_factor:.4f} -> shrinkage_derivado={derived_shrinkage:.4f} -> absentismo={ABSENTEEISM_RATE:.2f} -> shrinkage_efectivo={effective_shrinkage:.4f}")

    # Erlang por hora
    erlang_rows = []
    for r in payload:
        calls_h = int(r["pred_llamadas"])
        aht_s   = int(r["pred_tmo_seg"])
        dims = required_agents(
            calls_h=calls_h,
            aht_s=aht_s,
            sla_target=SLA_TARGET,
            asa_s=ASA_TARGET_S,
            max_occ=MAX_OCC,
            shrinkage=effective_shrinkage,
            use_erlang_a=USE_ERLANG_A,
            patience_s=MEAN_PATIENCE_S,
            abandon_max=ABANDON_MAX,
            awt_max_s=AWT_MAX_S,
            intercall_gap_s=INTERCALL_GAP_S,
            use_strict_occ_cap=USE_STRICT_OCC_CAP
        )
        erlang_rows.append({
            "ts": r["ts"],
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
                "ABSENTEEISM_RATE": ABSENTEEISM_RATE,
                "EFFECTIVE_SHRINKAGE": round(effective_shrinkage, 4)
            }
        })

    with open(OUT_JSON_ERLANG, "w", encoding="utf-8") as f:
        json.dump(erlang_rows, f, ensure_ascii=False, indent=2)
    with open(OUT_JSON_ERLANG_DO, "w", encoding="utf-8") as f:
        json.dump(erlang_rows, f, ensure_ascii=False, indent=2)

    # Stamp
    horizon_hours = len(payload)
    stamp = {
        "generated_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "horizon_hours": int(horizon_hours),
        "records": int(len(payload))
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
