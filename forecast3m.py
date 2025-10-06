# forecast3m.py
# Inferencia con Modelo NN (total+perfil) para llamadas + TMO clásico
# Descarga modelos desde el último Release y publica JSON/CSV listos para front.

import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils_release import download_asset_from_latest

# ---------- Parámetros ----------
OWNER = "Supervision-Inbound"
REPO  = "wf-Analytics-AI2.5"

ASSET_LLAMADAS_NN = "modelo_llamadas_nn.keras"
ASSET_TMO         = "modelo_tmo.pkl"

MODELS_DIR = "models"

OUT_CSV            = "data_out/predicciones.csv"
OUT_JSON_PUBLIC    = "public/predicciones.json"
OUT_JSON_DATAOUT   = "data_out/predicciones.json"
OUT_JSON_ERLANG    = "public/erlang_forecast.json"
OUT_JSON_ERLANG_DO = "data_out/erlang_forecast.json"
STAMP_JSON         = "public/last_update.json"

FREQ        = "H"

# Semillas/hiper de inferencia
SEQ_DIAS     = 28           # debe coincidir con entrenamiento
DEFAULT_LL   = 100.0        # seed para los primeros días
DEFAULT_TMO  = 180.0
PROFILE_TEMP = 1.15         # temperatura para suavizar/aguzar el perfil (>=1 suaviza)
EPS_PROFILE  = 1e-6         # anti-cero numérico

# Parámetros de operación (Erlang)
SLA_TARGET   = 0.90
ASA_TARGET_S = 22
MAX_OCC      = 0.85
SHRINKAGE    = 0.30

# Turno/productividad (tu realidad)
SHIFT_HOURS = 10.0
LUNCH_HOURS = 1.0
BREAKS_MIN  = [15, 15]
AUX_RATE    = 0.15

# Erlang-A (abandono/paciencia) + restricciones
USE_ERLANG_A       = True
MEAN_PATIENCE_S    = 60.0
ABANDON_MAX        = 0.06
AWT_MAX_S          = 120.0
USE_STRICT_OCC_CAP = True

# Pausa legal entre llamadas
INTERCALL_GAP_S    = 10.0

# Absentismo mensual aplicado a AGENDADOS
ABSENTEEISM_RATE   = 0.23

# ===== Erlang-C/A utils =====
def erlang_c_prob_wait(N: int, A: float) -> float:
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
    return float(pn / (summ + pn))

def service_level(A: float, N: int, AHT: float, T: float) -> float:
    if N <= 0 or A <= 0: return 0.0
    if A >= N: return 0.0
    pw = erlang_c_prob_wait(N, A)
    return 1.0 - pw * np.exp(-(N - A) * (T / AHT))

def erlang_c_awt(A: float, N: int, AHT: float) -> float:
    if N <= 0 or A <= 0 or N <= A:
        return float('inf')
    pw = erlang_c_prob_wait(N, A)
    return float(pw * (AHT / (N - A)))

def erlang_a_metrics(A: float, N: int, aht_s: float, patience_s: float, T: float):
    if N <= 0 or A <= 0:
        return 0.0, 1.0, float('inf')
    mu = 1.0 / max(aht_s, 1.0)
    theta = 1.0 / max(patience_s, 1.0)
    if N <= A:
        return 0.0, 1.0, float('inf')
    pw = erlang_c_prob_wait(N, A)
    r = mu * (N - A)
    rate = r + theta
    sl_a = (1.0 - pw) + pw * (r / rate) * (1.0 - np.exp(-rate * T))
    aban = pw * (theta / rate)
    awt  = pw * (1.0 / rate)
    sl_a = float(min(max(sl_a, 0.0), 1.0))
    aban = float(min(max(aban, 0.0), 1.0))
    return sl_a, aban, float(awt)

def scheduled_productivity_factor(shift_hours: float, lunch_hours: float, breaks_min: list, aux_rate: float) -> float:
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

    service = 0.0
    aban = 1.0
    awt = float('inf')

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

# ---------- Calendario ----------
def build_month_window_days():
    """Devuelve lista de fechas (naive) día a día: 1 mes anterior -> fin de mes siguiente (America/Santiago)."""
    tz = "America/Santiago"
    now_local = pd.Timestamp.now(tz=tz).normalize()
    current_period = now_local.to_period("M")
    prev_period    = current_period - 1
    next_period    = current_period + 1

    start = pd.Timestamp(year=prev_period.year, month=prev_period.month, day=1, tz=tz)
    last_day_next = pd.Timestamp(year=next_period.year, month=next_period.month, day=1, tz=tz) + pd.offsets.MonthEnd(1)
    end = last_day_next.tz_localize(tz).normalize()

    days = pd.date_range(start=start, end=end, freq="D", tz=tz).tz_localize(None)
    return [d.to_pydatetime().date() for d in days]

def day_context_feats(d):
    """dow, week, month, doy, sin_dow, cos_dow"""
    dts = pd.Timestamp(d)
    dow = dts.dayofweek
    week = int(dts.isocalendar().week)
    month = dts.month
    doy = dts.day_of_year if hasattr(dts, "day_of_year") else dts.timetuple().tm_yday
    sin_dow = np.sin(2*np.pi*dow/7)
    cos_dow = np.cos(2*np.pi*dow/7)
    return np.array([dow, week, month, doy, sin_dow, cos_dow], dtype=np.float32)

# ---------- Perfil helpers ----------
def apply_temperature(probs, T=1.0, eps=1e-6):
    """Aplica temperatura a un vector prob. T>1 suaviza; T<1 agudiza."""
    logits = np.log(np.clip(probs, eps, None))
    logits = logits / max(T, eps)
    x = np.exp(logits - logits.max())
    p = x / x.sum()
    return p

def distribute_total_to_hours(total, profile):
    """Enteros por hora que suman total (redondeo + ajuste de residuo)."""
    raw = total * profile
    ints = np.floor(raw).astype(int)
    resid = int(round(total - ints.sum()))
    if resid > 0:
        # reparte +1 a las horas con mayor parte fraccional
        frac_order = np.argsort(-(raw - ints))
        ints[frac_order[:resid]] += 1
    elif resid < 0:
        frac_order = np.argsort((raw - ints))  # quita donde menor fracción
        ints[frac_order[:(-resid)]] -= 1
    ints = np.maximum(ints, 0)
    return ints

# ---------- TMO features (idéntico a tu script previo) ----------
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

# ---------- Main ----------
def main():
    os.makedirs("public", exist_ok=True)
    os.makedirs("data_out", exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1) Descargar modelos desde el último Release
    print("Descargando modelos desde Release…")
    p_ll = download_asset_from_latest(OWNER, REPO, ASSET_LLAMADAS_NN, os.path.join(MODELS_DIR, ASSET_LLAMADAS_NN))
    p_tm = download_asset_from_latest(OWNER, REPO, ASSET_TMO,        os.path.join(MODELS_DIR, ASSET_TMO))
    print("Modelos descargados:", p_ll, p_tm)

    # 2) Cargar modelos
    # Importante: compile=False para no requerir objetos personalizados
    mdl_ll = load_model(p_ll, compile=False)
    mdl_tmo = joblib.load(p_tm) if os.path.exists(p_tm) else None

    # 3) Calendario diario del 1 mes anterior al fin del mes siguiente
    days = build_month_window_days()

    # 4) Autoregresión diaria (usa solo la propia predicción como historia)
    hist_totals = [DEFAULT_LL] * SEQ_DIAS
    daily_totals = []
    daily_profiles = []

    for d in days:
        # entrada secuencia
        x_seq = np.array(hist_totals[-SEQ_DIAS:], dtype=np.float32).reshape(1, SEQ_DIAS, 1)
        x_ctx = day_context_feats(d).reshape(1, -1)
        # predicción
        y_total_logp, y_profile, _ = mdl_ll.predict({"seq_totales": x_seq, "ctx": x_ctx}, verbose=0)
        total = float(np.expm1(y_total_logp)[0,0])
        total = max(0.0, total)

        profile = y_profile[0].astype(np.float64)
        # temperatura (opcional)
        profile = apply_temperature(profile, T=PROFILE_TEMP, eps=EPS_PROFILE)

        daily_totals.append(total)
        daily_profiles.append(profile)
        hist_totals.append(total)

    # 5) Expandir a horas
    hourly_rows = []
    for d, tot, prof in zip(days, daily_totals, daily_profiles):
        tot_int = int(round(tot))
        by_hour = distribute_total_to_hours(tot_int, prof)
        for h in range(24):
            ts = pd.Timestamp(d) + pd.Timedelta(hours=h)
            hourly_rows.append({"ts": ts.strftime("%Y-%m-%d %H:%M:%S"),
                                "pred_llamadas": int(by_hour[h])})

    # 6) TMO por hora (igual que antes)
    df = pd.DataFrame(hourly_rows)
    df["ts"] = pd.to_datetime(df["ts"])
    df = add_time_features(df)
    df["seed_tmo"] = DEFAULT_TMO
    TARGET_TMO = "tmo_seg"
    df[f"{TARGET_TMO}_lag1"]        = df["seed_tmo"].shift(1)
    df[f"{TARGET_TMO}_lag24"]       = df["seed_tmo"].shift(24)
    df[f"{TARGET_TMO}_ma24"]        = df["seed_tmo"].rolling(24, min_periods=1).mean()
    df[f"{TARGET_TMO}_ma168"]       = df["seed_tmo"].rolling(24*7, min_periods=1).mean()
    df[f"{TARGET_TMO}_samehour_7d"] = df["seed_tmo"].shift(24*7)

    if mdl_tmo is not None:
        X_tmo = build_feature_matrix(df, TARGET_TMO).fillna(method="bfill").fillna(method="ffill")
        pred_tmo = mdl_tmo.predict(X_tmo)
        pred_tmo_int = np.maximum(0, np.rint(pred_tmo).astype(int))
    else:
        pred_tmo_int = np.full(len(df), int(DEFAULT_TMO))

    out = pd.DataFrame({
        "ts": df["ts"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "pred_llamadas": df["pred_llamadas"].astype(int),
        "pred_tmo_seg": pred_tmo_int.astype(int)
    })

    # 7) Guardar CSV/JSON (predicciones)
    out.to_csv(OUT_CSV, index=False)
    payload = out.to_dict(orient="records")
    with open(OUT_JSON_PUBLIC, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(OUT_JSON_DATAOUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 8) Productividad y shrinkage efectivo (turno + absentismo)
    prod_factor = scheduled_productivity_factor(SHIFT_HOURS, LUNCH_HOURS, BREAKS_MIN, AUX_RATE)
    derived_shrinkage = max(0.0, min(1.0, 1.0 - prod_factor))
    effective_shrinkage = 1.0 - ((1.0 - derived_shrinkage) * (1.0 - ABSENTEEISM_RATE))
    print(f"[Turno] productividad={prod_factor:.4f} -> shrinkage_derivado={derived_shrinkage:.4f} -> absentismo={ABSENTEEISM_RATE:.2f} -> shrinkage_efectivo={effective_shrinkage:.4f}")

    # 9) Erlang por hora
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
            shrinkage=effective_shrinkage,
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

    # 10) Timestamp
    horizon_hours = len(out)
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
    # silenciar logs TF si quieres
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()
