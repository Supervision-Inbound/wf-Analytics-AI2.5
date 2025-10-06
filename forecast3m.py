# =======================================================================
# forecast3m.py (VERSIÓN FINAL PARA REDES NEURONALES)
# Inferencia con los modelos Keras (.h5) y los scalers (.pkl)
# =======================================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from utils_release import download_asset_from_latest

# ---------- Parámetros ----------
OWNER = "Supervision-Inbound"      # Tu usuario u organización de GitHub
REPO  = "wf-Analytics-AI2.5"       # El nombre de tu repositorio

# --- NOMBRES DE LOS NUEVOS ARCHIVOS DE MODELO ---
ASSET_LLAMADAS = "modelo_llamadas_nn.h5"
ASSET_SCALER_LLAMADAS = "scaler_llamadas.pkl"
ASSET_TMO = "modelo_tmo_nn.h5"
ASSET_SCALER_TMO = "scaler_tmo.pkl"

MODELS_DIR = "models"
OUT_JSON_PUBLIC = "public/predicciones.json"
OUT_JSON_ERLANG = "public/erlang_forecast.json"
OUT_JSON_DATAOUT = "data_out/predicciones.json" # Para auditoría
OUT_JSON_ERLANG_DO = "data_out/erlang_forecast.json" # Para auditoría
STAMP_JSON = "public/last_update.json"

HOURS_AHEAD = 24 * 90  # 90 días
FREQ        = "H"
TARGET_LLAMADAS = "recibidos"
TARGET_TMO      = "tmo_seg"

# --- Parámetros de operación (Erlang) - Sin cambios ---
SLA_TARGET = 0.90
ASA_TARGET_S = 22
MAX_OCC = 0.85
SHIFT_HOURS = 10.0
LUNCH_HOURS = 1.0
BREAKS_MIN = [15, 15]
AUX_RATE = 0.15
ABSENTEEISM_RATE = 0.23
USE_ERLANG_A = True
MEAN_PATIENCE_S = 60.0
ABANDON_MAX = 0.06
AWT_MAX_S = 120.0
INTERCALL_GAP_S = 10.0

# --- FUNCIÓN DE FEATURES (IDÉNTICA A LA DEL ENTRENAMIENTO) ---
def build_feature_matrix_nn(df, target_col, training_columns):
    """
    Construye la matriz de características para la inferencia, asegurando que
    tenga exactamente las mismas columnas que durante el entrenamiento.
    """
    # Features cíclicas
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_dow"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["cos_dow"] = np.cos(2 * np.pi * df["dow"] / 7)

    # Features de rolling: En inferencia no podemos calcularlas porque son del futuro.
    # Las creamos como placeholders (columnas con ceros) porque el modelo las espera.
    df[f"{target_col}_lag24"] = 0
    df[f"{target_col}_ma24"] = 0
    df[f"{target_col}_ma168"] = 0

    base_feats = [
        "sin_hour", "cos_hour", "sin_dow", "cos_dow",
        f"{target_col}_lag24", f"{target_col}_ma24", f"{target_col}_ma168"
    ]
    cat_feats = ["dow", "month"]
    df_dummies = pd.get_dummies(df[cat_feats], columns=cat_feats, drop_first=False)
    
    X = pd.concat([df[base_feats], df_dummies], axis=1)
    
    # Alinear columnas con las del entrenamiento para evitar errores
    missing_cols = set(training_columns) - set(X.columns)
    for c in missing_cols:
        X[c] = 0
    
    # Asegurar el mismo orden de columnas
    X = X[training_columns]
    
    return X.fillna(0)

# --- FUNCIONES DE ERLANG (SIN CAMBIOS) ---
def erlang_c(R, N):
    if N <= R: return 0.0
    inv_erlang_b = 1.0
    for i in range(1, int(N) + 1): inv_erlang_b = 1.0 + (i / R) * inv_erlang_b
    erlang_b = 1.0 / inv_erlang_b
    return (N * erlang_b) / (N - R * (1 - erlang_b))

def erlang_a(R, N, patience_s, aht_s):
    if N <= R: return 0.0, 0.0, R - N
    p_wait = erlang_c(R, N)
    asa = (p_wait * aht_s) / (N - R) if p_wait > 0 else 0
    sla = 1 - p_wait * np.exp(-(N - R) * patience_s / aht_s) if p_wait > 0 else 1.0
    abn = p_wait * (1 - (1 - np.exp(-patience_s / asa))) if asa > 0 else 0
    return sla, abn, asa

def calculate_agents(llamadas, tmo_seg, sla_target, asa_target_s):
    if llamadas == 0: return 0
    erlangs = (llamadas * tmo_seg) / 3600.0
    min_agents = int(np.ceil(erlangs)) + 1
    for agents in range(min_agents, min_agents + 100):
        if USE_ERLANG_A:
            sla, _, asa = erlang_a(erlangs, agents, asa_target_s, tmo_seg)
        else:
            p_wait = erlang_c(erlangs, agents)
            asa = (p_wait * tmo_seg) / (agents - erlangs) if p_wait > 0 else 0
            sla = 1 - p_wait * np.exp(-(agents - erlangs) * asa_target_s / tmo_seg) if p_wait > 0 else 1.0
        if sla >= sla_target and asa <= asa_target_s:
            return agents
    return min_agents + 100

def get_prod_factor(shift_h, lunch_h, breaks_m):
    total_paid = shift_h * 60
    non_prod = (lunch_h * 60) + sum(breaks_m)
    prod_minutes = total_paid - non_prod
    return (prod_minutes / total_paid) if total_paid > 0 else 0

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs("data_out", exist_ok=True)
    os.makedirs("public", exist_ok=True)

    # 1) Descargar los 4 artefactos desde el último Release
    print("Descargando modelos y scalers desde el último Release...")
    assets_to_download = [ASSET_LLAMADAS, ASSET_SCALER_LLAMADAS, ASSET_TMO, ASSET_SCALER_TMO]
    for asset_name in assets_to_download:
        download_asset_from_latest(OWNER, REPO, asset_name, MODELS_DIR)

    # 2) Cargar modelos y scalers
    print("Cargando modelos y scalers en memoria...")
    model_ll = tf.keras.models.load_model(os.path.join(MODELS_DIR, ASSET_LLAMADAS))
    scaler_ll = joblib.load(os.path.join(MODELS_DIR, ASSET_SCALER_LLAMADAS))
    model_tmo = tf.keras.models.load_model(os.path.join(MODELS_DIR, ASSET_TMO))
    scaler_tmo = joblib.load(os.path.join(MODELS_DIR, ASSET_SCALER_TMO))
    
    # Obtenemos las columnas exactas del entrenamiento desde el scaler
    training_cols_ll = scaler_ll.get_feature_names_out()
    training_cols_tmo = scaler_tmo.get_feature_names_out()

    # 3) Crear DataFrame con fechas futuras
    print(f"Generando timestamps para las próximas {HOURS_AHEAD} horas...")
    start_date = pd.Timestamp.now(tz="America/Santiago").floor(FREQ)
    future_dates = pd.date_range(start=start_date, periods=HOURS_AHEAD, freq=FREQ)
    df_pred = pd.DataFrame({"ts": future_dates})
    df_pred["dow"] = df_pred["ts"].dt.dayofweek; df_pred["month"] = df_pred["ts"].dt.month; df_pred["hour"] = df_pred["ts"].dt.hour

    # 4) Construir matrices, escalar y predecir
    print("Construyendo características y generando predicciones...")
    
    # Para llamadas
    X_pred_ll = build_feature_matrix_nn(df_pred.copy(), TARGET_LLAMADAS, training_cols_ll)
    X_pred_ll_scaled = scaler_ll.transform(X_pred_ll)
    pred_ll = model_ll.predict(X_pred_ll_scaled).flatten()

    # Para TMO
    X_pred_tmo = build_feature_matrix_nn(df_pred.copy(), TARGET_TMO, training_cols_tmo)
    X_pred_tmo_scaled = scaler_tmo.transform(X_pred_tmo)
    pred_tmo = model_tmo.predict(X_pred_tmo_scaled).flatten()
    
    # 5) Ensamblar predicciones
    out = pd.DataFrame({
        "ts": future_dates.strftime("%Y-%m-%d %H:%M:%S"),
        "pred_llamadas": np.round(np.maximum(0, pred_ll)).astype(int),
        "pred_tmo_seg": np.round(np.maximum(0, pred_tmo)).astype(int)
    })
    
    # 6) Guardar predicciones en JSON
    out_dict = out.to_dict(orient="records")
    with open(OUT_JSON_PUBLIC, "w", encoding="utf-8") as f: json.dump(out_dict, f, ensure_ascii=False, indent=2)
    with open(OUT_JSON_DATAOUT, "w", encoding="utf-8") as f: json.dump(out_dict, f, ensure_ascii=False, indent=2)

    # 7) Calcular dimensionamiento con Erlang
    print("Calculando dimensionamiento de agentes...")
    erlang_rows = []
    prod_factor = get_prod_factor(SHIFT_HOURS, LUNCH_HOURS, BREAKS_MIN)
    derived_shrinkage = 1 - prod_factor
    effective_shrinkage = 1 - (prod_factor * (1 - ABSENTEEISM_RATE))
    
    for row in out_dict:
        llamadas = row["pred_llamadas"]
        tmo = row["pred_tmo_seg"]
        agentes_prod = calculate_agents(llamadas, tmo, SLA_TARGET, ASA_TARGET_S)
        agentes_agendados = int(np.ceil(agentes_prod / (1 - effective_shrinkage))) if effective_shrinkage < 1 else agentes_prod
        
        erlangs = (llamadas * tmo) / 3600.0
        occupancy = erlangs / agentes_prod if agentes_prod > 0 else 0
        sla, abn, asa = erlang_a(erlangs, agentes_prod, ASA_TARGET_S, tmo)
        
        erlang_rows.append({
            "ts": row["ts"], "llamadas": llamadas, "tmo_seg": tmo,
            "erlangs": round(erlangs, 2), "agentes_productivos": agentes_prod,
            "agentes_agendados": agentes_agendados, "occupancy": round(occupancy, 4),
            "service_level": round(sla, 4), "abandon_rate": round(abn, 4), "avg_wait_s": round(asa, 2),
            "model": "Erlang-A", "params": {
                "SLA_TARGET": SLA_TARGET, "ASA_TARGET_S": ASA_TARGET_S, "MAX_OCC": MAX_OCC,
                "SHIFT_HOURS": SHIFT_HOURS, "LUNCH_HOURS": LUNCH_HOURS, "BREAKS_MIN": BREAKS_MIN,
                "AUX_RATE": AUX_RATE, "DERIVED_SHRINKAGE": round(derived_shrinkage, 4),
                "PRODUCTIVITY_FACTOR": round(prod_factor, 4), "USE_ERLANG_A": USE_ERLANG_A,
                "MEAN_PATIENCE_S": MEAN_PATIENCE_S, "ABANDON_MAX": ABANDON_MAX, "AWT_MAX_S": AWT_MAX_S,
                "INTERCALL_GAP_S": INTERCALL_GAP_S, "ABSENTEEISM_RATE": ABSENTEEISM_RATE,
                "EFFECTIVE_SHRINKAGE": round(effective_shrinkage, 4)
            }
        })
    with open(OUT_JSON_ERLANG, "w", encoding="utf-8") as f: json.dump(erlang_rows, f, ensure_ascii=False, indent=2)
    with open(OUT_JSON_ERLANG_DO, "w", encoding="utf-8") as f: json.dump(erlang_rows, f, ensure_ascii=False, indent=2)

    # 8) Timestamp de actualización
    stamp = {"generated_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}
    with open(STAMP_JSON, "w") as f: json.dump(stamp, f)

    print("Proceso de inferencia completado. Archivos JSON generados.")

if __name__ == "__main__":
    main()

