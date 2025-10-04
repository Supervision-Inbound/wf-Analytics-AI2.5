# -*- coding: utf-8 -*-
"""
Forecast 3 meses (multitarea) → public/forecast_3m.(json|csv)
Columnas: fecha, hora, llamadas_recibidas, tmo_pred_seg, ejecutivos_requeridos
"""
import os, json
import numpy as np
import pandas as pd
import joblib

# --- Erlang C básico ---
def erlang_c_prob_wait(N, A):
    if A<=0: return 0.0
    if N<=0: return 1.0
    if A>=N: return 1.0
    summ=0.0; term=1.0
    for k in range(1,N):
        summ += term
        term *= A/k
    summ += term
    pn = term*(A/N)/(1.0-(A/N))
    return float(pn/(summ+pn))

def service_level(N,A,AHT,ASA):
    if A<=0: return 1.0
    if N<=0: return 0.0
    Pw = 1.0 if A>=N else erlang_c_prob_wait(N,A)
    return 1.0 - Pw*np.exp(-(N-A)*(ASA/max(AHT,1e-9)))

def required_agents(cph, aht, sla=0.90, asa=22.0, occ=0.80):
    if cph<=0 or aht<=0: return 0
    lam = cph/3600.0; A = lam*aht
    N = max(1,int(np.ceil(A/occ)))
    while True:
        if service_level(N,A,aht,asa)>=sla and (A/N)<=occ:
            return N
        N+=1
        if N>10000: return N

# --- Paths ---
PUBLIC_DIR   = "./public"
os.makedirs(PUBLIC_DIR, exist_ok=True)
OUT_JSON     = f"{PUBLIC_DIR}/forecast_3m.json"
OUT_CSV      = f"{PUBLIC_DIR}/forecast_3m.csv"

MODEL_PATHS  = ["./model_multi.pkl", "./model-lgb.pkl", "./model_lgb.pkl"]  # por compatibilidad
FEATS_PATH   = "./features.json"
TARGETS_PATH = "./targets.json"  # necesario para nombres de salidas

DATA_PATHS   = ["./data/Hosting ia.xlsx","./Data/Hosting ia.xlsx","./Hosting ia.xlsx",
                "/kaggle/input/historico-trafico/Hosting ia.xlsx"]

CALLS_COL = "recibidos"
TMO_COL   = "tmo (segundos)"
SLA=0.90; ASA=22.0; OCC=0.80

def find_first(paths):
    for p in paths:
        if os.path.exists(p): return p
    return None

def ensure_datetime(df):
    for c in ["datetime","datatime","fecha_hora","ts"]:
        if c in df.columns:
            df["datetime"]=pd.to_datetime(df[c],errors="coerce"); return df
    fcol = next((c for c in ["fecha","date","dia"] if c in df.columns), None)
    hcol = next((c for c in ["hora","time"] if c in df.columns), None)
    if fcol and hcol:
        f = pd.to_datetime(df[fcol], errors="coerce", dayfirst=True)
        h = pd.to_timedelta(df[hcol].astype(str), errors="coerce")
        mask = df[hcol].apply(lambda x: str(x).isdigit())
        if mask.any():
            hn = pd.to_timedelta(pd.to_numeric(df.loc[mask,hcol], errors="coerce"), unit="h")
            h.loc[mask] = hn.values
        df["datetime"]=f+h.fillna(pd.Timedelta(0)); return df
    if fcol:
        df["datetime"]=pd.to_datetime(df[fcol],errors="coerce",dayfirst=True); return df
    raise ValueError("No pude construir 'datetime'.")

def aggregate_to_hour(df, calls_col, tmo_col):
    df=df.copy(); df["datetime_h"]=df["datetime"].dt.floor("h")
    grp=df.groupby("datetime_h",as_index=False).agg({calls_col:"sum"})
    if tmo_col in df.columns:
        df["_tmo_w"]=df[tmo_col]*df[calls_col]
        w=df.groupby("datetime_h",as_index=False).agg({"_tmo_w":"sum", calls_col:"sum"})
        grp[tmo_col]=np.where(w[calls_col]>0, w["_tmo_w"]/w[calls_col], 0.0)
    return grp.rename(columns={"datetime_h":"datetime"})

def add_time_cols(df):
    df["hour"]=df["datetime"].dt.hour
    df["dow"]=df["datetime"].dt.dayofweek
    df["week"]=df["datetime"].dt.isocalendar().week.astype(int)
    df["month"]=df["datetime"].dt.month
    df["hour_sin"]=np.sin(2*np.pi*df["hour"]/24); df["hour_cos"]=np.cos(2*np.pi*df["hour"]/24)
    df["dow_sin"]=np.sin(2*np.pi*df["dow"]/7);   df["dow_cos"]=np.cos(2*np.pi*df["dow"]/7)
    return df

def build_row(ts,hist,X_cols):
    row=pd.DataFrame({"datetime":[ts]}); row=add_time_cols(row)
    h=hist.set_index("datetime")
    yS=h["yhat_calls"].fillna(h[CALLS_COL])
    tS=h["yhat_tmo"].fillna(h[TMO_COL]) if TMO_COL in h.columns else pd.Series(dtype=float)

    for lag in range(1,25):
        row[f"{CALLS_COL}_lag{lag}"] = yS.reindex([ts-pd.Timedelta(hours=lag)]).values[0]
        if TMO_COL in h.columns:
            row[f"{TMO_COL}_lag{lag}"] = tS.reindex([ts-pd.Timedelta(hours=lag)]).values[0]
    row[f"{CALLS_COL}_roll3"]=yS.loc[ts-pd.Timedelta(hours=24*365):ts].tail(3).mean()
    row[f"{CALLS_COL}_roll6"]=yS.loc[ts-pd.Timedelta(hours=24*365):ts].tail(6).mean()
    row[f"{CALLS_COL}_roll24"]=yS.loc[ts-pd.Timedelta(hours=24*365):ts].tail(24).mean()
    if TMO_COL in h.columns:
        row[f"{TMO_COL}_roll3"]=tS.loc[ts-pd.Timedelta(hours=24*365):ts].tail(3).mean()
        row[f"{TMO_COL}_roll6"]=tS.loc[ts-pd.Timedelta(hours=24*365):ts].tail(6).mean()
        row[f"{TMO_COL}_roll24"]=tS.loc[ts-pd.Timedelta(hours=24*365):ts].tail(24).mean()
    return row.reindex(columns=set(["datetime"]+X_cols))

def end_of_month_plus2(ts):
    y,m=ts.year,ts.month
    m3=m+3; y3=y+(m3-1)//12; m3=((m3-1)%12)+1
    first_next=pd.Timestamp(year=y3,month=m3,day=1,hour=0)
    return first_next - pd.Timedelta(hours=1)

# --- Carga artefactos ---
model_path = find_first(MODEL_PATHS)
assert model_path and model_path.endswith("model_multi.pkl"), "Debes usar el modelo multitarea: model_multi.pkl"
model = joblib.load(model_path)
with open(FEATS_PATH,"r",encoding="utf-8") as f: X_cols=json.load(f)["feature_columns"]
with open(TARGETS_PATH,"r",encoding="utf-8") as f: meta=json.load(f)
assert "y_calls_next" in meta["targets"] and "y_tmo_next" in meta["targets"], "targets.json inválido."

# --- Seed histórico ---
data_path = find_first(DATA_PATHS); assert data_path, "Falta histórico."
raw = pd.read_excel(data_path, engine="openpyxl")
df  = ensure_datetime(raw)
df  = df[["datetime",CALLS_COL,TMO_COL]].dropna(subset=["datetime"]).sort_values("datetime")
hist = aggregate_to_hour(df,CALLS_COL,TMO_COL).sort_values("datetime").reset_index(drop=True)
hist["yhat_calls"]=np.nan; hist["yhat_tmo"]=np.nan

# --- Horizonte ---
now = pd.Timestamp.now(tz=None).floor("h")
start = max(now+pd.Timedelta(hours=1), hist["datetime"].max()+pd.Timedelta(hours=1))
end   = end_of_month_plus2(now)
H = max(0, int((end-start)/pd.Timedelta(hours=1))+1)
future_index = [start+pd.Timedelta(hours=i) for i in range(H)]

# --- Predicción recursiva (calls + tmo) ---
rows=[]
for ts in future_index:
    feat = build_row(ts, hist, X_cols)
    X    = feat[X_cols].values
    yhat = np.column_stack([est.predict(X) for est in model.estimators_])[0]
    calls_hat = float(yhat[0]); tmo_hat = float(yhat[1])

    # alimentar historial para siguientes lags
    new = {"datetime":ts, CALLS_COL:np.nan, TMO_COL:np.nan, "yhat_calls":calls_hat, "yhat_tmo":tmo_hat}
    hist = pd.concat([hist, pd.DataFrame([new])], ignore_index=True)

    N = required_agents(max(calls_hat,0.0), max(tmo_hat,1.0), SLA, ASA, OCC)

    rows.append({
        "datetime": ts,
        "llamadas_recibidas": max(calls_hat,0.0),
        "tmo_pred_seg": max(tmo_hat,0.0),
        "ejecutivos_requeridos": int(N)
    })

forecast = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
forecast["fecha"] = forecast["datetime"].dt.strftime("%Y-%m-%d")
forecast["hora"]  = forecast["datetime"].dt.strftime("%H:%M")
forecast = forecast[["fecha","hora","llamadas_recibidas","tmo_pred_seg","ejecutivos_requeridos","datetime"]]
forecast.drop(columns=["datetime"]).to_json(OUT_JSON, orient="records", force_ascii=False, indent=2)
forecast.drop(columns=["datetime"]).to_csv(OUT_CSV, index=False)

print("✅ Forecast multitarea listo:")
print(" -", OUT_JSON)
print(" -", OUT_CSV)
print(forecast.head(5))

