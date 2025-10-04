# -*- coding: utf-8 -*-
import math

def erlang_c_prob_wait(N: int, A: float) -> float:
    """
    Probabilidad de espera en M/M/N (Erlang C).
    N: agentes
    A: carga ofrecida en Erlangs (λ * AHT)
    """
    if A <= 0: return 0.0
    if N <= 0: return 1.0
    if A >= N: 
        return 1.0

    summ = 0.0
    term = 1.0
    for k in range(1, N):
        summ += term
        term *= A / k
    summ += term  # k = N-1

    pn = term * (A / N) / (1.0 - (A / N))
    P_wait = pn / (summ + pn)
    return float(P_wait)

def service_level(N: int, A: float, AHT_sec: float, ASA_target_sec: float) -> float:
    """
    SL = 1 - P(wait) * exp(-(N - A) * (ASA / AHT))
    """
    if A <= 0: return 1.0
    if N <= 0: return 0.0
    if A >= N:
        P_w = 1.0
    else:
        P_w = erlang_c_prob_wait(N, A)
    expo = - (N - A) * (ASA_target_sec / max(AHT_sec, 1e-9))
    return 1.0 - P_w * math.exp(expo)

def required_agents_erlang_c(calls_per_hour: float,
                             AHT_sec: float,
                             sla_target: float = 0.90,
                             asa_target_sec: float = 22.0,
                             max_occupancy: float = 0.80) -> int:
    """
    Devuelve N mínimo que cumpla:
      - Service Level(asa_target_sec) >= sla_target
      - A/N <= max_occupancy
    """
    if calls_per_hour <= 0 or AHT_sec <= 0:
        return 0

    lam = calls_per_hour / 3600.0
    A = lam * AHT_sec

    N_min_occ = max(1, int(math.ceil(A / max_occupancy)))
    N = max(1, N_min_occ)

    while True:
        sl = service_level(N, A, AHT_sec, asa_target_sec)
        occ = A / N
        if sl >= sla_target and occ <= max_occupancy:
            return N
        N += 1
        if N > 10000:
            return N
