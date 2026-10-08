import math
import numpy as np
import pandas as pd

from modelos.config import MERCADOS_CONTEO


def poisson_pmf(lam, values):
    lam = max(float(lam), 1e-6)
    vals = np.asarray(values, dtype=int)
    return np.exp(-lam) * np.power(lam, vals) / np.array([math.factorial(int(v)) for v in vals])


def _rate(row, market, side, calibration):
    value = row.get(f"{market}_{side}_expected", np.nan)
    if pd.isna(value) or value <= 0:
        value = 1.0
    factor = calibration.get(market, 1.0)
    return max(float(value) * factor, 0.05)


def predict_match(row, calibration=None):
    calibration = calibration or {}
    out = {
        "match_id": row["match_id"],
        "date": row["date"],
        "home_team": row["home_team"],
        "away_team": row["away_team"],
    }

    rates = {}
    for market in MERCADOS_CONTEO:
        home = _rate(row, market, "home", calibration)
        away = _rate(row, market, "away", calibration)
        rates[market] = (home, away)
        out[f"{market}_home_lambda"] = home
        out[f"{market}_away_lambda"] = away

    # Resultado 1X2: simulación analítica de goles.
    hg = np.arange(0, 10)
    ag = np.arange(0, 10)
    hp = poisson_pmf(rates["goles"][0], hg)
    ap = poisson_pmf(rates["goles"][1], ag)
    matrix = np.outer(hp, ap)
    out["prob_local_gana"] = float(np.tril(matrix, -1).sum())
    out["prob_empate"] = float(np.trace(matrix))
    out["prob_visitante_gana"] = float(np.triu(matrix, 1).sum())

    for market in MERCADOS_CONTEO:
        h, a = rates[market]
        total = h + a
        out[f"{market}_total_lambda"] = total
        out[f"{market}_media"] = total

    return out
