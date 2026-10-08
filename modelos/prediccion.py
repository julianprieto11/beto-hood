import math
import numpy as np

from modelos.config import MERCADOS_CONTEO


def poisson_pmf(lam, values):
    lam = max(float(lam), 1e-6)
    vals = np.asarray(values, dtype=int)
    return np.exp(-lam) * np.power(lam, vals) / np.array(
        [math.factorial(int(v)) for v in vals]
    )


def _calibration_factor(calibration, market):
    # La memoria puede guardar "goles_home" / "goles_away" o "goles".
    candidates = (
        market,
        f"{market}_home",
        f"{market}_away",
    )
    factors = [calibration[k] for k in candidates if k in calibration]
    return float(np.mean(factors)) if factors else 1.0


def _rate(row, market, side, calibration):
    value = row.get(f"{market}_{side}_expected", np.nan)
    if value is None or np.isnan(float(value)) or float(value) <= 0:
        value = 1.0
    factor = _calibration_factor(calibration, market)
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

    hg = np.arange(0, 12)
    ag = np.arange(0, 12)
    hp = poisson_pmf(rates["goles"][0], hg)
    ap = poisson_pmf(rates["goles"][1], ag)
    matrix = np.outer(hp, ap)

    # La matriz está truncada en 0..11 goles. Normalizamos para que
    # las probabilidades 1X2 sumen exactamente 1.
    matrix_sum = matrix.sum()
    if matrix_sum <= 0:
        matrix_sum = 1.0
    matrix = matrix / matrix_sum

    # Filas = goles local, columnas = goles visitante.
    out["prob_local_gana"] = float(np.tril(matrix, -1).sum())
    out["prob_empate"] = float(np.trace(matrix))
    out["prob_visitante_gana"] = float(np.triu(matrix, 1).sum())

    for market in MERCADOS_CONTEO:
        out[f"{market}_total_lambda"] = sum(rates[market])
        out[f"{market}_media"] = sum(rates[market])

    out["prob_btts"] = float(
        (1 - np.exp(-rates["goles"][0]))
        * (1 - np.exp(-rates["goles"][1]))
    )
    return out
