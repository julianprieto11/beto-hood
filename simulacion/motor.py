import numpy as np
import pandas as pd

from modelos.config import N_SIMULACIONES, MERCADOS_CONTEO


def simular_partido(prediccion, n=N_SIMULACIONES, seed=42):
    rng = np.random.default_rng(seed)

    home = {}
    away = {}
    for market in MERCADOS_CONTEO:
        home[market] = rng.poisson(
            max(float(prediccion[f"{market}_home_lambda"]), 0.05), n
        )
        away[market] = rng.poisson(
            max(float(prediccion[f"{market}_away_lambda"]), 0.05), n
        )

    goles_h = home["goles"]
    goles_a = away["goles"]

    data = {
        "prob_local_gana": np.mean(goles_h > goles_a),
        "prob_empate": np.mean(goles_h == goles_a),
        "prob_visitante_gana": np.mean(goles_h < goles_a),
    }

    for market in MERCADOS_CONTEO:
        total = home[market] + away[market]
        data[f"{market}_media_sim"] = float(np.mean(total))
        for line in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5):
            data[f"{market}_over_{str(line).replace('.', '_')}"] = float(np.mean(total > line))
            data[f"{market}_under_{str(line).replace('.', '_')}"] = float(np.mean(total < line))

    data["goles_local_media_sim"] = float(np.mean(goles_h))
    data["goles_visitante_media_sim"] = float(np.mean(goles_a))
    data["goles_total_media_sim"] = float(np.mean(goles_h + goles_a))
    data["goles_local_cero"] = float(np.mean(goles_h == 0))
    data["goles_visitante_cero"] = float(np.mean(goles_a == 0))
    data["btts"] = float(np.mean((goles_h > 0) & (goles_a > 0)))

    return data


def simular_lote(predicciones, n=N_SIMULACIONES):
    rows = []
    for i, pred in predicciones.iterrows():
        sim = simular_partido(pred, n=n, seed=42 + i)
        sim["match_id"] = pred["match_id"]
        sim["home_team"] = pred["home_team"]
        sim["away_team"] = pred["away_team"]
        rows.append(sim)
    return pd.DataFrame(rows)
