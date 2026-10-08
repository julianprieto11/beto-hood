import json
import numpy as np
import pandas as pd

from modelos.config import ERRORES, CALIBRACION, ESTADO


def _ensure():
    ERRORES.parent.mkdir(parents=True, exist_ok=True)
    if not ERRORES.exists():
        pd.DataFrame(columns=[
            "fecha", "match_id", "mercado", "prediccion", "real",
            "error_abs", "error", "brier"
        ]).to_csv(ERRORES, index=False)
    if not CALIBRACION.exists():
        pd.DataFrame(columns=[
            "mercado", "factor", "n_observaciones", "mae", "sesgo"
        ]).to_csv(CALIBRACION, index=False)
    if not ESTADO.exists():
        ESTADO.write_text(
            json.dumps({"version": 1, "fechas_aprendidas": [], "observaciones": 0}, indent=2),
            encoding="utf-8",
        )


def cargar_calibracion():
    _ensure()
    df = pd.read_csv(CALIBRACION)
    if df.empty:
        return {}
    return dict(zip(df["mercado"], df["factor"]))


def _load_errors():
    _ensure()
    return pd.read_csv(ERRORES)


def evaluar_fecha(predicciones, partidos, fecha):
    rows = []
    partidos = partidos.set_index("match_id")

    for _, p in predicciones.iterrows():
        mid = p["match_id"]
        if mid not in partidos.index:
            continue
        r = partidos.loc[mid]

        # Si todavía no existe el resultado real, NO aprendemos nada.
        if pd.isna(r["goles_favor"]) or pd.isna(r["goles_contra"]):
            continue

        objetivos = {
            "goles_home": r["goles_favor"],
            "goles_away": r["goles_contra"],
            "corners_home": r["sofascore_cornerKicks"],
            "corners_away": r["rival_cornerKicks"],
            "tarjetas_home": r["sofascore_yellowCards"],
            "tarjetas_away": r["rival_yellowCards"],
            "tiros_home": r["sofascore_totalShotsOnGoal"],
            "tiros_away": r["rival_totalShotsOnGoal"],
            "tiros_arco_home": r["sofascore_shotsOnGoal"],
            "tiros_arco_away": r["rival_shotsOnGoal"],
            "big_chances_home": r["sofascore_bigChanceCreated"],
            "big_chances_away": r["rival_bigChanceCreated"],
        }

        for key, real in objetivos.items():
            if pd.isna(real):
                continue
            pred_key = key.rsplit("_", 1)[0] + "_" + key.rsplit("_", 1)[1] + "_lambda"
            # goles_home -> goles_home_lambda; etc.
            pred = p.get(pred_key, np.nan)
            if pd.isna(pred):
                continue
            rows.append({
                "fecha": fecha,
                "match_id": mid,
                "mercado": key,
                "prediccion": float(pred),
                "real": float(real),
                "error_abs": abs(float(pred) - float(real)),
                "error": float(real) - float(pred),
                "brier": np.nan,
            })

        real_result = (
            "resultado_local" if r["goles_favor"] > r["goles_contra"]
            else "resultado_empate" if r["goles_favor"] == r["goles_contra"]
            else "resultado_visitante"
        )
        probs = {
            "resultado_local": p["prob_local_gana"],
            "resultado_empate": p["prob_empate"],
            "resultado_visitante": p["prob_visitante_gana"],
        }
        for name, prob in probs.items():
            y = 1.0 if name == real_result else 0.0
            rows.append({
                "fecha": fecha,
                "match_id": mid,
                "mercado": name,
                "prediccion": float(prob),
                "real": y,
                "error_abs": abs(float(prob) - y),
                "error": y - float(prob),
                "brier": (float(prob) - y) ** 2,
            })

    return pd.DataFrame(rows)


def actualizar_memoria(evaluacion):
    _ensure()
    if evaluacion.empty:
        return False

    old = _load_errors()
    keys = set(zip(old["fecha"], old["match_id"], old["mercado"])) if not old.empty else set()
    nuevos = evaluacion[
        ~evaluacion.apply(
            lambda x: (x["fecha"], x["match_id"], x["mercado"]) in keys,
            axis=1,
        )
    ]
    if nuevos.empty:
        return False

    all_errors = pd.concat([old, nuevos], ignore_index=True)
    all_errors.to_csv(ERRORES, index=False)

    calibraciones = []
    for mercado, g in all_errors.groupby("mercado"):
        mae = float(g["error_abs"].mean())
        sesgo = float(g["error"].mean())

        # Corrección pequeña y estable. No "persigue" un solo resultado.
        factor = float(np.clip(
            1.0 + 0.15 * np.tanh(sesgo / max(mae, 0.25)),
            0.75,
            1.25,
        ))
        calibraciones.append({
            "mercado": mercado,
            "factor": factor,
            "n_observaciones": len(g),
            "mae": mae,
            "sesgo": sesgo,
        })

    pd.DataFrame(calibraciones).to_csv(CALIBRACION, index=False)

    estado = {
        "version": 1,
        "fechas_aprendidas": sorted(
            pd.Series(all_errors["fecha"]).dropna().astype(str).unique().tolist()
        ),
        "observaciones": int(len(all_errors)),
    }
    ESTADO.write_text(json.dumps(estado, indent=2), encoding="utf-8")
    return True
