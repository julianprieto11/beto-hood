import argparse
import sys
from pathlib import Path

# Permite ejecutar este archivo directamente desde scripts\\validar_modelo.py
# sin depender de PYTHONPATH.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from modelos.config import PARTIDOS, ESTADISTICAS_EQUIPOS, HISTORICO
from modelos.datos import cargar_partidos, cargar_historico, cargar_estadisticas_equipos, partidos_de_fecha
from modelos.features import construir_features
from modelos.prediccion import predict_match
from simulacion.motor import simular_lote


def check(ok, msg):
    print(("OK   " if ok else "ERROR"), msg)
    return ok


def main():
    parser = argparse.ArgumentParser(description="Validacion estructural y anti-leakage de Beto Hood")
    parser.add_argument("--fecha", type=int, default=1)
    args = parser.parse_args()

    failures = 0

    for path in (PARTIDOS, ESTADISTICAS_EQUIPOS, HISTORICO):
        failures += not check(
            path.exists(),
            f"existe {path.relative_to(Path.cwd()) if path.is_absolute() else path}",
        )

    if failures:
        raise SystemExit(1)

    partidos = cargar_partidos()
    raw = cargar_estadisticas_equipos()
    historico = cargar_historico()

    required_partidos = {"match_id", "date", "round_name", "goles_favor", "goles_contra"}
    required_raw = {"match_id", "date", "team_id", "team_name", "local_visitante"}
    required_hist = {"match_id", "date", "team_id", "partidos_previos"}

    failures += not check(required_partidos <= set(partidos.columns), "columnas obligatorias de partidos")
    failures += not check(required_raw <= set(raw.columns), "columnas obligatorias de estadisticas de equipos")
    failures += not check(required_hist <= set(historico.columns), "columnas obligatorias de historico")

    failures += not check(partidos["match_id"].notna().all(), "partidos sin match_id nulo")
    failures += not check(partidos["date"].notna().all(), "partidos sin fecha nula")
    failures += not check(raw["date"].notna().all(), "estadisticas sin fecha nula")
    failures += not check(historico["date"].notna().all(), "historico sin fecha nula")

    dup_matches = partidos["match_id"].duplicated().sum()
    failures += not check(dup_matches == 0, f"match_id unicos ({dup_matches} duplicados)")

    dup_team_match = raw.duplicated(["match_id", "team_id"]).sum()
    failures += not check(dup_team_match == 0, f"un registro por equipo/partido ({dup_team_match} duplicados)")

    leakage = historico["partidos_previos"].lt(0).sum()
    failures += not check(leakage == 0, f"partidos_previos no negativos ({leakage})")

    fecha_partidos = partidos_de_fecha(partidos, args.fecha, "Clausura")
    failures += not check(not fecha_partidos.empty, f"Clausura Fecha {args.fecha:02d} encontrada")

    if not fecha_partidos.empty:
        corte = fecha_partidos["date"].min()
        history = historico[historico["date"] < corte]
        raw_history = raw[raw["date"] < corte]

        same_day_hist = history["date"].ge(corte).sum()
        same_day_raw = raw_history["date"].ge(corte).sum()
        failures += not check(same_day_hist == 0, "historico no usa datos del mismo dia")
        failures += not check(same_day_raw == 0, "estadisticas no usan datos del mismo dia")

        features = construir_features(fecha_partidos, history, raw_history)
        failures += not check(
            len(features) == len(fecha_partidos),
            f"features completas ({len(features)}/{len(fecha_partidos)})",
        )

        if not features.empty:
            preds = pd.DataFrame([predict_match(row, {}) for _, row in features.iterrows()])
            pcols = ["prob_local_gana", "prob_empate", "prob_visitante_gana"]
            sums = preds[pcols].sum(axis=1)
            finite = np.isfinite(sums).all()
            failures += not check(finite, "probabilidades 1X2 finitas")
            failures += not check(np.allclose(sums, 1.0, atol=1e-8), "probabilidades 1X2 suman 1")

            sim = simular_lote(preds, n_simulaciones=200)
            sim_prob_cols = [c for c in sim.columns if c.startswith("prob_")]
            bounded = all(sim[c].between(0, 1).all() for c in sim_prob_cols)
            failures += not check(bounded, "probabilidades de simulacion entre 0 y 1")

    print()
    if failures:
        print(f"VALIDACION FALLIDA: {failures} chequeo(s)")
        raise SystemExit(1)

    print("VALIDACION OK: arquitectura, datos, prediccion, simulacion y anti-leakage basicos funcionan.")


if __name__ == "__main__":
    main()
