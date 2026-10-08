import argparse
from pathlib import Path

import pandas as pd

from modelos.config import SALIDAS, PREDICCIONES, SIMULACIONES, N_SIMULACIONES
from modelos.datos import cargar_partidos, cargar_historico, partidos_de_fecha
from modelos.features import construir_features
from modelos.prediccion import predict_match
from modelos.aprendizaje import cargar_calibracion, evaluar_fecha, actualizar_memoria
from simulacion.motor import simular_lote


def main():
    parser = argparse.ArgumentParser(description="Motor Beto Hood fecha a fecha")
    parser.add_argument("fecha", type=int)
    parser.add_argument("--competencia", default="Clausura")
    parser.add_argument("--simulaciones", type=int, default=N_SIMULACIONES)
    parser.add_argument("--sin-aprender", action="store_true")
    args = parser.parse_args()

    SALIDAS.mkdir(parents=True, exist_ok=True)

    partidos = cargar_partidos()
    historico = cargar_historico()
    fecha_partidos = partidos_de_fecha(partidos, args.fecha, args.competencia)

    if fecha_partidos.empty:
        raise SystemExit(f"No se encontraron partidos para {args.competencia} Fecha {args.fecha}.")

    # Corte estricto: ninguna predicción ve partidos del mismo día.
    fecha_real = fecha_partidos["date"].min()
    history = historico[historico["date"] < fecha_real].copy()

    features = construir_features(fecha_partidos, history)
    if features.empty:
        raise SystemExit("No hay suficiente histórico para construir las predicciones.")

    calibration = cargar_calibracion()
    preds = pd.DataFrame([
        predict_match(row, calibration)
        for _, row in features.iterrows()
    ])

    sims = simular_lote(preds, n=args.simulaciones)

    # Guardar snapshot de predicción de esta fecha.
    pred_path = SALIDAS / f"predicciones_{args.competencia.lower()}_{args.fecha:02d}.csv"
    sim_path = SALIDAS / f"simulaciones_{args.competencia.lower()}_{args.fecha:02d}.csv"
    preds.to_csv(pred_path, index=False)
    sims.to_csv(sim_path, index=False)

    # También mantenemos un histórico consolidado.
    if PREDICCIONES.exists():
        old = pd.read_csv(PREDICCIONES)
        old = old[~old["match_id"].isin(preds["match_id"])]
        pd.concat([old, preds], ignore_index=True).to_csv(PREDICCIONES, index=False)
    else:
        preds.to_csv(PREDICCIONES, index=False)

    if not args.sin_aprender:
        evaluacion = evaluar_fecha(preds, fecha_partidos, args.fecha)
        actualizar_memoria(evaluacion)

    print("=" * 70)
    print(f"BETO HOOD | {args.competencia} FECHA {args.fecha}")
    print("=" * 70)
    print(f"PARTIDOS: {len(fecha_partidos)}")
    print(f"HISTÓRICO UTILIZADO HASTA: {fecha_real.date()}")
    print(f"SIMULACIONES POR PARTIDO: {args.simulaciones}")
    print()
    for _, p in preds.iterrows():
        print(
            f'{p["home_team"]} vs {p["away_team"]} | '
            f'1={p["prob_local_gana"]:.1%} '
            f'X={p["prob_empate"]:.1%} '
            f'2={p["prob_visitante_gana"]:.1%}'
        )
    print()
    print(f"Predicciones: {pred_path}")
    print(f"Simulaciones: {sim_path}")


if __name__ == "__main__":
    main()
