import argparse
import pandas as pd

from modelos.datos import (
    cargar_partidos,
    cargar_historico,
    cargar_estadisticas_equipos,
    partidos_de_fecha,
)
from modelos.features import construir_features
from modelos.prediccion import predict_match
from modelos.aprendizaje import cargar_calibracion, evaluar_fecha, actualizar_memoria
from modelos.config import SALIDAS, ERRORES, CALIBRACION, ESTADO


def main():
    parser = argparse.ArgumentParser(
        description="Backtest cronológico Beto Hood"
    )
    parser.add_argument("--desde", type=int, default=1)
    parser.add_argument("--hasta", type=int, required=True)
    parser.add_argument("--aprender", action="store_true")
    parser.add_argument(
        "--reiniciar-memoria",
        action="store_true",
        help="Borra la memoria de aprendizaje antes de comenzar el backtest.",
    )
    args = parser.parse_args()

    if args.reiniciar_memoria:
        for path in (ERRORES, CALIBRACION, ESTADO):
            if path.exists():
                path.unlink()
        print("MEMORIA DE APRENDIZAJE REINICIADA.")

    partidos = cargar_partidos()
    historico = cargar_historico()
    raw = cargar_estadisticas_equipos()

    pred_rows = []
    error_rows = []

    for fecha in range(args.desde, args.hasta + 1):
        pf = partidos_de_fecha(partidos, fecha, "Clausura")
        if pf.empty:
            continue

        corte = pf["date"].min()
        history = historico[historico["date"] < corte]
        raw_history = raw[raw["date"] < corte]

        features = construir_features(pf, history, raw_history)
        if features.empty:
            continue

        # La calibración disponible al comienzo de esta fecha.
        cal = cargar_calibracion()

        preds = pd.DataFrame([
            predict_match(row, cal)
            for _, row in features.iterrows()
        ])
        pred_rows.append(preds)

        # SOLO después de generar la predicción usamos el resultado.
        if args.aprender:
            evaluation = evaluar_fecha(preds, pf, fecha)
            if not evaluation.empty:
                error_rows.append(evaluation)
                actualizar_memoria(evaluation)

        print(
            f"Fecha {fecha:02d}: {len(preds)} partidos | "
            f"histórico hasta {corte.date()}"
        )

    SALIDAS.mkdir(exist_ok=True)
    out = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    path = SALIDAS / f"backtest_predicciones_{args.desde}_{args.hasta}.csv"
    out.to_csv(path, index=False)

    if error_rows:
        errors = pd.concat(error_rows, ignore_index=True)
        errors.to_csv(
            SALIDAS / f"backtest_errores_{args.desde}_{args.hasta}.csv",
            index=False,
        )

    print(f"BACKTEST GENERADO: {path}")
    print(f"REGISTROS: {len(out)}")


if __name__ == "__main__":
    main()
