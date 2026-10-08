import argparse
import pandas as pd

from modelos.datos import cargar_partidos, cargar_historico, partidos_de_fecha
from modelos.features import construir_features
from modelos.prediccion import predict_match
from modelos.aprendizaje import cargar_calibracion
from modelos.config import SALIDAS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--desde", type=int, default=1)
    parser.add_argument("--hasta", type=int, required=True)
    args = parser.parse_args()

    partidos = cargar_partidos()
    historico = cargar_historico()
    rows = []

    for fecha in range(args.desde, args.hasta + 1):
        pf = partidos_de_fecha(partidos, fecha, "Clausura")
        if pf.empty:
            continue
        corte = pf["date"].min()
        history = historico[historico["date"] < corte]
        features = construir_features(pf, history)
        if features.empty:
            continue

        cal = cargar_calibracion()
        for _, row in features.iterrows():
            p = predict_match(row, cal)
            rows.append(p)

    SALIDAS.mkdir(exist_ok=True)
    out = pd.DataFrame(rows)
    path = SALIDAS / f"backtest_predicciones_{args.desde}_{args.hasta}.csv"
    out.to_csv(path, index=False)
    print(f"BACKTEST GENERADO: {path}")
    print(f"REGISTROS: {len(out)}")


if __name__ == "__main__":
    main()
