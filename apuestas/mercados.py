import numpy as np


def mercado(probabilidad, cuota=None):
    p = float(np.clip(probabilidad, 0.0001, 0.9999))
    resultado = {
        "probabilidad": p,
        "probabilidad_implícita": None,
        "cuota_justa": 1.0 / p,
        "edge": None,
    }
    if cuota is not None and float(cuota) > 1:
        q = float(cuota)
        impl = 1.0 / q
        resultado["probabilidad_implícita"] = impl
        resultado["edge"] = p - impl
    return resultado


def ranking_value(predicciones, cuotas=None):
    rows = []
    cuotas = cuotas or {}
    for nombre, p in predicciones.items():
        info = mercado(p, cuotas.get(nombre))
        info["mercado"] = nombre
        rows.append(info)
    return sorted(rows, key=lambda x: (x["edge"] if x["edge"] is not None else x["probabilidad"]), reverse=True)
