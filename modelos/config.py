from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATOS = ROOT / "datos"
MODELOS = ROOT / "modelos"
SALIDAS = ROOT / "salidas"
MEMORIA = MODELOS / "memoria"

PARTIDOS = DATOS / "partidos_beto_hood.csv"
ESTADISTICAS_EQUIPOS = DATOS / "estadisticas_equipos_beto_hood.csv"
HISTORICO = DATOS / "historico_beto_hood.csv"

PREDICCIONES = SALIDAS / "predicciones.csv"
SIMULACIONES = SALIDAS / "simulaciones.csv"
ERRORES = MEMORIA / "errores_historicos.csv"
CALIBRACION = MEMORIA / "calibracion_mercados.csv"
ESTADO = MEMORIA / "estado_modelo.json"

VENTANAS = (5, 10)
N_SIMULACIONES = 10000

MERCADOS_CONTEO = (
    "goles",
    "corners",
    "tarjetas",
    "tiros",
    "tiros_arco",
    "big_chances",
)

MERCADOS_DIRECCIONALES = ("local_gana", "empate", "visitante_gana")
