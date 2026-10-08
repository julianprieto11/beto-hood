import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

ARCHIVO_ENTRADA = "datos/estadisticas_equipos_beto_hood.csv"
ARCHIVO_SALIDA = "datos/historico_beto_hood.csv"

VENTANAS = [5, 10]


# ============================================================
# CARGAR DATOS
# ============================================================

df = pd.read_csv(ARCHIVO_ENTRADA)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["team_id", "date", "match_id"]
).reset_index(drop=True)


# ============================================================
# COLUMNAS QUE EL MODELO PUEDE UTILIZAR
# ============================================================

columnas_estadisticas = [
    "goles_favor",
    "goles_contra",

    "corners_favor",
    "corners_contra",

    "tarjetas_favor",
    "tarjetas_contra",

    "tiros_favor",
    "tiros_contra",

    "tiros_arco_favor",
    "tiros_arco_contra",

    "tiros_desviados_favor",
    "tiros_desviados_contra",

    "big_chances_favor",
    "big_chances_contra",

    "big_chances_falladas_favor",
    "big_chances_falladas_contra",

    "toques_area_favor",
    "toques_area_contra",

    "faltas_favor",
    "faltas_contra",

    "pases_favor",
    "pases_contra",

    "pases_precisos_favor",
    "pases_precisos_contra",

    "tackles_favor",
    "tackles_contra",

    "intercepciones_favor",
    "intercepciones_contra",

    "recuperaciones_favor",
    "recuperaciones_contra",
]


# ============================================================
# HISTÓRICO SIN DATA LEAKAGE
# ============================================================

resultado = df[
    [
        "date",
        "match_id",
        "round_name",
        "team_id",
        "team_name",
        "local_visitante",
    ]
].copy()


for columna in columnas_estadisticas:

    grupo = df.groupby("team_id")[columna]

    # --------------------------------------------------------
    # MUY IMPORTANTE:
    # shift(1)
    #
    # El partido actual queda FUERA del cálculo.
    # --------------------------------------------------------

    previo = grupo.shift(1)

    for ventana in VENTANAS:

        nombre = f"{columna}_ultimos_{ventana}"

        resultado[nombre] = (
            previo
            .groupby(df["team_id"])
            .rolling(
                window=ventana,
                min_periods=1
            )
            .mean()
            .reset_index(level=0, drop=True)
        )


# ============================================================
# PROMEDIO HISTÓRICO TOTAL
# ============================================================

for columna in columnas_estadisticas:

    nombre = f"{columna}_historico"

    resultado[nombre] = (
        df.groupby("team_id")[columna]
        .transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )


# ============================================================
# CANTIDAD DE PARTIDOS PREVIOS
# ============================================================

resultado["partidos_previos"] = (
    df.groupby("team_id").cumcount()
)


# ============================================================
# GUARDAR
# ============================================================

resultado.to_csv(
    ARCHIVO_SALIDA,
    index=False
)


# ============================================================
# CONTROL
# ============================================================

print()
print("=" * 60)
print("HISTÓRICO BETO HOOD GENERADO")
print("=" * 60)

print()
print("REGISTROS:", len(resultado))
print("EQUIPOS:", resultado["team_id"].nunique())
print("PARTIDOS:", resultado["match_id"].nunique())
print("COLUMNAS:", len(resultado.columns))

print()
print("ARCHIVO:")
print(ARCHIVO_SALIDA)

print()
print("PRIMEROS REGISTROS:")

columnas_mostrar = [
    "date",
    "team_name",
    "local_visitante",
    "partidos_previos",
    "goles_favor_ultimos_5",
    "goles_contra_ultimos_5",
    "corners_favor_ultimos_5",
    "tiros_arco_favor_ultimos_5",
]

print(
    resultado[columnas_mostrar]
    .head(15)
    .to_string(index=False)
)