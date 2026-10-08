import re
import pandas as pd

from modelos.config import PARTIDOS, HISTORICO, ESTADISTICAS_EQUIPOS


def cargar_partidos():
    df = pd.read_csv(PARTIDOS)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "match_id"]).reset_index(drop=True)


def cargar_historico():
    df = pd.read_csv(HISTORICO)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "match_id"]).reset_index(drop=True)


def cargar_estadisticas_equipos():
    df = pd.read_csv(ESTADISTICAS_EQUIPOS)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "match_id"]).reset_index(drop=True)


def normalizar_competencia(round_name):
    texto = str(round_name).strip().lower()
    if "clausura" in texto:
        return "Clausura"
    if "apertura" in texto:
        return "Apertura"
    return "Otro"


def extraer_fecha(round_name):
    texto = str(round_name)
    m = re.search(r"(?:round|fecha)[ _-]*(\d+)", texto, flags=re.I)
    if m:
        return int(m.group(1))
    if texto.strip().isdigit():
        return int(texto.strip())
    return None


def preparar_partidos(df):
    out = df.copy()
    out["competencia"] = out["round_name"].map(normalizar_competencia)
    out["fecha_num"] = out["round_name"].map(extraer_fecha)
    return out


def partidos_de_fecha(df, fecha, competencia="Clausura"):
    df = preparar_partidos(df)
    return df[
        (df["competencia"] == competencia)
        & (df["fecha_num"] == int(fecha))
    ].copy()


def historico_anterior(df_historico, fecha):
    return df_historico[df_historico["date"] < pd.Timestamp(fecha)].copy()
