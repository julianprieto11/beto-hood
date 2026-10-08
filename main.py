import pandas as pd

archivo = "datos/partidos_beto_hood.csv"

df = pd.read_csv(archivo)

df["date"] = pd.to_datetime(df["date"])

# ============================================================
# ESTADÍSTICAS POR EQUIPO
# ============================================================

local = pd.DataFrame({
    "date": df["date"],
    "match_id": df["match_id"],
    "round_name": df["round_name"],
    "team_id": df["pitchapi_home_team_id"],
    "team_name": df["pitchapi_home_team"],
    "local_visitante": "LOCAL",

    "goles_favor": df["goles_favor"],
    "goles_contra": df["goles_contra"],

    "corners_favor": df["sofascore_cornerKicks"],
    "corners_contra": df["rival_cornerKicks"],

    "tarjetas_favor": df["sofascore_yellowCards"],
    "tarjetas_contra": df["rival_yellowCards"],

    "tiros_favor": df["sofascore_totalShotsOnGoal"],
    "tiros_contra": df["rival_totalShotsOnGoal"],

    "tiros_arco_favor": df["sofascore_shotsOnGoal"],
    "tiros_arco_contra": df["rival_shotsOnGoal"],

    "tiros_desviados_favor": df["sofascore_shotsOffGoal"],
    "tiros_desviados_contra": df["rival_shotsOffGoal"],

    "big_chances_favor": df["sofascore_bigChanceCreated"],
    "big_chances_contra": df["rival_bigChanceCreated"],

    "big_chances_falladas_favor": df["sofascore_bigChanceMissed"],
    "big_chances_falladas_contra": df["rival_bigChanceMissed"],

    "toques_area_favor": df["sofascore_touchesInOppBox"],
    "toques_area_contra": df["rival_touchesInOppBox"],

    "faltas_favor": df["sofascore_fouls"],
    "faltas_contra": df["rival_fouls"],

    "pases_favor": df["sofascore_passes"],
    "pases_contra": df["rival_passes"],

    "pases_precisos_favor": df["sofascore_accuratePasses"],
    "pases_precisos_contra": df["rival_accuratePasses"],

    "tackles_favor": df["sofascore_totalTackle"],
    "tackles_contra": df["rival_totalTackle"],

    "intercepciones_favor": df["sofascore_interceptionWon"],
    "intercepciones_contra": df["rival_interceptionWon"],

    "recuperaciones_favor": df["sofascore_ballRecovery"],
    "recuperaciones_contra": df["rival_ballRecovery"],
})

visitante = pd.DataFrame({
    "date": df["date"],
    "match_id": df["match_id"],
    "round_name": df["round_name"],
    "team_id": df["pitchapi_away_team_id"],
    "team_name": df["pitchapi_away_team"],
    "local_visitante": "VISITANTE",

    "goles_favor": df["goles_contra"],
    "goles_contra": df["goles_favor"],

    "corners_favor": df["rival_cornerKicks"],
    "corners_contra": df["sofascore_cornerKicks"],

    "tarjetas_favor": df["rival_yellowCards"],
    "tarjetas_contra": df["sofascore_yellowCards"],

    "tiros_favor": df["rival_totalShotsOnGoal"],
    "tiros_contra": df["sofascore_totalShotsOnGoal"],

    "tiros_arco_favor": df["rival_shotsOnGoal"],
    "tiros_arco_contra": df["sofascore_shotsOnGoal"],

    "tiros_desviados_favor": df["rival_shotsOffGoal"],
    "tiros_desviados_contra": df["sofascore_shotsOffGoal"],

    "big_chances_favor": df["rival_bigChanceCreated"],
    "big_chances_contra": df["sofascore_bigChanceCreated"],

    "big_chances_falladas_favor": df["rival_bigChanceMissed"],
    "big_chances_falladas_contra": df["sofascore_bigChanceMissed"],

    "toques_area_favor": df["rival_touchesInOppBox"],
    "toques_area_contra": df["sofascore_touchesInOppBox"],

    "faltas_favor": df["rival_fouls"],
    "faltas_contra": df["sofascore_fouls"],

    "pases_favor": df["rival_passes"],
    "pases_contra": df["sofascore_passes"],

    "pases_precisos_favor": df["rival_accuratePasses"],
    "pases_precisos_contra": df["sofascore_accuratePasses"],

    "tackles_favor": df["rival_totalTackle"],
    "tackles_contra": df["sofascore_totalTackle"],

    "intercepciones_favor": df["rival_interceptionWon"],
    "intercepciones_contra": df["sofascore_interceptionWon"],

    "recuperaciones_favor": df["rival_ballRecovery"],
    "recuperaciones_contra": df["sofascore_ballRecovery"],
})

equipos = pd.concat([local, visitante], ignore_index=True)

equipos = equipos.sort_values(["team_name", "date"])

# ============================================================
# GUARDAR
# ============================================================

salida = "datos/estadisticas_equipos_beto_hood.csv"

equipos.to_csv(salida, index=False)

print("ESTADISTICAS DE EQUIPOS GENERADAS")
print()
print("PARTIDOS:", df["match_id"].nunique())
print("REGISTROS EQUIPO:", len(equipos))
print("EQUIPOS:", equipos["team_name"].nunique())
print()
print("ARCHIVO:")
print(salida)
print()
print("EJEMPLO:")
print(equipos.head(10).to_string(index=False))