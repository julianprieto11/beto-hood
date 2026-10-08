import numpy as np
import pandas as pd

from modelos.config import ESTADISTICAS_EQUIPOS

STAT_MAP = {
    "goles": ("goles_favor", "goles_contra"),
    "corners": ("corners_favor", "corners_contra"),
    "tarjetas": ("tarjetas_favor", "tarjetas_contra"),
    "tiros": ("tiros_favor", "tiros_contra"),
    "tiros_arco": ("tiros_arco_favor", "tiros_arco_contra"),
    "big_chances": ("big_chances_favor", "big_chances_contra"),
}


def _mean(values):
    values = pd.Series(values).dropna()
    return float(values.mean()) if not values.empty else np.nan


def _team_snapshot(history, raw, team_id, venue):
    h = history[history["team_id"] == team_id].sort_values(["date", "match_id"])
    if h.empty:
        return None

    last = h.iloc[-1]
    result = {
        "team_id": team_id,
        "team_name": last["team_name"],
        "matches": int(last.get("partidos_previos", len(h))),
    }

    # El histórico ya contiene rolling/expanding calculados exclusivamente
    # con partidos anteriores. Tomamos SOLO la última fila disponible.
    for market, (fav, con) in STAT_MAP.items():
        for window in (5, 10):
            result[f"{market}_for_{window}"] = last.get(
                f"{fav}_ultimos_{window}", np.nan
            )
            result[f"{market}_against_{window}"] = last.get(
                f"{con}_ultimos_{window}", np.nan
            )
        result[f"{market}_for_all"] = last.get(f"{fav}_historico", np.nan)
        result[f"{market}_against_all"] = last.get(f"{con}_historico", np.nan)

    # Localía específica se calcula sobre la tabla de partidos de equipo,
    # siempre filtrada a fechas anteriores al partido objetivo.
    r = raw[
        (raw["team_id"] == team_id)
        & (raw["local_visitante"] == venue)
    ].sort_values(["date", "match_id"])

    result["venue_matches"] = len(r)
    for market, (fav, con) in STAT_MAP.items():
        result[f"{market}_for_venue"] = _mean(r[fav]) if not r.empty else np.nan
        result[f"{market}_against_venue"] = _mean(r[con]) if not r.empty else np.nan

    return result


def _fallback(*values, default=1.0):
    for value in values:
        if pd.notna(value) and float(value) > 0:
            return float(value)
    return float(default)


def construir_feature_partido(match, history, raw_stats=None):
    if raw_stats is None:
        raw_stats = pd.DataFrame()

    home = _team_snapshot(
        history, raw_stats,
        match["pitchapi_home_team_id"], "LOCAL"
    )
    away = _team_snapshot(
        history, raw_stats,
        match["pitchapi_away_team_id"], "VISITANTE"
    )

    if home is None or away is None:
        return None

    row = {
        "match_id": match["match_id"],
        "date": match["date"],
        "round_name": match["round_name"],
        "home_team": match["pitchapi_home_team"],
        "away_team": match["pitchapi_away_team"],
        "home_team_id": match["pitchapi_home_team_id"],
        "away_team_id": match["pitchapi_away_team_id"],
        "home_matches_history": home["matches"],
        "away_matches_history": away["matches"],
    }

    for market in STAT_MAP:
        h_for = _fallback(
            home[f"{market}_for_5"],
            home[f"{market}_for_10"],
            home[f"{market}_for_all"],
        )
        a_against = _fallback(
            away[f"{market}_against_5"],
            away[f"{market}_against_10"],
            away[f"{market}_against_all"],
        )
        a_for = _fallback(
            away[f"{market}_for_5"],
            away[f"{market}_for_10"],
            away[f"{market}_for_all"],
        )
        h_against = _fallback(
            home[f"{market}_against_5"],
            home[f"{market}_against_10"],
            home[f"{market}_against_all"],
        )

        # Mezcla ataque propio + defensa rival.
        row[f"{market}_home_expected"] = np.mean([h_for, a_against])
        row[f"{market}_away_expected"] = np.mean([a_for, h_against])

        # La localía específica pesa si hay muestra suficiente.
        h_venue = home[f"{market}_for_venue"]
        a_venue = away[f"{market}_for_venue"]
        if pd.notna(h_venue) and home["venue_matches"] >= 3:
            row[f"{market}_home_expected"] = (
                0.70 * row[f"{market}_home_expected"] + 0.30 * h_venue
            )
        if pd.notna(a_venue) and away["venue_matches"] >= 3:
            row[f"{market}_away_expected"] = (
                0.70 * row[f"{market}_away_expected"] + 0.30 * a_venue
            )

        row[f"{market}_home_venue"] = h_venue
        row[f"{market}_away_venue"] = a_venue

    return row


def construir_features(matches, history, raw_stats=None):
    rows = []
    for _, match in matches.iterrows():
        row = construir_feature_partido(match, history, raw_stats)
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)
