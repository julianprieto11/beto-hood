import numpy as np
import pandas as pd

STAT_MAP = {
    "goles": ("goles_favor", "goles_contra"),
    "corners": ("corners_favor", "corners_contra"),
    "tarjetas": ("tarjetas_favor", "tarjetas_contra"),
    "tiros": ("tiros_favor", "tiros_contra"),
    "tiros_arco": ("tiros_arco_favor", "tiros_arco_contra"),
    "big_chances": ("big_chances_favor", "big_chances_contra"),
}


def _safe_mean(row, names):
    values = [row.get(n) for n in names if n in row.index]
    values = [float(v) for v in values if pd.notna(v)]
    return float(np.mean(values)) if values else np.nan


def _team_snapshot(history, team_id, venue):
    h = history[history["team_id"] == team_id].sort_values("date")
    if h.empty:
        return None

    result = {"team_id": team_id, "team_name": h.iloc[-1]["team_name"]}
    for market, (fav, con) in STAT_MAP.items():
        for window in (5, 10):
            result[f"{market}_for_{window}"] = _safe_mean(
                h.tail(window), [f"{fav}_ultimos_{window}"]
            )
            result[f"{market}_against_{window}"] = _safe_mean(
                h.tail(window), [f"{con}_ultimos_{window}"]
            )
        result[f"{market}_for_all"] = _safe_mean(
            h, [f"{fav}_historico"]
        )
        result[f"{market}_against_all"] = _safe_mean(
            h, [f"{con}_historico"]
        )

    venue_h = h[h["local_visitante"] == venue]
    result["venue_matches"] = len(venue_h)
    for market, (fav, con) in STAT_MAP.items():
        result[f"{market}_for_venue"] = (
            float(venue_h[fav].mean()) if not venue_h.empty else np.nan
        )
        result[f"{market}_against_venue"] = (
            float(venue_h[con].mean()) if not venue_h.empty else np.nan
        )
    result["matches"] = len(h)
    return result


def _fallback(value, *fallbacks):
    if pd.notna(value):
        return float(value)
    valid = [float(x) for x in fallbacks if pd.notna(x)]
    return float(valid[0]) if valid else np.nan


def construir_feature_partido(match, history):
    home = _team_snapshot(history, match["pitchapi_home_team_id"], "LOCAL")
    away = _team_snapshot(history, match["pitchapi_away_team_id"], "VISITANTE")

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
        # Ataque local + defensa visitante y viceversa.
        h_for = _fallback(home[f"{market}_for_5"], home[f"{market}_for_10"], home[f"{market}_for_all"])
        a_against = _fallback(away[f"{market}_against_5"], away[f"{market}_against_10"], away[f"{market}_against_all"])
        a_for = _fallback(away[f"{market}_for_5"], away[f"{market}_for_10"], away[f"{market}_for_all"])
        h_against = _fallback(home[f"{market}_against_5"], home[f"{market}_against_10"], home[f"{market}_against_all"])

        row[f"{market}_home_expected"] = np.nanmean([h_for, a_against])
        row[f"{market}_away_expected"] = np.nanmean([a_for, h_against])

        row[f"{market}_home_venue"] = home[f"{market}_for_venue"]
        row[f"{market}_away_venue"] = away[f"{market}_for_venue"]

    return row


def construir_features(matches, history):
    rows = []
    for _, match in matches.iterrows():
        row = construir_feature_partido(match, history)
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)
