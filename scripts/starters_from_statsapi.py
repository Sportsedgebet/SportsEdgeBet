"""
Starting lineups from MLB's official Stats API via MLB-StatsAPI (import statsapi).

Uses schedule for the slate date, then per gamePk loads the game feed and reads
liveData.boxscore.teams.(home|away): batters list + players[ID{pid}].battingOrder.
Starters are rows where battingOrder is in {100, 200, ..., 900} (MLB encoding).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

# MLB boxscore uses 100=1st hitter, 200=2nd, ... 900=9th
STARTER_BAT_ORDERS = frozenset(range(100, 901, 100))


def _game_payload(game_pk: int) -> Dict[str, Any]:
    import statsapi

    data = statsapi.get("game", {"gamePk": game_pk})
    if isinstance(data, str):
        import json

        data = json.loads(data)
    if not isinstance(data, dict):
        return {}
    return data


def fetch_starters_for_date(schedule_date: str, sport_id: int = 1) -> Tuple[pd.DataFrame, int]:
    """
    schedule_date: 'YYYY-MM-DD'
    Returns (starters_df, num_scheduled_games).
    starters_df columns: game_id, team_name, player_id, full_name, batting_order
    """
    import statsapi

    games: List[dict] = statsapi.schedule(date=schedule_date, sportId=sport_id) or []
    num_scheduled = len(games)
    rows: List[dict] = []

    for g in games:
        gid = g.get("game_id")
        if gid is None:
            continue
        try:
            gid_int = int(gid)
        except (TypeError, ValueError):
            continue

        try:
            data = _game_payload(gid_int)
        except Exception:
            continue

        live = data.get("liveData") or {}
        box = live.get("boxscore") or {}
        teams = box.get("teams") or {}
        if not teams:
            continue

        home_blk = teams.get("home") or {}
        away_blk = teams.get("away") or {}
        home_name = (home_blk.get("team") or {}).get("name") or "Home"
        away_name = (away_blk.get("team") or {}).get("name") or "Away"
        matchup_label = f"{away_name} @ {home_name}"

        for side in ("home", "away"):
            t = teams.get(side) or {}
            team_info = t.get("team") or {}
            team_name = team_info.get("name") or side
            opponent_team = away_name if side == "home" else home_name
            batters = t.get("batters") or []
            players = t.get("players") or {}

            for pid in batters:
                try:
                    pid_int = int(pid)
                except (TypeError, ValueError):
                    continue
                key = f"ID{pid_int}"
                rec = players.get(key)
                if not rec:
                    continue
                bo = rec.get("battingOrder")
                if bo is None or bo == "":
                    continue
                try:
                    bo_int = int(bo)
                except (TypeError, ValueError):
                    continue
                if bo_int not in STARTER_BAT_ORDERS:
                    continue
                person = rec.get("person") or {}
                full_name = person.get("fullName") or ""
                rows.append(
                    {
                        "game_id": gid_int,
                        "team_name": team_name,
                        "side": side,
                        "opponent_team": opponent_team,
                        "away_team": away_name,
                        "home_team": home_name,
                        "matchup": matchup_label,
                        "player_id": pid_int,
                        "full_name": full_name,
                        "batting_order": bo_int,
                    }
                )

    if not rows:
        empty = pd.DataFrame(
            columns=[
                "game_id",
                "team_name",
                "side",
                "opponent_team",
                "away_team",
                "home_team",
                "matchup",
                "player_id",
                "full_name",
                "batting_order",
            ]
        )
        return empty, num_scheduled

    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset=["game_id", "player_id"], keep="first"), num_scheduled
