"""Builds the standard player-match table from a live FBref scrape.

A live scrape only gets FBref's match summary table, so the result is narrower than the
mirror: no xG, touches or passes, and tackles won instead of total tackles.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from footy.config import SCRAPED_TARGETS
from footy.ingest import parse_age, split_positions, validate_player_matches

log = logging.getLogger(__name__)

#: soccerdata flattens FBref's two-row header to `Group_Stat`.
COLUMN_MAP = {
    "Performance_Gls": "Gls",
    "Performance_Ast": "Ast",
    "Performance_PK": "PK",
    "Performance_PKatt": "PKatt",
    "Performance_Sh": "Sh",
    "Performance_SoT": "SoT",
    "Performance_CrdY": "CrdY",
    "Performance_CrdR": "CrdR",
    "Performance_Fls": "Fls",
    "Performance_Fld": "Fld",
    "Performance_Off": "Off",
    "Performance_Crs": "Crs",
    "Performance_TklW": "TklW",
    "Performance_Int": "Int",
    "Performance_OG": "OG",
    "Performance_PKwon": "PKwon",
    "Performance_PKcon": "PKcon",
}

IDENTITY_MAP = {
    "team": "Team",
    "player": "Player",
    "nation": "Nation",
    "pos": "Pos",
    "age": "Age",
    "min": "Min",
    "game_id": "MatchURL",
    "season": "Season",
}


def resolve_home_away(frame: pd.DataFrame) -> pd.DataFrame:
    """Work out which side each player was on, tolerating inconsistent club names.

    soccerdata spells clubs differently in its two tables -- the schedule says "QPR" and
    "Blackburn" where the player stats say "Queens Park Rangers" and "Blackburn Rovers".
    Comparing them directly marked 7 of 24 clubs as permanently away: 869 away rows to
    507 home, with 15 matches having no home side at all. Every home/away and opponent
    feature downstream would have been quietly wrong.

    Rather than hard-code aliases that rot, each match is resolved on its own: it has
    exactly two teams in the player table, and whichever is the closer string match to the
    scheduled home team is the home side.
    """
    from difflib import SequenceMatcher

    def similarity(a: str, b: str) -> float:
        a, b = str(a).casefold(), str(b).casefold()
        if a in b or b in a:
            return 1.0
        return SequenceMatcher(None, a, b).ratio()

    home_flags = pd.Series(0, index=frame.index, dtype=int)
    unresolved = []

    for match_id, group in frame.groupby("MatchURL", sort=False):
        teams = group["Team"].dropna().unique()
        scheduled_home = group["Home_Team"].dropna()
        if len(teams) != 2 or scheduled_home.empty:
            unresolved.append(match_id)
            continue

        target = scheduled_home.iloc[0]
        home_team = max(teams, key=lambda t: similarity(t, target))
        home_flags.loc[group.index[group["Team"] == home_team]] = 1

    if unresolved:
        log.warning(
            "could not resolve home/away for %d matches (e.g. %s)",
            len(unresolved), unresolved[:3],
        )

    frame = frame.copy()
    frame["is_home"] = home_flags
    frame["Home_Away"] = np.where(frame["is_home"] == 1, "Home", "Away")

    # Opponent is the other team in the same match -- taken from the player table so the
    # spelling matches `Team`, which is what the feature joins key on.
    opponents = {}
    for match_id, group in frame.groupby("MatchURL", sort=False):
        teams = list(group["Team"].dropna().unique())
        if len(teams) == 2:
            opponents[(match_id, teams[0])] = teams[1]
            opponents[(match_id, teams[1])] = teams[0]
    frame["Opponent"] = [
        opponents.get((m, t)) for m, t in zip(frame["MatchURL"], frame["Team"])
    ]

    balance = frame.groupby("MatchURL")["is_home"].mean()
    lopsided = balance[(balance < 0.2) | (balance > 0.8)]
    if len(lopsided):
        log.warning(
            "%d matches have a lopsided home/away split; name matching may have failed",
            len(lopsided),
        )

    return frame


def build_scraped_matches(
    player_stats: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    competition: str = "Championship",
) -> pd.DataFrame:
    """Turn a scraped summary table and its schedule into the standard player-match table."""
    frame = player_stats.rename(columns={**IDENTITY_MAP, **COLUMN_MAP}).copy()

    missing = sorted(set(COLUMN_MAP.values()) - set(frame.columns))
    if missing:
        log.warning("scrape is missing columns %s; filling with 0", missing)
        for column in missing:
            frame[column] = 0.0

    # -- match context from the schedule -----------------------------------------
    fixtures = schedule.rename(
        columns={
            "game_id": "MatchURL",
            "home_team": "Home_Team",
            "away_team": "Away_Team",
            "date": "Match_Date",
            "week": "Matchweek",
            "referee": "Referee",
        }
    )
    keep = ["MatchURL", "Match_Date", "Matchweek", "Home_Team", "Away_Team", "Referee"]
    fixtures = fixtures[[c for c in keep if c in fixtures.columns]].drop_duplicates("MatchURL")

    frame = frame.merge(fixtures, on="MatchURL", how="left")

    # -- types --------------------------------------------------------------------
    frame["Match_Date"] = pd.to_datetime(frame["Match_Date"], errors="coerce")
    frame["Matchweek"] = pd.to_numeric(frame.get("Matchweek"), errors="coerce")
    frame["Age"] = frame["Age"].map(parse_age)
    for column in COLUMN_MAP.values():
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["Min"] = pd.to_numeric(frame["Min"], errors="coerce")

    # FBref's season label is the starting year; the rest of the package keys on the
    # season's *end* year, so 2026/27 is 2027.
    frame["Season_End_Year"] = (
        pd.to_numeric(frame["Season"].astype(str).str[:2], errors="coerce") + 2001
    )
    frame["Competition_Name"] = competition

    # -- derived identity ---------------------------------------------------------
    frame = resolve_home_away(frame)
    frame["positions"] = frame["Pos"].map(split_positions)
    frame["is_gk"] = frame["positions"].map(lambda p: int("GK" in p))

    before = len(frame)
    frame = frame.dropna(subset=["Match_Date", "Min", "Player", "Team"])
    frame = frame[frame["Min"] > 0]
    if len(frame) != before:
        log.info("dropped %d unusable rows", before - len(frame))

    frame = frame.sort_values(["Match_Date", "MatchURL", "Team", "Player"])
    frame = frame.reset_index(drop=True)

    validate_player_matches(frame, targets=SCRAPED_TARGETS)
    return frame
