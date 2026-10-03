"""
Builds the player-match table from the downloaded FBref and football-data files.

One row per player per match. Everything after this reads that table, so the type fixes,
FBref's string formats and the checks on the table all live here.
"""

from __future__ import annotations

import logging
import re
from unicodedata import normalize

import numpy as np
import pandas as pd

from footy.config import PLAYER_MATCHES_PATH, TARGETS, ensure_dirs
from footy.sources import footballdata, worldfootballr
from footy.sources.base import CachedDownloader

log = logging.getLogger(__name__)

# columns that identify a row rather than measure anything, so they are never made numeric
ID_COLUMNS = [
    "MatchURL", "Match_Date", "Matchweek", "Season_End_Year", "Competition_Name",
    "Team", "Opponent", "Home_Away", "Player", "Player_Href", "Nation", "Pos",
    "Home_Team", "Away_Team", "Referee",
]

_MATCHWEEK_RE = re.compile(r"Matchweek\s+(\d+)")
_AGE_RE = re.compile(r"^(\d+)-(\d+)$")
_MATCH_ID_RE = re.compile(r"/matches/([0-9a-f]{8})/")

# letters that unicode normalisation cannot split into a plain letter plus an accent
_LETTER_SWAPS = str.maketrans({
    "ø": "o", "Ø": "O", "đ": "d", "Đ": "D", "ł": "l", "Ł": "L",
    "æ": "ae", "Æ": "AE", "ß": "ss", "ı": "i", "’": "'",
})


# --------------------------------------------------------------------------- #
# FBREF STRING FORMATS
# --------------------------------------------------------------------------- #

def parse_matchweek(value: object) -> float:
    """Matchweek number from text like 'Premier League (Matchweek 12)', which gives 12.0."""
    if not isinstance(value, str):
        return np.nan
    match = _MATCHWEEK_RE.search(value)
    return float(match.group(1)) if match else np.nan


def parse_age(value: object) -> float:
    """Age in years from FBref's 'years-days' text, e.g. '26-075' gives 26.205."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if pd.notna(value) else np.nan
    if not isinstance(value, str):
        return np.nan
    match = _AGE_RE.match(value.strip())
    if not match:
        return np.nan
    years, days = int(match.group(1)), int(match.group(2))
    return years + days / 365.25


def fbref_match_id(value: object) -> str:
    """FBref's 8 character match id, taken from a match URL or passed through if already an id."""
    text = str(value)
    match = _MATCH_ID_RE.search(text)
    return match.group(1) if match else text


def name_key(name: object) -> str:
    """Lowercase plain-ASCII form of a name for loose matching, e.g. 'Martin Ødegaard' gives 'martin odegaard'."""
    swapped = str(name).translate(_LETTER_SWAPS)
    stripped = normalize("NFKD", swapped).encode("ascii", "ignore").decode()
    return stripped.casefold().strip()


def split_positions(value: object) -> list[str]:
    """Every position a player played in a match, e.g. 'FW,LW,LM' gives ['FW', 'LW', 'LM']."""
    if not isinstance(value, str) or not value.strip():
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


# --------------------------------------------------------------------------- #
# BUILDING THE TABLE
# --------------------------------------------------------------------------- #

def add_position_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the `positions` list and the `is_gk` flag from FBref's `Pos` string."""
    frame["positions"] = frame["Pos"].map(split_positions)
    frame["is_gk"] = frame["positions"].map(lambda p: int("GK" in p))
    return frame


def drop_unusable_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with no date, player, team or minutes."""
    before = len(frame)
    frame = frame.dropna(subset=["Match_Date", "Min", "Player", "Team"])
    frame = frame[frame["Min"] > 0]
    if len(frame) != before:
        log.info("dropped %d rows with no date, player, team or minutes", before - len(frame))
    return frame


def build_player_matches(
    league: str = "ENG-PL",
    *,
    stat_types: tuple[str, ...] = worldfootballr.DEFAULT_STAT_TYPES,
    with_referee: bool = True,
    downloader: CachedDownloader | None = None,
    force: bool = False,
    write: bool = True,
) -> pd.DataFrame:
    """Download, clean and check the player-match table, saving it unless `write` is False."""
    ensure_dirs()
    downloader = downloader or CachedDownloader()

    frame = worldfootballr.load_player_match_stats(
        league, stat_types, downloader=downloader, force=force
    )

    # shots come from the shot-level file, which covers every season (see worldfootballr.py)
    shooting = worldfootballr.load_match_shooting(
        league, downloader=downloader, force=force
    )
    frame = worldfootballr.attach_shooting(frame, shooting)

    # types
    frame["Match_Date"] = pd.to_datetime(frame["Match_Date"], errors="coerce")
    frame["Matchweek"] = frame["Matchweek"].map(parse_matchweek)
    frame["Age"] = frame["Age"].map(parse_age)
    frame["Season_End_Year"] = pd.to_numeric(frame["Season_End_Year"], errors="coerce")

    # FBref sometimes leaves blanks or stray text in stat columns
    stat_columns = [c for c in frame.columns if c not in ID_COLUMNS and c != "Age"]
    for column in stat_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # home or away, opponent and positions
    frame["Opponent"] = np.where(
        frame["Home_Away"].str.lower().eq("home"), frame["Away_Team"], frame["Home_Team"]
    )
    frame["is_home"] = frame["Home_Away"].str.lower().eq("home").astype(int)
    frame = add_position_columns(frame)
    frame = drop_unusable_rows(frame)

    # a player missing from a joined table gets NaN, which means zero for a count. the
    # missing rate is logged first so a broken join can't hide behind the zeros
    count_columns = [c for c in stat_columns if not c.endswith("_pct")]
    na_rate = frame[count_columns].isna().mean()
    noisy = na_rate[na_rate > 0.05]
    if not noisy.empty:
        log.warning(
            "columns >5%% missing before zero-fill:\n%s",
            noisy.sort_values(ascending=False).head(20).to_string(),
        )
    frame[count_columns] = frame[count_columns].fillna(0.0)

    # referee from football-data, which is optional so a failure only logs a warning
    if with_referee:
        seasons = tuple(sorted(frame["Season_End_Year"].dropna().astype(int).unique()))
        try:
            results = footballdata.load_match_results(
                league, seasons, downloader=downloader, force=force
            )
            frame = footballdata.attach_referee(frame, results)
        except Exception as exc:  # noqa: BLE001 - the referee is optional
            log.warning("referee join skipped: %s", exc)
            frame["Referee"] = pd.NA
    else:
        frame["Referee"] = pd.NA

    frame = frame.sort_values(["Match_Date", "MatchURL", "Team", "Player"])
    frame = frame.reset_index(drop=True)

    validate_player_matches(frame)

    if write:
        frame.to_parquet(PLAYER_MATCHES_PATH, index=False)
        log.info("wrote %s (%d rows x %d cols)", PLAYER_MATCHES_PATH, *frame.shape)

    return frame


def validate_player_matches(
    frame: pd.DataFrame, targets: tuple[str, ...] = TARGETS
) -> None:
    """Raise if the table breaks anything later steps rely on, e.g. duplicate rows or negative counts."""
    required = {"MatchURL", "Match_Date", "Team", "Player", "Min", "Season_End_Year"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"player-match table is missing required columns: {sorted(missing)}")

    missing_targets = set(targets) - set(frame.columns)
    if missing_targets:
        raise ValueError(f"player-match table is missing targets: {sorted(missing_targets)}")

    duplicates = frame.duplicated(["MatchURL", "Team", "Player"]).sum()
    if duplicates:
        raise ValueError(f"{duplicates} duplicate player-match rows")

    if (frame["Min"] <= 0).any():
        raise ValueError("found rows with non-positive minutes")

    negative = [t for t in targets if (frame[t] < 0).any()]
    if negative:
        raise ValueError(f"negative counts in targets: {negative}")


def load_player_matches() -> pd.DataFrame:
    """Read the saved player-match table."""
    if not PLAYER_MATCHES_PATH.exists():
        raise FileNotFoundError(
            f"{PLAYER_MATCHES_PATH} not found. Run `footy fetch` first."
        )
    return pd.read_parquet(PLAYER_MATCHES_PATH)
