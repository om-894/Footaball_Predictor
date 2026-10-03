"""Raw source CSVs -> one tidy, validated Parquet table of player-matches.

Everything downstream reads the output of :func:`build_player_matches`, so this is where
type coercion, the awkward FBref string formats, and the schema contract all live.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from footy.config import PLAYER_MATCHES_PATH, TARGETS, ensure_dirs
from footy.sources import footballdata, worldfootballr
from footy.sources.base import CachedDownloader

log = logging.getLogger(__name__)

#: Columns that identify rather than measure. Never treated as numeric features.
ID_COLUMNS = [
    "MatchURL", "Match_Date", "Matchweek", "Season_End_Year", "Competition_Name",
    "Team", "Opponent", "Home_Away", "Player", "Player_Href", "Nation", "Pos",
    "Home_Team", "Away_Team", "Referee",
]

_MATCHWEEK_RE = re.compile(r"Matchweek\s+(\d+)")
_AGE_RE = re.compile(r"^(\d+)-(\d+)$")
_MATCH_ID_RE = re.compile(r"/matches/([0-9a-f]{8})/")


def parse_matchweek(value: object) -> float:
    """'Premier League (Matchweek 12)' -> 12.0."""
    if not isinstance(value, str):
        return np.nan
    match = _MATCHWEEK_RE.search(value)
    return float(match.group(1)) if match else np.nan


def parse_age(value: object) -> float:
    """FBref writes age as 'years-days'. '26-075' -> 26.205."""
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


def split_positions(value: object) -> list[str]:
    """'FW,LW,LM' -> ['FW', 'LW', 'LM'].

    FBref lists every position a player occupied in the match, so this is genuinely
    multi-label and must not be squeezed into one categorical.
    """
    if not isinstance(value, str) or not value.strip():
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def build_player_matches(
    league: str = "ENG-PL",
    *,
    stat_types: tuple[str, ...] = worldfootballr.DEFAULT_STAT_TYPES,
    with_referee: bool = True,
    downloader: CachedDownloader | None = None,
    force: bool = False,
    write: bool = True,
) -> pd.DataFrame:
    """Download, clean, validate and (optionally) persist the player-match table."""
    ensure_dirs()
    downloader = downloader or CachedDownloader()

    frame = worldfootballr.load_player_match_stats(
        league, stat_types, downloader=downloader, force=force
    )

    # Sh / SoT / xG come from the shot-level file rather than an aggregate table, which
    # is both wider in coverage and richer (distance, body part).
    shooting = worldfootballr.load_match_shooting(
        league, downloader=downloader, force=force
    )
    frame = worldfootballr.attach_shooting(frame, shooting)

    # -- types -------------------------------------------------------------------
    frame["Match_Date"] = pd.to_datetime(frame["Match_Date"], errors="coerce")
    frame["Matchweek"] = frame["Matchweek"].map(parse_matchweek)
    frame["Age"] = frame["Age"].map(parse_age)
    frame["Season_End_Year"] = pd.to_numeric(frame["Season_End_Year"], errors="coerce")

    # Every stat column is numeric; FBref occasionally emits blanks or stray strings.
    stat_columns = [c for c in frame.columns if c not in ID_COLUMNS and c != "Age"]
    for column in stat_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # -- derived identity --------------------------------------------------------
    frame["Opponent"] = np.where(
        frame["Home_Away"].str.lower().eq("home"), frame["Away_Team"], frame["Home_Team"]
    )
    frame["is_home"] = frame["Home_Away"].str.lower().eq("home").astype(int)
    frame["positions"] = frame["Pos"].map(split_positions)
    frame["is_gk"] = frame["positions"].map(lambda p: int("GK" in p))

    # -- rows we cannot use ------------------------------------------------------
    before = len(frame)
    frame = frame.dropna(subset=["Match_Date", "Min", "Player", "Team"])
    frame = frame[frame["Min"] > 0]
    if len(frame) != before:
        log.info("dropped %d rows with no date, player or minutes", before - len(frame))

    # A left-joined table contributes NaN only where a player has no entry in it. For
    # count stats that genuinely means zero, but we log the rate so a badly broken join
    # cannot hide behind a wall of zeros.
    count_columns = [c for c in stat_columns if not c.endswith("_pct")]
    na_rate = frame[count_columns].isna().mean()
    noisy = na_rate[na_rate > 0.05]
    if not noisy.empty:
        log.warning(
            "columns >5%% missing before zero-fill:\n%s",
            noisy.sort_values(ascending=False).head(20).to_string(),
        )
    frame[count_columns] = frame[count_columns].fillna(0.0)

    # -- referee -----------------------------------------------------------------
    if with_referee:
        seasons = tuple(sorted(frame["Season_End_Year"].dropna().astype(int).unique()))
        try:
            results = footballdata.load_match_results(
                league, seasons, downloader=downloader, force=force
            )
            frame = footballdata.attach_referee(frame, results)
        except Exception as exc:  # noqa: BLE001 - referee is a nice-to-have, not required
            log.warning("referee join skipped: %s", exc)
            frame["Referee"] = pd.NA
    else:
        frame["Referee"] = pd.NA

    frame = frame.sort_values(["Match_Date", "MatchURL", "Team", "Player"])
    frame = frame.reset_index(drop=True)

    validate_player_matches(frame)

    if write:
        # `positions` is a list column; Parquet handles it natively.
        frame.to_parquet(PLAYER_MATCHES_PATH, index=False)
        log.info("wrote %s (%d rows x %d cols)", PLAYER_MATCHES_PATH, *frame.shape)

    return frame


def validate_player_matches(
    frame: pd.DataFrame, targets: tuple[str, ...] = TARGETS
) -> None:
    """Fail loudly on the invariants the rest of the package relies on.

    ``targets`` is a parameter because the Championship supports a narrower set -- FBref
    publishes no total-tackles column for the second tier.
    """
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
    """Read the cached player-match table, with a useful error if it is absent."""
    if not PLAYER_MATCHES_PATH.exists():
        raise FileNotFoundError(
            f"{PLAYER_MATCHES_PATH} not found. Run `footy fetch` first."
        )
    return pd.read_parquet(PLAYER_MATCHES_PATH)
