"""Team-level match results from football-data.co.uk, mirrored on GitHub.

Two things this gives us that the FBref tables do not:

1. **The referee.** Referees vary a lot in how many fouls they call, and the name appears
   in no FBref table we load. It is one of the stronger available predictors for `Fls`
   and `CrdY`.
2. **Recency.** The FBref mirror stopped updating in September 2025; this one runs to the
   current season, so it can carry team-form features past that point.
"""

from __future__ import annotations

import logging

import pandas as pd

from footy.config import FOOTBALL_DATA_BASE, FOOTBALL_DATA_LEAGUE_DIRS
from footy.sources.base import CachedDownloader, SourceError

log = logging.getLogger(__name__)

#: football-data.co.uk abbreviates club names; FBref writes them out. Explicit rather
#: than fuzzy-matched, because a wrong join here silently attaches the wrong referee.
TEAM_NAME_TO_FBREF = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton & Hove Albion",
    "Burnley": "Burnley",
    "Cardiff": "Cardiff City",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Huddersfield": "Huddersfield Town",
    "Ipswich": "Ipswich Town",
    "Leeds": "Leeds United",
    "Leicester": "Leicester City",
    "Liverpool": "Liverpool",
    "Luton": "Luton Town",
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Newcastle": "Newcastle United",
    "Norwich": "Norwich City",
    "Nott'm Forest": "Nottingham Forest",
    "Sheffield United": "Sheffield United",
    "Southampton": "Southampton",
    "Stoke": "Stoke City",
    "Sunderland": "Sunderland",
    "Swansea": "Swansea City",
    "Tottenham": "Tottenham Hotspur",
    "Watford": "Watford",
    "West Brom": "West Bromwich Albion",
    "West Ham": "West Ham United",
    "Wolves": "Wolverhampton Wanderers",
}

COLUMNS = {
    "Date": "Match_Date",
    "HomeTeam": "Home_Team",
    "AwayTeam": "Away_Team",
    "Referee": "Referee",
    "HS": "Home_Sh", "AS": "Away_Sh",
    "HST": "Home_SoT", "AST": "Away_SoT",
    "HF": "Home_Fls", "AF": "Away_Fls",
    "HC": "Home_Corners", "AC": "Away_Corners",
    "HY": "Home_CrdY", "AY": "Away_CrdY",
    "HR": "Home_CrdR", "AR": "Away_CrdR",
}


def season_code(season_end_year: int) -> str:
    """2024 (i.e. 2023/24) -> "2324", matching the mirror's filenames."""
    return f"{(season_end_year - 1) % 100:02d}{season_end_year % 100:02d}"


def load_match_results(
    league: str = "ENG-PL",
    seasons: tuple[int, ...] = tuple(range(2018, 2027)),
    *,
    downloader: CachedDownloader | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Return one row per match, with team names normalised to FBref spelling."""
    if league not in FOOTBALL_DATA_LEAGUE_DIRS:
        raise KeyError(
            f"No football-data mapping for {league!r}. "
            f"Known: {sorted(FOOTBALL_DATA_LEAGUE_DIRS)}"
        )

    downloader = downloader or CachedDownloader()
    directory = FOOTBALL_DATA_LEAGUE_DIRS[league]
    frames: list[pd.DataFrame] = []

    for season in seasons:
        code = season_code(season)
        url = f"{FOOTBALL_DATA_BASE}/{directory}/season-{code}.csv"
        try:
            path = downloader.fetch(url, f"footballdata_{directory}_{code}.csv", force=force)
        except SourceError:
            # A season that has not started yet simply 404s. Skip it rather than fail.
            log.warning("no football-data file for %s season %s", league, season)
            continue

        frame = pd.read_csv(path)
        available = {src: dst for src, dst in COLUMNS.items() if src in frame.columns}
        frame = frame[list(available)].rename(columns=available)
        frame["Season_End_Year"] = season
        frames.append(frame)

    if not frames:
        raise SourceError(f"No football-data seasons loaded for {league}")

    out = pd.concat(frames, ignore_index=True)
    out["Match_Date"] = pd.to_datetime(out["Match_Date"], errors="coerce", format="mixed")

    unmapped = (
        set(out["Home_Team"].dropna()) | set(out["Away_Team"].dropna())
    ) - set(TEAM_NAME_TO_FBREF)
    if unmapped:
        log.warning(
            "unmapped team names dropped from the referee join: %s", sorted(unmapped)
        )

    for column in ("Home_Team", "Away_Team"):
        out[column] = out[column].map(TEAM_NAME_TO_FBREF)
    out = out.dropna(subset=["Home_Team", "Away_Team", "Match_Date"])

    log.info("football-data: %d matches across %d seasons", len(out), len(frames))
    return out


def attach_referee(player_matches: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Left-join the referee onto a player-match frame.

    Matched on date plus both team names. Kick-off dates occasionally differ by a day
    between sources (late reschedules, timezone rounding), so we retry unmatched rows
    against a +/-1 day window before giving up.
    """
    referees = results[["Match_Date", "Home_Team", "Away_Team", "Referee"]].copy()
    key = ["Match_Date", "Home_Team", "Away_Team"]

    out = player_matches.copy()
    out["Match_Date"] = pd.to_datetime(out["Match_Date"])
    out = out.merge(referees, on=key, how="left")

    unmatched = out["Referee"].isna()
    if unmatched.any():
        # Second pass on team pair alone, restricted to fixtures within one day.
        pairs = referees.rename(columns={"Match_Date": "ref_date"})
        retry = (
            out.loc[unmatched, ["Home_Team", "Away_Team", "Match_Date"]]
            .reset_index()
            .merge(pairs, on=["Home_Team", "Away_Team"], how="left")
        )
        retry = retry[
            (retry["ref_date"] - retry["Match_Date"]).abs() <= pd.Timedelta(days=1)
        ]
        resolved = retry.dropna(subset=["Referee"]).drop_duplicates("index")
        out.loc[resolved["index"], "Referee"] = resolved["Referee"].to_numpy()

    coverage = out["Referee"].notna().mean()
    log.info("referee attached to %.1f%% of player-match rows", 100 * coverage)
    return out
