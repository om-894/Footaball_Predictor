"""
Match results from football-data.co.uk, mirrored on GitHub.

Used for the referee, which none of the FBref tables we load include.
"""

from __future__ import annotations

import logging

import pandas as pd

from footy.config import FOOTBALL_DATA_BASE, FOOTBALL_DATA_LEAGUE_DIRS
from footy.sources.base import CachedDownloader, SourceError

log = logging.getLogger(__name__)

# football-data shortens club names where FBref writes them out. the list is explicit,
# since a wrong match would attach the wrong referee
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

# football-data column names and what they are called here
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
    """The mirror's season code, e.g. 2024 (meaning 2023/24) gives "2324"."""
    return f"{(season_end_year - 1) % 100:02d}{season_end_year % 100:02d}"


def load_match_results(
    league: str = "ENG-PL",
    seasons: tuple[int, ...] = tuple(range(2018, 2027)),
    *,
    downloader: CachedDownloader | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """One row per match, with team names changed to FBref's spelling."""
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
            # a season that hasn't started yet has no file, so it is skipped
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
    """Join the referee onto each player-match by date and both team names.

    The two sources sometimes disagree on the date by a day, so unmatched rows are tried
    again within a day either side.
    """
    referees = results[["Match_Date", "Home_Team", "Away_Team", "Referee"]].copy()
    key = ["Match_Date", "Home_Team", "Away_Team"]

    out = player_matches.copy()
    out["Match_Date"] = pd.to_datetime(out["Match_Date"])
    out = out.merge(referees, on=key, how="left")

    unmatched = out["Referee"].isna()
    if unmatched.any():
        # second pass on the two teams only, within a day of the date
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
