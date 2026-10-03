"""Paths, data sources and the constants the rest of the package shares."""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# PATHS
# --------------------------------------------------------------------------- #

# everything under data/ is gitignored and can be rebuilt, apart from data/legacy.
# setting FOOTY_ROOT moves the data and reports folders somewhere else
PROJECT_ROOT = Path(os.environ.get("FOOTY_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" # downloaded CSVs, exactly as published
INTERIM_DIR = DATA_DIR / "interim" # tidied parquet, one row per player-match
FEATURES_DIR = DATA_DIR / "features"
REPORTS_DIR = PROJECT_ROOT / "reports"

# tables shared between the cli and the scripts
PLAYER_MATCHES_PATH = INTERIM_DIR / "player_matches.parquet" # mirror history, made by `footy fetch`
COMBINED_PATH = INTERIM_DIR / "pl_combined_player_matches.parquet" # mirror plus live scrapes
FEATURES_PATH = FEATURES_DIR / "player_features.parquet" # made by `footy build`


def scraped_path(slug: str) -> Path:
    """Player-match table built from a live scrape, e.g. scraped_path("ENG-Championship")."""
    return INTERIM_DIR / f"{slug}_player_matches.parquet"


# leagues scraped live, keyed by their folder under data/raw, with the label used in the tables
SCRAPED_LEAGUES = {
    "ENG-Premier-League": "Premier League",
    "ENG-Championship": "Championship",
    "NED-Eredivisie": "Eredivisie",
}

# --------------------------------------------------------------------------- #
# DATA SOURCES
# --------------------------------------------------------------------------- #

# FBref's per-match player tables, republished as CSVs by worldfootballR. FBref itself
# blocks plain HTTP requests, so live scraping goes through scripts/scrape_fbref.py instead
WFR_RELEASE_BASE = (
    "https://github.com/JaseZiv/worldfootballR_data/releases/download"
    "/fb_advanced_match_stats"
)

# worldfootballR's league codes, e.g. ENG_M_1st is the Premier League
LEAGUES = {
    "ENG-PL": "ENG_M_1st",
    "ESP-LaLiga": "ESP_M_1st",
    "FRA-Ligue1": "FRA_M_1st",
    "GER-Bundesliga": "GER_M_1st",
    "ITA-SerieA": "ITA_M_1st",
    "USA-MLS": "USA_M_1st",
}
DEFAULT_LEAGUE = "ENG-PL"

# football-data.co.uk results, mirrored on GitHub. team level only, but they name the
# referee, which no FBref table we load does
FOOTBALL_DATA_BASE = (
    "https://raw.githubusercontent.com/datasets/football-datasets/main/datasets"
)
FOOTBALL_DATA_LEAGUE_DIRS = {
    "ENG-PL": "premier-league",
    "ESP-LaLiga": "la-liga",
    "FRA-Ligue1": "ligue-1",
    "GER-Bundesliga": "bundesliga",
    "ITA-SerieA": "serie-a",
}

# --------------------------------------------------------------------------- #
# MODELLING
# --------------------------------------------------------------------------- #

# per-match counts the models forecast
TARGETS = ("Sh", "SoT", "Fls", "Fld", "CrdY", "Tkl")

# a live scrape only gets the summary table, which has tackles won (TklW) but not total tackles
SCRAPED_TARGETS = ("Sh", "SoT", "Fls", "Fld", "CrdY", "TklW")

# the combined table keeps the targets both sources have
COMBINED_TARGETS = tuple(t for t in TARGETS if t in SCRAPED_TARGETS)

EWMA_HALFLIVES = (3, 6, 12) # half-lives of the recent form features, in appearances

# stats whose history feeds the form features
FORM_STATS = (
    "Sh", "SoT", "Fls", "Fld", "CrdY", "CrdR", "Tkl", "Int", "Blocks",
    "Touches", "Touches_AttThird", "Touches_AttPen", "Touches_DefThird",
    "Carries", "PrgC", "Carries_PenArea", "Miscontrols", "Dispossessed",
    "TakeOns_Att", "TakeOns_Succ", "TakeOns_Tkld",
    "Passes_Cmp", "Passes_Att", "PrgP", "KeyPasses", "Passes_PenArea",
    "xG", "xAG", "Recov", "AerialsWon", "AerialsLost",
    "TklW", "Clearances", "Challenges_Att", "Challenges_Lost", "Off", "Crs",
)

# percentages, which features.to_per90 never scales by minutes
RATIO_COLUMNS = frozenset({
    "Passes_Cmp_pct", "TakeOns_Succ_pct", "Aerials_Won_pct", "Save_pct",
    "Tkl_pct", "Launch_pct",
})

FULL_MATCH_MINUTES = 90.0 # used for per-90 rates and as the cap on predicted minutes


def ensure_dirs() -> None:
    """Create the data and reports folders if they are missing."""
    for path in (RAW_DIR, INTERIM_DIR, FEATURES_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
