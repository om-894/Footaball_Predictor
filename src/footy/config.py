"""Paths, source URLs and the small set of constants the rest of the package agrees on."""

from __future__ import annotations

import os
from pathlib import Path

# Repo layout. Everything derived lives under data/ and is gitignored; only the
# legacy Sunderland scrape is committed.
PROJECT_ROOT = Path(os.environ.get("FOOTY_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # downloaded CSVs, exactly as published
INTERIM_DIR = DATA_DIR / "interim"  # tidied parquet, one row per player-match
FEATURES_DIR = DATA_DIR / "features"
REPORTS_DIR = PROJECT_ROOT / "reports"

# --------------------------------------------------------------------------------------
# Source 1: FBref per-match player tables, republished as plain CSVs by worldfootballR.
#
# FBref itself now sits behind a Cloudflare bot gate, so the plain-HTTP scraping the v1
# scripts did no longer works at all. These release assets carry the identical schema and
# need no scraping. Live scraping is done by scripts/scrape_fbref.py.
# --------------------------------------------------------------------------------------
WFR_RELEASE_BASE = (
    "https://github.com/JaseZiv/worldfootballR_data/releases/download"
    "/fb_advanced_match_stats"
)

# worldfootballR's country/gender/tier naming. ENG_M_1st is the Premier League.
LEAGUES = {
    "ENG-PL": "ENG_M_1st",
    "ESP-LaLiga": "ESP_M_1st",
    "FRA-Ligue1": "FRA_M_1st",
    "GER-Bundesliga": "GER_M_1st",
    "ITA-SerieA": "ITA_M_1st",
    "USA-MLS": "USA_M_1st",
}
DEFAULT_LEAGUE = "ENG-PL"

# --------------------------------------------------------------------------------------
# Source 2: football-data.co.uk, mirrored on GitHub. Team-level only, but it runs to the
# current season and -- uniquely -- names the referee, who matters a great deal for fouls
# and cards and appears nowhere in the FBref tables.
# --------------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------------
# Modelling constants
# --------------------------------------------------------------------------------------

#: Counts we forecast. All are non-negative integers and all scale with minutes played.
TARGETS = ("Sh", "SoT", "Fls", "Fld", "CrdY", "Tkl")

#: Half-lives (in appearances) for the exponentially weighted form features.
EWMA_HALFLIVES = (3, 6, 12)

#: Per-90 rate columns that feed the form features. These are the stats that plausibly
#: drive the targets, and every one is available before kickoff via a player's history.
FORM_STATS = (
    "Sh", "SoT", "Fls", "Fld", "CrdY", "CrdR", "Tkl", "Int", "Blocks",
    "Touches", "Touches_AttThird", "Touches_AttPen", "Touches_DefThird",
    "Carries", "PrgC", "Carries_PenArea", "Miscontrols", "Dispossessed",
    "TakeOns_Att", "TakeOns_Succ", "TakeOns_Tkld",
    "Passes_Cmp", "Passes_Att", "PrgP", "KeyPasses", "Passes_PenArea",
    "xG", "xAG", "Recov", "AerialsWon", "AerialsLost",
    "TklW", "Clearances", "Challenges_Att", "Challenges_Lost", "Off", "Crs",
)

#: Ratio columns must never be rescaled by minutes -- doing so is what turned a 66.7%
#: pass completion into 84.6% in the v1 pipeline. features.to_per90 refuses to touch these.
RATIO_COLUMNS = frozenset({
    "Passes_Cmp_pct", "TakeOns_Succ_pct", "Aerials_Won_pct", "Save_pct",
    "Tkl_pct", "Launch_pct",
})

#: A full match. Used for the per-90 conversion and as the minutes cap.
FULL_MATCH_MINUTES = 90.0


def ensure_dirs() -> None:
    """Create the data/model/report tree. Safe to call repeatedly."""
    for path in (RAW_DIR, INTERIM_DIR, FEATURES_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
