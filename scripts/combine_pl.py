"""
Combine the mirrored Premier League history with live scrapes into one table.

The mirror runs from 2017/18 to September 2025 and the live scrape covers recent matches,
so neither alone can forecast a match this week. Only the columns both sources carry are
kept, which drops xG, touches and passes. Extra scraped leagues can be added too, e.g. the
Eredivisie before a European tie, so a foreign side's players have some history.

INPUTS        data/interim/player_matches.parquet (from `footy fetch`)
              data/interim/ENG-Premier-League_player_matches.parquet plus any extra slugs
              (from scripts/build_scraped.py)
OUTPUTS       data/interim/pl_combined_player_matches.parquet
REQUIREMENTS  pip install -e .

Example:  python scripts/combine_pl.py NED-Eredivisie
"""

import argparse
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from footy.config import COMBINED_PATH, COMBINED_TARGETS, PLAYER_MATCHES_PATH, scraped_path
from footy.ingest import drop_unusable_rows, fbref_match_id


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #

# columns both sources carry
KEEP = [
    "MatchURL", "Match_Date", "Matchweek", "Season_End_Year", "Competition_Name",
    "Team", "Opponent", "Home_Away", "is_home", "Player", "Nation", "Pos", "Age",
    "Min", "Referee", "positions", "is_gk", "CrdR", "Int", "Gls", "Ast",
    *COMBINED_TARGETS,
]


# --------------------------------------------------------------------------- #
# COMBINING
# --------------------------------------------------------------------------- #

def read_table(path, how: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"{path} not found, run {how} first")
    return pd.read_parquet(path)


def align(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Keep the shared columns, adding any a source lacks, and tag each row with its source."""
    out = frame.copy()
    for column in KEEP:
        if column not in out.columns:
            out[column] = pd.NA
    out = out[KEEP]
    out["source"] = source
    out["Match_Date"] = pd.to_datetime(out["Match_Date"], errors="coerce")
    for column in (*COMBINED_TARGETS, "CrdR", "Int", "Gls", "Ast", "Min"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine the mirrored PL history with live scrapes.")
    parser.add_argument("extra", nargs="*", help="extra scraped slugs to add, e.g. NED-Eredivisie")
    args = parser.parse_args()

    mirror = align(read_table(PLAYER_MATCHES_PATH, "`footy fetch`"), "mirror")
    print(f"mirror: {len(mirror):,} rows to {mirror['Match_Date'].max().date()}")
    scraped = []
    for slug in ["ENG-Premier-League", *args.extra]:
        live = align(read_table(scraped_path(slug), f"scripts/build_scraped.py {slug}"), slug)
        print(f"{slug}: {len(live):,} rows to {live['Match_Date'].max().date()}")
        scraped.append(live)

    # the mirror stores full match URLs and the scrape stores bare ids, so compare on the id
    scraped_ids = set(pd.concat(scraped)["MatchURL"].map(fbref_match_id))
    overlap = mirror["MatchURL"].map(fbref_match_id).isin(scraped_ids)
    if overlap.any():
        matches = mirror.loc[overlap, "MatchURL"].nunique()
        print(f"dropped {overlap.sum():,} mirror rows from {matches} matches the scrape also has")

    combined = pd.concat([mirror[~overlap], *scraped], ignore_index=True)
    # a match that appears in two scraped tables keeps its last copy
    before = len(combined)
    combined = combined.drop_duplicates(subset=["MatchURL", "Team", "Player"], keep="last")
    if len(combined) < before:
        print(f"dropped {before - len(combined):,} rows saved twice")
    combined = drop_unusable_rows(combined)
    combined = combined.sort_values(["Match_Date", "MatchURL", "Team", "Player"]).reset_index(drop=True)
    combined.to_parquet(COMBINED_PATH, index=False)

    seasons = combined.groupby("Season_End_Year").agg(rows=("Player", "size"), matches=("MatchURL", "nunique"))
    print(f"\ncombined: {len(combined):,} rows, {combined['MatchURL'].nunique():,} matches")
    print(seasons.tail(6).to_string())
    print(f"saved to {COMBINED_PATH}")


if __name__ == "__main__":
    main()
