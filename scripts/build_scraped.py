"""
Build a player-match table from a live FBref scrape.

Reads the CSVs that scrape_fbref.py saved for one league, joins each player row to its
fixture and writes the same table shape that `footy fetch` makes, so the features and
models run on it unchanged.

INPUTS        data/raw/<slug>/player_summary_*.csv and schedule_*.csv
OUTPUTS       data/interim/<slug>_player_matches.parquet
REQUIREMENTS  pip install -e .

Example:  python scripts/build_scraped.py ENG-Championship
"""

import argparse
import logging
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from footy.config import RAW_DIR, SCRAPED_LEAGUES, ensure_dirs, scraped_path
from footy.scraped import build_scraped_matches


def read_csvs(folder, pattern: str) -> pd.DataFrame:
    """Every CSV in a folder that matches a pattern, stacked into one frame."""
    paths = sorted(folder.glob(pattern))
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True) if paths else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a player-match table from a live FBref scrape.")
    parser.add_argument("slug", help=f"folder under data/raw, e.g. {', '.join(SCRAPED_LEAGUES)}")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    raw = RAW_DIR / args.slug
    players = read_csvs(raw, "player_summary_*.csv")
    schedule = read_csvs(raw, "schedule_*.csv")
    if players.empty or schedule.empty:
        raise SystemExit(f"no scraped CSVs in {raw} yet, run scripts/scrape_fbref.py first")

    # appended chunks can repeat the header row. a match can also be saved twice
    players = players[players["player"] != "player"]
    players = players.drop_duplicates(subset=["game_id", "team", "player"])
    print(f"loaded {len(players):,} player rows from {raw}")

    label = SCRAPED_LEAGUES.get(args.slug, args.slug)
    frame = build_scraped_matches(players, schedule, competition=label)
    ensure_dirs()
    out = scraped_path(args.slug)
    frame.to_parquet(out, index=False)

    seasons = frame.groupby("Season_End_Year").agg(rows=("Player", "size"), matches=("MatchURL", "nunique"))
    print(f"\nplayer-matches: {len(frame):,}  matches: {frame['MatchURL'].nunique()}")
    print(seasons.to_string())
    print(f"dates: {frame['Match_Date'].min().date()} to {frame['Match_Date'].max().date()}")
    print(f"home/away balance: {frame['is_home'].mean():.2f} (should be close to 0.50)")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
