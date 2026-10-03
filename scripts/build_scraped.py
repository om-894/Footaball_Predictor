"""Assemble a live FBref scrape into the standard player-match table.

Usage:  python scripts/build_scraped.py ENG-Premier-League
"""

import glob
import logging
import sys
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import pandas as pd

from footy.config import RAW_DIR, SCRAPED_LEAGUES, ensure_dirs, scraped_path
from footy.scraped import build_scraped_matches

slug = sys.argv[1]
raw = RAW_DIR / slug

stats = sorted(glob.glob(f"{raw}/player_summary_*.csv"))
schedules = sorted(glob.glob(f"{raw}/schedule_*.csv"))
if not stats:
    raise SystemExit(f"no player_summary_*.csv in {raw} yet")

players = pd.concat([pd.read_csv(f) for f in stats], ignore_index=True)
# Chunked appends repeat the header row; drop those and any duplicate player-matches.
players = players[players["player"] != "player"]
players = players.drop_duplicates(subset=["game_id", "team", "player"])
schedule = pd.concat([pd.read_csv(f) for f in schedules], ignore_index=True)

print(f"loaded {len(players):,} player rows from {len(stats)} season file(s)")
print("columns:", sorted(players.columns.tolist()))

label = SCRAPED_LEAGUES.get(slug, slug)
frame = build_scraped_matches(players, schedule, competition=label)
out = scraped_path(slug)
ensure_dirs()
frame.to_parquet(out, index=False)

print(f"\nplayer-matches: {len(frame):,}  matches: {frame.MatchURL.nunique()}")
print(frame.groupby("Season_End_Year").agg(
    rows=("Player", "size"), matches=("MatchURL", "nunique")).to_string())
print(f"date range: {frame.Match_Date.min().date()} -> {frame.Match_Date.max().date()}")
print(f"home/away balance: {frame.is_home.mean():.2f} (want ~0.50)")

for team in ("Sunderland", "Hull City", "AZ Alkmaar"):
    sub = frame[frame.Team == team]
    if len(sub):
        print(f"\n{team}: {len(sub)} rows, {sub.MatchURL.nunique()} matches, "
              f"{sub.Player.nunique()} players")
        print(sub.groupby("Player")[["Min", "Fls", "Fld", "Sh"]].sum()
              .sort_values("Min", ascending=False).head(10).to_string())

print(f"\nwrote {out}")
