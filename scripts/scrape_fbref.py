"""
Scrape per-match player summary tables from FBref using soccerdata.

Saves in chunks of 25 matches and skips any match already saved, so running it again
resumes a crashed run and retries matches that failed before. soccerdata waits between
requests to stay inside FBref's rate limit, so a full season takes over an hour.

INPUTS        a league name as soccerdata spells it, e.g. "ENG-Premier League"
              one or more season codes, e.g. 2627 2526
OUTPUTS       data/raw/<league with spaces as dashes>/schedule_<season>.csv
              data/raw/<league with spaces as dashes>/player_summary_<season>.csv
REQUIREMENTS  pip install -e ".[live]"   (soccerdata drives a real browser)
              ENG-Championship also needs the soccerdata config in docs/NOTES.md

Example:  python scripts/scrape_fbref.py "ENG-Premier League" 2627 2526
"""

import argparse
import time
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from footy.config import RAW_DIR

try:
    import soccerdata as sd
except ImportError:
    sd = None


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #

CHUNK = 25 # matches requested per call and saved per append


# --------------------------------------------------------------------------- #
# SCRAPING
# --------------------------------------------------------------------------- #

def flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Turn soccerdata's two-row FBref headers into single names like Performance_Sh."""
    df = df.reset_index()
    df.columns = [
        "_".join(str(p) for p in c if str(p) and "Unnamed" not in str(p)).strip()
        if isinstance(c, tuple) else str(c)
        for c in df.columns
    ]
    return df


def scrape_chunk(fbref, match_ids: list[str], season: str) -> pd.DataFrame | None:
    """Summary tables for a chunk of matches, retrying one at a time if the chunk fails."""
    try:
        return flatten(fbref.read_player_match_stats(stat_type="summary", match_id=match_ids))
    except Exception as exc: # noqa: BLE001 - one bad page must not end the run
        print(f"[{season}] chunk failed ({exc}); retrying one match at a time", flush=True)

    # pages fetched before the failure are cached by soccerdata, so the retry is cheap
    recovered = []
    for match_id in match_ids:
        try:
            stats = fbref.read_player_match_stats(stat_type="summary", match_id=[match_id])
            recovered.append(flatten(stats))
        except Exception as exc: # noqa: BLE001
            print(f"[{season}]   skipping {match_id}: {exc}", flush=True)
    return pd.concat(recovered, ignore_index=True) if recovered else None


def scrape_season(league: str, season: str, out_dir) -> None:
    """Scrape every played match of one season that isn't saved yet."""
    target = out_dir / f"player_summary_{season}.csv"
    fbref = sd.FBref(leagues=league, seasons=season)

    schedule = flatten(fbref.read_schedule())
    schedule.to_csv(out_dir / f"schedule_{season}.csv", index=False)
    played = schedule.loc[schedule["score"].notna(), "game_id"].dropna().astype(str).tolist()

    done = set()
    if target.exists():
        done = set(pd.read_csv(target)["game_id"].astype(str))
        print(f"[{season}] resuming: {len(done)} matches already saved", flush=True)
    todo = [match_id for match_id in played if match_id not in done]
    print(f"[{season}] {len(played)} played, {len(todo)} to scrape", flush=True)

    for start in range(0, len(todo), CHUNK):
        began = time.time()
        frame = scrape_chunk(fbref, todo[start:start + CHUNK], season)
        if frame is None:
            continue
        frame.to_csv(target, mode="a", header=not target.exists(), index=False)
        print(
            f"[{season}] {min(start + CHUNK, len(todo))}/{len(todo)} "
            f"(+{len(frame)} rows, {time.time() - began:.0f}s)",
            flush=True,
        )

    print(f"[{season}] done, saved to {target}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape FBref match summary tables with soccerdata.")
    parser.add_argument("league", help='league name as soccerdata spells it, e.g. "ENG-Premier League"')
    parser.add_argument("seasons", nargs="+", help="season codes, e.g. 2627 2526")
    args = parser.parse_args()

    if sd is None:
        raise SystemExit('soccerdata is not installed. Install it with: pip install -e ".[live]"')

    out_dir = RAW_DIR / args.league.replace(" ", "-").replace("/", "-")
    out_dir.mkdir(parents=True, exist_ok=True)
    for season in args.seasons:
        scrape_season(args.league, season, out_dir)


if __name__ == "__main__":
    main()
