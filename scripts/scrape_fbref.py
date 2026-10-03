"""Scrape player match stats live from FBref via soccerdata.

Writes incrementally. A 557-match season takes the better part of an hour at FBref's
rate limit, and the browser driver does occasionally drop its connection -- soccerdata
retries, but a fatal crash on match 500 would otherwise discard the whole run. Chunked
writes mean a restart resumes from the last completed chunk, and soccerdata's own page
cache makes re-covering that ground close to free.

Usage:  python scripts/scrape_fbref.py ENG-Championship 2627 2526
        python scripts/scrape_fbref.py "ENG-Premier League" 2627
"""

import sys
import time
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import soccerdata as sd

from footy.config import RAW_DIR

LEAGUE = sys.argv[1]
SLUG = LEAGUE.replace(" ", "-").replace("/", "-")
OUT = RAW_DIR / SLUG
OUT.mkdir(parents=True, exist_ok=True)
CHUNK = 25


def flatten(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index()
    df.columns = [
        "_".join(str(p) for p in c if str(p) and "Unnamed" not in str(p)).strip()
        if isinstance(c, tuple) else str(c)
        for c in df.columns
    ]
    return df


def scrape_season(season: str) -> None:
    target = OUT / f"player_summary_{season}.csv"
    fb = sd.FBref(leagues=LEAGUE, seasons=season)

    schedule = flatten(fb.read_schedule())
    schedule.to_csv(OUT / f"schedule_{season}.csv", index=False)
    ids = schedule.loc[schedule["score"].notna(), "game_id"].dropna().astype(str).tolist()

    done: set[str] = set()
    if target.exists():
        existing = pd.read_csv(target)
        done = set(existing["game_id"].astype(str))
        print(f"[{season}] resuming: {len(done)} matches already saved", flush=True)

    todo = [i for i in ids if i not in done]
    print(f"[{season}] {len(ids)} played, {len(todo)} to scrape", flush=True)

    for start in range(0, len(todo), CHUNK):
        batch = todo[start:start + CHUNK]
        began = time.time()
        try:
            frame = flatten(fb.read_player_match_stats(stat_type="summary", match_id=batch))
        except Exception as exc:  # noqa: BLE001 - one bad chunk must not end the run
            # A single unparseable match page otherwise costs the whole chunk, including
            # the matches already fetched before it. Retry one at a time so only the bad
            # page is lost -- the pages already retrieved are cached, so this is cheap.
            print(f"[{season}] chunk at {start} failed ({exc}); retrying individually",
                  flush=True)
            recovered = []
            for match_id in batch:
                try:
                    recovered.append(
                        flatten(fb.read_player_match_stats(stat_type="summary",
                                                           match_id=[match_id]))
                    )
                except Exception as inner:  # noqa: BLE001
                    print(f"[{season}]   skipping {match_id}: {inner}", flush=True)
            if not recovered:
                continue
            frame = pd.concat(recovered, ignore_index=True)

        frame.to_csv(target, mode="a", header=not target.exists(), index=False)
        print(
            f"[{season}] {min(start + CHUNK, len(todo))}/{len(todo)} "
            f"(+{len(frame)} rows, {time.time() - began:.0f}s)",
            flush=True,
        )

    print(f"[{season}] DONE -> {target}", flush=True)


if __name__ == "__main__":
    for season_code in sys.argv[2:]:
        scrape_season(season_code)
