"""Combine the mirrored Premier League history with a live 2026/27 scrape.

The mirror runs 2017/18 to September 2025 (81k player-matches) but stops there; the live
scrape covers the current season only (a few matchweeks). Neither alone is enough to
forecast a match tonight: the mirror has no current squads, and the scrape has no history.

They are joined on the columns both actually carry. That intersection is narrower than the
mirror's own schema -- FBref's match summary has no xG, touches or passes -- so the price
of current data is a thinner feature set. Stated here rather than discovered later.
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from footy.config import INTERIM_DIR
from footy.ingest import fbref_match_id

#: Targets present in both sources. The mirror has total tackles (`Tkl`) and the live
#: summary only tackles won (`TklW`), so neither survives the intersection.
COMMON_TARGETS = ("Sh", "SoT", "Fls", "Fld", "CrdY")

KEEP = [
    "MatchURL", "Match_Date", "Matchweek", "Season_End_Year", "Competition_Name",
    "Team", "Opponent", "Home_Away", "is_home", "Player", "Nation", "Pos", "Age",
    "Min", "Referee", "positions", "is_gk", "CrdR", "Int", "Gls", "Ast",
    *COMMON_TARGETS,
]


def _align(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    out = frame.copy()
    for column in KEEP:
        if column not in out.columns:
            out[column] = pd.NA
    out = out[KEEP]
    out["source"] = source
    out["Match_Date"] = pd.to_datetime(out["Match_Date"], errors="coerce")
    for column in (*COMMON_TARGETS, "CrdR", "Int", "Gls", "Ast", "Min"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def main() -> None:
    """Usage: combine_pl.py [extra-slug ...]

    Always joins the mirror with the Premier League scrape. Extra slugs -- e.g.
    ``NED-Eredivisie`` for a European opponent -- are appended too. Cross-league rows
    give a foreign side's players real history and give the opponent-form features
    something to work from; the caveat is that refereeing norms differ by league and
    there is no league indicator feature, so treat foreign-side legs with more suspicion.
    """
    mirror_path = INTERIM_DIR / "player_matches.parquet"
    slugs = ["ENG-Premier-League", *sys.argv[1:]]

    frames = [_align(pd.read_parquet(mirror_path), "mirror")]
    print(f"mirror: {len(frames[0]):,} rows to {frames[0].Match_Date.max().date()}")
    for slug in slugs:
        path = INTERIM_DIR / f"{slug}_player_matches.parquet"
        if not path.exists():
            raise SystemExit(f"{path} missing -- run scripts/build_scraped.py {slug} first")
        live = _align(pd.read_parquet(path), slug)
        print(f"{slug}: {len(live):,} rows to {live.Match_Date.max().date()}")
        frames.append(live)

    # the mirror stores full match URLs and the scrape stores bare ids, so compare on the id
    scraped_ids = set(pd.concat(frames[1:])["MatchURL"].map(fbref_match_id))
    overlap = frames[0]["MatchURL"].map(fbref_match_id).isin(scraped_ids)
    if overlap.any():
        matches = frames[0].loc[overlap, "MatchURL"].nunique()
        print(f"dropped {overlap.sum():,} mirror rows from {matches} matches the scrape also has")
    frames[0] = frames[0][~overlap]

    combined = pd.concat(frames, ignore_index=True)
    # a match that appears in two scraped tables keeps its last copy
    before = len(combined)
    combined = combined.drop_duplicates(subset=["MatchURL", "Team", "Player"], keep="last")
    if len(combined) != before:
        print(f"dropped {before - len(combined):,} overlapping rows")

    combined = combined.dropna(subset=["Match_Date", "Min", "Player", "Team"])
    combined = combined[combined["Min"] > 0]
    combined = combined.sort_values(["Match_Date", "MatchURL", "Team", "Player"])
    combined = combined.reset_index(drop=True)

    out = INTERIM_DIR / "pl_combined_player_matches.parquet"
    combined.to_parquet(out, index=False)

    print(f"\ncombined: {len(combined):,} rows, {combined.MatchURL.nunique():,} matches")
    print(combined.groupby("Season_End_Year").agg(
        rows=("Player", "size"), matches=("MatchURL", "nunique")).tail(6).to_string())

    for team in ("Sunderland", "Hull City", "AZ Alkmaar"):
        sub = combined[combined.Team == team]
        recent = sub[sub.Match_Date >= "2026-08-01"]
        print(f"\n{team}: {len(sub):,} rows total, {len(recent)} since Aug 2026, "
              f"{recent.Player.nunique()} current players")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
