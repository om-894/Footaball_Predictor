"""FBref per-match player stats, read from the worldfootballR_data GitHub releases.

FBref publishes a match's player stats across several tables. This module downloads the
ones we need, renames their columns to a single flat vocabulary, and joins them on
``(MatchURL, Team, Player)`` -- verified to be a unique key across all 81,328 rows of the
Premier League file.

Why not scrape FBref directly: it now serves a Cloudflare interstitial to plain HTTP
clients, so the v1 approach cannot work. scripts/scrape_fbref.py offers an optional
browser-driven path for topping up recent matches.
"""

from __future__ import annotations

import logging

import pandas as pd

from footy.config import LEAGUES, WFR_RELEASE_BASE
from footy.sources.base import CachedDownloader, SourceError

log = logging.getLogger(__name__)

#: Identify a player-match. Verified duplicate-free on the full ENG_M_1st misc table.
JOIN_KEY = ["MatchURL", "Team", "Player"]

#: The table we anchor the join on. It must be the one with the widest coverage, because
#: every other table is left-joined onto it.
#:
#: This is `misc`, not `summary`, and the difference matters: upstream, the published
#: `summary` asset covers only 2 seasons (8,464 rows) while `misc` covers 8 (81,328).
#: Anchoring on `summary` silently discards ~90% of the history.
ANCHOR_TABLE = "misc"

#: Match-level context, identical across every table, so we take it from the anchor only.
MATCH_CONTEXT = [
    "MatchURL", "Match_Date", "Matchweek", "Season_End_Year", "Competition_Name",
    "Home_Team", "Away_Team", "Home_Score", "Away_Score", "Home_xG", "Away_xG",
]

#: Per-player identity columns, also duplicated across tables.
PLAYER_CONTEXT = ["Team", "Home_Away", "Player", "Player_Href", "Nation", "Pos", "Age", "Min"]

# --------------------------------------------------------------------------------------
# Column vocabulary.
#
# FBref's own names are inconsistent between tables (`Touches` in summary vs
# `Touches_Touches` in possession) and some carry spaces. Each entry maps a source column
# to the single name the rest of the package uses. Anything not listed is dropped, so an
# upstream rename surfaces as a missing column rather than a silently absent feature.
# --------------------------------------------------------------------------------------
COLUMN_MAP: dict[str, dict[str, str]] = {
    # Anchor table, and the one that retires the hand-typed foul arrays.
    "misc": {
        "Fls": "Fls", "Fld": "Fld", "Off": "Off", "Crs": "Crs",
        "TklW": "TklW", "PKwon": "PKwon", "PKcon": "PKcon", "OG": "OG",
        "Recov": "Recov", "2CrdY": "CrdY2",
        "Won_Aerial_Duels": "AerialsWon", "Lost_Aerial_Duels": "AerialsLost",
        "Won_percent_Aerial_Duels": "Aerials_Won_pct",
        "CrdY": "CrdY", "CrdR": "CrdR",
    },
    "possession": {
        "Touches_Touches": "Touches", "Carries_Carries": "Carries",
        "PrgC_Carries": "PrgC", "Att_Take_Ons": "TakeOns_Att",
        "Succ_Take_Ons": "TakeOns_Succ",
        "Def Pen_Touches": "Touches_DefPen", "Def 3rd_Touches": "Touches_DefThird",
        "Mid 3rd_Touches": "Touches_MidThird", "Att 3rd_Touches": "Touches_AttThird",
        "Att Pen_Touches": "Touches_AttPen", "Live_Touches": "Touches_Live",
        "Succ_percent_Take_Ons": "TakeOns_Succ_pct", "Tkld_Take_Ons": "TakeOns_Tkld",
        "TotDist_Carries": "Carries_TotDist", "PrgDist_Carries": "Carries_PrgDist",
        "Final_Third_Carries": "Carries_FinalThird", "CPA_Carries": "Carries_PenArea",
        "Mis_Carries": "Miscontrols", "Dis_Carries": "Dispossessed",
        "Rec_Receiving": "PassesReceived", "PrgR_Receiving": "PrgPassesReceived",
    },
    "passing": {
        "Cmp_Total": "Passes_Cmp", "Att_Total": "Passes_Att",
        "Cmp_percent_Total": "Passes_Cmp_pct", "PrgP": "PrgP",
        "Ast": "Ast", "xAG": "xAG",
        "TotDist_Total": "Passes_TotDist", "PrgDist_Total": "Passes_PrgDist",
        "Att_Short": "Passes_Att_Short", "Att_Medium": "Passes_Att_Medium",
        "Att_Long": "Passes_Att_Long", "xA": "xA", "KP": "KeyPasses",
        "Final_Third": "Passes_FinalThird", "PPA": "Passes_PenArea",
        "CrsPA": "Crosses_PenArea",
    },
    "defense": {
        "Tkl_Tackles": "Tkl", "Int": "Int", "Blocks_Blocks": "Blocks",
        "TklW_Tackles": "TklW_def", "Def 3rd_Tackles": "Tkl_DefThird",
        "Mid 3rd_Tackles": "Tkl_MidThird", "Att 3rd_Tackles": "Tkl_AttThird",
        "Tkl_Challenges": "Challenges_Tkl", "Att_Challenges": "Challenges_Att",
        "Lost_Challenges": "Challenges_Lost", "Sh_Blocks": "Blocks_Sh",
        "Pass_Blocks": "Blocks_Pass", "Clr": "Clearances", "Err": "Errors",
    },
    "passing_types": {
        "Dead_Pass_Types": "Passes_Dead", "FK_Pass_Types": "Passes_FK",
        "TB_Pass_Types": "Passes_Through", "Sw_Pass_Types": "Passes_Switch",
        "Crs_Pass_Types": "Passes_Cross", "TI_Pass_Types": "ThrowIns",
        "CK_Pass_Types": "CornerKicks", "Off_Outcomes": "Passes_Offside",
    },
}

#: Loaded by default. `passing_types` is available but adds ~46MB for features that
#: matter much less. `summary` is deliberately excluded: upstream it covers only 2 of the
#: 8 seasons, and every column it holds is recoverable from the tables below plus the
#: shot-level file.
DEFAULT_STAT_TYPES = ("misc", "possession", "passing", "defense")


def league_code(league: str) -> str:
    """worldfootballR's code for a league, e.g. ENG_M_1st for ENG-PL."""
    if league not in LEAGUES:
        raise KeyError(f"Unknown league {league!r}. Known: {sorted(LEAGUES)}")
    return LEAGUES[league]


def asset_url(league: str, stat_type: str) -> str:
    """URL of one published FBref table."""
    return f"{WFR_RELEASE_BASE}/{league_code(league)}_{stat_type}_player_advanced_match_stats.csv"


def _load_table(
    downloader: CachedDownloader, league: str, stat_type: str, *, force: bool
) -> pd.DataFrame:
    """Download one table and reduce it to the join key plus its mapped columns."""
    if stat_type not in COLUMN_MAP:
        raise ValueError(
            f"No column map for stat_type {stat_type!r}. Known: {sorted(COLUMN_MAP)}"
        )

    url = asset_url(league, stat_type)
    path = downloader.fetch(url, f"{league_code(league)}_{stat_type}.csv", force=force)
    frame = pd.read_csv(path, low_memory=False)

    mapping = COLUMN_MAP[stat_type]
    missing = sorted(set(mapping) - set(frame.columns))
    if missing:
        # Loud, not silent: a renamed upstream column would otherwise quietly become a
        # column of NaNs that the model happily trains on.
        raise SourceError(
            f"{stat_type} table is missing expected columns {missing}. "
            "The upstream schema has probably changed; update COLUMN_MAP."
        )

    keep = JOIN_KEY + list(mapping)
    if stat_type == ANCHOR_TABLE:
        keep = list(dict.fromkeys(MATCH_CONTEXT + PLAYER_CONTEXT + keep))

    out = frame[keep].rename(columns=mapping)
    duplicates = out.duplicated(JOIN_KEY).sum()
    if duplicates:
        raise SourceError(
            f"{stat_type} has {duplicates} duplicate rows on {JOIN_KEY}; "
            "the join key assumption no longer holds."
        )

    log.info("%s: %d rows, %d mapped columns", stat_type, len(out), len(mapping))
    return out


def load_player_match_stats(
    league: str = "ENG-PL",
    stat_types: tuple[str, ...] = DEFAULT_STAT_TYPES,
    *,
    downloader: CachedDownloader | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Return one wide row per player-match, joined across the requested FBref tables."""
    if ANCHOR_TABLE not in stat_types:
        raise ValueError(
            f"stat_types must include {ANCHOR_TABLE!r} -- it anchors the join and "
            "supplies match context"
        )

    downloader = downloader or CachedDownloader()
    merged = _load_table(downloader, league, ANCHOR_TABLE, force=force)

    for stat_type in stat_types:
        if stat_type == ANCHOR_TABLE:
            continue
        table = _load_table(downloader, league, stat_type, force=force)
        before = len(merged)
        merged = merged.merge(table, on=JOIN_KEY, how="left", validate="one_to_one")
        if len(merged) != before:
            raise SourceError(
                f"Joining {stat_type} changed the row count {before} -> {len(merged)}"
            )

    log.info("joined player-match frame: %d rows x %d cols", *merged.shape)
    return merged


# --------------------------------------------------------------------------------------
# Shot-level events
# --------------------------------------------------------------------------------------

SHOOTING_URL = (
    "https://github.com/JaseZiv/worldfootballR_data/releases/download"
    "/fb_match_shooting/{code}_match_shooting.csv"
)

#: FBref counts a shot as on target when it is a goal or was saved. Blocked shots,
#: woodwork and "saved off target" are all off target.
ON_TARGET_OUTCOMES = frozenset({"Goal", "Saved"})


def load_match_shooting(
    league: str = "ENG-PL",
    *,
    downloader: CachedDownloader | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Aggregate the shot-level file to one row per player-match.

    This is where ``Sh``, ``SoT`` and ``xG`` come from. The published `summary` table
    would also carry them, but only for 2 of the 8 seasons -- and shot events give us
    shot distance and body part for free, which an aggregate never could.
    """
    code = league_code(league)
    downloader = downloader or CachedDownloader()
    path = downloader.fetch(
        SHOOTING_URL.format(code=code), f"{code}_shooting.csv", force=force
    )
    shots = pd.read_csv(path, low_memory=False)

    required = {"MatchURL", "Squad", "Player", "Outcome", "xG", "Distance"}
    missing = required - set(shots.columns)
    if missing:
        raise SourceError(f"shooting table missing columns {sorted(missing)}")

    shots = shots.dropna(subset=["MatchURL", "Squad", "Player"])
    shots["xG"] = pd.to_numeric(shots["xG"], errors="coerce")
    shots["Distance"] = pd.to_numeric(shots["Distance"], errors="coerce")
    shots["_on_target"] = shots["Outcome"].isin(ON_TARGET_OUTCOMES).astype(int)
    shots["_header"] = shots["Body Part"].eq("Head").astype(int)
    shots["_free_kick"] = (
        shots["Notes"].fillna("").str.contains("Free kick", case=False).astype(int)
    )

    aggregated = (
        shots.groupby(["MatchURL", "Squad", "Player"], as_index=False)
        .agg(
            Sh=("Outcome", "size"),
            SoT=("_on_target", "sum"),
            xG=("xG", "sum"),
            Sh_dist_mean=("Distance", "mean"),
            Sh_headers=("_header", "sum"),
            Sh_free_kicks=("_free_kick", "sum"),
        )
        .rename(columns={"Squad": "Team"})
    )

    log.info(
        "shooting: %d shots -> %d player-match rows", len(shots), len(aggregated)
    )
    return aggregated


def attach_shooting(player_matches: pd.DataFrame, shooting: pd.DataFrame) -> pd.DataFrame:
    """Left-join shot aggregates. A player with no shots is absent from the shot file,
    so the resulting NaNs are genuine zeros -- except ``Sh_dist_mean``, which is
    undefined with no shots and stays missing."""
    out = player_matches.merge(shooting, on=JOIN_KEY, how="left", validate="one_to_one")
    for column in ("Sh", "SoT", "xG", "Sh_headers", "Sh_free_kicks"):
        out[column] = out[column].fillna(0.0)
    return out
