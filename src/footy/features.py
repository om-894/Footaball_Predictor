"""
Builds the model features from the player-match table.

Every feature for a match on date t only uses matches played before t. Rolling stats are
shifted by one row within each player or team before they are averaged, and
tests/test_causality.py checks that nothing from the future leaks in.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from footy.config import (
    EWMA_HALFLIVES,
    FORM_STATS,
    FULL_MATCH_MINUTES,
    RATIO_COLUMNS,
    TARGETS,
)

log = logging.getLogger(__name__)

# coarse position groups for the shrinkage prior, since FBref's detailed positions are too sparse
POSITION_GROUPS = {
    "GK": "GK",
    "DF": "DF", "CB": "DF", "LB": "DF", "RB": "DF", "WB": "DF",
    "MF": "MF", "DM": "MF", "CM": "MF", "AM": "MF", "LM": "MF", "RM": "MF",
    "FW": "FW", "LW": "FW", "RW": "FW",
}

# weight of the positional prior, in full matches. a player with 5 full matches gets a
# career rate halfway between their own and their position's
PRIOR_STRENGTH = 5.0


# --------------------------------------------------------------------------- #
# PLAYER FEATURES
# --------------------------------------------------------------------------- #

def to_per90(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Count columns converted to per-90 rates, skipping percentage columns."""
    scalable = [c for c in columns if c not in RATIO_COLUMNS and c in frame.columns]
    refused = sorted(set(columns) & RATIO_COLUMNS)
    if refused:
        log.debug("not rescaling ratio columns: %s", refused)

    minutes = frame["Min"].to_numpy(dtype=float)
    if np.any(minutes <= 0):
        raise ValueError("per-90 conversion requires positive minutes on every row")

    scale = FULL_MATCH_MINUTES / minutes
    out = frame[scalable].astype(float).mul(scale, axis=0)
    out.columns = [f"{c}_p90" for c in scalable]
    return out


def _lagged_ewma(
    frame: pd.DataFrame, group: str, columns: list[str], halflife: int
) -> pd.DataFrame:
    """EWMA of each group's previous rows, so a first appearance gets NaN."""
    # shift(1) first so the current match is never in its own average
    grouped = frame.groupby(group, sort=False)[columns]
    return grouped.shift(1).groupby(frame[group], sort=False).ewm(
        halflife=halflife, ignore_na=True
    ).mean().reset_index(level=0, drop=True)


def add_player_form(frame: pd.DataFrame) -> pd.DataFrame:
    """Recent form at several half-lives, plus appearances, minutes and starts so far."""
    frame = frame.sort_values(["Player", "Match_Date", "MatchURL"]).copy()

    stats = [c for c in FORM_STATS if c in frame.columns]
    rates = to_per90(frame, stats)
    rate_columns = list(rates.columns)
    frame = pd.concat([frame, rates], axis=1)

    # smoothed per-90 rates, which the NaivePer90EWMA baseline uses
    for halflife in EWMA_HALFLIVES:
        ewma = _lagged_ewma(frame, "Player", rate_columns, halflife)
        ewma.columns = [f"{c}_ewm{halflife}" for c in rate_columns]
        frame = pd.concat([frame, ewma], axis=1)

    # smoothed counts over smoothed minutes. averaging per-90 rates instead lets one foul in
    # a five-minute cameo (18 per 90) outweigh a full match
    frame["_exposure90"] = frame["Min"] / FULL_MATCH_MINUTES
    for halflife in EWMA_HALFLIVES:
        counts = _lagged_ewma(frame, "Player", stats, halflife)
        exposure = _lagged_ewma(frame, "Player", ["_exposure90"], halflife)
        weighted = counts.div(exposure["_exposure90"].replace(0, np.nan), axis=0)
        weighted.columns = [f"{c}_rate{halflife}" for c in stats]
        frame = pd.concat([frame, weighted], axis=1)
    frame = frame.drop(columns=["_exposure90"])

    # recent minutes, which the minutes model leans on
    for halflife in EWMA_HALFLIVES:
        frame[f"Min_ewm{halflife}"] = _lagged_ewma(frame, "Player", ["Min"], halflife)["Min"]

    # experience before this match (cumcount already leaves out the current row)
    grouped = frame.groupby("Player", sort=False)
    frame["prior_appearances"] = grouped.cumcount()
    frame["prior_minutes"] = grouped["Min"].cumsum() - frame["Min"]
    started = (frame["Min"] >= 60).astype(float) # 60+ minutes counts as a start
    frame["prior_starts"] = (
        started.groupby(frame["Player"], sort=False).cumsum() - started
    )
    frame["start_rate"] = frame["prior_starts"] / frame["prior_appearances"].clip(lower=1)

    return frame


def add_rest_and_congestion(frame: pd.DataFrame) -> pd.DataFrame:
    """Days since the player's previous match plus how busy the last two weeks were."""
    frame = frame.sort_values(["Player", "Match_Date", "MatchURL"]).copy()

    previous = frame.groupby("Player", sort=False)["Match_Date"].shift(1)
    frame["days_since_last"] = (frame["Match_Date"] - previous).dt.days

    # the 15 days before the match, not counting the match itself (closed="left")
    window = (
        frame.set_index("Match_Date")
        .groupby("Player", sort=False)["Min"]
        .rolling("15D", closed="left")
    )
    frame["matches_last_14d"] = window.count().reset_index(level=0, drop=True).to_numpy()
    frame["minutes_last_14d"] = window.sum().reset_index(level=0, drop=True).to_numpy()
    # an empty window means no matches, so zero rather than unknown
    frame[["matches_last_14d", "minutes_last_14d"]] = frame[
        ["matches_last_14d", "minutes_last_14d"]
    ].fillna(0.0)
    return frame


def add_position_features(frame: pd.DataFrame) -> pd.DataFrame:
    """One flag per position played, plus the group of the first position listed."""
    frame = frame.copy()

    def _clean(row: object) -> tuple[str, ...]:
        # positions arrive as a list, a numpy array after a parquet round trip, NaN for a
        # fixture row or a raw "FW,LW" string, so turn them all into a tuple
        if row is None or isinstance(row, float):
            return ()
        if isinstance(row, str):
            return tuple(p.strip() for p in row.split(",") if p.strip())
        return tuple(row)

    positions = frame["positions"].map(_clean) if "positions" in frame.columns else None
    if positions is None:
        positions = frame["Pos"].map(_clean)
    frame["positions"] = positions

    detailed = sorted({p for row in positions for p in row})
    for position in detailed:
        frame[f"pos_{position}"] = positions.map(lambda row, p=position: int(p in row))

    primary = positions.map(lambda row: row[0] if row else None)
    frame["pos_group"] = primary.map(POSITION_GROUPS).fillna("MF")
    for group in ("GK", "DF", "MF", "FW"):
        frame[f"posgrp_{group}"] = (frame["pos_group"] == group).astype(int)

    return frame


def add_shrunk_career_rates(frame: pd.DataFrame) -> pd.DataFrame:
    """Career per-90 rate for each target, pulled towards the player's position average.

    rate = (player count + k * position rate) / (player 90s + k), with k = PRIOR_STRENGTH,
    so a player with little history mostly gets their position's rate.
    """
    frame = frame.sort_values(["Match_Date", "MatchURL", "Player"]).copy()
    frame["_exposure90"] = frame["Min"] / FULL_MATCH_MINUTES

    # only the targets this table has, e.g. a scraped table has TklW but no Tkl
    targets = [t for t in TARGETS if t in frame.columns]

    # position rates from earlier dates only. summing by date first stops players in the
    # same round of fixtures feeding each other's prior
    daily = (
        frame.groupby(["pos_group", "Match_Date"], as_index=False)[
            [*targets, "_exposure90"]
        ].sum()
    ).sort_values(["pos_group", "Match_Date"])

    by_group = daily.groupby("pos_group", sort=False)
    prior_exposure = by_group["_exposure90"].cumsum() - daily["_exposure90"]
    for target in targets:
        prior_count = by_group[target].cumsum() - daily[target]
        daily[f"{target}_posprior_p90"] = prior_count / prior_exposure.replace(0, np.nan)

    prior_columns = [f"{t}_posprior_p90" for t in targets]
    frame = frame.merge(
        daily[["pos_group", "Match_Date", *prior_columns]],
        on=["pos_group", "Match_Date"],
        how="left",
    )

    # the player's own record before this match, blended with the position rate
    frame = frame.sort_values(["Player", "Match_Date", "MatchURL"])
    by_player = frame.groupby("Player", sort=False)
    player_exposure = by_player["_exposure90"].cumsum() - frame["_exposure90"]

    for target in targets:
        player_count = by_player[target].cumsum() - frame[target]
        prior_mean = frame[f"{target}_posprior_p90"]
        frame[f"{target}_career_p90"] = (
            player_count + PRIOR_STRENGTH * prior_mean
        ) / (player_exposure + PRIOR_STRENGTH)

    frame = frame.drop(columns=["_exposure90"])
    return frame


# --------------------------------------------------------------------------- #
# TEAM, OPPONENT AND REFEREE FEATURES
# --------------------------------------------------------------------------- #

# team totals tracked for both what a team does and what it concedes
TEAM_STATS = (
    "Sh", "SoT", "xG", "Fls", "Fld", "CrdY", "Tkl", "Int",
    "Touches", "Passes_Cmp", "Passes_Att", "Touches_AttThird",
)


def build_team_matches(frame: pd.DataFrame) -> pd.DataFrame:
    """Player rows summed to one row per team per match."""
    stats = [c for c in TEAM_STATS if c in frame.columns]
    team = (
        frame.groupby(["MatchURL", "Match_Date", "Team", "Opponent"], as_index=False)[stats]
        .sum()
        .sort_values(["Team", "Match_Date", "MatchURL"])
    )
    return team


def add_team_form(team_matches: pd.DataFrame, halflife: int = 6) -> pd.DataFrame:
    """Each team's recent output (`_for`) and what its opponents managed against it (`_against`)."""
    stats = [c for c in TEAM_STATS if c in team_matches.columns]
    team = team_matches.sort_values(["Team", "Match_Date", "MatchURL"]).copy()

    produced = _lagged_ewma(team, "Team", stats, halflife)
    produced.columns = [f"team_{c}_for" for c in stats]
    team = pd.concat([team, produced], axis=1)

    # attach each match's opponent totals, then smooth those the same way
    opponent_totals = team_matches[["MatchURL", "Team", *stats]].rename(
        columns={"Team": "Opponent", **{c: f"_opp_{c}" for c in stats}}
    )
    team = team.merge(opponent_totals, on=["MatchURL", "Opponent"], how="left")
    team = team.sort_values(["Team", "Match_Date", "MatchURL"])

    conceded = _lagged_ewma(team, "Team", [f"_opp_{c}" for c in stats], halflife)
    conceded.columns = [f"team_{c}_against" for c in stats]
    team = pd.concat([team, conceded], axis=1)

    return team.drop(columns=[f"_opp_{c}" for c in stats])


def attach_team_and_opponent_form(
    frame: pd.DataFrame, team_form: pd.DataFrame
) -> pd.DataFrame:
    """Join the player's team form (`team_`) and the opponent's form (`opp_`) onto each row."""
    form_columns = [c for c in team_form.columns if c.startswith("team_")]
    lookup = team_form[["MatchURL", "Team", *form_columns]]

    out = frame.merge(lookup, on=["MatchURL", "Team"], how="left")

    opponent_lookup = lookup.rename(
        columns={"Team": "Opponent", **{c: c.replace("team_", "opp_") for c in form_columns}}
    )
    out = out.merge(opponent_lookup, on=["MatchURL", "Opponent"], how="left")
    return out


def add_referee_form(frame: pd.DataFrame) -> pd.DataFrame:
    """The referee's average fouls and yellow cards per match before this one."""
    frame = frame.copy()
    if frame["Referee"].isna().all():
        frame["ref_Fls_per_match"] = np.nan
        frame["ref_CrdY_per_match"] = np.nan
        frame["ref_matches"] = 0
        return frame

    per_match = (
        frame.groupby(["MatchURL", "Match_Date", "Referee"], as_index=False)
        .agg(match_Fls=("Fls", "sum"), match_CrdY=("CrdY", "sum"))
    )

    # summed by date first, so two matches a referee takes on one day can't feed each other
    daily = (
        per_match.groupby(["Referee", "Match_Date"], as_index=False)
        .agg(
            day_matches=("MatchURL", "size"),
            day_Fls=("match_Fls", "sum"),
            day_CrdY=("match_CrdY", "sum"),
        )
        .sort_values(["Referee", "Match_Date"])
    )

    by_referee = daily.groupby("Referee", sort=False)
    daily["ref_matches"] = by_referee["day_matches"].cumsum() - daily["day_matches"]
    for stat in ("Fls", "CrdY"):
        cumulative = by_referee[f"day_{stat}"].cumsum() - daily[f"day_{stat}"]
        daily[f"ref_{stat}_per_match"] = (
            cumulative / daily["ref_matches"].replace(0, np.nan)
        )

    columns = ["Referee", "Match_Date", "ref_matches", "ref_Fls_per_match", "ref_CrdY_per_match"]
    return frame.merge(daily[columns], on=["Referee", "Match_Date"], how="left")


# --------------------------------------------------------------------------- #
# FULL FEATURE TABLE
# --------------------------------------------------------------------------- #

def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Run every feature step over a player-match table."""
    log.info("building features for %d player-matches", len(frame))

    frame = frame.sort_values(["Match_Date", "MatchURL", "Team", "Player"]).copy()
    frame = add_position_features(frame)
    frame = add_player_form(frame)
    frame = add_rest_and_congestion(frame)
    frame = add_shrunk_career_rates(frame)

    team_form = add_team_form(build_team_matches(frame))
    frame = attach_team_and_opponent_form(frame, team_form)
    frame = add_referee_form(frame)

    frame["season_progress"] = frame["Matchweek"] / 38.0 # 38 rounds in a Premier League season
    frame["log_min_offset"] = np.log(frame["Min"] / FULL_MATCH_MINUTES)

    frame = frame.sort_values(["Match_Date", "MatchURL", "Team", "Player"])
    frame = frame.reset_index(drop=True)
    log.info("feature frame: %d rows x %d cols", *frame.shape)
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """The numeric columns a model may train on.

    Anything measured during the match itself is left out. `Min` is left out too, since
    the models take minutes as an offset from the minutes model instead.
    """
    banned = set(TARGETS) | {"Min", "log_min_offset"}
    banned |= {f"{t}_p90" for t in TARGETS}
    banned |= {f"{c}_p90" for c in FORM_STATS}
    banned |= {"Gls", "Ast", "xG", "xAG", "Sh_dist_mean", "Sh_headers", "Sh_free_kicks"}
    banned |= set(FORM_STATS)

    allowed_prefixes = (
        "pos_", "posgrp_", "team_", "opp_", "ref_", "prior_", "start_",
        "days_since_", "matches_last_", "minutes_last_", "season_progress",
        "is_home", "Age",
    )

    columns: list[str] = []
    for column in frame.columns:
        if column in banned or not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        if (
            column.endswith(tuple(f"_ewm{h}" for h in EWMA_HALFLIVES))
            or column.endswith(tuple(f"_rate{h}" for h in EWMA_HALFLIVES))
            or column.endswith("_career_p90")
            or column.endswith("_posprior_p90")
            or column.startswith(allowed_prefixes)
        ):
            columns.append(column)

    return sorted(set(columns))
