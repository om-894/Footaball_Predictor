"""Strictly causal feature construction.

The governing rule, and the thing the v1 model got wrong: **every feature for a match on
date *t* is computed only from matches played strictly before *t*.** The v1 model
regressed a match's fouls on that same match's touches and passes -- information nobody
has before kickoff -- so its reported accuracy could never be realised in practice.

Mechanically that means every rolling statistic here is ``.shift(1)``-ed within its group
before aggregation. ``tests/test_causality.py`` enforces it by perturbing the future and
asserting the past does not move.
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

#: Coarse position groups used for the shrinkage prior. FBref's detailed positions
#: (RW, WB, AM...) are too sparse to form a stable prior on their own.
POSITION_GROUPS = {
    "GK": "GK",
    "DF": "DF", "CB": "DF", "LB": "DF", "RB": "DF", "WB": "DF",
    "MF": "MF", "DM": "MF", "CM": "MF", "AM": "MF", "LM": "MF", "RM": "MF",
    "FW": "FW", "LW": "FW", "RW": "FW",
}

#: Strength of the empirical-Bayes prior, in 90-minute matches. A player with 5 full
#: matches behind them gets a feature halfway between their own rate and their position's.
PRIOR_STRENGTH = 5.0


def to_per90(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert count columns to per-90 rates.

    Two rules the v1 pipeline broke:

    * Every row is converted, not just those over 20 minutes. Scaling only some rows left
      one column holding raw counts for short appearances and per-90 rates for long ones.
    * Ratio columns are never touched. Multiplying a 66.7% pass completion by ``90/71``
      produced 84.6%, a number that cannot occur.
    """
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
    """EWMA over each group's *previous* rows.

    ``shift(1)`` before ``ewm`` is what makes this causal: at row *t* the window covers
    rows ``0..t-1`` only. The first appearance is therefore NaN, which is correct -- we
    know nothing about a debutant.
    """
    grouped = frame.groupby(group, sort=False)[columns]
    return grouped.shift(1).groupby(frame[group], sort=False).ewm(
        halflife=halflife, ignore_na=True
    ).mean().reset_index(level=0, drop=True)


def add_player_form(frame: pd.DataFrame) -> pd.DataFrame:
    """Player form: lagged EWMA per-90 rates at several half-lives, plus workload."""
    frame = frame.sort_values(["Player", "Match_Date", "MatchURL"]).copy()

    stats = [c for c in FORM_STATS if c in frame.columns]
    rates = to_per90(frame, stats)
    rate_columns = list(rates.columns)
    frame = pd.concat([frame, rates], axis=1)

    for halflife in EWMA_HALFLIVES:
        ewma = _lagged_ewma(frame, "Player", rate_columns, halflife)
        ewma.columns = [f"{c}_ewm{halflife}" for c in rate_columns]
        frame = pd.concat([frame, ewma], axis=1)

    # Exposure-weighted form: EWMA of counts divided by EWMA of exposure, rather than an
    # EWMA of per-90 rates.
    #
    # The distinction is not cosmetic. A player who commits one foul in a five-minute
    # cameo has a per-90 rate of 18, and averaging rates lets that cameo outweigh a full
    # match. Dividing the smoothed count by the smoothed exposure weights each appearance
    # by how much evidence it carries -- and it measurably outscores the naive version,
    # which is the same flaw the v1 per-90 pipeline built its whole dataset on.
    frame["_exposure90"] = frame["Min"] / FULL_MATCH_MINUTES
    for halflife in EWMA_HALFLIVES:
        counts = _lagged_ewma(frame, "Player", stats, halflife)
        exposure = _lagged_ewma(frame, "Player", ["_exposure90"], halflife)
        weighted = counts.div(exposure["_exposure90"].replace(0, np.nan), axis=0)
        weighted.columns = [f"{c}_rate{halflife}" for c in stats]
        frame = pd.concat([frame, weighted], axis=1)
    frame = frame.drop(columns=["_exposure90"])

    # Minutes get the same treatment; they are the exposure the count models need.
    for halflife in EWMA_HALFLIVES:
        frame[f"Min_ewm{halflife}"] = _lagged_ewma(frame, "Player", ["Min"], halflife)["Min"]

    # Experience so far. cumcount is already exclusive of the current row.
    grouped = frame.groupby("Player", sort=False)
    frame["prior_appearances"] = grouped.cumcount()
    frame["prior_minutes"] = grouped["Min"].cumsum() - frame["Min"]
    started = (frame["Min"] >= 60).astype(float)
    frame["prior_starts"] = (
        started.groupby(frame["Player"], sort=False).cumsum() - started
    )
    frame["start_rate"] = frame["prior_starts"] / frame["prior_appearances"].clip(lower=1)

    return frame


def add_rest_and_congestion(frame: pd.DataFrame) -> pd.DataFrame:
    """Days since the player's previous match, and how crowded the last fortnight was."""
    frame = frame.sort_values(["Player", "Match_Date", "MatchURL"]).copy()

    previous = frame.groupby("Player", sort=False)["Match_Date"].shift(1)
    frame["days_since_last"] = (frame["Match_Date"] - previous).dt.days

    # closed="left" makes the window cover [t - 15d, t), excluding the current match.
    window = (
        frame.set_index("Match_Date")
        .groupby("Player", sort=False)["Min"]
        .rolling("15D", closed="left")
    )
    frame["matches_last_14d"] = window.count().reset_index(level=0, drop=True).to_numpy()
    frame["minutes_last_14d"] = window.sum().reset_index(level=0, drop=True).to_numpy()
    # An empty window means the player genuinely played nothing in the fortnight, which
    # is zero rather than unknown. `days_since_last` stays NaN on debut, correctly.
    frame[["matches_last_14d", "minutes_last_14d"]] = frame[
        ["matches_last_14d", "minutes_last_14d"]
    ].fillna(0.0)
    return frame


def add_position_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Multi-hot positions plus a coarse primary group.

    ``Pos`` lists every position the player occupied ("FW,LW,LM"), so it is genuinely
    multi-label; collapsing it to one categorical throws information away.
    """
    frame = frame.copy()
    # Parquet round-trips the list column as numpy arrays, where truthiness raises. Coerce
    # to plain tuples so membership and indexing behave the same either way.
    def _clean(row: object) -> tuple[str, ...]:
        # Sources vary: a list from ingest, a numpy array after a Parquet round-trip
        # (where truthiness raises), a NaN float for a synthesised fixture row, or a raw
        # "FW,LW" string. Normalise all of them to a tuple.
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
    """Career-to-date per-90 rates, shrunk toward a causal positional prior.

    A player with two appearances has a career foul rate that is mostly noise. The
    Gamma-Poisson posterior mean

        rate = (prior_count + k * prior_mean) / (prior_exposure90 + k)

    interpolates between the player's own record and their position's, weighted by how
    much evidence the player has. Both the numerator counts and the positional prior are
    computed from matches strictly before the current one.
    """
    frame = frame.sort_values(["Match_Date", "MatchURL", "Player"]).copy()
    frame["_exposure90"] = frame["Min"] / FULL_MATCH_MINUTES

    # Intersect with what is actually present. The Championship supplies a narrower set
    # than the Premier League -- FBref publishes no total-tackles column for the second
    # tier -- and the same pipeline serves both.
    targets = [t for t in TARGETS if t in frame.columns]

    # League-wide positional prior, expanding over strictly earlier *dates*.
    #
    # Aggregating to the date before accumulating matters. A plain row-wise cumsum would
    # let players earlier in the sort order contribute to the prior used by others in the
    # same round of fixtures -- a small leak numerically, but the rule this whole module
    # rests on is "matches before t", and a matchday is not before itself.
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

    # Player's own record, then blend.
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


# --------------------------------------------------------------------------------------
# Team and opponent form
# --------------------------------------------------------------------------------------

#: Team-level totals worth tracking. "For" is what the team did; joining the same table
#: on the opponent gives what they concede.
TEAM_STATS = (
    "Sh", "SoT", "xG", "Fls", "Fld", "CrdY", "Tkl", "Int",
    "Touches", "Passes_Cmp", "Passes_Att", "Touches_AttThird",
)


def build_team_matches(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player rows up to one row per team per match."""
    stats = [c for c in TEAM_STATS if c in frame.columns]
    team = (
        frame.groupby(["MatchURL", "Match_Date", "Team", "Opponent"], as_index=False)[stats]
        .sum()
        .sort_values(["Team", "Match_Date", "MatchURL"])
    )
    return team


def add_team_form(team_matches: pd.DataFrame, halflife: int = 6) -> pd.DataFrame:
    """Lagged EWMA of each team's own output, and of what their opponents managed."""
    stats = [c for c in TEAM_STATS if c in team_matches.columns]
    team = team_matches.sort_values(["Team", "Match_Date", "MatchURL"]).copy()

    produced = _lagged_ewma(team, "Team", stats, halflife)
    produced.columns = [f"team_{c}_for" for c in stats]
    team = pd.concat([team, produced], axis=1)

    # What this team allows: attach each match's opponent totals, then roll those.
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
    """Join the team's own form and their opponent's form onto each player-match.

    The opponent's *conceded* rates are the interesting half: a side that fouls a lot and
    allows a lot of shots lifts every opposing player's expected counts.
    """
    form_columns = [c for c in team_form.columns if c.startswith("team_")]
    lookup = team_form[["MatchURL", "Team", *form_columns]]

    out = frame.merge(lookup, on=["MatchURL", "Team"], how="left")

    opponent_lookup = lookup.rename(
        columns={"Team": "Opponent", **{c: c.replace("team_", "opp_") for c in form_columns}}
    )
    out = out.merge(opponent_lookup, on=["MatchURL", "Opponent"], how="left")
    return out


def add_referee_form(frame: pd.DataFrame) -> pd.DataFrame:
    """The referee's expanding average fouls and cards per match, before this match.

    Referees differ substantially in how freely they whistle, and this is the only
    referee signal available anywhere in our sources. Computed as an expanding mean over
    that referee's earlier matches, so it leaks nothing.
    """
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

    # Accumulate by date, not by row. A referee taking two fixtures on one day -- a
    # midweek double-header, or the same official across competitions -- would otherwise
    # let the first inform the second's feature.
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


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Run the full causal feature pipeline over a player-match table."""
    log.info("building features for %d player-matches", len(frame))

    frame = frame.sort_values(["Match_Date", "MatchURL", "Team", "Player"]).copy()
    frame = add_position_features(frame)
    frame = add_player_form(frame)
    frame = add_rest_and_congestion(frame)
    frame = add_shrunk_career_rates(frame)

    team_form = add_team_form(build_team_matches(frame))
    frame = attach_team_and_opponent_form(frame, team_form)
    frame = add_referee_form(frame)

    frame["season_progress"] = frame["Matchweek"] / 38.0
    frame["log_min_offset"] = np.log(frame["Min"] / FULL_MATCH_MINUTES)

    frame = frame.sort_values(["Match_Date", "MatchURL", "Team", "Player"])
    frame = frame.reset_index(drop=True)
    log.info("feature frame: %d rows x %d cols", *frame.shape)
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """The numeric columns a model may train on.

    Deliberately exclusive: anything measured *during* the match -- the targets and their
    per-90 versions -- is excluded, because it is not knowable before kickoff. ``Min`` is
    excluded too; it enters the count models as an offset, supplied by the minutes model
    rather than read off the row.
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
