"""Checks that no feature can see the match it describes or anything after it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footy import features
from footy.config import FORM_STATS, TARGETS
from tests.conftest import make_player_matches


def test_features_ignore_the_future(player_matches: pd.DataFrame) -> None:
    """Scrambling every match from a cutoff onwards must not move any earlier feature."""
    cutoff = pd.Timestamp("2020-11-01")

    baseline = features.build_features(player_matches)

    perturbed = player_matches.copy()
    future = perturbed["Match_Date"] >= cutoff
    assert future.any() and (~future).any(), "cutoff must split the fixture"

    rng = np.random.default_rng(1234)
    stat_columns = sorted(set(FORM_STATS) | set(TARGETS))
    perturbed.loc[future, stat_columns] = rng.poisson(
        9.0, size=(int(future.sum()), len(stat_columns))
    ).astype(float)
    perturbed.loc[future, "Min"] = 90.0

    changed = features.build_features(perturbed)

    key = ["MatchURL", "Team", "Player"]
    columns = features.feature_columns(baseline)
    assert columns, "no feature columns were produced"

    before_baseline = baseline[baseline["Match_Date"] < cutoff].set_index(key)[columns]
    before_changed = changed[changed["Match_Date"] < cutoff].set_index(key)[columns]
    before_changed = before_changed.reindex(before_baseline.index)

    difference = (before_baseline - before_changed).abs()
    leaking = difference.max()[lambda s: s > 1e-9].index.tolist()
    assert not leaking, f"features changed when only the future changed: {leaking}"


def test_no_target_survives_into_the_feature_matrix(player_matches: pd.DataFrame) -> None:
    """Targets, their per-90 versions and same-match stats are never model features."""
    built = features.build_features(player_matches)
    columns = set(features.feature_columns(built))

    for target in TARGETS:
        assert target not in columns
        assert f"{target}_p90" not in columns

    # stats from the match itself aren't known before kickoff either
    for stat in FORM_STATS:
        assert stat not in columns, f"same-match stat {stat} leaked into features"
        assert f"{stat}_p90" not in columns

    # minutes come from the minutes model as an offset
    assert "Min" not in columns
    assert "log_min_offset" not in columns


def test_first_appearance_has_no_form(player_matches: pd.DataFrame) -> None:
    """A debut has no earlier matches, so its form features are NaN."""
    built = features.build_features(player_matches)
    debuts = built[built["prior_appearances"] == 0]
    assert not debuts.empty

    form_columns = [c for c in built.columns if c.endswith("_ewm3")]
    assert form_columns
    assert built.loc[debuts.index, form_columns].isna().all().all()


def test_rolling_windows_exclude_the_current_match(player_matches: pd.DataFrame) -> None:
    """`matches_last_14d` only counts earlier matches, so a debut gets zero."""
    built = features.build_features(player_matches)
    debuts = built[built["prior_appearances"] == 0]
    assert (built["matches_last_14d"].fillna(0) >= 0).all()
    assert (debuts["matches_last_14d"].fillna(0) == 0).all()


def test_features_ignore_the_rest_of_the_same_matchday(player_matches: pd.DataFrame) -> None:
    """A match's features must not use other matches played on the same day.

    League-wide sums are where this goes wrong: a cumulative sum in row order would let
    earlier rows on the same day feed the positional prior of later ones.
    """
    target_date = sorted(player_matches["Match_Date"].unique())[6]

    baseline = features.build_features(player_matches)

    # change every match on that date except the one that sorts last, which a row-order
    # cumulative sum would be the one to pick up the others
    on_date = player_matches["Match_Date"] == target_date
    protected_url = sorted(player_matches.loc[on_date, "MatchURL"].unique())[-1]
    to_perturb = on_date & (player_matches["MatchURL"] != protected_url)
    assert to_perturb.any(), "fixture needs at least two matches on the same day"

    rng = np.random.default_rng(99)
    perturbed = player_matches.copy()
    stat_columns = sorted(set(FORM_STATS) | set(TARGETS))
    perturbed.loc[to_perturb, stat_columns] = rng.poisson(
        12.0, size=(int(to_perturb.sum()), len(stat_columns))
    ).astype(float)

    changed = features.build_features(perturbed)

    key = ["MatchURL", "Team", "Player"]
    columns = features.feature_columns(baseline)
    before = baseline[baseline["MatchURL"] == protected_url].set_index(key)[columns]
    after = changed[changed["MatchURL"] == protected_url].set_index(key)[columns]
    after = after.reindex(before.index)

    difference = (before - after).abs()
    leaking = difference.max()[lambda s: s > 1e-9].index.tolist()
    assert not leaking, f"same-matchday leak via: {leaking}"


def test_career_rate_excludes_the_current_match() -> None:
    """The shrunk career rate for a match only uses the matches before it."""
    frame = make_player_matches(n_matchdays=8, seed=7)
    built = features.build_features(frame)

    # give one player an extreme final match, their feature for that match must not move
    player = built["Player"].iloc[0]
    history = built[built["Player"] == player].sort_values("Match_Date")
    assert len(history) >= 3

    spiked = frame.copy()
    last = history.iloc[-1]
    mask = (
        (spiked["Player"] == player)
        & (spiked["MatchURL"] == last["MatchURL"])
    )
    spiked.loc[mask, "Fls"] = 99.0
    rebuilt = features.build_features(spiked)

    original_value = history.iloc[-1]["Fls_career_p90"]
    new_value = rebuilt[
        (rebuilt["Player"] == player) & (rebuilt["MatchURL"] == last["MatchURL"])
    ]["Fls_career_p90"].iloc[0]

    assert original_value == pytest.approx(new_value, rel=1e-9, nan_ok=True)
