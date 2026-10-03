"""Forecasting a match that has not been played."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footy.fixture import build_fixture_rows, candidate_squad, forecast_fixture, match_lineup
from tests.conftest import TEAMS, make_player_matches

TARGETS = ("Fls", "Fld", "Sh")


@pytest.fixture(scope="module")
def history() -> pd.DataFrame:
    return make_player_matches(n_matchdays=30, players_per_team=6, seed=5)


@pytest.fixture(scope="module")
def kickoff(history: pd.DataFrame) -> pd.Timestamp:
    return history["Match_Date"].max() + pd.Timedelta(days=7)


def test_squad_is_drawn_from_recent_appearances_only(history, kickoff) -> None:
    squad = candidate_squad(history, TEAMS[0], kickoff, window_days=30)
    assert not squad.empty
    assert (squad["last_seen"] < kickoff).all()
    assert (squad["last_seen"] >= kickoff - pd.Timedelta(days=30)).all()


def test_unknown_team_fails_loudly(history, kickoff) -> None:
    """Silently returning an empty squad would produce an empty forecast table."""
    with pytest.raises(ValueError, match="No appearances"):
        candidate_squad(history, "Not A Real Club", kickoff)


def test_fixture_rows_cover_both_sides(history, kickoff) -> None:
    rows = build_fixture_rows(history, TEAMS[0], TEAMS[1], kickoff)
    assert set(rows["Team"]) == {TEAMS[0], TEAMS[1]}
    assert rows.loc[rows["Team"] == TEAMS[0], "is_home"].eq(1).all()
    assert rows.loc[rows["Team"] == TEAMS[1], "is_home"].eq(0).all()
    assert (rows["Opponent"] != rows["Team"]).all()
    assert (rows["Match_Date"] == kickoff).all()


def test_forecast_produces_a_distribution_per_player(history, kickoff) -> None:
    out = forecast_fixture(history, TEAMS[0], TEAMS[1], kickoff, TARGETS)

    assert not out.empty
    assert set(out["Team"]) == {TEAMS[0], TEAMS[1]}
    assert out["exp_minutes"].between(1, 90).all()

    for target in TARGETS:
        assert (out[f"{target}_exp"] >= 0).all()
        assert out[f"{target}_p1"].between(0, 1).all()
        # P(>=2) can never exceed P(>=1).
        assert (out[f"{target}_p2"] <= out[f"{target}_p1"] + 1e-9).all()


def test_forecast_ignores_matches_after_kickoff(history) -> None:
    """The whole point: a fixture forecast must not see anything at or after kickoff.

    We cut the history in half, forecast from the midpoint, then scramble everything after
    it. The forecast must be unchanged.
    """
    midpoint = history["Match_Date"].quantile(0.5)
    cut = pd.Timestamp(midpoint) + pd.Timedelta(days=1)

    baseline = forecast_fixture(history, TEAMS[0], TEAMS[1], cut, TARGETS)

    rng = np.random.default_rng(4)
    perturbed = history.copy()
    future = perturbed["Match_Date"] >= cut
    assert future.any()
    perturbed.loc[future, list(TARGETS)] = rng.poisson(
        15.0, size=(int(future.sum()), len(TARGETS))
    ).astype(float)

    changed = forecast_fixture(perturbed, TEAMS[0], TEAMS[1], cut, TARGETS)

    merged = baseline.merge(changed, on=["Team", "Player"], suffixes=("_a", "_b"))
    assert len(merged) == len(baseline)
    for target in TARGETS:
        np.testing.assert_allclose(
            merged[f"{target}_exp_a"], merged[f"{target}_exp_b"], rtol=1e-9
        )


def test_lineup_restricts_and_fixes_minutes(history, kickoff) -> None:
    """Naming the starters must replace the minutes model, not merely filter rows."""
    squad = candidate_squad(history, TEAMS[0], kickoff)["Player"].tolist()[:4]
    out = forecast_fixture(
        history, TEAMS[0], TEAMS[1], kickoff, TARGETS,
        lineup=squad, lineup_minutes=90.0,
    )

    named = out[out["Team"] == TEAMS[0]]
    assert set(named["Player"]) == set(squad)
    assert (named["exp_minutes"] == 90.0).all()


def test_lineup_matching_is_case_and_accent_insensitive(history, kickoff) -> None:
    """Team sheets rarely carry FBref's exact spelling."""
    candidates = build_fixture_rows(history, TEAMS[0], TEAMS[1], kickoff)
    full = candidates["Player"].iloc[0]
    matched = match_lineup(candidates, [full.upper()])
    assert full in set(matched["Player"])

    # Accents are dropped by hand-typed team sheets but kept by FBref.
    accented = candidates.copy()
    accented.loc[accented.index[0], "Player"] = "Enzo Le Fée"
    assert "Enzo Le Fée" in set(match_lineup(accented, ["Enzo Le Fee"])["Player"])


def test_lineup_matching_handles_letters_without_accents(history, kickoff) -> None:
    """ø, đ and ł have no accent to strip, so they are swapped for plain letters."""
    candidates = build_fixture_rows(history, TEAMS[0], TEAMS[1], kickoff)
    renamed = candidates.copy()
    renamed.loc[renamed.index[0], "Player"] = "Martin Ødegaard"
    assert "Martin Ødegaard" in set(match_lineup(renamed, ["Odegaard"])["Player"])


def test_ambiguous_surname_is_not_guessed(history, kickoff) -> None:
    """Two players sharing a surname must not be silently resolved to one of them."""
    candidates = build_fixture_rows(history, TEAMS[0], TEAMS[1], kickoff)
    shared = candidates.copy()
    shared.loc[shared.index[0], "Player"] = "Gary Neville"
    shared.loc[shared.index[-1], "Player"] = "Phil Neville"

    with pytest.raises(ValueError, match="none of the supplied lineup names"):
        match_lineup(shared, ["Neville"])


def test_unmatched_lineup_names_do_not_pass_silently(history, kickoff) -> None:
    candidates = build_fixture_rows(history, TEAMS[0], TEAMS[1], kickoff)
    with pytest.raises(ValueError, match="none of the supplied lineup names"):
        match_lineup(candidates, ["Nobody At All", "Also Nobody"])


def test_lineup_for_one_team_leaves_the_other_intact(history, kickoff) -> None:
    """Team news lands one side at a time; the other must not disappear."""
    home_xi = candidate_squad(history, TEAMS[0], kickoff)["Player"].tolist()[:4]
    out = forecast_fixture(
        history, TEAMS[0], TEAMS[1], kickoff, TARGETS,
        lineup=home_xi, lineup_minutes=90.0,
    )

    assert set(out.loc[out["Team"] == TEAMS[0], "Player"]) == set(home_xi)
    away = out[out["Team"] == TEAMS[1]]
    assert not away.empty, "the team without a lineup was dropped"

    # Named players get the fixed exposure; the rest keep modelled minutes.
    assert (out.loc[out["Team"] == TEAMS[0], "exp_minutes"] == 90.0).all()
    assert not (away["exp_minutes"] == 90.0).all()
