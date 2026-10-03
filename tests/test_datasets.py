"""Splitters must never place a test row before a training row."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footy.datasets import (
    Fold,
    assert_fold_is_ordered,
    season_folds,
)
from tests.conftest import make_player_matches


@pytest.fixture
def frame() -> pd.DataFrame:
    # 60 matchdays spans five synthetic seasons (12 matchdays each).
    return make_player_matches(n_matchdays=60, players_per_team=3, seed=3)


def test_season_folds_are_ordered(frame: pd.DataFrame) -> None:
    folds = season_folds(frame)
    assert folds
    for fold in folds:
        assert_fold_is_ordered(frame, fold)


def test_validation_precedes_test(frame: pd.DataFrame) -> None:
    for fold in season_folds(frame):
        seasons = frame["Season_End_Year"].to_numpy()
        assert seasons[fold.valid].max() < seasons[fold.test].min()
        assert seasons[fold.train].max() < seasons[fold.valid].min()


def test_a_shuffled_split_is_rejected(frame: pd.DataFrame) -> None:
    """The v1 mistake, caught.

    ``train_test_split(..., random_state=42)`` on a time series produces exactly this
    fold, and the guard must refuse it.
    """
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(len(frame))
    split = int(0.8 * len(frame))
    bad = Fold(
        name="shuffled",
        train=np.sort(shuffled[:split]),
        valid=np.array([], dtype=int),
        test=np.sort(shuffled[split:]),
    )

    with pytest.raises(ValueError, match="leaks"):
        assert_fold_is_ordered(frame, bad)
