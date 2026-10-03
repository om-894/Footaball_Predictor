"""
Splits the data into training, validation and test seasons, always in date order.

Every test row is later than every training row, so a model never trains on matches that
come after the ones it is scored on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fold:
    """Row positions of the training, validation and test parts of one fold."""

    name: str
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray


def testable_seasons(
    frame: pd.DataFrame, *, n_valid_seasons: int = 1, min_train_seasons: int = 2
) -> list[int]:
    """Seasons with enough earlier seasons before them to train and validate on."""
    seasons = sorted(frame["Season_End_Year"].dropna().unique().astype(int))
    return [int(s) for s in seasons[min_train_seasons + n_valid_seasons:]]


def season_folds(
    frame: pd.DataFrame,
    *,
    test_seasons: tuple[int, ...] | None = None,
    n_valid_seasons: int = 1,
    min_train_seasons: int = 2,
) -> list[Fold]:
    """One fold per test season: validate on the season before it, train on the rest before that."""
    seasons = sorted(frame["Season_End_Year"].dropna().unique().astype(int))
    if test_seasons is None:
        test_seasons = tuple(
            testable_seasons(
                frame, n_valid_seasons=n_valid_seasons, min_train_seasons=min_train_seasons
            )
        )

    folds: list[Fold] = []
    for season in test_seasons:
        earlier = [s for s in seasons if s < season]
        if len(earlier) < min_train_seasons + n_valid_seasons:
            log.info("skipping test season %s: not enough history", season)
            continue

        valid_seasons = earlier[-n_valid_seasons:] if n_valid_seasons else []
        train_seasons = earlier[: len(earlier) - n_valid_seasons]

        values = frame["Season_End_Year"].to_numpy()
        folds.append(
            Fold(
                name=f"test{season}",
                train=np.flatnonzero(np.isin(values, train_seasons)),
                valid=np.flatnonzero(np.isin(values, valid_seasons)),
                test=np.flatnonzero(values == season),
            )
        )

    log.info("built %d season folds", len(folds))
    return folds


def assert_fold_is_ordered(frame: pd.DataFrame, fold: Fold) -> None:
    """Raise unless every test date is after every training and validation date."""
    dates = pd.to_datetime(frame["Match_Date"]).to_numpy()

    if len(fold.train) == 0 or len(fold.test) == 0:
        raise ValueError(f"fold {fold.name} has an empty train or test side")

    latest_train = dates[fold.train].max()
    earliest_test = dates[fold.test].min()
    if earliest_test <= latest_train:
        raise ValueError(
            f"fold {fold.name} leaks: test starts {earliest_test} but training runs "
            f"to {latest_train}"
        )

    if len(fold.valid):
        latest_valid = dates[fold.valid].max()
        if dates[fold.test].min() <= latest_valid:
            raise ValueError(
                f"fold {fold.name} leaks: validation runs past the start of test"
            )

    overlap = (
        set(fold.train.tolist()) & set(fold.test.tolist())
        or set(fold.valid.tolist()) & set(fold.test.tolist())
        or set(fold.train.tolist()) & set(fold.valid.tolist())
    )
    if overlap:
        raise ValueError(f"fold {fold.name} reuses {len(overlap)} rows across splits")
