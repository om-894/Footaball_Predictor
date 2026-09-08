"""Time-respecting train/validation/test splits.

The v1 pipeline used ``train_test_split(X, y, test_size=0.2, random_state=42)`` on a time
series, so the model trained on future matches to predict past ones. Every splitter here
is ordered: a test row's date is always later than every training row's.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fold:
    """Index positions for one ordered fold."""

    name: str
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray

    def __len__(self) -> int:  # pragma: no cover - display only
        return len(self.test)


def season_folds(
    frame: pd.DataFrame,
    *,
    test_seasons: tuple[int, ...] | None = None,
    n_valid_seasons: int = 1,
    min_train_seasons: int = 2,
) -> list[Fold]:
    """One fold per test season, training on everything strictly earlier.

    The season immediately before the test season is held out for validation (early
    stopping, hyperparameters), so no tuning decision ever sees the test season.
    """
    seasons = sorted(frame["Season_End_Year"].dropna().unique().astype(int))
    if test_seasons is None:
        # Everything we can afford a training and validation history for.
        test_seasons = tuple(seasons[min_train_seasons + n_valid_seasons:])

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


def expanding_matchweek_folds(
    frame: pd.DataFrame, *, test_season: int, step: int = 4, min_matchweek: int = 5
) -> list[Fold]:
    """Refit repeatedly through a season, as you would in live use.

    Each fold trains on every prior season plus the completed matchweeks of the test
    season, then predicts the next ``step`` matchweeks.
    """
    seasons = frame["Season_End_Year"].to_numpy()
    matchweeks = frame["Matchweek"].to_numpy()
    in_season = seasons == test_season
    if not in_season.any():
        raise ValueError(f"season {test_season} is not present in the frame")

    available = sorted(
        int(w) for w in np.unique(matchweeks[in_season]) if not np.isnan(w)
    )
    folds: list[Fold] = []

    for start in range(min_matchweek, max(available) + 1, step):
        test_weeks = [w for w in available if start <= w < start + step]
        if not test_weeks:
            continue

        history = (seasons < test_season) | (in_season & (matchweeks < start))
        # The most recent completed matchweeks of the test season act as validation.
        valid_mask = in_season & (matchweeks >= start - step) & (matchweeks < start)
        train_mask = history & ~valid_mask

        folds.append(
            Fold(
                name=f"{test_season}-mw{start:02d}",
                train=np.flatnonzero(train_mask),
                valid=np.flatnonzero(valid_mask),
                test=np.flatnonzero(in_season & np.isin(matchweeks, test_weeks)),
            )
        )

    log.info("built %d expanding matchweek folds for %s", len(folds), test_season)
    return folds


def assert_fold_is_ordered(frame: pd.DataFrame, fold: Fold) -> None:
    """Raise unless every test date is later than every training date.

    Called by the splitter tests, and cheap enough to call before training too.
    """
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
