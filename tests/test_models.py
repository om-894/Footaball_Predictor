"""Model-level contracts, especially the ones that keep the comparison fair."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footy import features as feature_module
from footy.models.baselines import GlobalMean, PlayerEWMA, PositionMean, ShrunkCareerRate
from footy.models.base import clean_matrix
from footy.models.glm import NegativeBinomialGLM, PoissonGLM
from footy.models.minutes import MinutesModel
from tests.conftest import make_player_matches


@pytest.fixture(scope="module")
def built() -> pd.DataFrame:
    return feature_module.build_features(make_player_matches(n_matchdays=30, seed=11))


@pytest.fixture(scope="module")
def matrix(built: pd.DataFrame):
    columns = feature_module.feature_columns(built)
    return built[columns], built["Fls"].to_numpy(float), built["Min"].to_numpy(float)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: GlobalMean(),
        lambda: PositionMean(),
        lambda: ShrunkCareerRate("Fls"),
        lambda: PlayerEWMA("Fls"),
        lambda: PoissonGLM(max_features=20),
    ],
)
def test_predictions_are_non_negative_and_finite(factory, matrix) -> None:
    """Counts cannot be negative.

    The v1 model standardised its targets and minimised MSE, so it could and did emit
    negative shot counts. Working on the rate scale with a log link makes that
    unrepresentable rather than merely unlikely.
    """
    X, y, minutes = matrix
    model = factory().fit(X, y, minutes)
    predictions = model.predict(X, minutes)

    assert np.isfinite(predictions).all()
    assert (predictions >= 0).all()


@pytest.mark.parametrize(
    "factory",
    [lambda: GlobalMean(), lambda: PositionMean(), lambda: ShrunkCareerRate("Fls")],
)
def test_prediction_scales_with_minutes(factory, matrix) -> None:
    """Doubling exposure must double the expected count.

    This is the property the log-minutes offset guarantees, and the reason minutes are
    never a plain feature.
    """
    X, y, minutes = matrix
    model = factory().fit(X, y, minutes)

    single = model.predict(X, np.full(len(X), 45.0))
    double = model.predict(X, np.full(len(X), 90.0))
    np.testing.assert_allclose(double, 2 * single, rtol=1e-9)


def test_baseline_emits_a_real_distribution(matrix) -> None:
    """Baselines must estimate dispersion, or they lose on log-score by construction."""
    X, y, minutes = matrix
    model = GlobalMean().fit(X, y, minutes)
    distribution = model.predict_distribution(X, minutes)

    assert len(distribution) == len(X)
    assert np.isfinite(distribution.log_score(y)).all()
    assert model.alpha_ >= 0


def test_glm_scaler_is_fitted_on_training_data_only(built: pd.DataFrame) -> None:
    """The scaler comes from the training rows, and predicting on other rows never changes it."""
    columns = feature_module.feature_columns(built)
    split = len(built) // 2
    train, test = built.iloc[:split], built.iloc[split:]

    model = PoissonGLM(max_features=15).fit(
        train[columns], train["Fls"].to_numpy(float), train["Min"].to_numpy(float)
    )
    matrix, _ = clean_matrix(train[columns], model.selected_)
    pd.testing.assert_series_equal(model.mean_, matrix.mean())

    minutes = test["Min"].to_numpy(float)
    before = model.predict(test[columns], minutes)
    model.predict(test[columns] * 100.0, minutes) # wildly different rows
    after = model.predict(test[columns], minutes)
    np.testing.assert_allclose(before, after, rtol=1e-12)
    pd.testing.assert_series_equal(model.mean_, matrix.mean())


def test_minutes_model_respects_bounds(built: pd.DataFrame) -> None:
    columns = feature_module.feature_columns(built)
    X, y = built[columns], built["Min"].to_numpy(float)

    model = MinutesModel(seed=0).fit(X, y)
    predictions = model.predict(X)

    assert (predictions >= 1.0).all()
    assert (predictions <= 90.0).all()


def test_minutes_quantiles_are_monotone(built: pd.DataFrame) -> None:
    """Independently fitted quantile models can cross; the output must not."""
    columns = feature_module.feature_columns(built)
    quantiles = MinutesModel(seed=0).fit(built[columns], built["Min"].to_numpy(float))
    values = quantiles.predict_quantiles(built[columns]).to_numpy()

    assert (np.diff(values, axis=1) >= -1e-9).all()


def test_glm_rate_is_capped_at_a_plausible_value(matrix) -> None:
    """A diverging log-link fit must not emit impossible rates.

    The negative binomial fit for tackles once predicted a mean of 12.2 against an actual
    mean near 1.0 -- a blown fit whose log-score still looked ordinary, so only the MAE
    column gave it away.
    """
    X, y, minutes = matrix
    model = NegativeBinomialGLM(max_features=20).fit(X, y, minutes)
    rates = model._predict_rate(X)

    observed_max = (y / (minutes / 90.0)).max()
    assert rates.max() <= max(observed_max * 2.0, model.max_rate_) + 1e-9
    assert np.isfinite(rates).all()
