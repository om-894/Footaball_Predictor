"""
Poisson and negative binomial regressions with a log-minutes offset.

log(E[count]) = log(minutes / 90) + X.beta, so the weight on minutes is fixed at 1 and a
player who plays twice as long is expected to do twice as much.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

from footy.models.base import CountModel, clean_matrix, log_exposure

log = logging.getLogger(__name__)


class PoissonGLM(CountModel):
    """Ridge-penalised Poisson regression on standardised features."""

    name = "PoissonGLM"

    def __init__(self, max_features: int = 60, ridge: float = 1e-3) -> None:
        super().__init__()
        self.max_features = max_features
        self.ridge = ridge

    def _select(self, X: pd.DataFrame) -> list[str]:
        """The highest-variance columns, since a GLM struggles to converge on hundreds of correlated ones."""
        variance = X.var(numeric_only=True).sort_values(ascending=False)
        return list(variance.head(self.max_features).index)

    def _make_family(self):
        return sm.families.Poisson()

    def _fit_rate(self, X: pd.DataFrame, y: np.ndarray, exposure: np.ndarray) -> None:
        selected = self._select(X)
        matrix, self.medians_ = clean_matrix(X, selected)

        # standardise with the training fold's statistics only
        self.mean_ = matrix.mean()
        self.std_ = matrix.std().replace(0, 1.0)
        standardised = (matrix - self.mean_) / self.std_
        self.selected_ = selected

        design = sm.add_constant(standardised, has_constant="add")
        offset = log_exposure(exposure)

        model = sm.GLM(y, design, family=self._make_family(), offset=offset)
        try:
            self.result_ = model.fit_regularized(alpha=self.ridge, L1_wt=0.0)
        except Exception as exc:  # noqa: BLE001 - fall back to a plain fit rather than lose the fold
            log.warning("%s regularised fit failed (%s); using plain IRLS", self.name, exc)
            self.result_ = model.fit(maxiter=100)

        # cap rates at twice the 99.9th percentile seen in training. without the cap a
        # diverging fit once predicted a mean of 12 tackles a match
        observed_rate = y / np.clip(exposure, 1e-6, None)
        self.max_rate_ = float(np.quantile(observed_rate, 0.999) * 2.0) or 1.0

    def _predict_rate(self, X: pd.DataFrame) -> np.ndarray:
        matrix, _ = clean_matrix(X, self.selected_, self.medians_)
        standardised = (matrix - self.mean_) / self.std_
        design = sm.add_constant(standardised, has_constant="add")
        # no offset here gives the per-90 rate, the base class applies the minutes
        linear = np.asarray(design) @ np.asarray(self.result_.params)
        return np.clip(np.exp(np.clip(linear, -20, 20)), 0.0, self.max_rate_)


class NegativeBinomialGLM(PoissonGLM):
    """The same regression with a negative binomial family, which allows for extra variance."""

    name = "NegBinGLM"

    def __init__(self, max_features: int = 60, ridge: float = 1e-3, alpha: float = 0.5):
        super().__init__(max_features=max_features, ridge=ridge)
        self.nb_alpha = alpha

    def _make_family(self):
        return sm.families.NegativeBinomial(alpha=self.nb_alpha)
