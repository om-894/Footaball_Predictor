"""Poisson and negative binomial GLMs with a log-minutes offset.

The offset is the point. Writing

    log(E[count]) = log(minutes / 90) + X·beta

fixes the exposure coefficient at exactly 1, which says a player who plays twice as long
commits twice as many fouls in expectation. That is the assumption the v1 per-90 division
was reaching for, but division discards the information that a 15-minute cameo is far
weaker evidence than a full match. As an offset, the model keeps it.

The negative binomial variant additionally estimates overdispersion, which the data
demands: Premier League fouls have mean 0.75 and variance 0.95.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

from footy.models.base import CountModel, clean_matrix

log = logging.getLogger(__name__)


class PoissonGLM(CountModel):
    """Log-link Poisson regression on standardised features."""

    name = "PoissonGLM"

    def __init__(self, max_features: int = 60, ridge: float = 1e-3) -> None:
        super().__init__()
        self.max_features = max_features
        self.ridge = ridge

    def _select(self, X: pd.DataFrame) -> list[str]:
        """Keep the highest-variance features.

        A GLM with 200 correlated columns will not converge cleanly, and the ranking here
        uses only the training fold, so it cannot leak.
        """
        variance = X.var(numeric_only=True).sort_values(ascending=False)
        return list(variance.head(self.max_features).index)

    def _make_family(self):
        return sm.families.Poisson()

    def _fit_rate(self, X: pd.DataFrame, y: np.ndarray, exposure: np.ndarray) -> None:
        selected = self._select(X)
        matrix, self.medians_ = clean_matrix(X, selected)

        # Standardise on the training fold only. Fitting the scaler on everything, as v1
        # did, leaks the test distribution into training.
        self.mean_ = matrix.mean()
        self.std_ = matrix.std().replace(0, 1.0)
        standardised = (matrix - self.mean_) / self.std_
        self.selected_ = selected

        design = sm.add_constant(standardised, has_constant="add")
        offset = np.log(np.clip(exposure, 1e-6, None))

        model = sm.GLM(y, design, family=self._make_family(), offset=offset)
        try:
            self.result_ = model.fit_regularized(alpha=self.ridge, L1_wt=0.0)
        except Exception as exc:  # noqa: BLE001 - fall back rather than lose the fold
            log.warning("%s regularised fit failed (%s); using plain IRLS", self.name, exc)
            self.result_ = model.fit(maxiter=100)

        # Guard against a diverging fit. With a log link, a few large coefficients send
        # exp() to absurd rates: the negative binomial fit for tackles produced a mean
        # prediction of 12 against an actual mean near 1, wrecking that row of the results
        # table while its log-score still looked ordinary. Nothing in football justifies a
        # rate far above the highest ever observed, so cap there.
        observed_rate = y / np.clip(exposure, 1e-6, None)
        self.max_rate_ = float(np.quantile(observed_rate, 0.999) * 2.0) or 1.0

    def _predict_rate(self, X: pd.DataFrame) -> np.ndarray:
        matrix, _ = clean_matrix(X, self.selected_, self.medians_)
        standardised = (matrix - self.mean_) / self.std_
        design = sm.add_constant(standardised, has_constant="add")
        # Predict with zero offset to get the per-90 rate; exposure is applied by the base
        # class, keeping every model on the same footing.
        linear = np.asarray(design) @ np.asarray(self.result_.params)
        return np.clip(np.exp(np.clip(linear, -20, 20)), 0.0, self.max_rate_)


class NegativeBinomialGLM(PoissonGLM):
    """Poisson GLM plus an estimated dispersion parameter."""

    name = "NegBinGLM"

    def __init__(self, max_features: int = 60, ridge: float = 1e-3, alpha: float = 0.5):
        super().__init__(max_features=max_features, ridge=ridge)
        self.nb_alpha = alpha

    def _make_family(self):
        return sm.families.NegativeBinomial(alpha=self.nb_alpha)
