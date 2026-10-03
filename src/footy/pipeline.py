"""
Fits every model on each season fold and scores them on the held-out season.

Each model is scored twice. "known-minutes" uses the minutes actually played, which
isolates how good the per-90 rate model is. "forecast" uses the minutes model's
prediction instead, which is what a real pre-match forecast has to do.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from footy.config import FULL_MATCH_MINUTES, TARGETS
from footy.datasets import Fold, assert_fold_is_ordered, season_folds
from footy.evaluate import (
    CountDistribution,
    comparison_table,
    expected_calibration_error,
    score,
)
from footy.features import feature_columns
from footy.models.baselines import (
    GlobalMean,
    NaivePer90EWMA,
    PlayerEWMA,
    PositionMean,
    ShrunkCareerRate,
)
from footy.models.gbm import PoissonGBM
from footy.models.glm import NegativeBinomialGLM, PoissonGLM
from footy.models.minutes import MinutesModel
from footy.models.nn import NegBinMLP, TrainConfig

log = logging.getLogger(__name__)

# the model every other model is compared against in the results tables
BENCHMARK = "PlayerEWMA"


@dataclass
class FoldResult:
    fold: str
    rows: list[dict]
    predictions: pd.DataFrame
    # the trained network, kept so `footy predict` can get full distributions from it
    network: "NegBinMLP | None" = None
    feature_columns: list[str] = field(default_factory=list)


def _score_model(
    name: str,
    target: str,
    fold_name: str,
    y_true: np.ndarray,
    distribution: CountDistribution,
    mode: str,
) -> dict:
    """Every metric for one model on one target, labelled with the fold and mode."""
    row = score(y_true, distribution, label=name)
    row.update(
        target=target,
        fold=fold_name,
        mode=mode,
        ECE=expected_calibration_error(y_true, distribution, k=1),
    )
    return row


def _fit_network(train, valid, X_train, X_valid, targets, seed) -> NegBinMLP:
    """Train the network on every target at once, early stopping on the validation season."""
    return NegBinMLP(list(targets), TrainConfig(seed=seed)).fit(
        X_train,
        train[list(targets)],
        train["Min"].to_numpy(dtype=float),
        train["Player"],
        validation=(
            X_valid,
            valid[list(targets)],
            valid["Min"].to_numpy(dtype=float),
            valid["Player"],
        ),
    )


def _fit_count_models(target, train, valid, X_train, X_valid, fold_name, seed) -> dict:
    """Fit the baselines, GLMs and gradient boosting for one target, skipping any that fail."""
    y_train = train[target].to_numpy(dtype=float)
    minutes_train = train["Min"].to_numpy(dtype=float)

    fitted = {}
    for model in (
        GlobalMean(),
        PositionMean(),
        ShrunkCareerRate(target),
        NaivePer90EWMA(target),
        PlayerEWMA(target),
        PoissonGLM(),
        NegativeBinomialGLM(),
    ):
        try:
            fitted[model.name] = model.fit(X_train, y_train, minutes_train)
        except Exception as exc:  # noqa: BLE001 - one failed model shouldn't stop the run
            log.warning("%s failed on %s/%s: %s", model.name, target, fold_name, exc)

    gbm = PoissonGBM(seed=seed).set_validation(
        X_valid, valid[target].to_numpy(dtype=float), valid["Min"].to_numpy(dtype=float)
    )
    try:
        fitted[gbm.name] = gbm.fit(X_train, y_train, minutes_train)
    except Exception as exc:  # noqa: BLE001
        log.warning("PoissonGBM failed on %s/%s: %s", target, fold_name, exc)
    return fitted


def run_fold(
    frame: pd.DataFrame,
    fold: Fold,
    *,
    targets: tuple[str, ...] = TARGETS,
    modes: tuple[str, ...] = ("known-minutes", "forecast"),
    include_nn: bool = True,
    seed: int = 42,
) -> FoldResult:
    """Fit every model on one fold and score it on the held-out matches."""
    assert_fold_is_ordered(frame, fold)

    columns = feature_columns(frame)
    train = frame.iloc[fold.train]
    valid = frame.iloc[fold.valid] if len(fold.valid) else train
    test = frame.iloc[fold.test]

    X_train, X_valid, X_test = train[columns], valid[columns], test[columns]
    minutes_train = train["Min"].to_numpy(dtype=float)
    minutes_test = test["Min"].to_numpy(dtype=float)

    log.info(
        "fold %s: train=%d valid=%d test=%d, %d features",
        fold.name, len(train), len(valid), len(test), len(columns),
    )

    # stage 1: how many minutes each player will play
    minutes_model = MinutesModel(seed=seed).fit(X_train, minutes_train)
    predicted_minutes = np.clip(minutes_model.predict(X_test), 1.0, FULL_MATCH_MINUTES)
    minutes_mae = float(np.abs(predicted_minutes - minutes_test).mean())
    log.info("fold %s: minutes model MAE %.2f", fold.name, minutes_mae)

    exposure_by_mode = {
        "known-minutes": minutes_test,
        "forecast": predicted_minutes,
    }

    rows: list[dict] = []
    keep = ["MatchURL", "Match_Date", "Team", "Opponent", "Player", "Min", *targets]
    predictions = test[[c for c in keep if c in test.columns]].copy()
    predictions["pred_minutes"] = predicted_minutes
    predictions["fold"] = fold.name

    # the network learns all targets together, so it is trained once per fold
    network = _fit_network(train, valid, X_train, X_valid, targets, seed) if include_nn else None

    # stage 2: the per-90 rate models, one set per target
    for target in targets:
        y_test = test[target].to_numpy(dtype=float)
        fitted = _fit_count_models(target, train, valid, X_train, X_valid, fold.name, seed)

        for mode in modes:
            exposure = exposure_by_mode[mode]
            for name, model in fitted.items():
                distribution = model.predict_distribution(X_test, exposure)
                rows.append(
                    _score_model(name, target, fold.name, y_test, distribution, mode)
                )
                if mode == "forecast":
                    predictions[f"{target}__{name}"] = distribution.mu

            if network is not None:
                distribution = network.predict_distribution(
                    X_test, exposure, test["Player"], target
                )
                rows.append(
                    _score_model(
                        NegBinMLP.name, target, fold.name, y_test, distribution, mode
                    )
                )
                if mode == "forecast":
                    predictions[f"{target}__{NegBinMLP.name}"] = distribution.mu

    for row in rows:
        row["minutes_MAE"] = minutes_mae

    return FoldResult(
        fold=fold.name,
        rows=rows,
        predictions=predictions,
        network=network,
        feature_columns=columns,
    )


def run_walk_forward(
    frame: pd.DataFrame,
    *,
    targets: tuple[str, ...] = TARGETS,
    test_seasons: tuple[int, ...] | None = None,
    include_nn: bool = True,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every fold and return the per-fold scores and the held-out predictions."""
    folds = season_folds(frame, test_seasons=test_seasons)
    if not folds:
        raise ValueError("no usable folds; the data may not span enough seasons")

    all_rows: list[dict] = []
    all_predictions: list[pd.DataFrame] = []
    for fold in folds:
        result = run_fold(
            frame, fold, targets=targets, include_nn=include_nn, seed=seed
        )
        all_rows.extend(result.rows)
        all_predictions.append(result.predictions)

    return pd.DataFrame(all_rows), pd.concat(all_predictions, ignore_index=True)


def summarise(scores: pd.DataFrame, mode: str = "forecast") -> pd.DataFrame:
    """Each model's metrics averaged over the folds, weighted by fold size."""
    subset = scores[scores["mode"] == mode]
    metrics = ["LogScore", "CRPS", "PoissonDev", "MAE", "RMSE", "ECE", "pred_mean", "actual_mean"]
    available = [m for m in metrics if m in subset.columns]

    def weighted(group: pd.DataFrame) -> pd.Series:
        weights = group["n"] / group["n"].sum()
        return pd.Series({m: float((group[m] * weights).sum()) for m in available})

    aggregated = (
        subset.groupby(["target", "model"])
        .apply(weighted, include_groups=False)
        .reset_index()
    )
    totals = subset.groupby(["target", "model"], as_index=False)["n"].sum()
    aggregated = aggregated.merge(totals, on=["target", "model"], how="left")
    return comparison_table(aggregated.to_dict("records"))


def improvement_over_baseline(
    summary: pd.DataFrame, baseline: str = BENCHMARK, metric: str = "LogScore"
) -> pd.DataFrame:
    """Percentage improvement on the benchmark for each target, negative where a model is worse."""
    pivot = summary.pivot_table(index="target", columns="model", values=metric)
    if baseline not in pivot.columns:
        raise KeyError(f"baseline {baseline!r} not in summary")
    reference = pivot[baseline]
    return (100.0 * (reference.values[:, None] - pivot) / reference.values[:, None]).round(2)
