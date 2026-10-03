"""Multi-head negative binomial network.

This is the v1 model rebuilt around what the data actually is. The differences that
matter, in rough order of importance:

* **Negative binomial likelihood, not MSE on z-scores.** Counts are non-negative integers
  and overdispersed; standardising them and minimising squared error assumes a symmetric
  unbounded target, and produces negative "predictions" for players who rarely foul.
* **A log-minutes offset**, so the network learns a per-90 rate and exposure is applied
  exactly rather than learned approximately.
* **One shared trunk with a head per target.** Shots, fouls and tackles share most of
  their signal (role, opponent, minutes), so learning them jointly regularises all six.
  v1 trained a separate model per player on ~13 rows.
* **Player and position embeddings**, giving the model a per-player intercept that
  generalises -- the thing v1 was reaching for by training one model per player, but
  fitted across 81k rows instead of 13.
* **Real training.** Mini-batches, AdamW, a cosine schedule and early stopping with
  meaningful patience, against v1's 20 full-batch steps.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from footy.config import FULL_MATCH_MINUTES
from footy.evaluate import CountDistribution
from footy.models.base import log_exposure

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# macOS OpenMP guard.
#
# LightGBM and PyTorch both link `@rpath/libomp.dylib`, and pip wheels ship their own
# copies (torch/lib/, sklearn/.dylibs/, ...). With two OpenMP runtimes in one process and
# both libraries using thread pools, the pipeline deadlocks: LightGBM fits, then the first
# torch operation hangs at 0% CPU indefinitely, printing nothing.
#
# Capping *torch* is the fix that works. Capping the pool globally via OMP_NUM_THREADS
# does stop the deadlock, but then LightGBM segfaults instead, so the constraint has to be
# torch-side only. It costs nothing measurable here -- these are small dense models, and
# single-threaded training measured slightly *faster* than multi-threaded because the
# batches are too small to amortise the synchronisation.
# --------------------------------------------------------------------------------------
if sys.platform == "darwin":
    torch.set_num_threads(1)


def negative_binomial_nll(
    y: torch.Tensor, mu: torch.Tensor, log_alpha: torch.Tensor
) -> torch.Tensor:
    """Negative log-likelihood of a negative binomial with mean ``mu``.

    Parameterised so ``Var = mu + alpha * mu^2``; as ``alpha -> 0`` this becomes the
    Poisson NLL, so the network can settle on either as the data warrants.
    """
    alpha = torch.exp(log_alpha).clamp(min=1e-6, max=1e6)
    inverse = 1.0 / alpha
    mu = mu.clamp(min=1e-6)

    return -(
        torch.lgamma(y + inverse)
        - torch.lgamma(inverse)
        - torch.lgamma(y + 1.0)
        + inverse * (torch.log(inverse) - torch.log(inverse + mu))
        + y * (torch.log(mu) - torch.log(inverse + mu))
    )


def _offset_tensor(minutes: np.ndarray) -> torch.Tensor:
    """Log exposure as a column tensor, added to the network's log rate."""
    offset = log_exposure(np.asarray(minutes) / FULL_MATCH_MINUTES)
    return torch.as_tensor(offset, dtype=torch.float32).unsqueeze(1)


def _mean_nll(model: nn.Module, x, players, y, offset) -> torch.Tensor:
    """Mean negative binomial NLL of the network on one batch."""
    mu = torch.exp(torch.clamp(model(x, players) + offset, -20, 20))
    return negative_binomial_nll(y, mu, model.log_alpha).mean()


@dataclass
class TrainConfig:
    hidden: tuple[int, ...] = (256, 128)
    embedding_dim: int = 16
    dropout: float = 0.2
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    max_epochs: int = 120
    patience: int = 12
    seed: int = 42
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


class _Network(nn.Module):
    def __init__(
        self, n_features: int, n_players: int, n_targets: int, config: TrainConfig
    ) -> None:
        super().__init__()
        self.player_embedding = nn.Embedding(n_players, config.embedding_dim)
        nn.init.normal_(self.player_embedding.weight, std=0.01)

        layers: list[nn.Module] = []
        size = n_features + config.embedding_dim
        for width in config.hidden:
            layers += [
                nn.Linear(size, width),
                nn.LayerNorm(width),
                nn.SiLU(),
                nn.Dropout(config.dropout),
            ]
            size = width
        self.trunk = nn.Sequential(*layers)

        # One head per target predicts the log per-90 rate.
        self.heads = nn.ModuleList([nn.Linear(size, 1) for _ in range(n_targets)])
        # Dispersion is a free parameter per target rather than a function of the input;
        # it is a property of the count process, not of the individual match.
        self.log_alpha = nn.Parameter(torch.zeros(n_targets))

    def forward(self, x: torch.Tensor, player: torch.Tensor) -> torch.Tensor:
        hidden = self.trunk(torch.cat([x, self.player_embedding(player)], dim=1))
        return torch.cat([head(hidden) for head in self.heads], dim=1)


class NegBinMLP:
    """Trains all targets jointly and exposes a per-target distribution."""

    name = "NegBinMLP"

    def __init__(self, targets: list[str], config: TrainConfig | None = None) -> None:
        self.targets = list(targets)
        self.config = config or TrainConfig()
        self.columns_: list[str] = []
        self.player_index_: dict[str, int] = {}

    # -- preparation -------------------------------------------------------------

    def _encode_players(self, players: pd.Series) -> torch.Tensor:
        # Index 0 is reserved for players unseen in training.
        codes = players.map(self.player_index_).fillna(0).astype(int).to_numpy()
        return torch.as_tensor(codes, dtype=torch.long)

    def _prepare(self, X: pd.DataFrame, fit: bool = False) -> torch.Tensor:
        matrix = X.reindex(columns=self.columns_)
        if fit:
            self.medians_ = matrix.median()
            filled = matrix.fillna(self.medians_)
            self.mean_ = filled.mean()
            self.std_ = filled.std().replace(0, 1.0)
        else:
            filled = matrix.fillna(self.medians_)

        standardised = ((filled - self.mean_) / self.std_).to_numpy(dtype=np.float32)
        standardised = np.nan_to_num(standardised, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.as_tensor(np.clip(standardised, -10, 10), dtype=torch.float32)

    # -- training ----------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        minutes: np.ndarray,
        players: pd.Series,
        validation: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.Series] | None = None,
    ) -> "NegBinMLP":
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        self.columns_ = list(X.columns)
        # Only players with enough history get their own embedding; the rest share the
        # index-0 "unknown" vector, which is also what a genuinely new player receives.
        counts = players.value_counts()
        frequent = counts[counts >= 5].index
        self.player_index_ = {name: i + 1 for i, name in enumerate(frequent)}

        features = self._prepare(X, fit=True)
        targets = torch.as_tensor(y[self.targets].to_numpy(dtype=np.float32))
        offset = _offset_tensor(minutes)
        player_codes = self._encode_players(players)

        device = torch.device(self.config.device)
        model = _Network(
            features.shape[1], len(self.player_index_) + 1, len(self.targets), self.config
        ).to(device)

        loader = DataLoader(
            TensorDataset(features, player_codes, targets, offset),
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.max_epochs
        )

        validation_tensors = None
        if validation is not None:
            X_valid, y_valid, minutes_valid, players_valid = validation
            validation_tensors = (
                self._prepare(X_valid).to(device),
                self._encode_players(players_valid).to(device),
                torch.as_tensor(y_valid[self.targets].to_numpy(dtype=np.float32)).to(device),
                _offset_tensor(minutes_valid).to(device),
            )

        best_loss = float("inf")
        best_state: dict | None = None
        patience_left = self.config.patience

        for epoch in range(self.config.max_epochs):
            model.train()
            for batch_x, batch_p, batch_y, batch_offset in loader:
                batch_x = batch_x.to(device)
                batch_p = batch_p.to(device)
                batch_y = batch_y.to(device)
                batch_offset = batch_offset.to(device)

                optimizer.zero_grad(set_to_none=True)
                loss = _mean_nll(model, batch_x, batch_p, batch_y, batch_offset)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            scheduler.step()

            if validation_tensors is None:
                continue

            model.eval()
            with torch.no_grad():
                validation_loss = _mean_nll(model, *validation_tensors).item()

            if validation_loss < best_loss - 1e-5:
                best_loss = validation_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                patience_left = self.config.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    log.info(
                        "early stop at epoch %d (best validation NLL %.4f)",
                        epoch + 1, best_loss,
                    )
                    break

            if epoch % 10 == 0:
                log.info("epoch %3d | validation NLL %.4f", epoch, validation_loss)

        if best_state is not None:
            model.load_state_dict(best_state)

        self.model_ = model.eval()
        self.device_ = device
        return self

    # -- prediction --------------------------------------------------------------

    def predict_rates(self, X: pd.DataFrame, players: pd.Series) -> np.ndarray:
        """Per-90 rate for every target, shape ``(n_rows, n_targets)``."""
        features = self._prepare(X).to(self.device_)
        codes = self._encode_players(players).to(self.device_)
        with torch.no_grad():
            log_rate = self.model_(features, codes)
        return np.exp(np.clip(log_rate.cpu().numpy(), -20, 20))

    def predict_distribution(
        self, X: pd.DataFrame, minutes: np.ndarray, players: pd.Series, target: str
    ) -> CountDistribution:
        index = self.targets.index(target)
        rates = self.predict_rates(X, players)[:, index]
        exposure = np.asarray(minutes, dtype=float) / FULL_MATCH_MINUTES
        alpha = float(torch.exp(self.model_.log_alpha[index]).item())
        return CountDistribution(rates * exposure, alpha)
