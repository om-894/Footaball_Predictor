"""
A neural network that predicts every target at once with a negative binomial loss.

One shared set of layers feeds a separate output for each target, so shots, fouls and
tackles learn from each other. Each player also gets a learned embedding (a vector of
numbers), which acts like a player-specific starting rate. Minutes enter as an offset,
like the GLMs.
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

# on macOS, LightGBM and torch each bring their own OpenMP library and the run hangs once
# torch starts. keeping torch to one thread stops it (setting OMP_NUM_THREADS crashes LightGBM)
if sys.platform == "darwin":
    torch.set_num_threads(1)


def negative_binomial_nll(
    y: torch.Tensor, mu: torch.Tensor, log_alpha: torch.Tensor
) -> torch.Tensor:
    """Negative log-likelihood of a negative binomial with mean `mu`, where Var = mu + alpha * mu^2."""
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
    """Network size and training settings."""

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
    """Player embedding and features, through shared layers, to one log rate per target."""

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

        # one output per target, each predicting the log per-90 rate
        self.heads = nn.ModuleList([nn.Linear(size, 1) for _ in range(n_targets)])
        # one dispersion per target, the same for every match
        self.log_alpha = nn.Parameter(torch.zeros(n_targets))

    def forward(self, x: torch.Tensor, player: torch.Tensor) -> torch.Tensor:
        hidden = self.trunk(torch.cat([x, self.player_embedding(player)], dim=1))
        return torch.cat([head(hidden) for head in self.heads], dim=1)


class NegBinMLP:
    """Trains on all targets together and gives a distribution for any one of them."""

    name = "NegBinMLP"

    def __init__(self, targets: list[str], config: TrainConfig | None = None) -> None:
        self.targets = list(targets)
        self.config = config or TrainConfig()
        self.columns_: list[str] = []
        self.player_index_: dict[str, int] = {}

    def _encode_players(self, players: pd.Series) -> torch.Tensor:
        # index 0 is shared by players not seen in training
        codes = players.map(self.player_index_).fillna(0).astype(int).to_numpy()
        return torch.as_tensor(codes, dtype=torch.long)

    def _prepare(self, X: pd.DataFrame, fit: bool = False) -> torch.Tensor:
        """Fill gaps with training medians and standardise with training statistics."""
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

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        minutes: np.ndarray,
        players: pd.Series,
        validation: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.Series] | None = None,
    ) -> "NegBinMLP":
        """Train with mini-batches, keeping the weights from the best validation epoch."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        self.columns_ = list(X.columns)
        # players with 5+ appearances get their own embedding, the rest share index 0
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

            # stop once the validation loss hasn't improved for `patience` epochs
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

    def predict_rates(self, X: pd.DataFrame, players: pd.Series) -> np.ndarray:
        """Per-90 rate for every target, one column per target."""
        features = self._prepare(X).to(self.device_)
        codes = self._encode_players(players).to(self.device_)
        with torch.no_grad():
            log_rate = self.model_(features, codes)
        return np.exp(np.clip(log_rate.cpu().numpy(), -20, 20))

    def predict_distribution(
        self, X: pd.DataFrame, minutes: np.ndarray, players: pd.Series, target: str
    ) -> CountDistribution:
        """Distribution of one target's count for the given minutes."""
        index = self.targets.index(target)
        rates = self.predict_rates(X, players)[:, index]
        exposure = np.asarray(minutes, dtype=float) / FULL_MATCH_MINUTES
        alpha = float(torch.exp(self.model_.log_alpha[index]).item())
        return CountDistribution(rates * exposure, alpha)
