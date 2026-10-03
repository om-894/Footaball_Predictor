"""Per-90 conversion, including the two unit mistakes the first version of the project made."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footy.config import FULL_MATCH_MINUTES
from footy.features import to_per90


def test_every_row_is_converted_regardless_of_minutes() -> None:
    """Every row is converted, so short appearances don't keep raw counts while long ones become rates."""
    frame = pd.DataFrame({"Min": [5.0, 19.0, 20.0, 45.0, 90.0], "Touches": [1.0] * 5})

    result = to_per90(frame, ["Touches"])

    expected = FULL_MATCH_MINUTES / frame["Min"]
    np.testing.assert_allclose(result["Touches_p90"], expected)

    # the 19-minute row must not be left at its raw value
    assert result["Touches_p90"].iloc[1] != pytest.approx(1.0)


def test_ratio_columns_are_never_rescaled() -> None:
    """Percentages are never scaled by minutes (scaling turned 66.7% into 84.6% before)."""
    frame = pd.DataFrame({"Min": [71.0], "Passes_Cmp_pct": [66.7], "Touches": [13.0]})

    result = to_per90(frame, ["Passes_Cmp_pct", "Touches"])

    assert "Passes_Cmp_pct_p90" not in result.columns
    assert result["Touches_p90"].iloc[0] == pytest.approx(13.0 * 90 / 71)


def test_conversion_is_exactly_invertible() -> None:
    """rate * minutes / 90 gives back the original count."""
    rng = np.random.default_rng(0)
    minutes = rng.uniform(1, 90, size=200)
    counts = rng.poisson(2.0, size=200).astype(float)
    frame = pd.DataFrame({"Min": minutes, "Fls": counts})

    rates = to_per90(frame, ["Fls"])["Fls_p90"]
    recovered = rates * frame["Min"] / FULL_MATCH_MINUTES

    np.testing.assert_allclose(recovered, counts)


def test_zero_minutes_is_rejected_not_silently_infinite() -> None:
    frame = pd.DataFrame({"Min": [0.0], "Fls": [1.0]})
    with pytest.raises(ValueError, match="positive minutes"):
        to_per90(frame, ["Fls"])
