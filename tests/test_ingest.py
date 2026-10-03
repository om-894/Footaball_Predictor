"""Parsing FBref's string formats and checking the upstream column maps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footy.ingest import (
    fbref_match_id,
    name_key,
    parse_age,
    parse_matchweek,
    split_positions,
    validate_player_matches,
)
from footy.sources import worldfootballr
from footy.sources.footballdata import TEAM_NAME_TO_FBREF, season_code
from tests.conftest import make_player_matches


@pytest.mark.parametrize(
    "value,expected",
    [
        ("Premier League (Matchweek 1)", 1.0),
        ("Premier League (Matchweek 38)", 38.0),
        ("Championship (Matchweek 17)", 17.0),
        ("Some Cup Round", np.nan),
        (None, np.nan),
        (12, np.nan),
    ],
)
def test_parse_matchweek(value, expected) -> None:
    result = parse_matchweek(value)
    assert (np.isnan(result) and np.isnan(expected)) or result == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("26-075", 26 + 75 / 365.25),
        ("19-000", 19.0),
        ("40-364", 40 + 364 / 365.25),
        (24.5, 24.5),
        ("", np.nan),
        ("nonsense", np.nan),
    ],
)
def test_parse_age(value, expected) -> None:
    result = parse_age(value)
    assert (np.isnan(result) and np.isnan(expected)) or result == pytest.approx(expected)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("FW,LW,LM", ["FW", "LW", "LM"]),
        ("GK", ["GK"]),
        ("CM, DM", ["CM", "DM"]),
        ("", []),
        (None, []),
    ],
)
def test_split_positions(value, expected) -> None:
    """FBref lists every position occupied, so this is multi-label, not categorical."""
    assert split_positions(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("https://fbref.com/en/matches/e3c3ddf0/Arsenal-Leicester-City-August-11-2017-Premier-League", "e3c3ddf0"),
        ("fc5c9711", "fc5c9711"),
    ],
)
def test_fbref_match_id(value, expected) -> None:
    """The mirror stores full match URLs and the live scrape stores bare ids."""
    assert fbref_match_id(value) == expected


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Enzo Le Fée", "enzo le fee"),
        ("Martin Ødegaard", "martin odegaard"),
        ("Đorđe Petrović", "dorde petrovic"),
        ("Nico O’Reilly", "nico o'reilly"),
        ("  JACK GREALISH ", "jack grealish"),
    ],
)
def test_name_key(name, expected) -> None:
    """Typed names rarely carry FBref's accents, so matching drops them."""
    assert name_key(name) == expected


def test_season_code() -> None:
    assert season_code(2024) == "2324"
    assert season_code(2026) == "2526"
    assert season_code(2000) == "9900"


def test_team_name_map_is_injective() -> None:
    """Two source clubs mapping to one FBref name would silently merge their histories."""
    values = list(TEAM_NAME_TO_FBREF.values())
    assert len(values) == len(set(values))


def test_anchor_table_has_the_targets_we_depend_on() -> None:
    """The misc table is where the fouls and yellow cards come from."""
    mapped = set(worldfootballr.COLUMN_MAP[worldfootballr.ANCHOR_TABLE].values())
    assert {"Fls", "Fld", "CrdY"} <= mapped


def test_column_maps_do_not_collide() -> None:
    """Two tables mapping different columns to one name would overwrite each other on the join."""
    seen: dict[str, str] = {}
    for stat_type, mapping in worldfootballr.COLUMN_MAP.items():
        for destination in mapping.values():
            assert destination not in seen, (
                f"{destination!r} produced by both {seen[destination]!r} and {stat_type!r}"
            )
            seen[destination] = stat_type


def test_validation_rejects_duplicate_player_matches() -> None:
    frame = make_player_matches(n_matchdays=4)
    duplicated = pd.concat([frame, frame.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_player_matches(duplicated)


def test_validation_rejects_non_positive_minutes() -> None:
    frame = make_player_matches(n_matchdays=4)
    frame.loc[0, "Min"] = 0.0
    with pytest.raises(ValueError, match="non-positive minutes"):
        validate_player_matches(frame)


def test_validation_rejects_negative_counts() -> None:
    frame = make_player_matches(n_matchdays=4)
    frame.loc[0, "Fls"] = -1.0
    with pytest.raises(ValueError, match="negative counts"):
        validate_player_matches(frame)


def test_validation_accepts_a_well_formed_frame() -> None:
    validate_player_matches(make_player_matches(n_matchdays=4))
