"""Second-tier ingest, and the club-name mismatch that silently broke home/away."""

from __future__ import annotations

import pandas as pd

from footy.championship import CHAMPIONSHIP_TARGETS, build_championship_matches, resolve_home_away


def _frame(pairs: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    """(match_id, player-table team, other team, schedule's home-team spelling)."""
    rows = []
    for match_id, team_a, team_b, scheduled_home in pairs:
        for team in (team_a, team_b):
            for slot in range(3):
                rows.append({
                    "MatchURL": match_id,
                    "Team": team,
                    "Home_Team": scheduled_home,
                    "Player": f"{team} {slot}",
                })
    return pd.DataFrame(rows)


def test_home_away_survives_abbreviated_club_names() -> None:
    """The bug this exists to prevent.

    soccerdata's schedule says "QPR" and "Blackburn"; its player stats say "Queens Park
    Rangers" and "Blackburn Rovers". Comparing the two directly marked 7 of 24 clubs as
    permanently away -- 869 away rows against 507 home, with 15 matches having no home
    side at all -- and every home/away and opponent feature inherited the error.
    """
    frame = resolve_home_away(_frame([
        ("m1", "Queens Park Rangers", "Bolton Wanderers", "QPR"),
        ("m2", "Blackburn Rovers", "Queens Park Rangers", "Blackburn"),
        ("m3", "West Bromwich Albion", "Preston North End", "West Brom"),
    ]))

    for match_id, expected_home in (
        ("m1", "Queens Park Rangers"),
        ("m2", "Blackburn Rovers"),
        ("m3", "West Bromwich Albion"),
    ):
        match = frame[frame["MatchURL"] == match_id]
        home = set(match.loc[match["is_home"] == 1, "Team"])
        assert home == {expected_home}, f"{match_id}: got {home}"


def test_every_match_has_exactly_one_home_side() -> None:
    frame = resolve_home_away(_frame([
        ("m1", "Queens Park Rangers", "Bolton Wanderers", "QPR"),
        ("m2", "Cardiff City", "Burnley", "Cardiff City"),
    ]))
    per_match = frame.groupby("MatchURL")["is_home"].mean()
    assert (per_match > 0).all() and (per_match < 1).all()


def test_opponent_uses_player_table_spelling() -> None:
    """Opponent must match `Team` exactly, or the opponent-form join drops silently."""
    frame = resolve_home_away(_frame([
        ("m1", "Queens Park Rangers", "Bolton Wanderers", "QPR"),
    ]))
    assert set(frame["Opponent"]) == {"Queens Park Rangers", "Bolton Wanderers"}
    assert frame["Opponent"].notna().all()
    assert (frame["Opponent"] != frame["Team"]).all()


def test_championship_targets_exclude_total_tackles() -> None:
    """FBref publishes tackles *won* but not total tackles for the second tier."""
    assert "Tkl" not in CHAMPIONSHIP_TARGETS
    assert "TklW" in CHAMPIONSHIP_TARGETS
    assert {"Fls", "Fld"} <= set(CHAMPIONSHIP_TARGETS)


def test_build_produces_the_standard_shape() -> None:
    players = pd.DataFrame({
        "game_id": ["g1"] * 4,
        "season": ["2627"] * 4,
        "team": ["QPR Full"] * 2 + ["Cardiff City"] * 2,
        "player": ["A", "B", "C", "D"],
        "nation": ["ENG"] * 4,
        "pos": ["CB", "FW", "GK", "MF"],
        "age": ["25-100"] * 4,
        "min": [90, 70, 90, 60],
        "Performance_Sh": [0, 3, 0, 1],
        "Performance_SoT": [0, 1, 0, 0],
        "Performance_Fls": [2, 1, 0, 3],
        "Performance_Fld": [1, 2, 0, 1],
        "Performance_CrdY": [1, 0, 0, 0],
        "Performance_TklW": [2, 0, 0, 1],
    })
    schedule = pd.DataFrame({
        "game_id": ["g1"],
        "date": ["2026-09-02"],
        "home_team": ["QPR"],
        "away_team": ["Cardiff City"],
        "week": [4],
        "referee": ["A Taylor"],
    })

    frame = build_championship_matches(players, schedule, write=False)

    assert len(frame) == 4
    assert frame["Season_End_Year"].eq(2027).all()
    assert set(frame.loc[frame["is_home"] == 1, "Team"]) == {"QPR Full"}
    assert frame["Age"].between(25, 26).all()
    for target in CHAMPIONSHIP_TARGETS:
        assert target in frame.columns
