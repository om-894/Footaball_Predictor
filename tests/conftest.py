"""Synthetic player-match fixtures.

Deliberately synthetic rather than a sample of the real Parquet: the tests need to be
able to *change the future* and check the past is unaffected, which means generating the
frame rather than reading one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footy.config import FORM_STATS, TARGETS

TEAMS = ["Arsenal", "Chelsea", "Everton", "Fulham"]
REFEREES = ["A Taylor", "M Oliver", "P Tierney"]


def make_player_matches(
    n_matchdays: int = 24, players_per_team: int = 11, seed: int = 0
) -> pd.DataFrame:
    """Build a small but structurally faithful player-match table."""
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2020-08-01")

    for matchday in range(n_matchdays):
        date = start + pd.Timedelta(days=7 * matchday)
        # Rotate the pairings so every team meets every other.
        pairs = [(TEAMS[0], TEAMS[1]), (TEAMS[2], TEAMS[3])]
        if matchday % 2:
            pairs = [(TEAMS[1], TEAMS[2]), (TEAMS[3], TEAMS[0])]

        for home, away in pairs:
            match_url = f"https://example.test/{matchday}-{home}-{away}"
            referee = REFEREES[matchday % len(REFEREES)]
            for team, side in ((home, "Home"), (away, "Away")):
                for slot in range(players_per_team):
                    minutes = float(rng.integers(15, 91))
                    row = {
                        "MatchURL": match_url,
                        "Match_Date": date,
                        "Matchweek": float(matchday + 1),
                        "Season_End_Year": 2021 + matchday // 12,
                        "Competition_Name": "Premier League",
                        "Team": team,
                        "Opponent": away if team == home else home,
                        "Home_Team": home,
                        "Away_Team": away,
                        "Home_Away": side,
                        "is_home": int(side == "Home"),
                        "Player": f"{team} Player {slot}",
                        "Nation": "eng ENG",
                        "Pos": "GK" if slot == 0 else ["DF", "MF", "FW"][slot % 3],
                        "Age": 20.0 + slot,
                        "Min": minutes,
                        "Referee": referee,
                    }
                    row["positions"] = [row["Pos"]]
                    row["is_gk"] = int(row["Pos"] == "GK")
                    for stat in sorted(set(FORM_STATS) | set(TARGETS)):
                        row[stat] = float(rng.poisson(1.0))
                    rows.append(row)

    frame = pd.DataFrame(rows)
    return frame.sort_values(["Match_Date", "MatchURL", "Team", "Player"]).reset_index(
        drop=True
    )


@pytest.fixture
def player_matches() -> pd.DataFrame:
    return make_player_matches()
