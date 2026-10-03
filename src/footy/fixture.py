"""
Forecasts a match that hasn't been played yet.

An upcoming match has no rows in the data, so one row is made up per likely player and
the normal feature pipeline fills it in from that player's earlier matches. The squad is
everyone who has played for the club recently, so a dropped or injured player still
appears until a confirmed lineup is passed in.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from footy.config import FULL_MATCH_MINUTES
from footy.evaluate import CountDistribution
from footy.features import build_features, feature_columns
from footy.ingest import add_position_columns, name_key
from footy.models.gbm import PoissonGBM
from footy.models.minutes import MinutesModel

log = logging.getLogger(__name__)

# a player counts as available if they have played for the club within this many days
SQUAD_WINDOW_DAYS = 75


def candidate_squad(
    history: pd.DataFrame, team: str, as_of: pd.Timestamp, window_days: int = SQUAD_WINDOW_DAYS
) -> pd.DataFrame:
    """Players who have played for `team` in the `window_days` before `as_of`, most minutes first."""
    recent = history[
        (history["Team"] == team)
        & (history["Match_Date"] < as_of)
        & (history["Match_Date"] >= as_of - pd.Timedelta(days=window_days))
    ]
    if recent.empty:
        last_played = history.loc[history["Team"] == team, "Match_Date"].max()
        if pd.isna(last_played):
            raise ValueError(f"No appearances for {team!r} anywhere in this history.")
        raise ValueError(
            f"No appearances for {team!r} in the {window_days} days before {as_of.date()} "
            f"(last played {last_played.date()}), so the history does not reach this fixture."
        )

    squad = (
        recent.sort_values("Match_Date")
        .groupby("Player", as_index=False)
        .agg(
            appearances=("Min", "size"),
            minutes=("Min", "sum"),
            last_seen=("Match_Date", "max"),
            Pos=("Pos", "last"),
            Nation=("Nation", "last"),
            Age=("Age", "last"),
        )
    )
    return squad.sort_values("minutes", ascending=False)


def build_fixture_rows(
    history: pd.DataFrame,
    home_team: str,
    away_team: str,
    kickoff: pd.Timestamp,
    *,
    referee: str | None = None,
    match_id: str = "UPCOMING",
) -> pd.DataFrame:
    """One made-up player-match row per likely player on both sides."""
    rows = []
    for team, opponent, is_home in (
        (home_team, away_team, 1),
        (away_team, home_team, 0),
    ):
        squad = candidate_squad(history, team, kickoff)
        for _, player in squad.iterrows():
            rows.append({
                "MatchURL": match_id,
                "Match_Date": kickoff,
                "Team": team,
                "Opponent": opponent,
                "Home_Team": home_team,
                "Away_Team": away_team,
                "Home_Away": "Home" if is_home else "Away",
                "is_home": is_home,
                "Player": player["Player"],
                "Pos": player["Pos"],
                "Nation": player["Nation"],
                "Age": player["Age"],
                "Referee": referee,
                "Min": FULL_MATCH_MINUTES, # placeholder, minutes are never a feature
            })

    frame = pd.DataFrame(rows)
    frame["Season_End_Year"] = history["Season_End_Year"].max()
    frame["Matchweek"] = history["Matchweek"].max()
    frame["Competition_Name"] = history["Competition_Name"].iloc[0]

    # same position columns as ingest, so the history and fixture rows concatenate cleanly
    return add_position_columns(frame)


def match_lineup(candidates: pd.DataFrame, lineup: list[str]) -> pd.DataFrame:
    """Keep the named starters, matching names on case, accents and unambiguous surnames.

    A team with no named players keeps all its candidates, since team news for the two
    sides often comes out at different times. Names that match nobody are logged.
    """
    known = {name_key(p): p for p in candidates["Player"]}
    surnames: dict[str, list[str]] = {}
    for k, original in known.items():
        surnames.setdefault(k.split()[-1] if k.split() else k, []).append(original)

    resolved, missing = [], []
    for raw in lineup:
        k = name_key(raw)
        if k in known:
            resolved.append(known[k])
            continue
        # a surname only counts if exactly one candidate has it
        matches = surnames.get(k.split()[-1] if k.split() else k, [])
        if len(matches) == 1:
            resolved.append(matches[0])
        else:
            missing.append(raw)

    if missing:
        log.warning(
            "lineup names not found in the squad (ignored): %s", ", ".join(missing)
        )
    if not resolved:
        raise ValueError(
            "none of the supplied lineup names matched the candidate squad; "
            f"known players include: {sorted(candidates['Player'])[:8]}"
        )

    out = candidates.copy()
    out["is_named"] = out["Player"].isin(resolved)

    covered = set(out.loc[out["is_named"], "Team"])
    uncovered = sorted(set(out["Team"]) - covered)
    if uncovered:
        log.info("no lineup supplied for %s; using modelled minutes", ", ".join(uncovered))

    return out[out["is_named"] | ~out["Team"].isin(covered)]


def forecast_fixture(
    history: pd.DataFrame,
    home_team: str,
    away_team: str,
    kickoff: str | pd.Timestamp,
    targets: tuple[str, ...],
    *,
    referee: str | None = None,
    lineup: list[str] | None = None,
    lineup_minutes: float | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Train on every match before kickoff, then forecast each likely player with PoissonGBM.

    With a `lineup`, named starters get `lineup_minutes` instead of the minutes model's
    guess. A team with no named players keeps its modelled minutes.
    """
    kickoff = pd.Timestamp(kickoff)
    history = history[history["Match_Date"] < kickoff].copy()
    if history.empty:
        raise ValueError(f"No history before {kickoff.date()}")

    fixture = build_fixture_rows(
        history, home_team, away_team, kickoff, referee=referee
    )
    if lineup:
        fixture = match_lineup(fixture, lineup)
        log.info("restricted to %d confirmed players", len(fixture))
    log.info(
        "forecasting %s vs %s on %s for %d candidate players",
        home_team, away_team, kickoff.date(), len(fixture),
    )

    # the fixture rows have no results yet, so their targets are zero-filled. they never
    # reach a feature, since features only use earlier rows
    combined = pd.concat([history, fixture], ignore_index=True)
    for target in targets:
        combined[target] = pd.to_numeric(combined[target], errors="coerce").fillna(0.0)

    built = build_features(combined)
    columns = feature_columns(built)

    is_fixture = built["MatchURL"] == "UPCOMING"
    train, upcoming = built[~is_fixture], built[is_fixture].copy()
    if upcoming.empty:
        raise ValueError("fixture rows vanished during feature construction")

    minutes_train = train["Min"].to_numpy(dtype=float)
    minutes_model = MinutesModel(seed=seed).fit(train[columns], minutes_train)
    expected_minutes = np.clip(
        minutes_model.predict(upcoming[columns]), 1.0, FULL_MATCH_MINUTES
    )
    if lineup_minutes is not None and "is_named" in upcoming.columns:
        named = upcoming["is_named"].to_numpy(dtype=bool)
        expected_minutes = np.where(named, float(lineup_minutes), expected_minutes)

    keep = ["Team", "Opponent", "Player", "Pos", "prior_appearances"]
    if "is_named" in upcoming.columns:
        keep.append("is_named")
    out = upcoming[keep].copy()
    out["exp_minutes"] = expected_minutes

    for target in targets:
        model = PoissonGBM(seed=seed).fit(
            train[columns], train[target].to_numpy(dtype=float), minutes_train
        )
        distribution: CountDistribution = model.predict_distribution(
            upcoming[columns], expected_minutes
        )
        out[f"{target}_exp"] = distribution.mu
        out[f"{target}_p1"] = distribution.prob_at_least(1)
        out[f"{target}_p2"] = distribution.prob_at_least(2)

    return out.sort_values("exp_minutes", ascending=False).reset_index(drop=True)
