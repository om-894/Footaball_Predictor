"""Forecast a fixture that has not been played yet.

Everything else in this package scores matches that already happened, where the row
exists and only the target is hidden. A real upcoming fixture has no row at all, so we
synthesise one per candidate player and let the causal feature pipeline fill it from that
player's history. Because every feature is built from strictly earlier matches, a row with
no result is not a special case -- it is just a row whose history happens to be all of it.

Two honest caveats, both stated in the output rather than buried:

* **We do not know the lineup.** The squad is taken to be everyone who has appeared for
  the club recently; the minutes model then supplies expected minutes for each. A player
  who is dropped or injured will still appear in the table with a plausible-looking
  number. Team news resolves this about an hour before kickoff.
* **Expected minutes carry real error** -- around 19 minutes MAE in Premier League
  backtesting -- and that error flows into every count below it.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from footy.config import FULL_MATCH_MINUTES
from footy.evaluate import CountDistribution
from footy.features import build_features, feature_columns
from footy.ingest import add_position_columns
from footy.models.gbm import PoissonGBM
from footy.models.minutes import MinutesModel

log = logging.getLogger(__name__)

#: A player is considered available if they have played for the club within this many
#: days. Wide enough to survive a rotation or a minor knock, tight enough to drop players
#: who have left.
SQUAD_WINDOW_DAYS = 75


def candidate_squad(
    history: pd.DataFrame, team: str, as_of: pd.Timestamp, window_days: int = SQUAD_WINDOW_DAYS
) -> pd.DataFrame:
    """Players who have appeared for ``team`` recently enough to be plausible starters."""
    recent = history[
        (history["Team"] == team)
        & (history["Match_Date"] < as_of)
        & (history["Match_Date"] >= as_of - pd.Timedelta(days=window_days))
    ]
    if recent.empty:
        raise ValueError(
            f"No appearances for {team!r} in the {window_days} days before {as_of.date()}. "
            "Either the team name is wrong or the history does not reach this fixture."
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
    """One synthetic player-match row per candidate player for both sides."""
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
                # Placeholder exposure. The row's own features come from shifted history,
                # and `Min` is never a model feature -- it enters only as an offset, which
                # the minutes model supplies. Nothing downstream reads this value.
                "Min": FULL_MATCH_MINUTES,
            })

    frame = pd.DataFrame(rows)
    frame["Season_End_Year"] = history["Season_End_Year"].max()
    frame["Matchweek"] = history["Matchweek"].max()
    frame["Competition_Name"] = history["Competition_Name"].iloc[0]

    # same position columns as ingest, so the history and fixture rows concatenate cleanly
    return add_position_columns(frame)


def match_lineup(candidates: pd.DataFrame, lineup: list[str]) -> pd.DataFrame:
    """Restrict candidates to a confirmed lineup, matching names loosely.

    Team sheets are typed by hand and rarely carry FBref's exact spelling -- accents get
    dropped, first names abbreviated. A surname match is enough to be unambiguous within
    one squad, and anything unmatched is reported rather than silently ignored.
    """
    from unicodedata import normalize

    def key(name: str) -> str:
        stripped = normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
        return stripped.casefold().strip()

    known = {key(p): p for p in candidates["Player"]}
    surnames: dict[str, list[str]] = {}
    for k, original in known.items():
        surnames.setdefault(k.split()[-1] if k.split() else k, []).append(original)

    resolved, missing = [], []
    for raw in lineup:
        k = key(raw)
        if k in known:
            resolved.append(known[k])
            continue
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

    # Team news for the two sides rarely lands together. Filter only the teams the lineup
    # actually covers; a side with no named players keeps its full candidate list and its
    # modelled minutes, rather than vanishing from the output.
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
    """Fit on everything before kickoff, then forecast every candidate player.

    Uses ``PoissonGBM``, which won or tied for the win on most targets in the Premier
    League backtest, and needs no validation fold to be useful.

    Pass ``lineup`` once team news is out. That is worth far more than any modelling
    change: the minutes model is the largest single error source in a forecast, and it is
    at its worst for cup ties, where selection stops resembling league football. Naming
    the starters replaces a prediction with a fact.
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

    # Targets are absent for an unplayed match; zero-fill so the shared feature pipeline
    # runs. These values never reach a feature -- the causal shift excludes the row from
    # its own history, and `feature_columns` bans same-match stats outright.
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
        # A named starter is assumed to play the full match unless told otherwise. Blunt,
        # but far closer to the truth than a league-trained minutes model applied to a
        # rotated cup XI. Players on a side with no team news keep their modelled minutes.
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
