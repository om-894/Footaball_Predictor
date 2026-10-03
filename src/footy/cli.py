"""
Command line interface for the footy package.

The usual order is fetch, build, evaluate, then predict or predict-fixture. Run
`footy <command> --help` to see each command's options.
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime
from difflib import get_close_matches
from enum import Enum
from pathlib import Path
from typing import NoReturn

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from footy import features as feature_module
from footy.config import (
    COMBINED_PATH,
    COMBINED_TARGETS,
    DEFAULT_LEAGUE,
    FEATURES_PATH,
    FULL_MATCH_MINUTES,
    INTERIM_DIR,
    LEAGUES,
    PLAYER_MATCHES_PATH,
    REPORTS_DIR,
    SCRAPED_TARGETS,
    TARGETS,
    ensure_dirs,
    scraped_path,
)
from footy.datasets import season_folds, testable_seasons
from footy.ingest import build_player_matches, load_player_matches, name_key

app = typer.Typer(
    add_completion=False, help="Forecast per-match football player stats from FBref data."
)
console = Console()


class Mode(str, Enum):
    forecast = "forecast"
    known_minutes = "known-minutes"


class League(str, Enum):
    pl = "pl"
    championship = "championship"
    combined = "combined"


# history table, targets and the command that makes the table, for each --league choice
HISTORY = {
    League.pl: (PLAYER_MATCHES_PATH, TARGETS, "footy fetch"),
    League.championship: (
        scraped_path("ENG-Championship"),
        SCRAPED_TARGETS,
        "scripts/scrape_fbref.py ENG-Championship <season> then "
        "scripts/build_scraped.py ENG-Championship",
    ),
    League.combined: (COMBINED_PATH, COMBINED_TARGETS, "scripts/combine_pl.py"),
}


# --------------------------------------------------------------------------- #
# HELPERS
# --------------------------------------------------------------------------- #

def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    warnings.filterwarnings("ignore", category=FutureWarning)


def _fail(message: str) -> NoReturn:
    """Print a one-line error and stop with exit code 1."""
    console.print(f"[red]error:[/] {message}")
    raise typer.Exit(1)


def _render(frame: pd.DataFrame, title: str, highlight: str | None = None) -> None:
    """Print a results frame as a table, with the `highlight` model's row in green."""
    table = Table(title=title, header_style="bold")
    for column in frame.columns:
        table.add_column(str(column), justify="right" if column != "model" else "left")

    for _, row in frame.iterrows():
        cells = []
        for column in frame.columns:
            value = row[column]
            cells.append(f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value))
        style = "bold green" if highlight and row.get("model") == highlight else None
        table.add_row(*cells, style=style)
    console.print(table)


def _load_features() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        _fail(f"{FEATURES_PATH} not found, run `footy build` first")
    return pd.read_parquet(FEATURES_PATH)


def _load_history(league: League) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """The history table behind a --league choice with the targets it supports."""
    path, targets, how = HISTORY[league]
    if not path.exists():
        _fail(f"{path.name} not found, run {how} first")
    return pd.read_parquet(path), targets


def _parse_lineup(text: str) -> list[str]:
    """Starter names from a comma-separated list or a file with one name per line."""
    path = Path(text)
    lines = path.read_text().splitlines() if path.exists() else text.split(",")
    return [name.strip() for name in lines if name.strip()]


def _check_team(history: pd.DataFrame, team: str) -> None:
    """Stop with suggestions if a team never appears in the history."""
    teams = sorted(history["Team"].dropna().unique())
    if team not in teams:
        close = get_close_matches(team, teams, n=3, cutoff=0.5)
        if close:
            hint = f" Did you mean: {', '.join(close)}?"
        else:
            hint = " Names follow FBref's player tables."
        _fail(f"no team called {team!r} in this history.{hint}")


def _check_referee(history: pd.DataFrame, referee: str) -> None:
    """Warn if the referee has no earlier matches, since their features will be blank."""
    referees = sorted(history["Referee"].dropna().unique())
    if referee not in referees:
        close = get_close_matches(referee, referees, n=3, cutoff=0.5)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        console.print(
            f"[yellow]warning:[/] no earlier matches for referee {referee!r}, "
            f"so the referee features will be blank.{hint}"
        )


def _find_player(frame: pd.DataFrame, player: str) -> list[str]:
    """FBref spellings of a typed player name, ignoring case and accents."""
    names = frame["Player"].dropna().unique()
    key = name_key(player)
    matches = [name for name in names if name_key(name) == key]
    if not matches:
        close = [name for name in names if key in name_key(name)][:8]
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        _fail(f"no player called {player!r}.{hint}")
    return matches


def _print_player_forecast(played: pd.DataFrame, exposure: np.ndarray, distributions: dict) -> None:
    """One table per match: expected count, P(at least 1), P(at least 2) and what happened."""
    for position, (_, row) in enumerate(played.iterrows()):
        opponent = row.get("Opponent", "?")
        console.rule(f"{row['Player']}, {row['Team']} v {opponent} ({row['Match_Date'].date()})")
        console.print(
            f"expected minutes: [bold]{exposure[position]:.0f}[/] "
            f"(actually played {row['Min']:.0f})"
        )

        table = Table(header_style="bold")
        table.add_column("target")
        for column in ("E[count]", "P(>=1)", "P(>=2)", "actual"):
            table.add_column(column, justify="right")
        for name, distribution in distributions.items():
            table.add_row(
                name,
                f"{distribution.mu[position]:.2f}",
                f"{distribution.prob_at_least(1)[position]:.0%}",
                f"{distribution.prob_at_least(2)[position]:.0%}",
                f"{row[name]:.0f}",
            )
        console.print(table)


def _print_fixture_table(side: pd.DataFrame, title: str, targets: tuple[str, ...]) -> None:
    """One row per player; each target cell shows the expected count and P(at least 1)."""
    table = Table(title=title, header_style="bold", title_style="bold cyan")
    table.add_column("player", no_wrap=True)
    table.add_column("pos")
    table.add_column("mins", justify="right")
    for name in targets:
        table.add_column(name, justify="right")

    for _, row in side.iterrows():
        cells = [row["Player"], str(row["Pos"]), f"{row['exp_minutes']:.0f}"]
        cells += [f"{row[f'{name}_exp']:.2f} ({row[f'{name}_p1']:.0%})" for name in targets]
        table.add_row(*cells)
    console.print(table)
    console.print("[dim]each cell: expected count (probability of at least one)[/]")


# --------------------------------------------------------------------------- #
# COMMANDS
# --------------------------------------------------------------------------- #

@app.command()
def fetch(
    league: str = typer.Option(
        DEFAULT_LEAGUE, help=f"League to download, one of: {', '.join(LEAGUES)}."
    ),
    force: bool = typer.Option(False, help="Download again even if the cached copy is current."),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Show progress logs."),
) -> None:
    """Download the FBref mirror and referee data into the player-match table."""
    if league not in LEAGUES:
        _fail(f"unknown league {league!r}, choose from {', '.join(LEAGUES)}")
    _setup_logging(verbose)
    ensure_dirs()
    frame = build_player_matches(league=league, force=force)

    console.print(
        f"[bold green]{len(frame):,}[/] player-matches | "
        f"{frame['MatchURL'].nunique():,} matches | "
        f"{frame['Player'].nunique():,} players | "
        f"seasons {int(frame['Season_End_Year'].min())}-{int(frame['Season_End_Year'].max())}"
    )
    console.print(f"referee coverage: {frame['Referee'].notna().mean():.1%}")


@app.command()
def build(
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Show progress logs."),
) -> None:
    """Build the feature table from the player-match table."""
    _setup_logging(verbose)
    if not PLAYER_MATCHES_PATH.exists():
        _fail(f"{PLAYER_MATCHES_PATH} not found, run `footy fetch` first")
    ensure_dirs()

    frame = feature_module.build_features(load_player_matches())
    frame.to_parquet(FEATURES_PATH, index=False)

    columns = feature_module.feature_columns(frame)
    console.print(
        f"[bold green]{len(frame):,}[/] rows | {len(columns)} model features | "
        f"saved to {FEATURES_PATH}"
    )


@app.command()
def evaluate(
    target: list[str] = typer.Option(
        None, help=f"Target to score, can be repeated. Default: {' '.join(TARGETS)}."
    ),
    test_season: list[int] = typer.Option(
        None,
        help="Season to hold out as its end year (2024 means 2023/24), can be repeated. "
        "Default: all.",
    ),
    mode: Mode = typer.Option(
        Mode.forecast, case_sensitive=False,
        help="forecast uses predicted minutes; known-minutes uses the minutes actually played.",
    ),
    skip_nn: bool = typer.Option(False, help="Skip the neural network, which is much faster."),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Show progress logs."),
) -> None:
    """Backtest every model season by season and print the comparison tables."""
    unknown = sorted(set(target or []) - set(TARGETS))
    if unknown:
        _fail(f"unknown target {', '.join(unknown)}, choose from {', '.join(TARGETS)}")
    _setup_logging(verbose)
    frame = _load_features()

    usable = testable_seasons(frame)
    unusable = sorted(set(test_season or []) - set(usable))
    if unusable:
        _fail(f"can't hold out {unusable}, seasons with enough history before them: {usable}")

    # torch is slow to import, so the pipeline is only loaded by the commands that train
    from footy.pipeline import BENCHMARK, improvement_over_baseline, run_walk_forward, summarise

    ensure_dirs()
    targets = tuple(target) if target else TARGETS
    seasons = tuple(test_season) if test_season else None
    scores, predictions = run_walk_forward(
        frame, targets=targets, test_seasons=seasons, include_nn=not skip_nn
    )
    scores.to_csv(REPORTS_DIR / "fold_scores.csv", index=False)
    predictions.to_parquet(REPORTS_DIR / "predictions.parquet", index=False)

    summary = summarise(scores, mode=mode.value)
    summary.to_csv(REPORTS_DIR / f"summary_{mode.value}.csv", index=False)

    for name in targets:
        subset = summary[summary["target"] == name].drop(columns=["target"])
        columns = [c for c in ["model", "LogScore", "CRPS", "PoissonDev", "MAE", "ECE",
                               "pred_mean", "actual_mean"] if c in subset.columns]
        best = subset.sort_values("LogScore")["model"].iloc[0]
        _render(subset[columns], f"{name} ({mode.value}), lower is better", highlight=best)

    console.rule(f"Improvement over the {BENCHMARK} benchmark (LogScore, %)")
    delta = improvement_over_baseline(summary)
    console.print(delta.to_string())

    beaten = [
        name for name in targets
        if delta.loc[name].drop(BENCHMARK, errors="ignore").max() > 0
    ]
    if beaten:
        console.print(f"\n[green]Beat the benchmark on:[/] {', '.join(beaten)}")
    losing = [name for name in targets if name not in beaten]
    if losing:
        console.print(
            f"[yellow]Nothing beat the player's own recent average on:[/] {', '.join(losing)}"
        )
    console.print(
        f"\nfold_scores.csv, predictions.parquet and summary_{mode.value}.csv "
        f"saved to {REPORTS_DIR}"
    )


@app.command()
def predict(
    player: str = typer.Option(
        ..., help="Player name as FBref spells it. Case and accents are ignored."
    ),
    last: int = typer.Option(1, help="How many of the player's latest matches to show."),
    minutes: float = typer.Option(
        None, help="Assume this many minutes instead of the predicted minutes."
    ),
    verbose: bool = typer.Option(False, "--verbose/--quiet", help="Show progress logs."),
) -> None:
    """Re-score a player's latest matches using the network trained on every earlier season."""
    _setup_logging(verbose)
    frame = _load_features()
    names = _find_player(frame, player)
    is_player = frame["Player"].isin(names)

    latest_season = int(frame.loc[is_player, "Season_End_Year"].max())
    folds = season_folds(frame, test_seasons=(latest_season,))
    if not folds:
        _fail(f"not enough seasons before {latest_season} to train on")

    # torch is slow to import, so the pipeline is only loaded by the commands that train
    from footy.pipeline import run_fold

    console.print(
        f"training on everything before season {latest_season}, this takes a few minutes..."
    )
    result = run_fold(frame, folds[0], targets=TARGETS, include_nn=True)

    played = result.predictions[result.predictions["Player"].isin(names)]
    played = played.sort_values("Match_Date").tail(last)
    # the network gives the full distribution, so P(>= k) uses its fitted dispersion
    rows = (
        frame.loc[frame["MatchURL"].isin(played["MatchURL"]) & is_player]
        .set_index("MatchURL").loc[played["MatchURL"]].reset_index()
    )
    if minutes is not None:
        exposure = np.full(len(rows), minutes)
    else:
        exposure = played["pred_minutes"].to_numpy(dtype=float)

    distributions = {
        name: result.network.predict_distribution(
            rows[result.feature_columns], exposure, rows["Player"], name
        )
        for name in TARGETS
    }
    _print_player_forecast(played, exposure, distributions)


@app.command("predict-fixture")
def predict_fixture(
    home: str = typer.Option(..., help="Home team as FBref spells it, e.g. 'Queens Park Rangers'."),
    away: str = typer.Option(..., help="Away team as FBref spells it."),
    date: str = typer.Option(..., help="Kick-off date, YYYY-MM-DD."),
    league: League = typer.Option(
        League.championship, case_sensitive=False, help="Which history table to use."
    ),
    referee: str = typer.Option(None, help="Referee, if the appointment is known."),
    lineup: str = typer.Option(
        None, help="Confirmed starters, comma-separated or a file with one name per line."
    ),
    lineup_minutes: float = typer.Option(
        FULL_MATCH_MINUTES, help="Minutes to assume for each named starter."
    ),
    top: int = typer.Option(14, help="Players per side to show."),
    verbose: bool = typer.Option(False, "--verbose/--quiet", help="Show progress logs."),
) -> None:
    """Forecast every likely player in a match that hasn't been played yet."""
    _setup_logging(verbose)
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        _fail(f"date {date!r} should be YYYY-MM-DD, e.g. 2026-09-08")

    history, targets = _load_history(league)
    _check_team(history, home)
    _check_team(history, away)
    if referee:
        _check_referee(history, referee)
    names = _parse_lineup(lineup) if lineup else None

    console.print(
        f"[dim]history: {len(history):,} player-matches up to "
        f"{pd.to_datetime(history['Match_Date']).max().date()}[/]"
    )

    from footy.fixture import forecast_fixture

    try:
        result = forecast_fixture(
            history, home, away, date, targets,
            referee=referee,
            lineup=names,
            lineup_minutes=lineup_minutes if names else None,
        )
    except ValueError as exc:
        if verbose:
            raise
        _fail(str(exc))

    for team in (home, away):
        _print_fixture_table(result[result["Team"] == team].head(top), f"{team}, {date}", targets)

    if names:
        console.print(
            f"\n[green]Conditioned on the confirmed lineup[/] ({int(result['is_named'].sum())} of "
            f"{len(names)} names matched) at {lineup_minutes:.0f} minutes each. "
            "Substitutions will still move these."
        )
    else:
        console.print(
            "\n[yellow]Caveats:[/] the lineup is not known, so this is everyone who has played "
            "recently rather than a predicted XI. Expected minutes are off by about 19 minutes "
            "on average, which carries into every count above."
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"fixture_{home}_{away}_{date}.csv".replace(" ", "_")
    result.to_csv(out, index=False)
    console.print(f"full table saved to {out}")


@app.command()
def info() -> None:
    """Show which data tables are on disk and which --league choices are ready."""
    tables = sorted(INTERIM_DIR.glob("*.parquet")) if INTERIM_DIR.exists() else []
    if not tables:
        console.print("[yellow]No player-match tables yet. Run `footy fetch`.[/]")
        return

    table = Table(header_style="bold")
    for column in ("table", "rows", "first match", "last match"):
        table.add_column(column, justify="right" if column == "rows" else "left")
    for path in tables:
        dates = pd.to_datetime(pd.read_parquet(path, columns=["Match_Date"])["Match_Date"])
        first, last = str(dates.min().date()), str(dates.max().date())
        table.add_row(path.name, f"{len(dates):,}", first, last)
    console.print(table)

    for league, (path, _, how) in HISTORY.items():
        status = "ready" if path.exists() else f"missing, run {how}"
        console.print(f"predict-fixture --league {league.value}: {status}")
    console.print(f"features built: {'yes' if FEATURES_PATH.exists() else 'no, run `footy build`'}")


if __name__ == "__main__":  # pragma: no cover
    app()
