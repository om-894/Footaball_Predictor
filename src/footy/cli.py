"""Command line entry points: fetch -> build -> evaluate -> predict."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from footy import features as feature_module
from footy.config import (
    DEFAULT_LEAGUE,
    FEATURES_DIR,
    FULL_MATCH_MINUTES,
    REPORTS_DIR,
    TARGETS,
    ensure_dirs,
)
from footy.ingest import build_player_matches, load_player_matches
from footy.pipeline import improvement_over_baseline, run_walk_forward, summarise

app = typer.Typer(add_completion=False, help="Football player-prop forecasting.")
console = Console()

FEATURES_PATH = FEATURES_DIR / "player_features.parquet"


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    warnings.filterwarnings("ignore", category=FutureWarning)


def _render(frame: pd.DataFrame, title: str, highlight: str | None = None) -> None:
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


@app.command()
def fetch(
    league: str = typer.Option(DEFAULT_LEAGUE, help="League code, e.g. ENG-PL."),
    force: bool = typer.Option(False, help="Re-download even if the cache is current."),
    verbose: bool = typer.Option(True, "--verbose/--quiet"),
) -> None:
    """Download FBref and football-data sources into a validated player-match table."""
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
def build(verbose: bool = typer.Option(True, "--verbose/--quiet")) -> None:
    """Turn the player-match table into the causal feature store."""
    _setup_logging(verbose)
    ensure_dirs()

    frame = feature_module.build_features(load_player_matches())
    frame.to_parquet(FEATURES_PATH, index=False)

    columns = feature_module.feature_columns(frame)
    console.print(
        f"[bold green]{len(frame):,}[/] rows | {len(columns)} model features "
        f"-> {FEATURES_PATH}"
    )


def _load_features() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise typer.BadParameter(f"{FEATURES_PATH} not found. Run `footy build` first.")
    return pd.read_parquet(FEATURES_PATH)


@app.command()
def evaluate(
    target: list[str] = typer.Option(None, help="Targets to score. Defaults to all."),
    test_season: list[int] = typer.Option(None, help="Seasons to hold out."),
    mode: str = typer.Option("forecast", help="'forecast' or 'known-minutes'."),
    skip_nn: bool = typer.Option(False, help="Skip the neural net (much faster)."),
    verbose: bool = typer.Option(True, "--verbose/--quiet"),
) -> None:
    """Run the walk-forward comparison across the whole model ladder."""
    _setup_logging(verbose)
    ensure_dirs()

    frame = _load_features()
    targets = tuple(target) if target else TARGETS
    seasons = tuple(test_season) if test_season else None

    scores, predictions = run_walk_forward(
        frame, targets=targets, test_seasons=seasons, include_nn=not skip_nn
    )
    scores.to_csv(REPORTS_DIR / "fold_scores.csv", index=False)
    predictions.to_parquet(REPORTS_DIR / "predictions.parquet", index=False)

    summary = summarise(scores, mode=mode)
    summary.to_csv(REPORTS_DIR / f"summary_{mode}.csv", index=False)

    for name in targets:
        subset = summary[summary["target"] == name].drop(columns=["target"])
        columns = [c for c in ["model", "LogScore", "CRPS", "PoissonDev", "MAE", "ECE",
                               "pred_mean", "actual_mean"] if c in subset.columns]
        best = subset.sort_values("LogScore")["model"].iloc[0]
        _render(subset[columns], f"{name} ({mode}) - lower is better", highlight=best)

    console.rule("Improvement over the PlayerEWMA benchmark (LogScore, %)")
    delta = improvement_over_baseline(summary)
    console.print(delta.to_string())

    beaten = [
        name for name in targets
        if delta.loc[name].drop("PlayerEWMA", errors="ignore").max() > 0
    ]
    if beaten:
        console.print(f"\n[green]Beat the benchmark on:[/] {', '.join(beaten)}")
    losing = [name for name in targets if name not in beaten]
    if losing:
        console.print(
            f"[yellow]Nothing beat a plain player average on:[/] {', '.join(losing)}"
        )
    console.print(f"\nwrote {REPORTS_DIR}/summary_{mode}.csv")


@app.command()
def predict(
    player: str = typer.Option(..., help="Player name, as spelled by FBref."),
    last: int = typer.Option(1, help="How many of the player's most recent matches."),
    minutes: float = typer.Option(None, help="Assume this many minutes instead."),
    verbose: bool = typer.Option(False, "--verbose/--quiet"),
) -> None:
    """Forecast a player's next match as a distribution, not a point estimate."""
    _setup_logging(verbose)

    frame = _load_features()
    rows = frame[frame["Player"].str.casefold() == player.casefold()]
    if rows.empty:
        candidates = (
            frame.loc[frame["Player"].str.contains(player, case=False, na=False), "Player"]
            .drop_duplicates().head(8).tolist()
        )
        hint = f" Did you mean: {', '.join(candidates)}?" if candidates else ""
        raise typer.BadParameter(f"No player named {player!r}.{hint}")

    from footy.datasets import season_folds
    from footy.pipeline import run_fold

    latest_season = int(rows["Season_End_Year"].max())
    folds = season_folds(frame, test_seasons=(latest_season,))
    if not folds:
        raise typer.BadParameter("Not enough history to fit a model.")

    console.print(f"Training on everything before season {latest_season}...")
    result = run_fold(frame, folds[0], targets=TARGETS, include_nn=True)

    subset = result.predictions
    subset = subset[subset["Player"].str.casefold() == player.casefold()]
    if subset.empty:
        raise typer.BadParameter(
            f"{player} has no matches in the held-out season {latest_season}."
        )

    if result.network is None:
        raise typer.BadParameter("The network did not train; cannot produce a forecast.")

    subset = subset.sort_values("Match_Date").tail(last)
    # Ask the fitted network for the real predictive distribution rather than rebuilding
    # one from the stored mean -- the dispersion is a fitted parameter, and guessing it
    # would misstate every P(>= k) in the table.
    rows = frame.loc[
        frame["MatchURL"].isin(subset["MatchURL"])
        & (frame["Player"].str.casefold() == player.casefold())
    ].set_index("MatchURL").loc[subset["MatchURL"]].reset_index()

    exposure = (
        np.full(len(rows), minutes)
        if minutes is not None
        else subset["pred_minutes"].to_numpy(dtype=float)
    )
    distributions = {
        name: result.network.predict_distribution(
            rows[result.feature_columns], exposure, rows["Player"], name
        )
        for name in TARGETS
    }

    for position, (_, row) in enumerate(subset.iterrows()):
        console.rule(
            f"{row['Player']} - {row['Team']} vs {row.get('Opponent', '?')} "
            f"({row['Match_Date'].date()})"
        )
        console.print(
            f"expected minutes: [bold]{exposure[position]:.0f}[/] "
            f"(actually played {row['Min']:.0f})"
        )

        table = Table(header_style="bold")
        table.add_column("target")
        table.add_column("E[count]", justify="right")
        table.add_column("P(>=1)", justify="right")
        table.add_column("P(>=2)", justify="right")
        table.add_column("actual", justify="right")

        for name, distribution in distributions.items():
            table.add_row(
                name,
                f"{distribution.mu[position]:.2f}",
                f"{distribution.prob_at_least(1)[position]:.0%}",
                f"{distribution.prob_at_least(2)[position]:.0%}",
                f"{row[name]:.0f}",
            )
        console.print(table)


@app.command("predict-fixture")
def predict_fixture(
    home: str = typer.Option(..., help="Home team, as FBref spells it."),
    away: str = typer.Option(..., help="Away team."),
    date: str = typer.Option(..., help="Kick-off date, YYYY-MM-DD."),
    league: str = typer.Option("championship", help="'championship', 'pl', or 'combined'."),
    referee: str = typer.Option(None, help="Referee, if the appointment is known."),
    lineup: str = typer.Option(
        None,
        help="Confirmed starters, comma-separated or a path to a file with one name per "
             "line. Removes the largest error source in the forecast.",
    ),
    lineup_minutes: float = typer.Option(
        90.0, help="Minutes to assume for a named starter."
    ),
    top: int = typer.Option(14, help="Players per side to show."),
    verbose: bool = typer.Option(False, "--verbose/--quiet"),
) -> None:
    """Forecast an upcoming fixture that has not been played yet."""
    _setup_logging(verbose)

    from footy.championship import CHAMPIONSHIP_TARGETS, load_championship_matches
    from footy.config import INTERIM_DIR
    from footy.fixture import forecast_fixture

    choice = league.lower()
    if choice in {"championship", "efl", "eng-championship"}:
        history = load_championship_matches()
        targets = CHAMPIONSHIP_TARGETS
    elif choice == "combined":
        # Mirrored history plus the live current-season scrape. Narrower schema than the
        # mirror alone -- see scripts/combine_pl.py -- but it is the only source that
        # covers both deep history and current squads.
        path = INTERIM_DIR / "pl_combined_player_matches.parquet"
        if not path.exists():
            raise typer.BadParameter(
                f"{path} not found. Run scripts/combine_pl.py first."
            )
        history = pd.read_parquet(path)
        targets = ("Sh", "SoT", "Fls", "Fld", "CrdY")
    else:
        history = load_player_matches()
        targets = TARGETS

    names: list[str] | None = None
    if lineup:
        candidate = Path(lineup)
        if candidate.exists():
            names = [n.strip() for n in candidate.read_text().splitlines() if n.strip()]
        else:
            names = [n.strip() for n in lineup.split(",") if n.strip()]
        console.print(f"[dim]lineup supplied: {len(names)} players[/]")

    console.print(
        f"[dim]history: {len(history):,} player-matches to "
        f"{pd.to_datetime(history['Match_Date']).max().date()}[/]"
    )

    result = forecast_fixture(
        history, home, away, date, targets,
        referee=referee,
        lineup=names,
        lineup_minutes=lineup_minutes if names else None,
    )

    for team in (home, away):
        side = result[result["Team"] == team].head(top)
        table = Table(
            title=f"{team} — {date}", header_style="bold", title_style="bold cyan"
        )
        table.add_column("player", no_wrap=True)
        table.add_column("pos")
        table.add_column("mins", justify="right")
        # One column per target holding "expected count (probability of at least one)".
        # Two columns each made the table unreadable at any sane terminal width; the CSV
        # carries the split values plus P(>=2).
        for name in targets:
            table.add_column(name, justify="right")

        for _, row in side.iterrows():
            cells = [row["Player"], str(row["Pos"]), f"{row['exp_minutes']:.0f}"]
            for name in targets:
                cells.append(f"{row[f'{name}_exp']:.2f} ({row[f'{name}_p1']:.0%})")
            table.add_row(*cells)
        console.print(table)
        console.print("[dim]each cell: expected count (probability of at least one)[/]")

    if names:
        console.print(
            f"\n[green]Conditioned on the confirmed lineup[/] at {lineup_minutes:.0f} "
            "minutes each. Substitutions will still move these."
        )
    else:
        console.print(
            "\n[yellow]Caveats:[/] the lineup is not known — this is everyone who has "
            "played recently, not a predicted XI. Expected minutes carry roughly ±19 "
            "minutes of error, and that propagates into every count above."
        )

    out = REPORTS_DIR / f"fixture_{home}_{away}_{date}.csv".replace(" ", "_")
    result.to_csv(out, index=False)
    console.print(f"full table -> {out}")


@app.command()
def info() -> None:
    """Summarise what is currently on disk."""
    ensure_dirs()
    try:
        frame = load_player_matches()
    except FileNotFoundError:
        console.print("[yellow]No player-match table yet. Run `footy fetch`.[/]")
        return

    console.print(f"player-matches: [bold]{len(frame):,}[/] rows")
    console.print(
        frame.groupby("Season_End_Year")
        .agg(rows=("Player", "size"), matches=("MatchURL", "nunique"))
        .to_string()
    )
    console.print(
        f"\nfeatures built: {'yes' if FEATURES_PATH.exists() else 'no'}"
        f"  ({FEATURES_PATH})"
    )


if __name__ == "__main__":  # pragma: no cover
    app()
