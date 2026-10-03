"""Command line checks that should fail fast, before any data is loaded or model trained."""

from __future__ import annotations

from typer.testing import CliRunner

from footy import cli
from tests.conftest import make_player_matches

runner = CliRunner()


def run(*args: str) -> tuple[int, str]:
    result = runner.invoke(cli.app, list(args))
    return result.exit_code, " ".join(result.output.split())


def test_unknown_league_is_rejected() -> None:
    code, output = run(
        "predict-fixture", "--home", "A", "--away", "B", "--date", "2026-09-02", "--league", "champ"
    )
    assert code != 0
    assert "championship" in output


def test_date_must_be_iso_format() -> None:
    code, output = run("predict-fixture", "--home", "A", "--away", "B", "--date", "02/09/2026")
    assert code == 1
    assert "YYYY-MM-DD" in output


def test_unknown_mode_is_rejected() -> None:
    code, _ = run("evaluate", "--mode", "known_minutes")
    assert code != 0


def test_unknown_target_is_rejected_before_loading_anything() -> None:
    code, output = run("evaluate", "--target", "Tackles")
    assert code == 1
    assert "unknown target Tackles" in output


def test_unknown_fetch_league_is_rejected() -> None:
    code, output = run("fetch", "--league", "EPL")
    assert code == 1
    assert "ENG-PL" in output


def test_misspelt_team_gets_a_suggestion(tmp_path, monkeypatch) -> None:
    path = tmp_path / "history.parquet"
    make_player_matches(n_matchdays=4).to_parquet(path)
    monkeypatch.setitem(cli.HISTORY, cli.League.pl, (path, cli.TARGETS, "footy fetch"))

    code, output = run(
        "predict-fixture", "--home", "Arsenl", "--away", "Chelsea", "--date", "2020-09-01",
        "--league", "pl",
    )
    assert code == 1
    assert "Did you mean: Arsenal?" in output


def test_lineup_can_be_a_list_or_a_file(tmp_path) -> None:
    assert cli._parse_lineup("Roefs, Xhaka ,") == ["Roefs", "Xhaka"]
    sheet = tmp_path / "xi.txt"
    sheet.write_text("Roefs\n\nXhaka\n")
    assert cli._parse_lineup(str(sheet)) == ["Roefs", "Xhaka"]
