"""Optional live FBref scraping via `soccerdata`.

Install with ``pip install -e ".[live]"``.

**Why this is optional.** FBref sits behind a Cloudflare challenge, so the plain
``requests.get(url, verify=False)`` the v1 scripts used returns a "Just a moment..."
holding page and nothing else. ``soccerdata`` gets through by driving a real browser
(``seleniumbase``), which is a heavyweight dependency and can break without warning. The
GitHub-release mirrors in :mod:`footy.sources.worldfootballr` are therefore the default
and this is a top-up.

**It fails loudly.** A scraper that silently returns three matches instead of thirty is
worse than one that raises, because the model trains on the short dataset and nobody
notices. Every failure here raises :class:`SourceError`.

Two things only this path can provide:

* matches more recent than the mirrors, which stop around September 2025
* ``read_lineup()``, which lists unused substitutes -- the only way to observe that a
  player was available and *not* selected, and so the only route to a genuine selection
  model rather than one conditional on appearing
"""

from __future__ import annotations

import logging
import re
import time

import pandas as pd

from footy.sources.base import SourceError

log = logging.getLogger(__name__)

#: FBref asks for no more than ten requests a minute. Six seconds is inside that with
#: room to spare; do not lower it.
REQUEST_DELAY_S = 6.0

#: soccerdata's league names differ from ours.
LEAGUE_NAMES = {
    "ENG-PL": "ENG-Premier League",
    "ESP-LaLiga": "ESP-La Liga",
    "FRA-Ligue1": "FRA-Ligue 1",
    "GER-Bundesliga": "GER-Bundesliga",
    "ITA-SerieA": "ITA-Serie A",
}

_UNNAMED_LEVEL = re.compile(r"Unnamed:.*?_level_0_")


def flatten_fbref_columns(columns) -> list[str]:
    """Flatten FBref's two-level table headers.

    Carried over from the v1 scraper ([legacy/matchday_1_to_10.py]) -- it is the one part
    of it worth keeping. FBref renders stat tables with a grouping row above the real
    header, which pandas reads as a MultiIndex whose upper level is "Unnamed: 4_level_0"
    for ungrouped columns.
    """
    flattened = []
    for column in columns:
        joined = "_".join(str(part) for part in column).strip() if isinstance(column, tuple) else str(column)
        flattened.append(_UNNAMED_LEVEL.sub("", joined).strip())
    return flattened


def _reader(league: str, seasons: list[str] | str):
    try:
        import soccerdata
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SourceError(
            "soccerdata is not installed. Install the optional extra with:\n"
            '    pip install -e ".[live]"'
        ) from exc

    if league not in LEAGUE_NAMES:
        raise SourceError(f"No soccerdata mapping for {league!r}")

    return soccerdata.FBref(leagues=LEAGUE_NAMES[league], seasons=seasons)


def read_player_match_stats(
    league: str = "ENG-PL",
    seasons: str | list[str] = "2526",
    stat_type: str = "misc",
) -> pd.DataFrame:
    """Scrape one FBref per-match player table.

    Returns the same flat vocabulary the mirrors produce, so the result can be
    concatenated onto the cached table.
    """
    reader = _reader(league, seasons)
    log.info("scraping FBref %s %s (%s) -- this is slow by design", league, seasons, stat_type)

    try:
        frame = reader.read_player_match_stats(stat_type=stat_type)
    except Exception as exc:  # noqa: BLE001 - surface the real cause
        raise SourceError(
            f"FBref scrape failed for {league}/{seasons}/{stat_type}: {exc}\n"
            "This is usually the Cloudflare challenge. Fall back to the mirrors with "
            "`footy fetch`, which needs no scraping."
        ) from exc

    if frame is None or frame.empty:
        raise SourceError(
            f"FBref returned no rows for {league}/{seasons}/{stat_type}. "
            "Refusing to continue with an empty scrape."
        )

    frame = frame.reset_index()
    frame.columns = flatten_fbref_columns(frame.columns)
    time.sleep(REQUEST_DELAY_S)

    log.info("scraped %d rows", len(frame))
    return frame


def read_lineups(league: str = "ENG-PL", seasons: str | list[str] = "2526") -> pd.DataFrame:
    """Scrape match lineups, including unused substitutes.

    This is the piece the mirrors cannot supply. With it, ``P(selected)`` becomes
    identifiable and the minutes model stops being conditional on appearing -- see the
    limitation noted in :mod:`footy.models.minutes`.
    """
    reader = _reader(league, seasons)
    try:
        frame = reader.read_lineup()
    except Exception as exc:  # noqa: BLE001
        raise SourceError(f"FBref lineup scrape failed for {league}/{seasons}: {exc}") from exc

    if frame is None or frame.empty:
        raise SourceError(f"FBref returned no lineups for {league}/{seasons}")

    frame = frame.reset_index()
    frame.columns = flatten_fbref_columns(frame.columns)
    log.info("scraped %d lineup rows", len(frame))
    return frame
