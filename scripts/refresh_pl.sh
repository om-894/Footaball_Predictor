#!/usr/bin/env bash
# Refresh the Premier League data and forecast one fixture.
#
# Safe to run again: the scraper skips matches it already has and retries any that failed.
#
#   scripts/refresh_pl.sh Sunderland "Hull City" 2026-09-08
#
# Once the lineups are out, run the forecast again with them:
#
#   footy predict-fixture --home Sunderland --away "Hull City" --date 2026-09-08 \
#       --league combined --lineup "Roefs, Xhaka, Ballard, Le Fée, ..."

set -euo pipefail
cd "$(dirname "$0")/.."

HOME_TEAM="${1:?usage: refresh_pl.sh <home> <away> <YYYY-MM-DD>}"
AWAY_TEAM="${2:?usage: refresh_pl.sh <home> <away> <YYYY-MM-DD>}"
DATE="${3:?usage: refresh_pl.sh <home> <away> <YYYY-MM-DD>}"

echo "==> scraping any missing Premier League matches"
python3 -u scripts/scrape_fbref.py "ENG-Premier League" 2627 2526

echo "==> rebuilding the scraped player-match table"
python3 -u scripts/build_scraped.py ENG-Premier-League

echo "==> combining with the mirrored history"
python3 -u scripts/combine_pl.py

echo "==> forecasting ${HOME_TEAM} v ${AWAY_TEAM} on ${DATE}"
python3 -u -m footy.cli predict-fixture \
    --home "${HOME_TEAM}" --away "${AWAY_TEAM}" --date "${DATE}" \
    --league combined --top 12
