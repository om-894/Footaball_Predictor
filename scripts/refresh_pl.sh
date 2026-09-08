#!/usr/bin/env bash
# Refresh Premier League data and re-forecast a fixture.
#
# Safe to re-run. The scraper resumes from whatever is already in the CSV, so running it
# again also retries any matches an earlier pass dropped -- no separate repair step.
#
#   scripts/refresh_pl.sh Sunderland "Hull City" 2026-09-08
#
# Add a lineup once team news is out (this is worth more than any modelling change):
#
#   footy predict-fixture --home Sunderland --away "Hull City" --date 2026-09-08 \
#       --league combined --lineup "Roefs, Xhaka, Ballard, Le Fée, ..."

set -euo pipefail
cd "$(dirname "$0")/.."

HOME_TEAM="${1:?usage: refresh_pl.sh <home> <away> <YYYY-MM-DD>}"
AWAY_TEAM="${2:?}"
DATE="${3:?}"

echo "==> scraping any missing matches (resumes; retries earlier failures)"
python3 -u scripts/scrape_fbref.py "ENG-Premier League" 2627 2526

echo "==> rebuilding the scraped player-match table"
python3 -u scripts/build_scraped.py ENG-Premier-League

echo "==> combining with the mirrored history"
python3 -u scripts/combine_pl.py

echo "==> forecasting ${HOME_TEAM} vs ${AWAY_TEAM} on ${DATE}"
python3 -u -m footy.cli predict-fixture \
    --home "${HOME_TEAM}" --away "${AWAY_TEAM}" --date "${DATE}" \
    --league combined --top 12
