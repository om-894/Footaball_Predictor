# Handover

State of the project as of 2026-09-16, written so someone can pick it up cold — including
you in three months. The README explains *what* the project is; this explains *where
things are*, *what's fragile*, and *what I'd do next*.

Branch: `v2-rebuild`, three commits ahead of `main`, working tree clean, 78 tests green.
Nothing is pushed.

---

## 1. What exists and what it does

A player-prop forecaster: shots, shots on target, fouls committed, fouls won, cards, and
tackles, as calibrated probability distributions rather than point estimates. Two ways to
use it:

| command | what it answers |
|---|---|
| `footy evaluate` | "How good is each model?" — walk-forward backtest on 6 held-out seasons |
| `footy predict-fixture` | "What happens in *this upcoming match*?" — including one that hasn't been played |

The second is the one that actually got used. It was run for two real fixtures:

- **QPR vs Cardiff, 2026-09-02** (Championship) — on 44 matches of current-season data
- **Sunderland vs Hull, 2026-09-08** (EFL Cup) — on 93,718 rows, then re-run against the
  confirmed XIs

Outputs are in `reports/fixture_*.csv`. I don't know how either match went — check the
predictions against what happened; that's the single most useful thing to do next (§6).

---

## 2. Data: what's on disk, and how stale it is

Everything under `data/` and `reports/` is gitignored and regenerable. Sizes as of today:

| path | rows | covers | how it got there |
|---|---|---|---|
| `data/interim/player_matches.parquet` | 81,327 | PL 2017/18 → **Sept 2025** | `footy fetch` (mirror) |
| `data/interim/ENG-Premier-League_player_matches.parquet` | 12,391 | PL 2025/26 + 2026/27 to 6 Sept | live scrape |
| `data/interim/pl_combined_player_matches.parquet` | **93,718** | both of the above, joined | `scripts/combine_pl.py` |
| `data/interim/champ_player_matches.parquet` | 1,376 | Championship 2026/27, 44 matches | live scrape |
| `data/features/player_features.parquet` | 81,327 × 450 | feature store for `evaluate` | `footy build` |

**The mirror stopped updating in September 2025 and won't restart.** Anything more recent
comes from the live scraper. The combined PL table is missing exactly one match
(Tottenham v Man City, 1 Feb 2026 — FBref's page for it won't parse).

**Two sources, two schemas.** The mirror carries the full FBref table set (xG, touches,
passes, 317 features). The live scrape gets only the match *summary* — no xG, touches,
passes or carries. `combine_pl.py` joins on the intersection, so the combined table is
narrower than the mirror alone. That's the price of current data; it's stated in the
script's docstring.

**soccerdata's page cache** is at `~/soccerdata/data/FBref` (118 MB). Re-scraping a season
that's cached is nearly free; that's why re-running the scraper is also the retry pass.

**Championship needs a one-off config** at `~/soccerdata/config/league_dict.json` — it's
there now, but it's outside the repo, so a fresh machine needs it again:

```json
{"ENG-Championship": {"FBref": "EFL Championship", "MatchHistory": "E1",
                      "season_start": "Aug", "season_end": "May"}}
```

Note the name. FBref calls it **"EFL Championship"**. `"Championship"` returns zero rows
with no error.

---

## 3. Running it

```bash
pip install -e ".[dev]"          # core
pip install -e ".[live]"         # + soccerdata, for anything after Sept 2025
pytest -q                        # 78 tests, ~45s
```

Backtest (Premier League, mirror data — no scraping):

```bash
footy fetch && footy build && footy evaluate
```

Forecast an upcoming Premier League fixture with current data:

```bash
scripts/refresh_pl.sh "Home Team" "Away Team" 2026-09-20
```

That scrapes anything missing (resumes; retries earlier failures), rebuilds, combines with
the mirror, and forecasts. Roughly **10.6 s per match** — FBref's rate limit, not the code.
A full season is ~65 minutes; a top-up of a few matchweeks is a couple of minutes.

Once team news is out:

```bash
footy predict-fixture --home Sunderland --away "Hull City" --date 2026-09-08 \
    --league combined --lineup "Isidor, Rigg, Hume, Talbi, Targett"
```

Names match on case, accents and unambiguous surnames. Unmatched names are **warned
about, never silently dropped**; a surname two players share is refused rather than
guessed. Supplying one side's XI is fine — that side gets fixed 90-minute exposure, the
other keeps modelled minutes.

Team names must be as FBref spells them in the *player* table: `Queens Park Rangers`,
`Cardiff City`, `Sunderland`, `Hull City`. (The schedule table abbreviates — see §5.)

---

## 4. What the backtest says — honestly

Six-fold walk-forward, log-score, lower is better. Full tables in the README.

| target | winner | NegBinMLP | PlayerEWMA (benchmark) |
|---|---|---|---|
| Sh | PoissonGBM 0.748 | 0.755 | 0.944 |
| SoT | PoissonGBM 0.395 | 0.402 | 0.551 |
| Fls | ShrunkCareerRate 1.112 | 1.127 | 1.342 |
| Fld | ShrunkCareerRate 1.087 | 1.097 | 1.294 |
| CrdY | PoissonGBM 0.386 | 0.392 | 0.606 |
| Tkl | ShrunkCareerRate 1.365 | 1.374 | 1.602 |

**The neural net never wins outright.** LightGBM takes the attacking targets; an
empirical-Bayes career average takes fouls and tackles. Everything beats the naive
recent-form baseline comfortably. For fouls specifically, the whole ladder sits within
about 2% of a positional mean — they're close to irreducible from this data.

`predict-fixture` uses `PoissonGBM`, which won or placed second everywhere and needs no
validation fold. Out-of-sample on 2026 matches it's calibrated to within 3% on shots.

---

## 5. Things that will bite you

Every one of these cost real time. In rough order of how likely you are to hit them:

**FBref blocks plain HTTP.** `requests.get` gets a Cloudflare "Just a moment..." page.
This is why the v1 scripts in `legacy/` cannot run. `soccerdata` drives a real browser
and gets through — don't try to replicate that with `requests`.

**soccerdata spells clubs two different ways.** Schedule table: `QPR`, `Blackburn`,
`West Brom`. Player table: `Queens Park Rangers`, `Blackburn Rovers`, `West Bromwich
Albion`. Comparing them directly marked 7 of 24 Championship clubs as permanently away —
869 away rows to 507 home. `championship.resolve_home_away` fixes it per match by string
similarity. If you add a league, check the home/away balance is ~0.50 before trusting
anything downstream.

**Live scraping is fragile in two specific ways.** The browser driver occasionally drops
its localhost connection (soccerdata retries; it recovered on its own). And a single
unparseable match page used to cost the whole 25-match chunk around it — now it retries
match-by-match on failure. If a chunk fails, just re-run the scraper.

**macOS OpenMP deadlock.** LightGBM and PyTorch both link `@rpath/libomp.dylib` and pip
ships multiple copies. Run both multi-threaded in one process and the pipeline hangs at 0%
CPU printing nothing. `models/nn.py` caps torch to one thread on Darwin — and note that
capping *globally* via `OMP_NUM_THREADS` makes LightGBM segfault instead, so the fix has
to be torch-side. Single-threaded torch measured faster anyway.

**`.gitignore` had `models/` unanchored** and was silently excluding `src/footy/models/`
— all seven model files. Fixed to `/models/`. Worth knowing the failure mode exists.

**The published FBref `summary` asset only covers 2 of 8 seasons.** Anchoring the join on
it discards 90% of the history. The pipeline anchors on `misc` and derives shots from the
shot-level file instead. Don't "simplify" this.

**Per-90 rates are a trap for short appearances.** One foul in a five-minute cameo is 18
per 90. The form features use smoothed counts over smoothed minutes, not smoothed rates;
`NaivePer90EWMA` exists in the ladder purely to show what the naive version costs (it's
last on every target).

---

## 6. Limitations to keep in mind when reading a forecast

**The model can't see who'll be picked.** The data contains only players who appeared, so
`P(selected)` isn't identifiable. The minutes model answers "how long, if picked". For a
league game that's a modest error (~19 min MAE); for a **cup tie it's the dominant one** —
Sunderland made nine changes for Hull, and three of my four pre-lineup legs were players
who ended up on the bench. Always re-run with `--lineup` once team news is out.

**Probability correlates with lack of data.** A player with one substitute appearance
falls back to a positional prior, and for an attacker that prior is high. The eye-catching
numbers on a team sheet are often the least-evidenced ones. Filter on
`prior_appearances` (≥10 is a reasonable bar) before trusting a leg. Hull's cup XI had
eight of eleven players with ≤3 appearances.

**Two strikers competing for one shirt are modelled as independent.** The model gave
Isidor 94% and Brobbey 91% for 1+ shots; only one was ever going to start. It doesn't
know that. You do.

**Fixed 90 minutes is optimistic.** `--lineup` assumes every starter finishes. Young
players in a cup tie often come off around the hour; that pushes every count down and
hits the marginal legs hardest.

**Hull have no history.** Newly promoted, so the PL archive doesn't contain them. Only
Targett (131 apps, from earlier clubs) had real data. A Championship 2025/26 scrape
(~557 matches, ~95 min) would fix this for any promoted side.

---

## 7. What I'd do next

In order of value:

1. **Score the two forecasts against what actually happened.** `reports/fixture_*.csv`
   has every prediction; FBref has the results. This is the first real out-of-sample test
   of the fixture path and it costs an hour.

2. **Scrape Championship 2025/26** so promoted clubs arrive with history.
   `python scripts/scrape_fbref.py ENG-Championship 2526` — long, but cached forever.

3. **Wire `read_lineup()` in.** soccerdata exposes FBref lineups *including unused subs*,
   which makes `P(selected)` identifiable and would let the minutes model stop being
   conditional on appearing. `sources/fbref_live.py` already has the reader; it's not
   plumbed into features yet.

4. **Give the fixture path a `--model` flag.** It's hardcoded to `PoissonGBM`. The
   backtest says `ShrunkCareerRate` is marginally better for fouls; it'd be a small change
   to let the fixture forecast use the per-target winner.

5. **Persist trained models.** `footy predict` retrains from scratch every call (~3 min).
   A saved-model path would make it usable interactively.

Longer-term, the thing the backtest is actually telling you: on fouls, nothing beats a
shrunk average by much. If the goal is fouls specifically, the win is more likely in
*data* (referee tendencies are in, but opponent style, match state and derby intensity
aren't) than in model architecture.

---

## 8. Layout, briefly

```
src/footy/            the package — see README for the module map
scripts/              thin wrappers for live data: scrape, build, combine, refresh
tests/                78 tests; the causality suite is the one that matters
legacy/               v1, archived with a README explaining why it can't run
data/legacy/          the original hand-assembled Sunderland scrape (committed; fouls were typed by hand)
reports/              backtest tables + fixture forecasts (gitignored)
```
