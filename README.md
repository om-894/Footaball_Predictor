# Football Predictor

Forecasts what a footballer will do in their **next** match — shots, shots on target,
fouls committed, fouls won, cards, tackles — as a probability distribution rather than a
single number.

```
Chris Rigg — Sunderland vs Crystal Palace
expected minutes: 63

  target   E[count]   P(>=1)   P(>=2)
  Sh           0.71      45%      17%
  Fls          0.84      51%      21%
  Fld          0.79      49%      20%
  CrdY         0.11       10%       1%
```

`P(fouls >= 1) = 51%` is something you can act on. "Predicted fouls: 0.84" is not.

Built on **81,327 player-matches** from 2,857 Premier League games (2017/18–2025/26),
using FBref's per-match player tables.

---

## What this is

This is v2 of a project that started as a small PyTorch model for Sunderland players. The
original worked end to end, but it had a problem no amount of tuning could fix: it
regressed a match's fouls on *that same match's* touches, passes and xG. None of that is
known before kickoff, so its accuracy was not something anyone could ever realise in
practice.

The rebuild is organised around three ideas.

**1. Every feature is strictly causal.** A feature for a match on date *t* is computed
only from matches played before *t*. This is enforced by a property test that scrambles
all data from a cutoff onward and asserts that no feature before the cutoff moves
([`tests/test_causality.py`](tests/test_causality.py)). It is the single most valuable
test in the repo.

**2. Counts are modelled as counts.** Fouls in this dataset have mean 0.75, variance 0.95
and are zero 56% of the time. Standardising them and minimising squared error — what v1
did — assumes a symmetric, unbounded target and will happily predict negative shots. Here
everything is a negative binomial with a **log-minutes offset**, so a player who plays
twice as long is expected to do twice as much, exactly rather than approximately.

**3. Nothing is believed without a baseline.** Every model is scored against a plain
exponentially weighted average of the player's own recent form — roughly what an analyst
does by eye. The comparison table below is the point of the project, and it reports
losses as readily as wins.

---

## Results

Walk-forward across six held-out seasons: train on everything earlier, validate on the
preceding season, test on the next. Never a random split.

Six folds (2021, 2022, 2023, 2024, 2025, 2026), 9 models, 6 targets. Lower is better throughout; **bold** is the best log-score for that target.

Stage one, the minutes model, has a mean absolute error of **18.8 minutes** across folds — that error propagates into every `forecast` number below.

#### Sh

| model | LogScore | CRPS | Poisson dev. | MAE | ECE |
|---|---|---|---|---|---|
| GlobalMean | 1.0124 | 0.4542 | 1.5850 | 0.8167 | 0.0667 |
| PositionMean | 0.9565 | 0.4247 | 1.3523 | 0.7477 | 0.0763 |
| ShrunkCareerRate | 0.8297 | 0.3493 | 0.9642 | 0.5818 | 0.0647 |
| NaivePer90EWMA | 0.9591 | 0.3666 | 1.2643 | 0.5741 | 0.0443 |
| PlayerEWMA | 0.9437 | 0.3446 | 1.1984 | 0.5419 | 0.0466 |
| PoissonGLM | 0.9618 | 0.4164 | 1.3265 | 0.7129 | 0.0739 |
| NegBinGLM | 0.9460 | 0.4171 | 1.3658 | 0.7342 | 0.0436 |
| **PoissonGBM** | 0.7477 | 0.3248 | 0.7724 | 0.5173 | 0.0493 |
| NegBinMLP | 0.7553 | 0.3278 | 0.8155 | 0.5163 | 0.0304 |

#### SoT

| model | LogScore | CRPS | Poisson dev. | MAE | ECE |
|---|---|---|---|---|---|
| GlobalMean | 0.5209 | 0.1690 | 0.7802 | 0.3379 | 0.0367 |
| PositionMean | 0.4836 | 0.1599 | 0.6775 | 0.3177 | 0.0301 |
| ShrunkCareerRate | 0.4270 | 0.1426 | 0.5438 | 0.2689 | 0.0259 |
| NaivePer90EWMA | 0.5596 | 0.1529 | 0.8268 | 0.2631 | 0.0306 |
| PlayerEWMA | 0.5510 | 0.1458 | 0.7984 | 0.2544 | 0.0313 |
| PoissonGLM | 0.4826 | 0.1579 | 0.6681 | 0.3011 | 0.0299 |
| NegBinGLM | 0.4794 | 0.1577 | 0.6698 | 0.3024 | 0.0224 |
| **PoissonGBM** | 0.3951 | 0.1373 | 0.4770 | 0.2480 | 0.0181 |
| NegBinMLP | 0.4020 | 0.1387 | 0.4940 | 0.2423 | 0.0167 |

#### Fls

| model | LogScore | CRPS | Poisson dev. | MAE | ECE |
|---|---|---|---|---|---|
| GlobalMean | 1.1821 | 0.4831 | 1.2937 | 0.7810 | 0.0695 |
| PositionMean | 1.1304 | 0.4574 | 1.1727 | 0.7237 | 0.0513 |
| **ShrunkCareerRate** | 1.1124 | 0.4464 | 1.1294 | 0.7177 | 0.0546 |
| NaivePer90EWMA | 1.3755 | 0.5207 | 1.7784 | 0.8336 | 0.0794 |
| PlayerEWMA | 1.3419 | 0.4774 | 1.6341 | 0.7628 | 0.0530 |
| PoissonGLM | 1.1664 | 0.4754 | 1.2617 | 0.7654 | 0.0635 |
| NegBinGLM | 1.1789 | 0.4838 | 1.9535 | 1.1153 | 0.0579 |
| PoissonGBM | 1.1148 | 0.4480 | 1.1315 | 0.7213 | 0.0634 |
| NegBinMLP | 1.1274 | 0.4552 | 1.1649 | 0.7387 | 0.0473 |

#### Fld

| model | LogScore | CRPS | Poisson dev. | MAE | ECE |
|---|---|---|---|---|---|
| GlobalMean | 1.1685 | 0.4859 | 1.3620 | 0.7968 | 0.0691 |
| PositionMean | 1.1298 | 0.4633 | 1.2546 | 0.7503 | 0.0519 |
| **ShrunkCareerRate** | 1.0865 | 0.4364 | 1.1419 | 0.7169 | 0.0592 |
| NaivePer90EWMA | 1.3229 | 0.4947 | 1.6945 | 0.7923 | 0.0661 |
| PlayerEWMA | 1.2938 | 0.4591 | 1.5833 | 0.7391 | 0.0614 |
| PoissonGLM | 1.1387 | 0.4684 | 1.2770 | 0.7608 | 0.0512 |
| NegBinGLM | 1.1424 | 0.4749 | 1.6693 | 0.9605 | 0.0383 |
| PoissonGBM | 1.0916 | 0.4394 | 1.1468 | 0.7215 | 0.0659 |
| NegBinMLP | 1.0974 | 0.4440 | 1.1740 | 0.7238 | 0.0369 |

#### CrdY

| model | LogScore | CRPS | Poisson dev. | MAE | ECE |
|---|---|---|---|---|---|
| GlobalMean | 0.3943 | 0.1131 | 0.5316 | 0.2216 | 0.0270 |
| PositionMean | 0.3902 | 0.1121 | 0.5234 | 0.2193 | 0.0197 |
| ShrunkCareerRate | 0.3898 | 0.1117 | 0.5226 | 0.2222 | 0.0188 |
| NaivePer90EWMA | 0.6227 | 0.1390 | 1.0238 | 0.2591 | 0.0659 |
| PlayerEWMA | 0.6062 | 0.1248 | 0.9612 | 0.2361 | 0.0568 |
| PoissonGLM | 0.3919 | 0.1126 | 0.5269 | 0.2268 | 0.0160 |
| NegBinGLM | 0.3985 | 0.1130 | 0.5269 | 0.2259 | 0.0182 |
| **PoissonGBM** | 0.3858 | 0.1108 | 0.5146 | 0.2295 | 0.0149 |
| NegBinMLP | 0.3916 | 0.1116 | 0.5199 | 0.2258 | 0.0185 |

#### Tkl

| model | LogScore | CRPS | Poisson dev. | MAE | ECE |
|---|---|---|---|---|---|
| GlobalMean | 1.4857 | 0.7111 | 1.7117 | 1.1056 | 0.0893 |
| PositionMean | 1.3978 | 0.6506 | 1.4587 | 1.0106 | 0.0645 |
| **ShrunkCareerRate** | 1.3647 | 0.6220 | 1.3581 | 0.9641 | 0.0699 |
| NaivePer90EWMA | 1.6432 | 0.7205 | 2.0954 | 1.1142 | 0.0530 |
| PlayerEWMA | 1.6016 | 0.6557 | 1.8976 | 1.0092 | 0.0504 |
| PoissonGLM | 1.4412 | 0.6797 | 1.6050 | 1.0488 | 0.0639 |
| NegBinGLM | 1.4411 | 0.6990 | 23.9408 | 12.2229 | 0.0835 |
| PoissonGBM | 1.3780 | 0.6240 | 1.3472 | 0.9582 | 0.0898 |
| NegBinMLP | 1.3741 | 0.6314 | 1.3963 | 0.9880 | 0.0485 |

Two evaluation modes, because the difference is informative:

- **`known-minutes`** conditions on the minutes actually played, isolating how well the
  *rate* model works.
- **`forecast`** is the honest pre-kickoff number: a minutes model supplies expected
  minutes and the count model works from those. It is always worse, and the gap is the
  price of not knowing the team sheet.

```bash
footy evaluate --mode forecast
```

---

## Data

**FBref now sits behind a Cloudflare bot gate.** The v1 approach —
`requests.get(url, verify=False)` then `pd.read_html` — returns a "Just a moment..."
holding page. Bulk history therefore comes from published mirrors, and live top-ups go
through `soccerdata`, which drives a real browser and rate-limits itself.

| Source | What it gives | Coverage |
|---|---|---|
| [`worldfootballR_data`](https://github.com/JaseZiv/worldfootballR_data) GitHub releases | FBref per-match player tables (`misc`, `possession`, `passing`, `defense`) and shot-level events, republished as plain CSVs | 81,327 player-matches, 2,857 matches, 2017/18–2025/26 |
| [football-data.co.uk](https://www.football-data.co.uk/) via [a GitHub mirror](https://github.com/datasets/football-datasets) | Team-level results and, uniquely, the **referee** | 33 seasons, through 2025/26 |
| `soccerdata` (optional extra) | Live FBref scraping — the only route to the current season, and the only route to the Championship at all | Current, but heavyweight |

Two details worth knowing:

**Fouls come from FBref's `misc` table** (`Fls`, `Fld`). In v1 these were typed in by hand
as index-aligned Python lists — `md1_df['Fouls_Committed'] = [1, 1, 0, 0, ...]` — repeated
across 17 files, silently wrong if FBref ever reordered a row. They were always
publishable data.

**Shots come from shot-level events**, not an aggregate table. FBref's published `summary`
asset covers only 2 of the 8 seasons, so anchoring the join on it would discard 90% of the
history. Deriving `Sh`/`SoT`/`xG` from the shot file covers everything *and* yields shot
distance and body part for free.

The referee matters more than it might seem: referees vary a lot in how freely they
whistle, and the name appears in no FBref table we load.

### The Championship

The second tier is a separate path ([`championship.py`](src/footy/championship.py)),
because neither mirror serves it usefully: the `worldfootballR_data` releases publish only
match events and shot events for `ENG_M_2nd`, and both stopped updating in January 2025.
Live scraping is the only option.

FBref serves the Championship a **narrower** table than the top flight — no `misc`,
`possession`, `passing` or `defense` tab, and no xG, touches, passes or carries in the
summary. What survives:

| | Premier League | Championship |
|---|---|---|
| targets | `Sh` `SoT` `Fls` `Fld` `CrdY` `Tkl` | `Sh` `SoT` `Fls` `Fld` `CrdY` `TklW` |
| features | 317 | ~90 |
| history | 2017/18 onward | whatever you scrape |

`Tkl` has no second-tier equivalent — FBref publishes tackles *won* only.

Setting it up needs a one-off custom-league entry, since `soccerdata` ships only the top
five leagues. Note the name: FBref calls it **"EFL Championship"**, and "Championship"
silently returns zero rows.

```json
// ~/soccerdata/config/league_dict.json
{"ENG-Championship": {"FBref": "EFL Championship", "MatchHistory": "E1",
                      "season_start": "Aug", "season_end": "May"}}
```

Worth recording: the Championship summary table **does** carry `Fls` and `Fld`. The v1
scripts typed those in by hand for all 17 Sunderland matches, and never needed to.

---

## Install and run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

```bash
footy fetch       # download and validate sources  -> data/interim/
footy build       # causal feature store           -> data/features/
footy evaluate    # walk-forward model comparison  -> reports/
footy predict --player "Chris Rigg"
```

### Forecasting a fixture that hasn't happened

`footy evaluate` and `footy predict` both score matches that already exist in the data.
An upcoming fixture has no row at all, so `predict-fixture`
([`fixture.py`](src/footy/fixture.py)) synthesises one per candidate player and lets the
causal pipeline fill it from history — which needs no special case, because a row with no
result is just a row whose history is all of it.

```bash
footy predict-fixture --home "QPR" --away "Cardiff City" --date 2026-09-02
```

Two things it cannot know, both printed with the output rather than buried:

- **The lineup.** The "squad" is everyone who has played for the club in the last 75 days,
  not a predicted XI. A dropped or injured player still appears with a plausible number.
- **Minutes.** The minutes model carries ~19 minutes MAE, and that propagates into every
  count derived from it.

Once team news is out, `--lineup` removes the second of those and most of the first —
worth more than any modelling change, and more still for a cup tie, where selection stops
resembling league football:

```bash
footy predict-fixture --home Sunderland --away "Hull City" --date 2026-09-08 \
    --league combined --lineup "Roefs, Xhaka, Ballard, Le Fée, Isidor, Brobbey"
```

Names match on case, accents and unambiguous surnames; anything unresolved is **warned
about, never silently dropped**, and a surname shared by two players is refused rather
than guessed. Supplying one side's XI is fine — that team gets fixed exposure, the other
keeps modelled minutes.

### Helper scripts

`src/footy` is the package; `scripts/` holds the thin wrappers around it for live data.

| script | what it does |
|---|---|
| `scrape_fbref.py <league> <seasons...>` | Live FBref scrape. Writes in chunks and resumes from what's already saved, so re-running it is also the retry pass. |
| `build_scraped.py <slug>` | Scraped CSVs → validated player-match Parquet |
| `combine_pl.py` | Mirrored history + live scrape → one table |
| `refresh_pl.sh <home> <away> <date>` | All of the above, then a forecast |

A full Premier League season is ~380 matches at roughly 10.6s each — about 65 minutes,
which is FBref's rate limit rather than anything slow in the code.

`footy info` shows what is currently on disk. Downloads are cached with ETags, so
re-running `fetch` costs one conditional request per file.

Optional live scraping (heavyweight — drives a real browser):

```bash
pip install -e ".[live]"
```

---

## How it works

### Two stages

Every count scales with minutes, so minutes are modelled separately and then applied as
exposure:

```
E[count] = rate_per_90(features) x (expected_minutes / 90)
```

Stage one is a quantile gradient-boosting model over minutes — bounded to [1, 90] and
sharply bimodal, since players tend to either start and finish or come off the bench, so
a regression to the conditional mean lands in the empty middle. Stage two predicts the
per-90 rate. `log(minutes/90)` enters as a fixed offset rather than a free coefficient.

### Features (317 of them, all causal)

- **Player form** — exponentially weighted rates at half-lives of 3, 6 and 12
  appearances, over ~35 per-match stats
- **Career rate with empirical-Bayes shrinkage** toward a positional prior, so a player
  with two appearances gets a usable number instead of noise
- **Role and workload** — share of team touches, days since last match, matches in the
  last fortnight, start rate
- **Team and opponent form** — rolling xG for and against, and what the opponent
  concedes: a side that fouls a lot lifts every opposing player's expected fouls
- **Match context** — home/away, matchweek, referee's historical fouls and cards per match

One detail worth pulling out. Form is computed as **smoothed counts over smoothed
minutes**, not as a smoothed per-90 rate. A player who commits one foul in a five-minute
cameo has a per-90 rate of 18, and averaging rates lets that cameo outweigh a full match.
The `NaivePer90EWMA` row in the results table is the same model using the naive version,
so the gap between it and `PlayerEWMA` is exactly what that mistake costs — and it was the
assumption v1's entire dataset was built on.

### The model ladder

| Model | What it is |
|---|---|
| `GlobalMean` | One rate for everyone. The floor. |
| `PositionMean` | Per-90 rate by position group |
| `ShrunkCareerRate` | Empirical-Bayes career rate |
| `NaivePer90EWMA` | Smoothed per-90 rates — the v1 assumption |
| **`PlayerEWMA`** | **Exposure-weighted recent form. The benchmark to beat.** |
| `PoissonGLM` / `NegBinGLM` | Log-link GLMs with a minutes offset |
| `PoissonGBM` | LightGBM, Poisson objective, offset via `init_score` |
| `NegBinMLP` | PyTorch: shared trunk, one head per target, player embeddings, NB likelihood |

`NegBinMLP` learns all six targets jointly — they share most of their signal (role,
opponent, minutes), so learning them together regularises all six. Player embeddings give
a per-player intercept that generalises: the thing v1 was reaching for by training one
model per player on ~13 rows, fitted here across 81k.

### Metrics

Reported per target: **log-score** and **CRPS** (both proper scoring rules — they cannot
be gamed by hedging toward the mean), Poisson deviance, MAE, RMSE, and **expected
calibration error** on `P(X >= 1)`.

MAE is in the table but it is the weakest column. For a target that is zero 56% of the
time, always predicting zero scores respectably on MAE while being useless.

---

## Known limitations

**The model is conditional on the player appearing.** The source data contains only
players who actually played, so `P(selected)` is not identifiable from it. The minutes
model answers "how long will he play *if picked*", not "will he be picked". Lifting this
needs lineup data including unused substitutes — available through the optional
`soccerdata` path (`read_lineup`), not yet wired into the feature pipeline.

**The FBref mirror stops around September 2025**, so 2025/26 has 40 matches rather than a
full season. Team-level data runs to the present via football-data.co.uk; player-level
does not.

**The Championship needs live scraping and gets thinner features.** The mirrors are
useless for the second tier, so it depends entirely on `soccerdata` reaching FBref — which
works today but is not something to depend on unattended. Feature count drops from 317 to
around 90, and `Tkl` is unavailable. The original hand-assembled Sunderland scrape is
preserved in [`data/legacy/`](data/legacy/sunderland_championship_2024_25/README.md).

---

## Layout

```
src/footy/
  config.py          paths, league codes, targets, constants
  sources/           base.py (cached HTTP), worldfootballr.py, footballdata.py
  ingest.py          raw CSV -> validated player-match Parquet
  championship.py    second-tier ingest, live-scraped, narrower schema
  features.py        causal feature construction
  fixture.py         forecasting a match that has not been played
  datasets.py        walk-forward splitters
  models/            minutes.py, baselines.py, glm.py, gbm.py, nn.py
  evaluate.py        predictive distributions and proper scoring rules
  pipeline.py        fits the ladder across folds
  cli.py             fetch / build / evaluate / predict / info
tests/               causality, per-90 units, splitters, metrics, models
```

## Tests

```bash
pytest -q
```

The suite is aimed squarely at the failure modes that are invisible in output: features
reading the future, splitters shuffling a time series, scalers fitted before the split,
per-90 conversion mixing units, and improper scoring rules.

## Licence

MIT.
