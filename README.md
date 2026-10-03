# Football Predictor

Forecasts what each player will do in a football match: shots, shots on target, fouls
committed, fouls won, yellow cards and tackles. Every forecast is a probability for each
count rather than a single number, so it can say a player has a 63% chance of at least one
shot instead of "1.01 shots".

```
$ footy predict-fixture --home Sunderland --away "Hull City" --date 2026-09-08 \
    --league combined --lineup "Roefs, Xhaka, Ballard, Le Fée, Isidor, Brobbey"

                           Sunderland, 2026-09-08
 player          pos    mins          Sh         SoT         Fls         Fld        CrdY
 Wilson Isidor   FW,MF    90  4.51 (99%)  1.50 (78%)  1.31 (73%)  1.48 (77%)  0.21 (19%)
 Robin Roefs     GK       90   0.01 (1%)   0.00 (0%)   0.02 (2%)  0.39 (32%)   0.07 (7%)
 Granit Xhaka    DM       90  1.01 (63%)  0.23 (21%)  0.72 (51%)  1.21 (70%)  0.11 (11%)
 Enzo Le Fée     AM       90  2.14 (88%)  0.80 (55%)  0.80 (55%)  1.08 (66%)   0.08 (8%)
 ...
 each cell: expected count (probability of at least one)
```

The models learn from 81,327 Premier League player-matches (2,857 matches from 2017/18 to
September 2025) of FBref data, topped up with live-scraped matches for the current season.

## What it does

- **Features only use the past.** Every feature for a match on date t comes from matches
  before t. `tests/test_causality.py` checks this by scrambling later matches and making
  sure no earlier feature changes.
- **Counts are modelled as counts.** The models predict Poisson and negative binomial
  distributions with minutes played as an offset, so a player who plays twice as long is
  expected to do twice as much.
- **Every model faces a simple benchmark.** Each one is scored against the player's own
  recent average on six seasons it never trained on.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

footy fetch       # download the FBref mirror and referee data (about 200 MB)
footy build       # build the feature table
footy evaluate    # backtest every model (about 15 minutes, --skip-nn is faster)
footy info        # show which tables are on disk
```

`footy predict --player "Bukayo Saka"` re-scores a player's latest matches with the neural
network and shows the full distribution next to what actually happened. It retrains first,
so it takes a few minutes.

### Forecasting a match this week

The mirror stops in September 2025, so a current fixture needs live data from FBref first.
[docs/NOTES.md](docs/NOTES.md) covers the setup.

```bash
pip install -e ".[live]"
python scripts/scrape_fbref.py "ENG-Premier League" 2627
python scripts/build_scraped.py ENG-Premier-League
python scripts/combine_pl.py
footy predict-fixture --home Sunderland --away "Hull City" --date 2026-09-08 --league combined
```

Without a lineup the squad is everyone who has played for the club in the last 75 days, so a
dropped or injured player still appears. Once team news is out, pass the starters with
`--lineup`. They get 90 minutes each (change it with `--lineup-minutes`) and the rest of that
side is dropped. Names match on case, accents and unambiguous surnames.

`--league` picks the history table: `combined` (mirror plus live scrapes), `pl` (mirror only)
or `championship` (live scrape only). The full forecast, including the chance of at least
two, is saved as a CSV in `reports/`.

## Results

Every model was backtested on six Premier League seasons it never trained on (2020/21 to
2025/26, 49,786 player-matches). Each season is predicted by models trained only on the
seasons before it. Log score, lower is better:

| target | best model | log score | neural net | PlayerEWMA (benchmark) | best v benchmark |
|---|---|---|---|---|---|
| shots | PoissonGBM | 0.748 | 0.755 | 0.944 | 21% better |
| shots on target | PoissonGBM | 0.395 | 0.402 | 0.551 | 28% better |
| fouls committed | ShrunkCareerRate | 1.112 | 1.127 | 1.342 | 17% better |
| fouls won | ShrunkCareerRate | 1.087 | 1.097 | 1.294 | 16% better |
| yellow cards | PoissonGBM | 0.386 | 0.392 | 0.606 | 36% better |
| tackles | ShrunkCareerRate | 1.365 | 1.374 | 1.602 | 15% better |

- LightGBM wins shots, shots on target and yellow cards. A player's career rate pulled
  towards their position's average wins fouls and tackles.
- The neural network never wins outright but is within 2% of the best model on every target.
- Fouls committed are hard to predict from this data. The best model is only 1.6% better
  than one average per position.
- With the minutes actually played known, LightGBM wins every target. Real forecasts score
  1 to 10% worse than that, since the minutes model is off by 18.6 minutes on average.

The full tables for every model and metric are in [docs/RESULTS.md](docs/RESULTS.md).

## How it works

### Two stages

```
E[count] = per-90 rate x expected minutes / 90
```

Stage one predicts minutes with quantile LightGBM models. Minutes bunch up at 90 and at
short substitute spells, so averaging several quantiles works better than predicting the
mean directly. Stage two predicts the per-90 rate, with log(minutes / 90) as a fixed offset.

### Features

317 features on the mirror, all built from earlier matches only:

- **Recent form**: exponentially weighted averages with half-lives of 3, 6 and 12
  appearances, for 37 per-match stats. They are smoothed counts divided by smoothed minutes,
  so one foul in a five-minute cameo doesn't count as 18 per 90.
- **Career rate**: each player's rate pulled towards their position's average, so a player
  with two appearances still gets a sensible number.
- **Workload**: days since the last match, matches and minutes in the last 15 days and how
  often the player has played at least 60 minutes.
- **Team and opponent form**: what each side has recently produced and conceded.
- **Match context**: home or away, how far through the season it is and the referee's
  average fouls and yellow cards per match.

### Models

| model | what it is |
|---|---|
| `GlobalMean` | one rate for every player |
| `PositionMean` | one rate per position group |
| `ShrunkCareerRate` | the player's career rate pulled towards their position's |
| `NaivePer90EWMA` | the player's smoothed per-90 rate, to show what averaging rates costs |
| `PlayerEWMA` | the player's recent form, smoothed counts over smoothed minutes (the benchmark) |
| `PoissonGLM`, `NegBinGLM` | regressions with a log-minutes offset |
| `PoissonGBM` | LightGBM with a Poisson objective, with the offset passed as `init_score` |
| `NegBinMLP` | a PyTorch network with one output per target, a learned vector per player and a negative binomial loss |

### Metrics

Log score and CRPS judge the whole distribution and reward one that is well calibrated.
Poisson deviance, MAE and the calibration error of P(at least one) are reported too. Lower is
better for all of them. MAE is the weakest: for a target that is zero most of the time,
always predicting zero does well on it.

## Limitations

- **It can't see who will be picked.** The data only has players who played, so the minutes
  model predicts minutes if picked. For a cup tie with lots of changes that is the biggest
  source of error, which is what `--lineup` is for.
- **Players are treated separately.** Two strikers competing for one place can both get a
  high chance of a shot.
- **Live data is thinner.** A live scrape has no xG, touches or passes, so the combined table
  gets 111 features and the Championship table 128, against 317 on the mirror.
- **Some names differ between sources**, which splits a few players' and referees' history.
  [docs/NOTES.md](docs/NOTES.md) lists these.

## Background

The first version of this project (2024, see commit
[`8fdd2b8`](https://github.com/om-894/Footaball_Predictor/tree/8fdd2b8)) was a small PyTorch
model for Sunderland's Championship season, trained on hand-collected FBref data. It ran end
to end, but rebuilding it showed four problems that this version fixes:

- It predicted a match's fouls from that same match's touches, passes and xG, none of which
  are known before kickoff.
- It split matches at random and fitted its scaler on all the data, so it trained on matches
  played after the ones it was tested on.
- Its per-90 conversion only scaled appearances over 20 minutes and scaled percentages too.
- Fouls were typed in by hand from match pages, although FBref publishes them in its tables.

The first three now have tests. Fouls now come straight from FBref's own tables. The
original Sunderland data is kept in
[`data/legacy/`](data/legacy/sunderland_championship_2024_25/README.md).

## Project layout

```
src/footy/
  cli.py          the `footy` command
  config.py       paths, data sources, targets and other settings
  ingest.py       builds the player-match table from the mirror
  scraped.py      builds the same table from a live scrape
  features.py     builds features from earlier matches only
  datasets.py     season by season splits
  models/         minutes model, baselines, GLMs, LightGBM and the neural network
  evaluate.py     count distributions and scoring
  pipeline.py     fits and scores every model on each season
  fixture.py      forecasts a match that hasn't been played
  sources/        downloads from the FBref mirror and football-data.co.uk
scripts/          live scraping: scrape, build, combine and refresh
tests/            90 tests
docs/             full results tables and setup notes
data/legacy/      the original hand-collected Sunderland data
```

## Tests

```bash
pytest -q
```

The 90 tests take about 40 seconds. They focus on mistakes that don't show up in the output:
features that see the future, splits that shuffle a time series, a scaler fitted on test
data, per-90 conversions that mix units and scoring rules that can be gamed.

## Licence

MIT, see [LICENSE](LICENSE).
