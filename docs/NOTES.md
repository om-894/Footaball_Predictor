# Notes

Practical notes for running the project, plus the things that caught me out. The README
covers what the project does and how well it works.

## Data on disk

Everything in `data/` apart from `data/legacy/` is gitignored and can be rebuilt. `footy info`
shows what is there.

| file in `data/interim/` | what it holds | made by |
|---|---|---|
| `player_matches.parquet` | Premier League 2017/18 to September 2025, from the FBref mirror | `footy fetch` |
| `ENG-Premier-League_player_matches.parquet` | Premier League matches scraped live from FBref | `scripts/scrape_fbref.py` then `scripts/build_scraped.py` |
| `ENG-Championship_player_matches.parquet` | Championship matches scraped live | the same two scripts |
| `NED-Eredivisie_player_matches.parquet` | Eredivisie matches scraped live, for a European tie | the same two scripts |
| `pl_combined_player_matches.parquet` | the mirror plus the live scrapes, on the columns they share | `scripts/combine_pl.py` |

`data/features/player_features.parquet` is the feature table that `footy evaluate` and
`footy predict` read. `footy build` makes it from the mirror.

The mirror stopped updating in September 2025, so anything newer has to be scraped live. A
live scrape only gets FBref's match summary table, which has no xG, touches or passes. That
is why the combined table has far fewer features than the mirror on its own.

## Scraping live data

```bash
pip install -e ".[live]"
python scripts/scrape_fbref.py "ENG-Premier League" 2627
python scripts/build_scraped.py ENG-Premier-League
python scripts/combine_pl.py
```

FBref blocks plain HTTP requests with a Cloudflare "Just a moment..." page, so the scraper
uses soccerdata, which drives a real browser. FBref also limits how fast it can be scraped,
so a full Premier League season takes over an hour. The scraper saves as it goes and skips
matches it already has, so running it again after a crash carries on where it stopped.

The Championship needs a one-off soccerdata setting, because soccerdata only ships the top
five leagues. Put this in `~/soccerdata/config/league_dict.json`:

```json
{"ENG-Championship": {"FBref": "EFL Championship", "MatchHistory": "E1",
                      "season_start": "Aug", "season_end": "May"}}
```

The name has to be "EFL Championship". Plain "Championship" returns no rows and no error.

`scripts/refresh_pl.sh <home> <away> <date>` runs the three Premier League steps above and
then forecasts one fixture.

## Things that caught me out

**soccerdata spells clubs two ways.** Its schedule says "QPR", "Blackburn" and "West Brom"
while its player table says "Queens Park Rangers", "Blackburn Rovers" and "West Bromwich
Albion". `scraped.resolve_home_away` finds each match's home side by string similarity. For a
new league, check that the home/away balance `build_scraped.py` prints is close to 0.50.

**Team names in `predict-fixture` follow the player table**, so it is "Queens Park Rangers"
rather than "QPR". A misspelt name gets suggestions.

**LightGBM and PyTorch hang together on macOS.** They each bring their own copy of OpenMP.
`models/nn.py` keeps torch to one thread, which fixes it. Setting `OMP_NUM_THREADS` instead
makes LightGBM crash.

**FBref's published summary table only covers two of the mirror's eight seasons.** So the
mirror tables are joined on `misc`, with shots taken from the shot-level file. Joining on
`summary` would throw away most of the history.

**Per-90 rates mislead for short appearances.** One foul in a five-minute cameo is 18 per 90.
The form features divide smoothed counts by smoothed minutes instead. The `NaivePer90EWMA`
baseline shows what the naive version costs.

## Known issues

- A few players are spelt differently in the mirror and the live scrape ("Andrew Robertson"
  and "Andy Robertson", "Sasa Kalajdzic" and "Saša Kalajdžić"), so their history splits in
  two at September 2025.
- Referees are "A Taylor" in the mirror but "Anthony Taylor" in a scrape, so the referee
  features start again from zero at the same point.
- `season_progress` divides the matchweek by 38, which is wrong for the Championship's 46
  rounds.
- The shrunk career rates are only built for the six mirror targets, so a scraped table gets
  no career feature for `TklW`.
- The minutes model only sees players who played. It predicts minutes if a player is picked,
  not whether they will be.

## Ideas for next steps

1. Score past fixture forecasts against what actually happened.
2. Scrape Championship 2025/26, so promoted clubs arrive with some history.
3. Use soccerdata's `read_lineup()`, which lists unused substitutes, to model whether a
   player is picked.
4. Add a `--model` option to `predict-fixture`. It always uses PoissonGBM now, even for
   targets where another model scored better.
5. Save trained models, so `footy predict` doesn't retrain everything each time.
6. Match player and referee names across sources (see known issues).
