# v1 scripts (archived)

The original project, kept for reference. Nothing here is imported by `src/footy`.

| File | What it did |
|---|---|
| `matchday_1_to_10.py`, `matchday_11_to_20.py` | Scraped 17 Sunderland match pages from FBref — one copy-pasted block per match, ~23k characters of near-identical code |
| `per_90_min_data_processing.py` | Normalised counts to per-90 |
| `per_min_data_processing.py` | Same, divided by minutes instead |
| `main.py` | A 3-layer PyTorch MLP predicting fouls, shots and shots on target |

## Why it was replaced rather than extended

**The scrapers no longer run at all.** They used `requests.get(url, verify=False)`, and
FBref now serves a Cloudflare challenge page to plain HTTP clients. `pd.read_html` gets a
"Just a moment..." holding page with no tables in it. (`verify=False` was also disabling
TLS certificate checks on the data the model trained on.)

**Fouls were never scraped.** `Fouls_Committed`, `Fouls_Won` and `Saves` — three of the
four prediction targets — were typed in by hand as index-aligned Python lists in every one
of the 17 blocks. FBref publishes all of them in its `misc` table.

**The model was not forecasting.** It regressed a match's fouls on that same match's
touches, passes and xG, none of which are known before kickoff. At prediction time it fed
the mean of all history, so it could not respond to the fixture at all.

**Three silent correctness bugs:**

- The per-90 mask was `(Min > 20) & (Min <= 90)`, so appearances of 20 minutes or less
  kept raw counts while longer ones became rates — one column, two units. In
  `normalized_sunderland_md1_stats.csv`, a 19-minute substitute's `Touches` stayed at
  `2.0` while a 71-minute starter's became `16.48`.
- `Passes_Cmp%` is a percentage but was multiplied by `90/Min` anyway, turning a genuine
  66.7% into 84.55%.
- `StandardScaler` was fitted on the full dataset before splitting, and the split itself
  was `train_test_split(..., random_state=42)` — a shuffle of a time series, so the model
  trained on future matches to predict past ones.

Each of these now has a regression test: `tests/test_per90.py` and
`tests/test_causality.py`.
