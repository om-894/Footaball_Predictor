# Sunderland, Championship 2024/25, matchdays 1 to 17

The original dataset from the first version of this project, scraped from FBref match pages in
late 2024. The scripts that built it are in the repo history at commit
[`8fdd2b8`](https://github.com/om-894/Footaball_Predictor/tree/8fdd2b8).

Nothing in `src/footy` reads these files. They are kept as a record of where the project started.

## Before using it

Three columns were typed in by hand rather than scraped:

| column | how it was made |
|---|---|
| `Fouls_Committed` | a Python list per matchday, e.g. `[1, 1, 0, 0, 0, 0, 1, 2, 0, 0, 1, 1, 1, 0]` |
| `Fouls_Won` | same |
| `Saves` | same |

Each list was matched to the scraped rows by position, so the values are only right if FBref
returned the players in the order assumed when they were typed. They have not been checked since.

Everything else came from FBref's match summary table. The current pipeline reads fouls from
FBref's own `Fls` and `Fld` columns instead, either from the mirror (`footy fetch`) or from a live
scrape (`scripts/scrape_fbref.py`).
