# Sunderland AFC — Championship 2024/25, matchdays 1–17

The original hand-assembled dataset from v1 of this project, scraped from FBref match
pages by the scripts now in [`legacy/`](../../../legacy/).

**Kept deliberately.** The `worldfootballR_data` mirrors that supply the rest of the
project cover first tiers only (`ENG_M_1st`), so there is no published source for
Championship *player-level* per-match stats. This scrape is genuinely not reproducible
from anywhere else, which is why it survives while the derived `normalized_*` folders
were dropped.

## Read this before using it

Three of the columns were **typed in by hand**, not scraped:

| Column | How it was produced |
|---|---|
| `Fouls_Committed` | A literal Python list per matchday, e.g. `[1, 1, 0, 0, 0, 0, 1, 2, 0, 0, 1, 1, 1, 0]` |
| `Fouls_Won` | Same |
| `Saves` | Same |

Each list was positionally zipped onto the scraped rows, so the values are correct only
if FBref returned the players in the exact order assumed when they were transcribed. They
have not been independently verified, and there is no way to check them after the fact.

Everything else in these files came from FBref's match summary table.

For Premier League data, `footy fetch` pulls FBref's `misc` table, which publishes the
same quantities as `Fls` and `Fld` — measured rather than transcribed. That is the source
the models actually train on.
