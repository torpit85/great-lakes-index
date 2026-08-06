# GLI site data

These files are compact public-site inputs. They do not replace or modify any
accepted reconstruction, final-lock addendum, close chain, divisor control, or
live anchor.

## Daily index history

`gli_historical_ohlcv_through_2025.csv` contains 5,138 accepted daily rows from
2005-08-01 through 2025-12-31. Its exact-value columns were extracted from the
accepted cumulative OHLCV workbook whose SHA-256 is recorded in
`site_data_manifest.json`.

At publication time, `gli_site_build.py` merges this immutable historical base
with the newly generated `gli_levels.csv`. Live rows win only on duplicate
dates, allowing the accepted 2025-12-31 carry and the current 2026 chain to
remain controlled by the live engine invocation.

## Component weights

`weights/weights_2005.json` through `weights_2025.json` are compact daily
component-weight snapshots generated from the accepted annual component OHLCV
files. A weight is stored as parts per million of that date's component sum.
The site converts this to percent for display.

The builder may add a current-year file from `gli_prices.csv`, but only for
sessions after the accepted cutoff recorded in `GLI_2026_live_anchor.json`.
It does not substitute public-vendor component bars for accepted historical
component weights.

## Component history

`component_roster_history_through_2025.json` combines:

- the workbook-style company-name checkpoints through 2014-05-01; and
- later ticker-based checkpoints derived from accepted component OHLCV rows.

The builder extends 2026 from `constituents_great_lakes.csv`, whose start and
end dates define the published membership chronology.

See `site_data_manifest.json` for row counts, date coverage, file sizes, and
SHA-256 values.
