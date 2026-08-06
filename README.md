# Great Lakes Index

Static GitHub Pages site and live publication engine for the price-weighted
Great Lakes Index.

## Public pages

- `index.html` — live summary and interactive full-history candlestick chart
- `history.html` — complete daily history beginning 2005-08-01 with year tabs
- `market-moves.html` — largest point and percentage gains and losses
- `milestones.html` — first close strictly above every 10-point level
- `weights.html` — date-selectable component-weight heatmap
- `components.html` — component roster history and membership changes
- `ohlcv.html` — latest component OHLCV snapshot

## Publication flow

`run_gli_publish.sh` runs the strict live engine, verifies the accepted
2026-08-04 close and divisor controls, builds all static pages, validates the
required output set, mirrors `report/` into `docs/`, and stages only `docs/` for
the routine site update commit.

Historical presentation inputs live under `site_data/`. They are compact,
hash-documented derivatives of accepted files and do not alter the locked
calculation layer.
