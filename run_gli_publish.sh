#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/torrey/GLI"
PYTHON="/home/torrey/gli-venv/bin/python"
LOG="$ROOT/gli_cron.log"
ACCEPTED_DATE="2026-08-04"
ACCEPTED_CLOSE="406.1515153985296153749960846365457989226027982051953548107515900777419980376593723529889298388725903952290474868732354842535747483769848300438652485357535102201540018579262180907897986109188086989671518496615272908859559"
ACCEPTED_DIVISOR="33.74533759045927377465246800383713665138133227577660810672970718185826639465173091980018495330191953257489466764274260559637039586358311654493131851189982917414129213024362246776822669357241805361423418122909215117016846"
LIVE_CHECKPOINT_DATE="2026-08-05"
LIVE_CHECKPOINT_LEVELS="$ROOT/GLI_2026_live_checkpoint_levels.csv"
LIVE_CHECKPOINT_PRICES="$ROOT/GLI_2026_live_checkpoint_prices.csv"
LIVE_CHECKPOINT_OPEN="408.9403451071589775758101156779756052470137849566459123036260857001694443473260023098346692956203059097802717683403630916629705276260690111270500909621054102186085663109319293109609554671272799858940552637522606732507844"
LIVE_CHECKPOINT_HIGH="415.5940050208615670450352181084530721901654038632785089870296613255401224889628998149943109938836633051918793330817737992565486204408269664836274224889228802712385031575302965436145507165639823347804395328606696238657556"
LIVE_CHECKPOINT_LOW="402.4363354965971625100927261022343036628917642127465566715349763684259960841697814732763556629884006804347962046765105013514819968685618804912259573340561592180686895056435062039248752758549892829430749075240791541621375"
LIVE_CHECKPOINT_CLOSE="408.0697063138370068876800951488372808414118851504869772367824699166510257671750556715164477969565307337361216313782526984455861052019048071080938623946182269016964942290138781405366797425984851032045172517876899924105053"
LIVE_CHECKPOINT_SUM_OPEN="13799.83"
LIVE_CHECKPOINT_SUM_HIGH="14024.36"
LIVE_CHECKPOINT_SUM_LOW="13580.35"
LIVE_CHECKPOINT_SUM_CLOSE="13770.45"
LIVE_CHECKPOINT_VOLUME="232993793"
LIVE_CHECKPOINT_ROSTER="86"

if ! "$PYTHON" - <<'PY'
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
import exchange_calendars as xcals
calendar = xcals.get_calendar("XNYS")
today = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
sys.exit(0 if calendar.is_session(today) else 1)
PY
then
  echo "$(date -Is) NYSE closed today; publishing latest available session." >> "$LOG"
fi

echo "$(date -Is) run_gli_publish.sh fired" >> "$LOG"
END_DATE="$(date +%F)"
cd "$ROOT"

"$PYTHON" great_lakes_index_PRO.py \
  --tickers constituents_great_lakes.csv \
  --accepted-chain GLI_2026_accepted_daily_close_chain.csv \
  --accepted-ohlcv-chain GLI_2026_accepted_daily_ohlcv_chain.csv \
  --live-checkpoint-levels "$LIVE_CHECKPOINT_LEVELS" \
  --live-checkpoint-prices "$LIVE_CHECKPOINT_PRICES" \
  --fetch yfinance \
  --start 2025-12-31 \
  --end "$END_DATE" \
  --events divisor_events.csv \
  --prices-out gli_prices.csv \
  --out gli_levels.csv \
  --db gli.sqlite \
  --report-dir report \
  --strict >> "$LOG" 2>&1

"$PYTHON" - \
  "$ACCEPTED_DATE" "$ACCEPTED_CLOSE" "$ACCEPTED_DIVISOR" \
  "$LIVE_CHECKPOINT_DATE" \
  "$LIVE_CHECKPOINT_OPEN" "$LIVE_CHECKPOINT_HIGH" \
  "$LIVE_CHECKPOINT_LOW" "$LIVE_CHECKPOINT_CLOSE" \
  "$LIVE_CHECKPOINT_SUM_OPEN" "$LIVE_CHECKPOINT_SUM_HIGH" \
  "$LIVE_CHECKPOINT_SUM_LOW" "$LIVE_CHECKPOINT_SUM_CLOSE" \
  "$LIVE_CHECKPOINT_VOLUME" "$LIVE_CHECKPOINT_ROSTER" <<'PY'
from decimal import Decimal
import csv
import sys

(
    accepted_date, accepted_close, accepted_divisor, checkpoint_date,
    checkpoint_open, checkpoint_high, checkpoint_low, checkpoint_close,
    checkpoint_sum_open, checkpoint_sum_high, checkpoint_sum_low,
    checkpoint_sum_close, checkpoint_volume, checkpoint_roster,
) = sys.argv[1:15]
with open("gli_levels.csv", newline="", encoding="utf-8") as stream:
    rows = {row["Date"]: row for row in csv.DictReader(stream)}
if accepted_date not in rows:
    raise SystemExit(f"Hard guard: {accepted_date} is missing")
row = rows[accepted_date]
if Decimal(row["GLI_Close"]) != Decimal(accepted_close):
    raise SystemExit("Hard guard: accepted August 4 close drifted")
if Decimal(row["Divisor"]) != Decimal(accepted_divisor):
    raise SystemExit("Hard guard: accepted August 4 divisor drifted")

if checkpoint_date not in rows:
    raise SystemExit(f"Hard guard: live checkpoint {checkpoint_date} is missing")
checkpoint = rows[checkpoint_date]
expected = {
    "GLI_Open": checkpoint_open,
    "GLI_High": checkpoint_high,
    "GLI_Low": checkpoint_low,
    "GLI_Close": checkpoint_close,
    "SumOpen": checkpoint_sum_open,
    "SumHigh": checkpoint_sum_high,
    "SumLow": checkpoint_sum_low,
    "SumClose": checkpoint_sum_close,
    "TotalVolume": checkpoint_volume,
}
for field, value in expected.items():
    if Decimal(checkpoint[field]) != Decimal(value):
        raise SystemExit(
            f"Hard guard: August 5 live checkpoint {field} drifted"
        )
if int(checkpoint["RowsLoaded"]) != int(checkpoint_roster):
    raise SystemExit("Hard guard: August 5 roster coverage drifted")
if Decimal(checkpoint["Divisor"]) != Decimal(accepted_divisor):
    raise SystemExit("Hard guard: August 5 divisor drifted")
if checkpoint.get("CloseSource") != "PINNED_LIVE_CHECKPOINT":
    raise SystemExit("Hard guard: August 5 close is not pinned checkpoint data")
if checkpoint.get("OHLCVSource") != "PINNED_LIVE_CHECKPOINT":
    raise SystemExit("Hard guard: August 5 OHLCV is not pinned checkpoint data")

with open("constituents_great_lakes.csv", newline="", encoding="utf-8") as stream:
    constituents = list(csv.DictReader(stream))

# Membership is date-effective. Do not use the Active flag here: during a
# staged transition, future additions may already be Active=Y and outgoing
# members may already be Active=N while StartDate/EndDate remain authoritative.
import datetime
guard_date = datetime.date.today().isoformat()

def roster_member_on(row, day):
    start = row.get("StartDate", "").strip()
    end = row.get("EndDate", "").strip()
    return (not start or start <= day) and (not end or day <= end)

active = [
    row["Ticker"].strip().upper()
    for row in constituents
    if roster_member_on(row, guard_date)
]

expected_count = 90 if guard_date >= "2026-08-10" else 86

if len(active) != expected_count or "ELV" not in active:
    raise SystemExit(
        f"Hard guard: {guard_date} roster is not accepted "
        f"{expected_count}-name roster"
    )

gli90_removed = {"LE", "RAIL", "WNC"}
gli90_added = {"CDW", "DTE", "DTM", "MPLX", "TWI", "VTR", "ZBRA"}

if guard_date < "2026-08-10":
    if not gli90_removed.issubset(active) or gli90_added.intersection(active):
        raise SystemExit(
            "Hard guard: pre-2026-08-10 GLI90 membership transition is incorrect"
        )
else:
    if gli90_removed.intersection(active) or not gli90_added.issubset(active):
        raise SystemExit(
            "Hard guard: 2026-08-10 GLI90 membership transition is incorrect"
        )
PY

HOME=/home/torrey "$PYTHON" "$ROOT/gli_site_build.py" >> "$LOG" 2>&1
# GLI_COMPONENT_HISTORY_EVENT_RENDER_GUARD
if ! grep -q "Company names + symbols" "$ROOT/report/components.html"; then
  echo "ERROR: Component History historical-name/symbol template was not rendered; refusing publication." >> "$LOG"
  exit 1
fi
if ! grep -q "security_replacement_same_ticker" "$ROOT/report/data/component_history.json"; then
  echo "ERROR: Component History same-ticker security replacement control is missing; refusing publication." >> "$LOG"
  exit 1
fi
if ! grep -q "name_change" "$ROOT/report/data/component_history.json"; then
  echo "ERROR: Component History explicit name-change events are missing; refusing publication." >> "$LOG"
  exit 1
fi
# GLI_COMPONENT_HISTORY_SYMBOL_RENDER_GUARD
if ! grep -q "Company names + symbols" "$ROOT/report/components.html"; then
  echo "ERROR: Component History symbol template was not rendered; refusing publication." >> "$LOG"
  exit 1
fi

for required in \
  index.html history.html market-moves.html milestones.html \
  weights.html components.html ohlcv.html ticker.txt \
  data/gli_history.json data/market_moves.json data/milestones.json \
  data/component_history.json data/weights_manifest.json
 do
  test -s "$ROOT/report/$required"
 done
grep -q "<th>Company</th>" "$ROOT/report/ohlcv.html"
grep -q "2005-08-01" "$ROOT/report/data/gli_history.json"
grep -q "GLI_INTERACTIVE_CHART_V5_DATE_INPUTS" "$ROOT/report/index.html"

rsync -av --delete "$ROOT/report/" "$ROOT/docs/" >> "$LOG" 2>&1
for required in \
  index.html history.html market-moves.html milestones.html \
  weights.html components.html ohlcv.html ticker.txt \
  data/gli_history.json data/market_moves.json data/milestones.json \
  data/component_history.json data/weights_manifest.json
 do
  test -s "$ROOT/docs/$required"
 done
grep -q "<th>Company</th>" "$ROOT/docs/ohlcv.html"
grep -q "2005-08-01" "$ROOT/docs/data/gli_history.json"

git add docs >> "$LOG" 2>&1
if ! git diff --cached --quiet; then
  git commit -m "GLI site update $(date +%F)" >> "$LOG" 2>&1
fi
git push >> "$LOG" 2>&1
