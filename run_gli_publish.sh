#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/torrey/GLI"
PYTHON="/home/torrey/gli-venv/bin/python"
LOG="$ROOT/gli_cron.log"
ACCEPTED_DATE="2026-08-04"
ACCEPTED_CLOSE="406.159408458803179844897657378502135518639147137075209285508535155764771406776716293291760336461828662105856724655057717691832524012300276339908366368991792628118743265856437494210"
ACCEPTED_DIVISOR="33.7446818036474799215589752183343431726902836865115467882999574496548904970795480197887599042738354723971462729623867931356914640348219979541621121036975056894497367037834426615724"
LIVE_CHECKPOINT_DATE="2026-08-05"
LIVE_CHECKPOINT_LEVELS="$ROOT/GLI_2026_live_checkpoint_levels.csv"
LIVE_CHECKPOINT_PRICES="$ROOT/GLI_2026_live_checkpoint_prices.csv"
LIVE_CHECKPOINT_OPEN="408.948292364942949755212716969380311635885021862158615917619741881388016492557817670699272496732461849956679751563333056879556945555626535668593672676065468261666798744061869192258"
LIVE_CHECKPOINT_HIGH="415.602081584426134729849209690025055909662688975355696898471184300941594423813058264358912336766095572920714330584851087990169679262172545732027014777088233005057988803794794271026"
LIVE_CHECKPOINT_LOW="402.444156356872004054267554085457858185527731620300087731287071047897528431488471901775664283618036474656513584887865291021287303218688405847534794466772828499142867000819619175387"
LIVE_CHECKPOINT_CLOSE="408.077636651815902257974841602469241459233403549301847382379839069797208495317898868571627110814374472854807681320371315668895554577590986852561643143584067877928175112676516059874"
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
if checkpoint.get("CloseSource") != "SUPERSEDING_BIG_OTC_REPAIR_PINNED_LIVE_CHECKPOINT":
    raise SystemExit("Hard guard: August 5 close is not pinned checkpoint data")
if checkpoint.get("OHLCVSource") != "SUPERSEDING_BIG_OTC_REPAIR_PINNED_LIVE_CHECKPOINT":
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
