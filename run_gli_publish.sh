#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/torrey/GLI"
PYTHON="/home/torrey/gli-venv/bin/python"
LOG="$ROOT/gli_cron.log"
ACCEPTED_DATE="2026-08-04"
ACCEPTED_CLOSE="405.0118976801850143497649371642440627923880222166590572694949116266828896677209632892154283916918526716542208146086709750087250992217390189455069968501599154857441656409868612674282624278964817526389377487766844636287644"
ACCEPTED_DIVISOR="33.84028982482542230086864766969518211131551391766690360193547571752682257130067996939078458758662067456588134300172407782359397606202903520495006769360726522427869786486876676194144552324419611170034107745495074455890385"
LIVE_CHECKPOINT_DATE="2026-08-05"
LIVE_CHECKPOINT_CLOSE="406.924706356915487172707495773543064828366524380579255608367616280622644999669345275255630196470748900632791638566085782272576598863685824125850872830853439892297941717102605608465"
LIVE_CHECKPOINT_SUM="13770.45"
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
  "$LIVE_CHECKPOINT_DATE" "$LIVE_CHECKPOINT_CLOSE" \
  "$LIVE_CHECKPOINT_SUM" "$LIVE_CHECKPOINT_ROSTER" <<'PY'
from decimal import Decimal
import csv
import sys

(
    accepted_date, accepted_close, accepted_divisor,
    checkpoint_date, checkpoint_close, checkpoint_sum, checkpoint_roster,
) = sys.argv[1:8]
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
if Decimal(checkpoint["GLI_Close"]) != Decimal(checkpoint_close):
    raise SystemExit("Hard guard: August 5 live checkpoint close drifted")
if Decimal(checkpoint["SumClose"]) != Decimal(checkpoint_sum):
    raise SystemExit("Hard guard: August 5 component sum drifted")
if int(checkpoint["RowsLoaded"]) != int(checkpoint_roster):
    raise SystemExit("Hard guard: August 5 roster coverage drifted")
if Decimal(checkpoint["Divisor"]) != Decimal(accepted_divisor):
    raise SystemExit("Hard guard: August 5 divisor drifted")

with open("constituents_great_lakes.csv", newline="", encoding="utf-8") as stream:
    constituents = list(csv.DictReader(stream))
active = [row["Ticker"] for row in constituents if row["Active"].upper() == "Y"]
if len(active) != 86 or "ELV" not in active:
    raise SystemExit("Hard guard: current roster is not accepted 86-name roster")
PY

HOME=/home/torrey "$PYTHON" "$ROOT/gli_site_build.py" >> "$LOG" 2>&1

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
grep -q "GLI_INTERACTIVE_CHART_V2" "$ROOT/report/index.html"

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
