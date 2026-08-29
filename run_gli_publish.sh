#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/torrey/GLI"
PYTHON="/home/torrey/gli-venv/bin/python"
LOG="$ROOT/gli_cron.log"
ACCEPTED_DATE="2026-08-13"
ACCEPTED_CLOSE="425.04456187724249443489003651850675905719432777811786740931063721540297097059631846549244283773981414565479874027146762257479841895096762245717921595101792966218949508350528619739148276384893908286284732337707477432499761204717042224782643087403614604914026206753084302067100782769031708712262547946346269639145575742107"
ACCEPTED_DIVISOR="34.742780671822254470035109586734603950651205273959262183171050410031797439187210494407655932395575762840884736246341241271391689364658894316945081694107068529249495548807675352421485698211987411474345727478646292125575950671742758686036175277659419199418150845777429829246135833493240170568268072899636767188373341170598"
LIVE_CHECKPOINT_DATE="2026-08-14"
LIVE_CHECKPOINT_LEVELS="$ROOT/GLI_2026_live_checkpoint_levels.csv"
LIVE_CHECKPOINT_PRICES="$ROOT/GLI_2026_live_checkpoint_prices.csv"
LIVE_CHECKPOINT_OPEN="424.70103554476499117104238484361138033788582298232698380445087060727640552028664361427106164972375341179017094466311778128297681286788435784963192989738957456013251059285732911046502835306275068701473010392337539573272426087513042864456643664508707635914677515367566717594934345059801739167629810153621477315473504968683"
LIVE_CHECKPOINT_HIGH="428.61091795618997550573666275187350413818511085079707422750756732368685311601341616515933043224004732770868409920888949242928698457315468217851194188350633191490845869151914423082451923639342336069148558174722394824155804062199786434677682734973060168343352701805737395243365935371549260377984029924577059686012360560152"
LIVE_CHECKPOINT_LOW="420.17390716251647281233540967266361666105376759453178641561232122042750227118474491099183891640410207099927296624619196628572511408946266574443869096944630796985498458907848209270285103957407072646008100918724303757907885564633231479687055617211354503418876951475327711796717182466453756645053239510410763869852833993669"
LIVE_CHECKPOINT_CLOSE="424.76248792860010766285259968966526156307235360329970168110712852277226149287268743593873658346391376382738655051952063212901112124638708987863557648927586501999974981399893548736525364565109132685465016938868947072450241505752211547077456565480547812484807243305406976699427558419757685479602674930176975922248481289236"
LIVE_CHECKPOINT_SUM_OPEN="14755.2949290275574154"
LIVE_CHECKPOINT_SUM_HIGH="14891.135116100311149"
LIVE_CHECKPOINT_SUM_LOW="14598.0099005699156407"
LIVE_CHECKPOINT_SUM_CLOSE="14757.429955720901503"
LIVE_CHECKPOINT_VOLUME="212752200"
LIVE_CHECKPOINT_ROSTER="90"

readarray -t MARKET_STATE < <("$PYTHON" - <<'PY'
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import exchange_calendars as xcals

calendar = xcals.get_calendar("XNYS")
today = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
is_session = bool(calendar.is_session(today))
if is_session:
    target = today
else:
    target = calendar.date_to_session(
        pd.Timestamp(today), direction="previous"
    ).date().isoformat()

print(today)
print("1" if is_session else "0")
print(target)
PY
)

NY_TODAY="${MARKET_STATE[0]}"
NYSE_IS_SESSION="${MARKET_STATE[1]}"
END_DATE="${MARKET_STATE[2]}"

cd "$ROOT"
echo "$(date -Is) run_gli_publish.sh fired; NY_DATE=$NY_TODAY TARGET_SESSION=$END_DATE" >> "$LOG"

REUSE_LOCAL_LATEST=0

if [[ "$NYSE_IS_SESSION" != "1" ]]; then
  echo "$(date -Is) NYSE closed on $NY_TODAY; target session is $END_DATE." >> "$LOG"

  if "$PYTHON" - "$END_DATE" <<'PY'
from decimal import Decimal
import csv, sys

target = sys.argv[1]

def pos(v):
    try:
        d = Decimal(str(v))
        return d.is_finite() and d > 0
    except Exception:
        return False

def nonneg(v):
    try:
        d = Decimal(str(v))
        return d.is_finite() and d >= 0
    except Exception:
        return False

with open("constituents_great_lakes.csv", newline="", encoding="utf-8-sig") as f:
    constituents = list(csv.DictReader(f))

def member_on(row, day):
    start = (row.get("StartDate") or "").strip()
    end = (row.get("EndDate") or "").strip()
    return (not start or start <= day) and (not end or day <= end)

active = sorted({
    (row.get("Ticker") or "").strip().upper()
    for row in constituents
    if member_on(row, target) and (row.get("Ticker") or "").strip()
})

with open("gli_levels.csv", newline="", encoding="utf-8-sig") as f:
    levels = list(csv.DictReader(f))
if not levels or max(r["Date"] for r in levels) != target:
    raise SystemExit(1)
matches = [r for r in levels if r["Date"] == target]
if len(matches) != 1:
    raise SystemExit(1)
level = matches[0]

for field in ("GLI_Open","GLI_High","GLI_Low","GLI_Close","Divisor",
              "SumOpen","SumHigh","SumLow","SumClose"):
    if field not in level or not pos(level[field]):
        raise SystemExit(1)
if "TotalVolume" not in level or not nonneg(level["TotalVolume"]):
    raise SystemExit(1)
if int(Decimal(level["RowsLoaded"])) != len(active):
    raise SystemExit(1)

with open("gli_prices.csv", newline="", encoding="utf-8-sig") as f:
    prices = list(csv.DictReader(f))
if not prices or max(r["Date"] for r in prices) != target:
    raise SystemExit(1)

by_ticker = {}
for r in prices:
    if r["Date"] != target:
        continue
    t = (r.get("Ticker") or "").strip().upper()
    if t not in active:
        continue
    if t in by_ticker:
        raise SystemExit(1)
    by_ticker[t] = r

if set(by_ticker) != set(active):
    raise SystemExit(1)

for t, r in by_ticker.items():
    if any(not pos(r.get(f)) for f in ("Open","High","Low","Close")):
        raise SystemExit(1)
    if not nonneg(r.get("Volume")):
        raise SystemExit(1)
print(f"LOCAL_LATEST_COMPLETE target={target} roster={len(active)} price_rows={len(by_ticker)}")
PY
  then
    REUSE_LOCAL_LATEST=1
    echo "$(date -Is) local target session $END_DATE is complete; skipping Yahoo refetch." >> "$LOG"
  else
    echo "$(date -Is) local target session $END_DATE is incomplete; attempting guarded Yahoo refetch." >> "$LOG"
  fi
fi

if [[ "$REUSE_LOCAL_LATEST" != "1" ]]; then
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
fi

"$PYTHON" - \
  "$ACCEPTED_DATE" "$ACCEPTED_CLOSE" "$ACCEPTED_DIVISOR" \
  "$LIVE_CHECKPOINT_DATE" \
  "$LIVE_CHECKPOINT_OPEN" "$LIVE_CHECKPOINT_HIGH" \
  "$LIVE_CHECKPOINT_LOW" "$LIVE_CHECKPOINT_CLOSE" \
  "$LIVE_CHECKPOINT_SUM_OPEN" "$LIVE_CHECKPOINT_SUM_HIGH" \
  "$LIVE_CHECKPOINT_SUM_LOW" "$LIVE_CHECKPOINT_SUM_CLOSE" \
  "$LIVE_CHECKPOINT_VOLUME" "$LIVE_CHECKPOINT_ROSTER" \
  "$END_DATE" <<'PY'
from decimal import Decimal
import csv
import sys

(
    accepted_date, accepted_close, accepted_divisor, checkpoint_date,
    checkpoint_open, checkpoint_high, checkpoint_low, checkpoint_close,
    checkpoint_sum_open, checkpoint_sum_high, checkpoint_sum_low,
    checkpoint_sum_close, checkpoint_volume, checkpoint_roster,
    guard_date,
) = sys.argv[1:16]
with open("gli_levels.csv", newline="", encoding="utf-8") as stream:
    rows = {row["Date"]: row for row in csv.DictReader(stream)}
if accepted_date not in rows:
    raise SystemExit(f"Hard guard: {accepted_date} is missing")
row = rows[accepted_date]
if Decimal(row["GLI_Close"]) != Decimal(accepted_close):
    raise SystemExit("Hard guard: accepted August 13 close drifted")
if Decimal(row["Divisor"]) != Decimal(accepted_divisor):
    raise SystemExit("Hard guard: accepted August 13 divisor drifted")

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
            f"Hard guard: August 14 live checkpoint {field} drifted"
        )
if int(checkpoint["RowsLoaded"]) != int(checkpoint_roster):
    raise SystemExit("Hard guard: August 14 roster coverage drifted")
if Decimal(checkpoint["Divisor"]) != Decimal(accepted_divisor):
    raise SystemExit("Hard guard: August 14 divisor drifted")
if checkpoint.get("CloseSource") != "JCI_GAS_REBASE_PINNED_LIVE_CHECKPOINT":
    raise SystemExit("Hard guard: August 5 close is not pinned checkpoint data")
if checkpoint.get("OHLCVSource") != "JCI_GAS_REBASE_PINNED_LIVE_CHECKPOINT":
    raise SystemExit("Hard guard: August 5 OHLCV is not pinned checkpoint data")

with open("constituents_great_lakes.csv", newline="", encoding="utf-8") as stream:
    constituents = list(csv.DictReader(stream))

# Membership is date-effective. Do not use the Active flag here: during a
# staged transition, future additions may already be Active=Y and outgoing
# members may already be Active=N while StartDate/EndDate remain authoritative.
# Use the resolved NYSE target session rather than the host machine's
# calendar date. This keeps membership guards aligned across time zones,
# weekends, and exchange holidays.

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
  git commit -m "GLI site update $END_DATE" >> "$LOG" 2>&1
fi
git push >> "$LOG" 2>&1
