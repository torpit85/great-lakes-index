#!/usr/bin/env bash
set -euo pipefail


# --- NYSE market-holiday / weekend detection ---
# Skip run if today is NOT a NYSE trading session.
if ! /home/torrey/gli-venv/bin/python - <<'PY'
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
import exchange_calendars as xcals

cal = xcals.get_calendar("XNYS")

# Determine "today" in New York, but pass a naive YYYY-MM-DD string to is_session
ny_now = datetime.now(ZoneInfo("America/New_York"))
date_str = ny_now.date().isoformat()  # 'YYYY-MM-DD'

sys.exit(0 if cal.is_session(date_str) else 1)
PY
then
  echo "$(date -Is) NYSE closed today (weekend/holiday). Skipping." >> /home/torrey/GLI/gli_cron.log
  exit 0
fi
# --- end detection ---

LOG="/home/torrey/GLI/gli_cron.log"

END_DATE="$(date -d 'tomorrow' +%F)"

cd /home/torrey/GLI

/home/torrey/gli-venv/bin/python great_lakes_index_PRO.py \
  --tickers constituents_great_lakes.csv \
  --fetch yfinance \
  --start 2024-12-31 \
  --end "$END_DATE" \
  --events divisor_events.csv \
  --prices-out gli_prices.csv \
  --out gli_levels.csv \
  --db gli.sqlite \
  --report-dir report >> "$LOG" 2>&1

/home/torrey/gli-venv/bin/python /home/torrey/GLI/gli_site_build.py >> "$LOG" 2>&1

/home/torrey/gli-venv/bin/python /home/torrey/GLI/build_component_ohlcv_page.py \
  --tickers /home/torrey/GLI/constituents_great_lakes.csv \
  --out /home/torrey/GLI/report/ohlcv.html >> "$LOG" 2>&1

rsync -a --delete /home/torrey/GLI/report/ /home/torrey/GLI/docs/ >> "$LOG" 2>&1

git add docs
git commit -m "GLI site update $(date +%F)" >> "$LOG" 2>&1 || true
git push >> "$LOG" 2>&1
