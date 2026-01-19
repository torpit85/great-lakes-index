#!/usr/bin/env python3
"""
build_component_ohlcv_page.py

Generates a static HTML page containing the most recent DAILY OHLCV bar
for each component ticker.

Primary source: Yahoo Finance via yfinance (bulk download + per-ticker retry).
Fallback source: Stooq daily CSV (per-ticker).

Output: a single HTML file (e.g. report/ohlcv.html) suitable for publishing
as part of a static site.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd


# ----------------------------
# Utilities
# ----------------------------

@contextlib.contextmanager
def suppress_output():
    """Suppress noisy stdout/stderr (yfinance likes to print warnings)."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def _dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def sanitize_ticker(raw: str) -> str:
    """
    Make tickers safer for data sources:
    - trim whitespace
    - strip leading '$'
    - remove internal whitespace
    """
    t = str(raw).strip()
    if not t or t.lower() == "nan":
        return ""
    t = t.lstrip("$").strip()
    t = re.sub(r"\s+", "", t)
    return t


def read_tickers(csv_path: Path) -> list[str]:
    df = pd.read_csv(csv_path)
    # Common column names; else first column
    for col in ("ticker", "Tickers", "TICKER", "symbol", "Symbol", "SYMBOL"):
        if col in df.columns:
            raw = df[col].astype(str).tolist()
            break
    else:
        raw = df.iloc[:, 0].astype(str).tolist()

    tickers = [sanitize_ticker(x) for x in raw]
    tickers = [t for t in tickers if t]
    return _dedupe_keep_order(tickers)


@dataclass
class Bar:
    ticker: str
    date: Optional[str] = None  # YYYY-MM-DD
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    adj_close: Optional[float] = None
    volume: Optional[int] = None
    source: Optional[str] = None  # "yahoo" or "stooq" or None


def bar_to_row(b: Bar) -> dict:
    return {
        "Ticker": b.ticker,
        "Date": b.date,
        "Open": b.open,
        "High": b.high,
        "Low": b.low,
        "Close": b.close,
        "Adj Close": b.adj_close,
        "Volume": b.volume,
        "Source": b.source,
    }


def _row_from_df_last(ticker: str, tdf: pd.DataFrame, source: str) -> Bar:
    tdf = tdf.dropna(how="all")
    if tdf.empty:
        return Bar(ticker=ticker, source=source)

    last = tdf.iloc[-1]
    idx = tdf.index[-1]
    date_str: Optional[str]
    try:
        date_str = pd.Timestamp(idx).date().isoformat()
    except Exception:
        date_str = None

    def fget(col: str) -> Optional[float]:
        v = last.get(col)
        return float(v) if pd.notna(v) else None

    def iget(col: str) -> Optional[int]:
        v = last.get(col)
        return int(v) if pd.notna(v) else None

    return Bar(
        ticker=ticker,
        date=date_str,
        open=fget("Open"),
        high=fget("High"),
        low=fget("Low"),
        close=fget("Close"),
        adj_close=fget("Adj Close") if "Adj Close" in tdf.columns else None,
        volume=iget("Volume"),
        source=source,
    )


# ----------------------------
# Yahoo (yfinance) fetch
# ----------------------------

def fetch_yahoo_lastbars(tickers: list[str], lookback_days: int) -> Tuple[list[Bar], list[str]]:
    """
    Fetch last daily bar for each ticker from Yahoo via yfinance.
    Strategy:
      1) One bulk download for speed
      2) For any ticker missing/empty, retry that ticker individually
    Returns: (bars, missing_tickers_after_retry)
    """
    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError("yfinance is not installed in this environment.") from e

    end = dt.date.today() + dt.timedelta(days=1)
    start = dt.date.today() - dt.timedelta(days=lookback_days)

    # Bulk download
    with suppress_output():
        bulk = yf.download(
            tickers=tickers,
            start=start.isoformat(),
            end=end.isoformat(),
            group_by="ticker",
            auto_adjust=False,
            actions=False,
            threads=True,
            progress=False,
        )

    bars: list[Bar] = []
    missing: list[str] = []

    if isinstance(bulk.columns, pd.MultiIndex):
        avail = set(bulk.columns.get_level_values(0))

        for t in tickers:
            b: Bar
            # Use bulk if present and non-empty
            if t in avail:
                b = _row_from_df_last(t, bulk[t], source="yahoo")
                if b.date is not None:
                    bars.append(b)
                    continue

            # Retry individually
            with suppress_output():
                solo = yf.download(
                    tickers=t,
                    start=start.isoformat(),
                    end=end.isoformat(),
                    auto_adjust=False,
                    actions=False,
                    progress=False,
                )
            b = _row_from_df_last(t, solo, source="yahoo")
            if b.date is None:
                missing.append(t)
            bars.append(b)

    else:
        # Single-ticker case
        t = tickers[0] if tickers else ""
        b = _row_from_df_last(t, bulk, source="yahoo")
        if b.date is None:
            with suppress_output():
                solo = yf.download(
                    tickers=t,
                    start=start.isoformat(),
                    end=end.isoformat(),
                    auto_adjust=False,
                    actions=False,
                    progress=False,
                )
            b = _row_from_df_last(t, solo, source="yahoo")
            if b.date is None:
                missing.append(t)
        bars.append(b)

    return bars, missing


# ----------------------------
# Stooq fallback fetch
# ----------------------------

def _stooq_symbol_for_us_equity(ticker: str) -> str:
    """
    Stooq uses lowercase symbols and often '.us' suffix for US equities, e.g. aapl.us
    Stooq also tends to use '.' instead of '-' for class shares in some cases.
    We'll attempt a reasonable normalization:
      - lowercase
      - replace '-' with '.'
      - append '.us' if no exchange suffix is present
    """
    sym = ticker.lower()
    sym = sym.replace("-", ".")
    # If user already has suffix like ".us" or ".de", keep it.
    if "." in sym:
        # If it's a class share like brk.b, we still need .us -> brk.b.us
        # Detect if last segment looks like an exchange code; if not, append .us.
        parts = sym.split(".")
        last = parts[-1]
        # Common stooq suffixes: us, de, uk, pl, etc.
        if len(last) in (2, 3) and last.isalpha():
            return sym
        return sym + ".us"
    return sym + ".us"


def fetch_stooq_lastbar(ticker: str, timeout_s: int = 15) -> Bar:
    """
    Fetch daily history from Stooq CSV endpoint and return the last available row.
    URL format:
      https://stooq.com/q/d/l/?s={symbol}&i=d
    """
    sym = _stooq_symbol_for_us_equity(ticker)
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
        # Stooq CSV is typically utf-8
        text = raw.decode("utf-8", errors="replace")
        if not text.strip():
            return Bar(ticker=ticker, source="stooq")

        from io import StringIO
        df = pd.read_csv(StringIO(text))

        # Expected columns: Date, Open, High, Low, Close, Volume
        # Sometimes Volume absent; handle gracefully.
        if "Date" not in df.columns or df.empty:
            return Bar(ticker=ticker, source="stooq")

        # Ensure sorted by date
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")
        if df.empty:
            return Bar(ticker=ticker, source="stooq")

        last = df.iloc[-1]
        date_str = pd.Timestamp(last["Date"]).date().isoformat()

        def fcol(name: str) -> Optional[float]:
            if name not in df.columns:
                return None
            v = last.get(name)
            return float(v) if pd.notna(v) else None

        def icol(name: str) -> Optional[int]:
            if name not in df.columns:
                return None
            v = last.get(name)
            return int(v) if pd.notna(v) else None

        return Bar(
            ticker=ticker,
            date=date_str,
            open=fcol("Open"),
            high=fcol("High"),
            low=fcol("Low"),
            close=fcol("Close"),
            adj_close=None,  # Stooq daily endpoint doesn’t provide Adj Close
            volume=icol("Volume"),
            source="stooq",
        )

    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return Bar(ticker=ticker, source="stooq")
    except Exception:
        # Any parsing edge-case: treat as unavailable
        return Bar(ticker=ticker, source="stooq")


# ----------------------------
# HTML generation
# ----------------------------

def format_html(df: pd.DataFrame, title: str, asof: str, missing: list[str]) -> str:
    dff = df.copy()

    # Numeric formatting
    for c in ["Open", "High", "Low", "Close", "Adj Close"]:
        if c in dff.columns:
            dff[c] = pd.to_numeric(dff[c], errors="coerce").round(4)

    if "Volume" in dff.columns:
        dff["Volume"] = pd.to_numeric(dff["Volume"], errors="coerce").astype("Int64")

    # Put Source at the end (or keep if already)
    cols = [c for c in dff.columns if c != "Source"] + (["Source"] if "Source" in dff.columns else [])
    dff = dff[cols]

    table_html = dff.to_html(index=False, classes="table", border=0, na_rep="")

    missing_html = ""
    if missing:
        missing_html = f'<div class="warn"><strong>No OHLCV from Yahoo or Stooq:</strong> {", ".join(missing)}</div>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, "Helvetica Neue", Arial, sans-serif; margin: 24px; }}
    header {{ display: flex; align-items: baseline; justify-content: space-between; gap: 16px; flex-wrap: wrap; }}
    h1 {{ margin: 0 0 6px 0; font-size: 22px; }}
    .meta {{ color: #555; font-size: 14px; }}
    .nav a {{ margin-right: 12px; }}
    .wrap {{ overflow-x: auto; }}
    .warn {{ margin-top: 12px; padding: 10px 12px; background: #fff6d6; border: 1px solid #f2d184; border-radius: 10px; }}
    table.table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    table.table th, table.table td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
    table.table th:first-child, table.table td:first-child {{ text-align: left; }}
    table.table th {{ background: #f6f6f6; position: sticky; top: 0; }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>{title}</h1>
      <div class="meta">As of: {asof}</div>
    </div>
    <div class="nav">
      <a href="./index.html">Home</a>
      <a href="./ohlcv.html">Component OHLCV</a>
    </div>
  </header>

  {missing_html}

  <div class="wrap">
    {table_html}
  </div>
</body>
</html>
"""


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", required=True, help="Path to constituents CSV")
    ap.add_argument("--out", required=True, help="Output HTML path (e.g., report/ohlcv.html)")
    ap.add_argument("--title", default="Component OHLCV (Most Recent Session)")
    ap.add_argument("--lookback-days", type=int, default=14, help="Lookback window for daily bars (handles weekends/holidays)")
    ap.add_argument("--stooq-timeout", type=int, default=15, help="Timeout seconds per ticker for Stooq fallback")
    args = ap.parse_args()

    tickers_path = Path(args.tickers).expanduser()
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tickers = read_tickers(tickers_path)
    if not tickers:
        print(f"No tickers found in {tickers_path}", file=sys.stderr)
        return 2

    # 1) Yahoo primary
    yahoo_bars, yahoo_missing = fetch_yahoo_lastbars(tickers, lookback_days=args.lookback_days)

    # Convert to dict for easy replacement
    bars_by_ticker = {b.ticker: b for b in yahoo_bars}

    # 2) Stooq fallback for those missing from Yahoo
    final_missing: list[str] = []
    for t in yahoo_missing:
        b2 = fetch_stooq_lastbar(t, timeout_s=args.stooq_timeout)
        if b2.date is None:
            final_missing.append(t)
            # keep the yahoo bar (empty), but mark source as yahoo
            b = bars_by_ticker.get(t, Bar(ticker=t, source="yahoo"))
            bars_by_ticker[t] = b
        else:
            bars_by_ticker[t] = b2

    # Ensure order matches constituents
    final_bars = [bars_by_ticker.get(t, Bar(ticker=t)) for t in tickers]
    df = pd.DataFrame([bar_to_row(b) for b in final_bars])

    # As-of: max Date present
    asof = "N/A"
    if "Date" in df.columns:
        dates = pd.to_datetime(df["Date"], errors="coerce")
        if dates.notna().any():
            asof = dates.max().date().isoformat()

    html = format_html(df, args.title, asof, final_missing)
    out_path.write_text(html, encoding="utf-8")

    # Exit non-zero ONLY if you want to fail the pipeline when data missing.
    # Right now we keep it successful and show missing in the HTML.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
