#!/usr/bin/env python3
"""
The Great Lakes Index (GLI) — PRO engine (Yahoo via yfinance)

Price-weighted index with daily Open/High/Low/Close + TotalVolume.
Base: 100.00 on 2024-12-31.

Key features:
- Fetch daily OHLCV from Yahoo (yfinance)
- Strict completeness checks (all tickers present per date)
- Divisor events for membership changes/corporate actions (DeltaSum continuity)
- Outputs: CSV, SQLite, HTML+PNG report
"""
from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

BASE_DATE = "2024-12-31"
BASE_VALUE = 100.00

try:
    import yfinance as yf
except Exception:
    yf = None


def read_tickers(path: Path) -> list[str]:
    df = pd.read_csv(path)
    if "Ticker" in df.columns:
        tickers = df["Ticker"].astype(str).tolist()
    else:
        tickers = df.iloc[:, 0].astype(str).tolist()
    tickers = [t.strip().upper() for t in tickers if str(t).strip()]
    seen = set()
    out: list[str] = []
    for t in tickers:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def normalize_prices_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}

    def col(name: str) -> Optional[str]:
        return cols.get(name.lower())

    required = ["date", "ticker", "open", "high", "low", "close"]
    missing = [r for r in required if r not in cols]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Present: {list(df.columns)}")

    out = df.copy()
    out.rename(
        columns={
            col("date"): "Date",
            col("ticker"): "Ticker",
            col("open"): "Open",
            col("high"): "High",
            col("low"): "Low",
            col("close"): "Close",
        },
        inplace=True,
    )

    vcol = col("volume")
    if vcol is not None and vcol != "Volume":
        out.rename(columns={vcol: "Volume"}, inplace=True)
    if "Volume" not in out.columns:
        out["Volume"] = 0

    out["Date"] = pd.to_datetime(out["Date"]).dt.date.astype(str)
    out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip()

    for c in ["Open", "High", "Low", "Close"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce").fillna(0)

    out = out.dropna(subset=["Date", "Ticker"])
    return out[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]]


def validate_completeness(prices_df: pd.DataFrame, tickers: list[str]) -> None:
    expected = set(tickers)
    problems = []
    for d, g in prices_df.groupby("Date"):
        found = set(g["Ticker"].unique())
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        if missing or extra:
            problems.append((d, missing, extra))
    if problems:
        lines = []
        for d, missing, extra in problems[:50]:
            if missing:
                lines.append(
                    f"{d}: missing {len(missing)} tickers (e.g., {', '.join(missing[:12])}{'...' if len(missing) > 12 else ''})"
                )
            if extra:
                lines.append(
                    f"{d}: unexpected {len(extra)} tickers (e.g., {', '.join(extra[:12])}{'...' if len(extra) > 12 else ''})"
                )
        raise ValueError("Strict mode: incomplete/invalid daily coverage detected.\n" + "\n".join(lines))


def fetch_yahoo_daily(tickers: list[str], start: str, end: str, auto_adjust: bool) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not installed. Run: pip install yfinance")

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=auto_adjust,
        actions=False,
        threads=True,
        progress=False,
    )

    rows = []
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if t not in data.columns.get_level_values(0):
                continue
            sub = data[t].copy().reset_index()
            keep = ["Date", "Open", "High", "Low", "Close"]
            if "Volume" in sub.columns:
                keep.append("Volume")
            sub = sub[keep].copy()
            sub["Ticker"] = t
            if "Volume" not in sub.columns:
                sub["Volume"] = 0
            rows.append(sub[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]])
        if not rows:
            raise RuntimeError("No data returned from Yahoo for the requested tickers/date range.")
        df = pd.concat(rows, ignore_index=True)
    else:
        df = data.reset_index()
        df["Ticker"] = tickers[0]
        if "Volume" not in df.columns:
            df["Volume"] = 0
        df = df.rename(columns={"Date": "Date"})
        df = df[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]]

    df["Date"] = pd.to_datetime(df["Date"]).dt.date.astype(str)
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    return df


@dataclass
class DivisorEvent:
    date: str
    event_type: str
    ticker: str
    note: str
    delta_sum: float


def read_divisor_events(path: Optional[Path]) -> list[DivisorEvent]:
    if path is None or not path.exists():
        return []
    df = pd.read_csv(path)
    req = {"Date", "Type", "Ticker", "DeltaSum", "Note"}
    if not req.issubset(set(df.columns)):
        raise ValueError(f"Divisor events file must contain columns: {sorted(req)}")
    df["Date"] = pd.to_datetime(df["Date"]).dt.date.astype(str)
    df["Type"] = df["Type"].astype(str).str.lower().str.strip()
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["DeltaSum"] = pd.to_numeric(df["DeltaSum"], errors="coerce").fillna(0.0)
    df["Note"] = df["Note"].astype(str)
    events = [
        DivisorEvent(str(r["Date"]), str(r["Type"]), str(r["Ticker"]), str(r["Note"]), float(r["DeltaSum"]))
        for _, r in df.iterrows()
    ]
    events.sort(key=lambda e: e.date)
    return events


def compute_base_divisor(prices_df: pd.DataFrame) -> float:
    base = prices_df.loc[prices_df["Date"] == BASE_DATE]
    if base.empty:
        raise ValueError(f"No prices found for base date {BASE_DATE}.")
    s = float(base["Close"].sum())
    if s <= 0:
        raise ValueError("Base sum of closes not positive.")
    return s / BASE_VALUE


def apply_divisor_events(prev_sum_close: float, events_for_date: list[DivisorEvent], prev_index_close: float) -> float:
    delta = sum(e.delta_sum for e in events_for_date)
    return (prev_sum_close + delta) / prev_index_close


def aggregate_index(prices_df: pd.DataFrame, tickers: list[str], events: list[DivisorEvent]) -> pd.DataFrame:
    prices_df = normalize_prices_df(prices_df)
    prices_df = prices_df[prices_df["Ticker"].isin(set(tickers))].copy()

    divisor = compute_base_divisor(prices_df)

    grouped = (
        prices_df.groupby("Date", as_index=False)
        .agg(
            SumOpen=("Open", "sum"),
            SumHigh=("High", "sum"),
            SumLow=("Low", "sum"),
            SumClose=("Close", "sum"),
            TotalVolume=("Volume", "sum"),
            Rows=("Ticker", "nunique"),
        )
        .sort_values("Date")
    )

    ev_by_date: dict[str, list[DivisorEvent]] = {}
    for e in events:
        ev_by_date.setdefault(e.date, []).append(e)

    out = []
    prev_sum_close = None
    prev_index_close = None
    current_divisor = divisor

    for _, r in grouped.iterrows():
        d = str(r["Date"])
        if d in ev_by_date:
            if prev_sum_close is None or prev_index_close is None:
                raise ValueError(f"Divisor event on {d} but no prior day available.")
            current_divisor = apply_divisor_events(prev_sum_close, ev_by_date[d], prev_index_close)

        sum_open = float(r["SumOpen"])
        sum_high = float(r["SumHigh"])
        sum_low = float(r["SumLow"])
        sum_close = float(r["SumClose"])
        total_vol = float(r["TotalVolume"])

        idx_open = sum_open / current_divisor
        idx_high = sum_high / current_divisor
        idx_low = sum_low / current_divisor
        idx_close = sum_close / current_divisor

        out.append(
            {
                "Date": d,
                "GLI_Open": idx_open,
                "GLI_High": idx_high,
                "GLI_Low": idx_low,
                "GLI_Close": idx_close,
                "TotalVolume": total_vol,
                "Divisor": current_divisor,
                "SumOpen": sum_open,
                "SumHigh": sum_high,
                "SumLow": sum_low,
                "SumClose": sum_close,
                "RowsLoaded": int(r["Rows"]),
            }
        )
        prev_sum_close = sum_close
        prev_index_close = idx_close

    return pd.DataFrame(out)


SQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
  date TEXT NOT NULL,
  ticker TEXT NOT NULL,
  open REAL,
  high REAL,
  low REAL,
  close REAL,
  volume REAL,
  PRIMARY KEY(date, ticker)
);

CREATE TABLE IF NOT EXISTS index_levels (
  date TEXT PRIMARY KEY,
  gli_open REAL,
  gli_high REAL,
  gli_low REAL,
  gli_close REAL,
  total_volume REAL,
  divisor REAL,
  sum_open REAL,
  sum_high REAL,
  sum_low REAL,
  sum_close REAL,
  rows_loaded INTEGER
);

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT
);
"""


def _table_columns(con: sqlite3.Connection, table: str) -> set[str]:
    cur = con.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


def sqlite_init_and_migrate(db_path: Path) -> None:
    con = sqlite3.connect(db_path)
    try:
        con.executescript(SQL_SCHEMA)

        cols_prices = _table_columns(con, "prices")
        if "volume" not in cols_prices:
            con.execute("ALTER TABLE prices ADD COLUMN volume REAL")

        cols_idx = _table_columns(con, "index_levels")
        if "total_volume" not in cols_idx:
            con.execute("ALTER TABLE index_levels ADD COLUMN total_volume REAL")

        con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", ("base_date", BASE_DATE))
        con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", ("base_value", str(BASE_VALUE)))
        con.commit()
    finally:
        con.close()


def sqlite_upsert_prices(db_path: Path, prices_df: pd.DataFrame) -> None:
    prices_df = normalize_prices_df(prices_df)
    con = sqlite3.connect(db_path)
    try:
        con.executemany(
            "INSERT OR REPLACE INTO prices(date,ticker,open,high,low,close,volume) VALUES(?,?,?,?,?,?,?)",
            [
                (
                    r.Date,
                    r.Ticker,
                    float(r.Open) if pd.notna(r.Open) else None,
                    float(r.High) if pd.notna(r.High) else None,
                    float(r.Low) if pd.notna(r.Low) else None,
                    float(r.Close) if pd.notna(r.Close) else None,
                    float(r.Volume) if pd.notna(r.Volume) else 0.0,
                )
                for r in prices_df.itertuples(index=False)
            ],
        )
        con.commit()
    finally:
        con.close()


def sqlite_upsert_index(db_path: Path, idx_df: pd.DataFrame) -> None:
    con = sqlite3.connect(db_path)
    try:
        con.executemany(
            """INSERT OR REPLACE INTO index_levels
            (date,gli_open,gli_high,gli_low,gli_close,total_volume,divisor,sum_open,sum_high,sum_low,sum_close,rows_loaded)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (
                    r.Date,
                    float(r.GLI_Open),
                    float(r.GLI_High),
                    float(r.GLI_Low),
                    float(r.GLI_Close),
                    float(r.TotalVolume),
                    float(r.Divisor),
                    float(r.SumOpen),
                    float(r.SumHigh),
                    float(r.SumLow),
                    float(r.SumClose),
                    int(r.RowsLoaded),
                )
                for r in idx_df.itertuples(index=False)
            ],
        )
        con.commit()
    finally:
        con.close()


def make_chart_png(idx_df: pd.DataFrame, out_png: Path) -> None:
    """
    Render a candlestick chart for GLI (Open/High/Low/Close) with volume bars.

    Note:
      - The "volume" plotted is GLI TotalVolume (sum of constituent volumes),
        not exchange volume for a single traded instrument.
      - Requires: mplfinance (pip install mplfinance)
    """
    try:
        import mplfinance as mpf
    except Exception as e:
        raise RuntimeError(
            "mplfinance is required for candlestick charts. "
            "Install it in your venv with: pip install mplfinance"
        ) from e

    df = idx_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # Build OHLCV frame for mplfinance
    ohlcv = df.set_index("Date")[["GLI_Open", "GLI_High", "GLI_Low", "GLI_Close", "TotalVolume"]].copy()
    ohlcv.columns = ["Open", "High", "Low", "Close", "Volume"]

    # mplfinance expects numeric Volume; fill any missing with 0
    ohlcv["Volume"] = pd.to_numeric(ohlcv["Volume"], errors="coerce").fillna(0)

    mpf.plot(
        ohlcv,
        type="candle",
        style="yahoo",
        title="The Great Lakes Index (GLI) - Candlestick",
        ylabel="Index Level",
        volume=True,
        savefig=dict(fname=str(out_png), dpi=150, bbox_inches="tight"),
    )



def _fmt_int(x: float) -> str:
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return "0"


def make_html_report(idx_df: pd.DataFrame, out_html: Path, chart_png_name: str) -> None:
    df = idx_df.sort_values("Date").copy()
    latest = df.iloc[-1].to_dict()
    prev = df.iloc[-2].to_dict() if len(df) >= 2 else None

    change = None
    change_pct = None
    if prev:
        change = latest["GLI_Close"] - prev["GLI_Close"]
        change_pct = (change / prev["GLI_Close"]) * 100 if prev["GLI_Close"] else None

    tail = df.tail(20).copy()
    for c in ["GLI_Open", "GLI_High", "GLI_Low", "GLI_Close"]:
        tail[c] = tail[c].map(lambda x: f"{float(x):,.2f}")
    tail["Divisor"] = tail["Divisor"].map(lambda x: f"{float(x):,.6f}")
    tail["TotalVolume"] = tail["TotalVolume"].map(_fmt_int)

    tail_html = tail[["Date", "GLI_Open", "GLI_High", "GLI_Low", "GLI_Close", "TotalVolume", "Divisor"]].to_html(
        index=False, escape=True
    )

    vol_latest = _fmt_int(latest.get("TotalVolume", 0))

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"/><title>The Great Lakes Index (GLI)</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin:24px; }}
.card {{ border:1px solid #ddd; border-radius:12px; padding:16px; max-width:980px; }}
.kpi {{ display:flex; gap:24px; flex-wrap:wrap; }}
.kpi div {{ min-width:170px; }}
table {{ border-collapse:collapse; width:100%; }}
th,td {{ border:1px solid #ddd; padding:8px; text-align:right; }}
th:first-child,td:first-child {{ text-align:left; }}
.muted {{ color:#666; }}
</style></head>
<body><div class="card">
<h1>The Great Lakes Index (GLI)</h1>
<div class="muted">Price-weighted • Base {BASE_VALUE:.2f} on {BASE_DATE}</div>
<div class="muted" style="margin-top:8px;">
  <a href="./ohlcv.html">Component OHLCV</a>
</div>

<div class="kpi" style="margin-top:16px;">
  <div><b>Latest Date</b><br/>{latest["Date"]}</div>
  <div><b>Close</b><br/>{latest["GLI_Close"]:,.2f}</div>
  <div><b>High</b><br/>{latest["GLI_High"]:,.2f}</div>
  <div><b>Low</b><br/>{latest["GLI_Low"]:,.2f}</div>
  <div><b>Total Volume</b><br/>{vol_latest}</div>
  <div><b>Day Change</b><br/>{("" if change is None else f"{change:+.2f} ({change_pct:+.2f}%)")}</div>
</div>

<div style="margin-top:16px;">
  <img src="{chart_png_name}" style="max-width:100%; border:1px solid #eee; border-radius:8px;"/>
</div>

<h2 style="margin-top:20px;">Recent Levels</h2>
{tail_html}

</div></body></html>"""
    out_html.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute and track The Great Lakes Index (GLI).")
    p.add_argument("--tickers", required=True, type=Path, help="CSV containing a 'Ticker' column (or first column).")
    p.add_argument("--prices", type=Path, help="CSV input with OHLC(V) if not fetching.")
    p.add_argument("--fetch", choices=["yfinance"], help="Fetch OHLCV from Yahoo via yfinance.")
    p.add_argument("--start", default=BASE_DATE, help="Start date (YYYY-MM-DD). Default is base date.")
    p.add_argument("--end", default=None, help="End date (YYYY-MM-DD). Default is today.")
    p.add_argument("--auto-adjust", action="store_true", default=False, help="Use adjusted OHLC from Yahoo.")
    p.add_argument("--strict", action="store_true", default=True, help="Fail if any date is missing a ticker.")
    p.add_argument("--no-strict", dest="strict", action="store_false", help="Disable strict completeness checks.")
    p.add_argument("--events", type=Path, default=None, help="Divisor events CSV (membership/splits/spinoffs).")
    p.add_argument("--out", type=Path, default=Path("gli_output.csv"), help="Output index levels CSV.")
    p.add_argument("--prices-out", type=Path, default=None, help="If fetching, write normalized OHLCV to this CSV.")
    p.add_argument("--db", type=Path, default=None, help="SQLite DB path to store prices and index levels.")
    p.add_argument("--report-dir", type=Path, default=None, help="Directory to write HTML+PNG report.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    tickers = read_tickers(args.tickers)

    # Interpret --end as an INCLUSIVE date (human-friendly).
    # yfinance's end parameter is end-EXCLUSIVE, so add +1 day when fetching.
    end_inclusive = args.end or pd.Timestamp.today().date().isoformat()
    end_fetch = (pd.to_datetime(end_inclusive) + pd.Timedelta(days=1)).date().isoformat()

    if args.fetch == "yfinance":
        prices_df = fetch_yahoo_daily(tickers, args.start, end_fetch, auto_adjust=args.auto_adjust)
        prices_df = normalize_prices_df(prices_df)
        if args.prices_out:
            prices_df.to_csv(args.prices_out, index=False)
    elif args.prices:
        prices_df = normalize_prices_df(pd.read_csv(args.prices))
    else:
        raise ValueError("Provide either --prices (CSV) or --fetch yfinance.")

    if args.strict:
        validate_completeness(prices_df, tickers)

    events = read_divisor_events(args.events)
    idx_df = aggregate_index(prices_df, tickers, events)
    idx_df.to_csv(args.out, index=False)

    if args.db:
        sqlite_init_and_migrate(args.db)
        sqlite_upsert_prices(args.db, prices_df)
        sqlite_upsert_index(args.db, idx_df)

    if args.report_dir:
        args.report_dir.mkdir(parents=True, exist_ok=True)
        png = args.report_dir / "gli_close.png"
        html = args.report_dir / "index.html"
        make_chart_png(idx_df, png)
        make_html_report(idx_df, html, png.name)

    print(f"Wrote index CSV: {args.out}")
    if args.prices_out:
        print(f"Wrote prices CSV: {args.prices_out}")
    if args.db:
        print(f"Updated SQLite DB: {args.db}")
    if args.report_dir:
        print(f"Wrote report: {args.report_dir/'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
