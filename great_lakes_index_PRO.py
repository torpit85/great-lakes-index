#!/usr/bin/env python3
"""Great Lakes Index live engine with an immutable accepted 2026 close anchor.

Original index base: 100 on 2005-08-01.
The published live series is anchored to the accepted 2025-12-31 carry.
Accepted closes and divisors through 2026-08-04 are never recalculated from
a public market-data vendor. Later sessions roll forward from the accepted
divisor using the active roster and unadjusted daily Yahoo bars.
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Optional

import pandas as pd

INDEX_BASE_DATE = "2005-08-01"
INDEX_BASE_VALUE = Decimal("100")
LIVE_SERIES_START = "2025-12-31"
DECIMAL_PRECISION = 180

try:
    import yfinance as yf
except Exception:
    yf = None


def _decimal(value: object) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if pd.isna(value):
        raise ValueError("Missing numeric value")
    return Decimal(str(value))


def read_constituents(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    if "Ticker" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Ticker"})
    for column, default in (
        ("Active", "Y"),
        ("StartDate", LIVE_SERIES_START),
        ("EndDate", ""),
    ):
        if column not in df.columns:
            df[column] = default
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Active"] = df["Active"].astype(str).str.upper().str.strip()
    df["StartDate"] = (
        df["StartDate"].astype(str).str.strip()
        .replace({"": LIVE_SERIES_START, "nan": LIVE_SERIES_START})
    )
    df["EndDate"] = (
        df["EndDate"].astype(str).str.strip()
        .replace({"nan": "", "NaT": ""})
    )
    df = df[df["Ticker"].ne("") & df["Ticker"].str.lower().ne("nan")].copy()
    if df["Ticker"].duplicated().any():
        duplicates = sorted(df.loc[df["Ticker"].duplicated(), "Ticker"].unique())
        raise ValueError(f"Duplicate constituent identities: {duplicates}")
    return df.reset_index(drop=True)


def active_tickers_for_date(constituents: pd.DataFrame, date_str: str) -> list[str]:
    day = str(pd.to_datetime(date_str).date())
    mask = (
        (constituents["StartDate"] <= day)
        & (
            constituents["EndDate"].eq("")
            | (constituents["EndDate"] >= day)
        )
    )
    return constituents.loc[mask, "Ticker"].tolist()


def tickers_intersecting_range(
    constituents: pd.DataFrame,
    start: str,
    end: str,
) -> list[str]:
    mask = (
        (constituents["StartDate"] <= end)
        & (
            constituents["EndDate"].eq("")
            | (constituents["EndDate"] >= start)
        )
    )
    return list(dict.fromkeys(constituents.loc[mask, "Ticker"].tolist()))


def normalize_prices_df(df: pd.DataFrame) -> pd.DataFrame:
    columns = {str(column).lower(): column for column in df.columns}
    required = ["date", "ticker", "open", "high", "low", "close"]
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(
            f"Missing required price columns: {missing}; present={list(df.columns)}"
        )
    rename = {
        columns["date"]: "Date",
        columns["ticker"]: "Ticker",
        columns["open"]: "Open",
        columns["high"]: "High",
        columns["low"]: "Low",
        columns["close"]: "Close",
    }
    if "volume" in columns:
        rename[columns["volume"]] = "Volume"
    out = df.rename(columns=rename).copy()
    if "Volume" not in out.columns:
        out["Volume"] = 0
    out["Date"] = pd.to_datetime(out["Date"]).dt.date.astype(str)
    out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip()
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["Volume"] = out["Volume"].fillna(0)
    out = out.dropna(subset=["Date", "Ticker"])
    return out[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]]


def fetch_yahoo_daily(
    tickers: list[str],
    start: str,
    end_exclusive: str,
    auto_adjust: bool,
) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not installed")
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end_exclusive,
        group_by="ticker",
        auto_adjust=auto_adjust,
        actions=False,
        threads=True,
        progress=False,
    )
    rows = []
    if isinstance(data.columns, pd.MultiIndex):
        available = set(data.columns.get_level_values(0))
        for ticker in tickers:
            if ticker not in available:
                continue
            sub = data[ticker].copy().reset_index()
            keep = ["Date", "Open", "High", "Low", "Close"]
            if "Volume" in sub.columns:
                keep.append("Volume")
            sub = sub[keep].copy()
            sub["Ticker"] = ticker
            if "Volume" not in sub.columns:
                sub["Volume"] = 0
            rows.append(
                sub[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]]
            )
    else:
        if not tickers:
            raise RuntimeError("No tickers requested")
        sub = data.reset_index()
        sub["Ticker"] = tickers[0]
        if "Volume" not in sub.columns:
            sub["Volume"] = 0
        rows.append(
            sub[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]]
        )
    if not rows:
        raise RuntimeError("Yahoo returned no daily bars")
    return normalize_prices_df(pd.concat(rows, ignore_index=True))


@dataclass(frozen=True)
class DivisorEvent:
    date: str
    event_type: str
    ticker: str
    delta_sum: Decimal
    reset_required: bool
    exact_new_divisor: Optional[Decimal]
    control_id: str
    reference_date: str
    note: str


def read_divisor_events(path: Optional[Path]) -> list[DivisorEvent]:
    if path is None or not path.exists():
        return []
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    required = {
        "Date", "Type", "Ticker", "DeltaSum", "ResetRequired",
        "ExactNewDivisor", "ControlID", "ReferenceDate", "Note",
    }
    if not rows:
        return []
    if not required.issubset(rows[0]):
        raise ValueError(
            f"Divisor event file must contain {sorted(required)}"
        )
    events = []
    for row in rows:
        exact = row["ExactNewDivisor"].strip()
        events.append(
            DivisorEvent(
                date=str(pd.to_datetime(row["Date"]).date()),
                event_type=row["Type"].strip().lower(),
                ticker=row["Ticker"].strip().upper(),
                delta_sum=Decimal(row["DeltaSum"]),
                reset_required=row["ResetRequired"].strip().upper()
                in {"Y", "YES", "TRUE", "1"},
                exact_new_divisor=Decimal(exact) if exact else None,
                control_id=row["ControlID"].strip(),
                reference_date=str(pd.to_datetime(row["ReferenceDate"]).date()),
                note=row["Note"].strip(),
            )
        )
    return sorted(events, key=lambda event: (event.date, event.control_id))


def read_accepted_chain(path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    required = {
        "Date", "IndexClose", "Divisor", "ComponentSum",
        "RosterCount", "Status",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            f"Accepted-chain file must contain {sorted(required)}"
        )
    dates = [row["Date"] for row in rows]
    if dates != sorted(dates) or len(dates) != len(set(dates)):
        raise ValueError("Accepted-chain dates must be unique and sorted")
    return rows


def _validate_live_day_rows(
    rows: pd.DataFrame,
    active_tickers: list[str],
    day: str,
) -> None:
    """Reject incomplete or internally invalid public-vendor bars.

    A batch download can contain a row for a ticker while one or more OHLC
    fields are still missing. Identity-only coverage is therefore not enough:
    strict mode requires exactly one complete, finite and positive OHLC bar
    for every active component before an index session may be published.
    """
    active = set(active_tickers)
    selected = rows[rows["Ticker"].isin(active)].copy()
    present = set(selected["Ticker"].unique())
    problems: list[str] = []

    missing = sorted(active - present)
    if missing:
        problems.append("missing tickers: " + ", ".join(missing))

    duplicate_counts = selected.groupby("Ticker").size()
    duplicates = sorted(
        ticker for ticker, count in duplicate_counts.items() if int(count) != 1
    )
    if duplicates:
        problems.append("non-unique ticker rows: " + ", ".join(duplicates))

    for ticker in sorted(active & present):
        ticker_rows = selected[selected["Ticker"] == ticker]
        if len(ticker_rows) != 1:
            continue
        row = ticker_rows.iloc[0]
        values: dict[str, Decimal] = {}
        ticker_problems: list[str] = []
        for column in ["Open", "High", "Low", "Close"]:
            value = row[column]
            if pd.isna(value):
                ticker_problems.append(f"{column}=missing")
                continue
            try:
                numeric = _decimal(value)
            except ValueError:
                ticker_problems.append(f"{column}=invalid")
                continue
            if not numeric.is_finite() or numeric <= 0:
                ticker_problems.append(f"{column}={numeric}")
                continue
            values[column] = numeric

        volume = row["Volume"]
        if pd.isna(volume):
            ticker_problems.append("Volume=missing")
        else:
            try:
                numeric_volume = _decimal(volume)
                if not numeric_volume.is_finite() or numeric_volume < 0:
                    ticker_problems.append(f"Volume={numeric_volume}")
            except ValueError:
                ticker_problems.append("Volume=invalid")

        if ticker_problems:
            problems.append(f"{ticker}: " + ", ".join(ticker_problems))

    if problems:
        raise ValueError(
            f"{day}: incomplete or invalid live component bars; "
            + " | ".join(problems)
        )


def _aggregate_raw_day(
    rows: pd.DataFrame,
    active_tickers: list[str],
) -> tuple[dict[str, Decimal], set[str]]:
    active = set(active_tickers)
    selected = rows[rows["Ticker"].isin(active)].copy()
    found = set(selected["Ticker"].unique())
    sums: dict[str, Decimal] = {}
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        values = [
            _decimal(value)
            for value in selected[column].tolist()
            if pd.notna(value)
        ]
        sums[column] = sum(values, Decimal(0))
    return sums, found



def read_accepted_ohlcv_chain(
    path: Path,
) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    required = {
        "Date", "GLI_Open", "GLI_High", "GLI_Low",
        "GLI_Close", "TotalVolume", "Divisor",
        "SumOpen", "SumHigh", "SumLow", "SumClose",
        "RosterCount", "Status",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            f"Accepted OHLCV chain must contain {sorted(required)}"
        )
    dates = [row["Date"] for row in rows]
    if dates != sorted(dates) or len(dates) != len(set(dates)):
        raise ValueError(
            "Accepted OHLCV chain dates must be unique and sorted"
        )
    for row in rows:
        open_value = Decimal(row["GLI_Open"])
        high = Decimal(row["GLI_High"])
        low = Decimal(row["GLI_Low"])
        close = Decimal(row["GLI_Close"])
        if high < max(open_value, close) or low > min(open_value, close):
            raise ValueError(
                f"Accepted aggregate candle constraint failed on {row['Date']}"
            )
    return rows


def aggregate_index(
    prices_df: pd.DataFrame,
    constituents: pd.DataFrame,
    events: list[DivisorEvent],
    accepted_chain: list[dict[str, str]],
    accepted_ohlcv_chain: list[dict[str, str]],
    requested_start: str,
    requested_end: str,
    strict: bool,
) -> pd.DataFrame:
    prices = normalize_prices_df(prices_df)
    prices = prices[
        (prices["Date"] >= requested_start)
        & (prices["Date"] <= requested_end)
    ].copy()

    raw_by_date = {
        day: group.copy()
        for day, group in prices.groupby("Date", sort=True)
    }
    chain_by_date = {row["Date"]: row for row in accepted_chain}
    accepted_dates = [
        day for day in chain_by_date
        if requested_start <= day <= requested_end
    ]
    if not accepted_dates:
        raise ValueError("Requested range does not include the accepted anchor")

    accepted_cutoff = max(chain_by_date)
    events_by_date: dict[str, list[DivisorEvent]] = {}
    for event in events:
        events_by_date.setdefault(event.date, []).append(event)

    output: list[dict[str, object]] = []

    # Immutable accepted portion. Both close and OHLCV chains are
    # frozen accepted artifacts; public vendor bars are never allowed to
    # reshape accepted history.
    ohlcv_by_date = {
        row["Date"]: row
        for row in accepted_ohlcv_chain
    }
    if set(ohlcv_by_date) != set(chain_by_date):
        raise ValueError(
            "Accepted close and OHLCV chain date sets differ"
        )

    for day in sorted(accepted_dates):
        accepted = chain_by_date[day]
        accepted_ohlcv = ohlcv_by_date[day]

        if Decimal(accepted["IndexClose"]) != Decimal(
            accepted_ohlcv["GLI_Close"]
        ):
            raise ValueError(
                f"Accepted close/OHLCV mismatch on {day}"
            )
        if Decimal(accepted["Divisor"]) != Decimal(
            accepted_ohlcv["Divisor"]
        ):
            raise ValueError(
                f"Accepted divisor/OHLCV mismatch on {day}"
            )
        if Decimal(accepted["ComponentSum"]) != Decimal(
            accepted_ohlcv["SumClose"]
        ):
            raise ValueError(
                f"Accepted component sum/OHLCV mismatch on {day}"
            )
        if int(accepted["RosterCount"]) != int(
            accepted_ohlcv["RosterCount"]
        ):
            raise ValueError(
                f"Accepted roster/OHLCV mismatch on {day}"
            )

        output.append({
            "Date": day,
            "GLI_Open": accepted_ohlcv["GLI_Open"],
            "GLI_High": accepted_ohlcv["GLI_High"],
            "GLI_Low": accepted_ohlcv["GLI_Low"],
            "GLI_Close": accepted_ohlcv["GLI_Close"],
            "TotalVolume": accepted_ohlcv["TotalVolume"],
            "Divisor": accepted_ohlcv["Divisor"],
            "SumOpen": accepted_ohlcv["SumOpen"],
            "SumHigh": accepted_ohlcv["SumHigh"],
            "SumLow": accepted_ohlcv["SumLow"],
            "SumClose": accepted_ohlcv["SumClose"],
            "RowsLoaded": int(accepted_ohlcv["RosterCount"]),
            "CloseSource": accepted["Status"],
            "OHLCVSource": accepted_ohlcv["Status"],
        })

    # Live roll-forward after the immutable accepted cutoff.
    live_dates = sorted(
        day for day in raw_by_date
        if accepted_cutoff < day <= requested_end
    )
    current_divisor = Decimal(chain_by_date[accepted_cutoff]["Divisor"])
    previous_sum = Decimal(chain_by_date[accepted_cutoff]["ComponentSum"])
    previous_index = Decimal(chain_by_date[accepted_cutoff]["IndexClose"])

    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        for day in live_dates:
            day_events = events_by_date.get(day, [])
            reset_events = [event for event in day_events if event.reset_required]
            if reset_events:
                exact_values = {
                    event.exact_new_divisor
                    for event in reset_events
                    if event.exact_new_divisor is not None
                }
                if len(exact_values) > 1:
                    raise ValueError(
                        f"Conflicting exact divisors on {day}: {exact_values}"
                    )
                if exact_values:
                    current_divisor = next(iter(exact_values))
                else:
                    delta = sum(
                        (event.delta_sum for event in reset_events),
                        Decimal(0),
                    )
                    current_divisor = (previous_sum + delta) / previous_index

            active = active_tickers_for_date(constituents, day)
            if strict:
                _validate_live_day_rows(raw_by_date[day], active, day)
            sums, found = _aggregate_raw_day(raw_by_date[day], active)
            missing = sorted(set(active) - found)
            if strict and missing:
                raise ValueError(
                    f"{day}: missing {len(missing)} active tickers: "
                    + ", ".join(missing)
                )
            if not found:
                continue

            index_open = sums["Open"] / current_divisor
            index_high = sums["High"] / current_divisor
            index_low = sums["Low"] / current_divisor
            index_close = sums["Close"] / current_divisor
            index_high = max(index_high, index_open, index_close)
            index_low = min(index_low, index_open, index_close)

            output.append({
                "Date": day,
                "GLI_Open": format(index_open, "f"),
                "GLI_High": format(index_high, "f"),
                "GLI_Low": format(index_low, "f"),
                "GLI_Close": format(index_close, "f"),
                "TotalVolume": format(sums["Volume"], "f"),
                "Divisor": format(current_divisor, "f"),
                "SumOpen": format(sums["Open"], "f"),
                "SumHigh": format(sums["High"], "f"),
                "SumLow": format(sums["Low"], "f"),
                "SumClose": format(sums["Close"], "f"),
                "RowsLoaded": len(found),
                "CloseSource": "LIVE_YAHOO_UNADJUSTED",
                "OHLCVSource": "LIVE_YAHOO_UNADJUSTED",
            })
            previous_sum = sums["Close"]
            previous_index = index_close

    frame = pd.DataFrame(output).sort_values("Date").reset_index(drop=True)
    if frame["Date"].duplicated().any():
        raise ValueError("Duplicate output dates")
    return frame


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


def sqlite_init_and_migrate(db_path: Path) -> None:
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(SQL_SCHEMA)
        connection.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)",
            ("original_base_date", INDEX_BASE_DATE),
        )
        connection.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)",
            ("original_base_value", str(INDEX_BASE_VALUE)),
        )
        connection.commit()
    finally:
        connection.close()


def sqlite_upsert_prices(db_path: Path, prices_df: pd.DataFrame) -> None:
    connection = sqlite3.connect(db_path)
    try:
        connection.executemany(
            "INSERT OR REPLACE INTO prices(date,ticker,open,high,low,close,volume) "
            "VALUES(?,?,?,?,?,?,?)",
            [
                (
                    row.Date,
                    row.Ticker,
                    float(row.Open) if pd.notna(row.Open) else None,
                    float(row.High) if pd.notna(row.High) else None,
                    float(row.Low) if pd.notna(row.Low) else None,
                    float(row.Close) if pd.notna(row.Close) else None,
                    float(row.Volume) if pd.notna(row.Volume) else 0.0,
                )
                for row in normalize_prices_df(prices_df).itertuples(index=False)
            ],
        )
        connection.commit()
    finally:
        connection.close()


def sqlite_upsert_index(db_path: Path, index_df: pd.DataFrame) -> None:
    connection = sqlite3.connect(db_path)
    try:
        connection.executemany(
            """INSERT OR REPLACE INTO index_levels
            (date,gli_open,gli_high,gli_low,gli_close,total_volume,divisor,
             sum_open,sum_high,sum_low,sum_close,rows_loaded)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (
                    row.Date,
                    float(row.GLI_Open),
                    float(row.GLI_High),
                    float(row.GLI_Low),
                    float(row.GLI_Close),
                    float(row.TotalVolume),
                    float(row.Divisor),
                    float(row.SumOpen) if str(row.SumOpen) else None,
                    float(row.SumHigh) if str(row.SumHigh) else None,
                    float(row.SumLow) if str(row.SumLow) else None,
                    float(row.SumClose),
                    int(row.RowsLoaded),
                )
                for row in index_df.itertuples(index=False)
            ],
        )
        connection.commit()
    finally:
        connection.close()


def make_chart_png(index_df: pd.DataFrame, output_path: Path) -> None:
    try:
        import mplfinance as mpf
    except Exception as exc:
        raise RuntimeError("mplfinance is required") from exc
    frame = index_df.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    for column in [
        "GLI_Open", "GLI_High", "GLI_Low", "GLI_Close", "TotalVolume"
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    ohlcv = frame.set_index("Date")[
        ["GLI_Open", "GLI_High", "GLI_Low", "GLI_Close", "TotalVolume"]
    ].copy()
    ohlcv.columns = ["Open", "High", "Low", "Close", "Volume"]
    ohlcv["Volume"] = ohlcv["Volume"].fillna(0)
    mpf.plot(
        ohlcv,
        type="candle",
        style="yahoo",
        title="The Great Lakes Index (GLI) - Candlestick",
        ylabel="Index Level",
        volume=True,
        savefig=dict(fname=str(output_path), dpi=150, bbox_inches="tight"),
    )


def _format_volume(value: object) -> str:
    try:
        return f"{int(round(float(value))):,}"
    except Exception:
        return "0"


def make_html_report(
    index_df: pd.DataFrame,
    output_path: Path,
    chart_name: str,
    accepted_cutoff: str,
) -> None:
    frame = index_df.sort_values("Date").copy()
    for column in ["GLI_Open", "GLI_High", "GLI_Low", "GLI_Close"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    latest = frame.iloc[-1].to_dict()
    previous = frame.iloc[-2].to_dict() if len(frame) >= 2 else None
    change = latest["GLI_Close"] - previous["GLI_Close"] if previous else None
    change_pct = (
        change / previous["GLI_Close"] * 100
        if previous and previous["GLI_Close"]
        else None
    )
    tail = frame.tail(20).copy()
    for column in ["GLI_Open", "GLI_High", "GLI_Low", "GLI_Close"]:
        tail[column] = tail[column].map(lambda value: f"{float(value):,.2f}")
    tail["Divisor"] = tail["Divisor"].map(lambda value: f"{float(value):,.6f}")
    tail["TotalVolume"] = tail["TotalVolume"].map(_format_volume)
    table = tail[
        [
            "Date", "GLI_Open", "GLI_High", "GLI_Low",
            "GLI_Close", "TotalVolume", "Divisor",
        ]
    ].to_html(index=False, escape=True)
    volume = _format_volume(latest.get("TotalVolume", 0))
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
.nav {{ margin-top:10px; padding:10px 12px; background:#f6f6f6; border:1px solid #ddd; border-radius:10px; }}
.nav a {{ margin-right:14px; text-decoration:none; }}
</style></head>
<body><div class="card">
<h1>The Great Lakes Index (GLI)</h1>
<div class="muted">Price-weighted • Original base 100.00 on 2005-08-01</div>
<div class="muted">Accepted close chain through {accepted_cutoff}; later sessions roll forward live.</div>
<div class="nav">
  <a href="./index.html"><b>Home</b></a>
  <a href="./history.html">Historical Values</a>
  <a href="./ohlcv.html">Component OHLCV</a>
</div>
<div class="kpi" style="margin-top:16px;">
  <div><b>Latest Date</b><br/>{latest["Date"]}</div>
  <div><b>Close</b><br/>{latest["GLI_Close"]:,.2f}</div>
  <div><b>High</b><br/>{latest["GLI_High"]:,.2f}</div>
  <div><b>Low</b><br/>{latest["GLI_Low"]:,.2f}</div>
  <div><b>Total Volume</b><br/>{volume}</div>
  <div><b>Day Change</b><br/>{("" if change is None else f"{change:+.2f} ({change_pct:+.2f}%)")}</div>
</div>
<div style="margin-top:16px;">
  <img src="{chart_name}" style="max-width:100%; border:1px solid #eee; border-radius:8px;"/>
</div>
<h2 style="margin-top:20px;">Recent Levels</h2>
{table}
</div></body></html>"""
    output_path.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", required=True, type=Path)
    parser.add_argument("--accepted-chain", required=True, type=Path)
    parser.add_argument("--accepted-ohlcv-chain", required=True, type=Path)
    parser.add_argument("--prices", type=Path)
    parser.add_argument("--fetch", choices=["yfinance"])
    parser.add_argument("--start", default=LIVE_SERIES_START)
    parser.add_argument("--end", default=None)
    parser.add_argument("--auto-adjust", action="store_true", default=False)
    parser.add_argument("--strict", action="store_true", default=True)
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    parser.add_argument("--events", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("gli_levels.csv"))
    parser.add_argument("--prices-out", type=Path, default=None)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    end_inclusive = args.end or pd.Timestamp.today().date().isoformat()
    if args.start > end_inclusive:
        raise ValueError("Start date is after end date")

    constituents = read_constituents(args.tickers)
    accepted_chain = read_accepted_chain(args.accepted_chain)
    accepted_ohlcv_chain = read_accepted_ohlcv_chain(
        args.accepted_ohlcv_chain
    )
    accepted_cutoff = max(row["Date"] for row in accepted_chain)
    events = read_divisor_events(args.events)

    if args.fetch == "yfinance":
        tickers = tickers_intersecting_range(
            constituents, args.start, end_inclusive
        )
        end_exclusive = (
            pd.to_datetime(end_inclusive) + pd.Timedelta(days=1)
        ).date().isoformat()
        prices_df = fetch_yahoo_daily(
            tickers, args.start, end_exclusive, args.auto_adjust
        )
    elif args.prices:
        prices_df = normalize_prices_df(pd.read_csv(args.prices))
    else:
        raise ValueError("Provide --prices or --fetch yfinance")

    if args.prices_out:
        normalize_prices_df(prices_df).to_csv(
            args.prices_out, index=False
        )

    index_df = aggregate_index(
        prices_df=prices_df,
        constituents=constituents,
        events=events,
        accepted_chain=accepted_chain,
        accepted_ohlcv_chain=accepted_ohlcv_chain,
        requested_start=args.start,
        requested_end=end_inclusive,
        strict=args.strict,
    )
    index_df.to_csv(args.out, index=False)

    if args.db:
        sqlite_init_and_migrate(args.db)
        sqlite_upsert_prices(args.db, prices_df)
        sqlite_upsert_index(args.db, index_df)

    if args.report_dir:
        args.report_dir.mkdir(parents=True, exist_ok=True)
        chart = args.report_dir / "gli_close.png"
        report = args.report_dir / "index.html"
        make_chart_png(index_df, chart)
        make_html_report(index_df, report, chart.name, accepted_cutoff)

    print(f"Wrote index CSV: {args.out}")
    if args.prices_out:
        print(f"Wrote prices CSV: {args.prices_out}")
    if args.db:
        print(f"Updated SQLite DB: {args.db}")
    if args.report_dir:
        print(f"Wrote report: {args.report_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
