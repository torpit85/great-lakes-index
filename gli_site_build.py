#!/usr/bin/env python3
"""Build the Great Lakes Index static site.

The builder preserves the accepted calculation layer and only produces public
presentation artifacts under ``report/``. Historical pages are based on the
compact, hash-documented files in ``site_data/`` and are merged with the live
``gli_levels.csv`` output generated immediately before this script runs.
"""

from __future__ import annotations

import csv
import html
import json
import math
import re
import shutil
from collections import defaultdict
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

BASE_DATE = "2005-08-01"
BASE_VALUE = Decimal("100")

ROOT = Path(__file__).resolve().parent
LEVELS = ROOT / "gli_levels.csv"
PRICES = ROOT / "gli_prices.csv"
CONSTITUENTS = ROOT / "constituents_great_lakes.csv"
COMPANY_CACHE = ROOT / "company_names.csv"
LIVE_ANCHOR = ROOT / "GLI_2026_live_anchor.json"
SITE_DATA = ROOT / "site_data"
HISTORICAL_COMPANY_NAMES = SITE_DATA / "historical_company_names.csv"
HISTORICAL_BASE = SITE_DATA / "gli_historical_ohlcv_through_2025.csv"
ROSTER_BASE = SITE_DATA / "component_roster_history_through_2025.json"
WEIGHTS_BASE = SITE_DATA / "weights"
REPORT = ROOT / "report"
REPORT_DATA = REPORT / "data"

NAV_ITEMS = [
    ("index.html", "Home"),
    ("history.html", "Historical Values"),
    ("market-moves.html", "Market Moves"),
    ("milestones.html", "Closing Milestones"),
    ("weights.html", "Component Weights"),
    ("components.html", "Component History"),
    ("ohlcv.html", "Component OHLCV"),
]

CSS_BLOCK = """
<style>
:root{
  --ink:#172033; --muted:#667085; --line:#d9dee8; --panel:#ffffff;
  --soft:#f5f7fa; --blue:#175cd3; --green:#087443; --red:#b42318;
  --gold:#946200;
}
*{box-sizing:border-box}
body{margin:24px; color:var(--ink); background:#fff; font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif}
a{color:var(--blue)}
.gli-nav{display:flex;flex-wrap:wrap;gap:7px;margin:0 0 18px;padding:10px;border:1px solid var(--line);border-radius:14px;background:var(--soft)}
.gli-nav a{padding:7px 10px;border-radius:9px;text-decoration:none;font-weight:750;color:#344054}
.gli-nav a:hover,.gli-nav a.active{background:#fff;color:var(--blue);box-shadow:0 1px 2px rgba(16,24,40,.08)}
#gli-tickerbar{font-weight:800;font-size:18px;padding:12px 16px;margin:0 0 14px;border:1px solid var(--line);border-radius:12px;background:linear-gradient(180deg,#fafafa,#f0f0f0);display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap}
#gltxt .up,.positive{color:var(--green)} #gltxt .down,.negative{color:var(--red)} #gltxt .flat{color:#475467}
.gli-pill{font-size:12px;font-weight:800;border:1px solid var(--line);border-radius:999px;padding:6px 10px;background:#fff}
.gli-symbolbox,.panel{border:1px solid var(--line);border-radius:14px;padding:16px;margin:0 0 18px;background:var(--panel);box-shadow:0 1px 2px rgba(16,24,40,.04)}
.gli-grid,.summary-grid{display:grid;grid-template-columns:repeat(5,minmax(130px,1fr));gap:10px 16px}
.gli-k{color:var(--muted);font-size:12px;font-weight:800;text-transform:uppercase;letter-spacing:.55px}
.gli-v{font-size:18px;font-weight:900;margin-top:2px}
.gli-mini,.muted{color:var(--muted);font-size:13px;line-height:1.4}
.gli-linkbar{margin:8px 0 18px}.gli-linkbar a{text-decoration:none;font-weight:800}
h1{margin:0 0 4px;font-size:clamp(28px,4vw,42px)} h2{margin:0 0 10px}
.page-head{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;flex-wrap:wrap;margin-bottom:18px}
.tabs{display:flex;gap:7px;overflow-x:auto;padding:3px 0 10px;scrollbar-width:thin}
.tab{border:1px solid var(--line);background:#fff;border-radius:999px;padding:7px 12px;font-weight:750;cursor:pointer;white-space:nowrap;color:#344054}
.tab:hover,.tab.active{border-color:#84adff;background:#eff4ff;color:#1849a9}
.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:12px 0}
select,input[type="search"],input[type="date"]{border:1px solid #cbd3df;border-radius:9px;padding:8px 10px;background:#fff;color:var(--ink)}
input[type="range"]{min-width:240px;accent-color:var(--blue)}
.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:12px;background:#fff}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid #e7eaf0;padding:9px 10px;text-align:right;white-space:nowrap}
th{position:sticky;top:0;z-index:1;background:#f8fafc;color:#475467;font-size:12px;text-transform:uppercase;letter-spacing:.35px}
th:first-child,td:first-child{text-align:left} tbody tr:hover{background:#f8fbff}
.rank{color:var(--muted);font-weight:700}.empty{padding:24px;text-align:center;color:var(--muted)}
.metric-cards{display:grid;grid-template-columns:repeat(4,minmax(160px,1fr));gap:12px;margin:14px 0 18px}
.metric-card{border:1px solid var(--line);border-radius:12px;padding:14px;background:#fff}.metric-card strong{display:block;font-size:23px;margin-top:3px}
.milestone-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}
.milestone{border:1px solid var(--line);border-radius:12px;padding:15px;background:linear-gradient(180deg,#fff,#fafbfd)}
.milestone .level{font-size:27px;font-weight:900;color:var(--gold)}
.heatmap{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:8px;margin-top:14px}
.heat-cell{min-height:86px;border-radius:11px;padding:11px;border:1px solid rgba(0,0,0,.08);display:flex;flex-direction:column;justify-content:space-between;box-shadow:inset 0 1px rgba(255,255,255,.25)}
.heat-symbol{font-weight:900;font-size:17px}.heat-company{font-size:11px;line-height:1.25;opacity:.9;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.heat-weight{font-size:15px;font-weight:850}
.timeline{display:grid;gap:14px}.snapshot{border:1px solid var(--line);border-radius:14px;padding:16px;background:#fff}.snapshot-head{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;flex-wrap:wrap}.snapshot-date{font-size:20px;font-weight:900}.snapshot-title{color:var(--muted);font-weight:700}.change-row{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}.badge{border-radius:999px;padding:5px 9px;font-size:12px;font-weight:800}.badge.add{background:#ecfdf3;color:#027a48}.badge.remove{background:#fef3f2;color:#b42318}.event-list{display:grid;gap:7px;margin:10px 0}.component-event{border-left:4px solid #7f56d9;background:#f9f5ff;padding:8px 10px;border-radius:8px;font-size:13px;color:#344054}.component-event.security{border-left-color:#f79009;background:#fffaeb}.event-label{font-weight:900;color:#53389e}.component-event.security .event-label{color:#b54708}.event-detail{color:#667085;margin-left:5px}.component-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:7px;margin-top:12px}.component{border:1px solid #e2e7ef;border-radius:8px;padding:7px 9px;background:#f9fafb;font-size:13px}.component.added{border-color:#86d5ad;background:#ecfdf3;font-weight:750}.component-symbol{font-weight:850}.component-name{color:var(--muted);font-size:11px;margin-top:2px}
.chart-shell{border:1px solid var(--line);border-radius:14px;padding:8px;background:#fff;margin:14px 0 18px}.chart-status{padding:24px;color:var(--muted);text-align:center}
.source-note{border-left:4px solid #84adff;background:#f5f8ff;padding:10px 12px;margin:12px 0;color:#344054;font-size:13px}
@media(max-width:920px){.gli-grid,.summary-grid{grid-template-columns:repeat(2,minmax(130px,1fr))}.metric-cards{grid-template-columns:repeat(2,minmax(140px,1fr))}}
@media(max-width:560px){body{margin:14px}.metric-cards{grid-template-columns:1fr}.component-grid{grid-template-columns:1fr}.heatmap{grid-template-columns:repeat(2,minmax(0,1fr))}th,td{padding:8px 7px}}
</style>
"""


def nav_html(active: str) -> str:
    links = []
    for href, label in NAV_ITEMS:
        cls = " class=\"active\"" if href == active else ""
        links.append(f'<a href="./{href}"{cls}>{html.escape(label)}</a>')
    return '<nav class="gli-nav" aria-label="Site navigation">' + "".join(links) + "</nav>"


def page(title: str, body: str, active: str, extra_head: str = "") -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(title)}</title>
{CSS_BLOCK}
{extra_head}
</head>
<body>
{nav_html(active)}
{body}
</body>
</html>
"""


def dec(value: Any, default: Decimal = Decimal(0)) -> Decimal:
    try:
        text = str(value).strip()
        if not text or text.lower() in {"none", "nan", "null"}:
            return default
        return Decimal(text)
    except (InvalidOperation, ValueError, TypeError):
        return default


def completed_close(value: Any) -> Decimal | None:
    """Return a usable completed-session close, or ``None``.

    The live engine can emit an in-progress row whose close is blank or zero
    before the source has published a completed session. Such a row must not
    enter historical tables, rankings, milestones, ticker calculations, or the
    interactive chart.
    """
    try:
        text = str(value).strip()
        if not text or text.lower() in {"none", "nan", "null"}:
            return None
        close = Decimal(text)
    except (InvalidOperation, ValueError, TypeError):
        return None
    return close if close.is_finite() and close > 0 else None


def number(value: Any) -> float:
    return float(dec(value))


def fmt_int(value: Any) -> str:
    try:
        return f"{int(dec(value).quantize(Decimal('1'), rounding=ROUND_HALF_UP)):,}"
    except Exception:
        return "0"


def load_levels() -> list[dict[str, str]]:
    if not LEVELS.exists():
        raise SystemExit("gli_levels.csv not found. Run the GLI engine first.")
    with LEVELS.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit("gli_levels.csv is empty.")
    return rows


def canonical_live_row(row: dict[str, str]) -> dict[str, str]:
    return {
        "Date": row.get("Date", ""),
        "GLI_Open": row.get("GLI_Open", ""),
        "GLI_High": row.get("GLI_High", ""),
        "GLI_Low": row.get("GLI_Low", ""),
        "GLI_Close": row.get("GLI_Close", ""),
        "TotalVolume": row.get("TotalVolume", "0"),
        "Divisor": row.get("Divisor", ""),
        "ComponentSum": row.get("ComponentSum") or row.get("SumClose", ""),
        "RosterCount": row.get("RosterCount") or row.get("RowsLoaded", ""),
        "SourceYear": row.get("Date", "")[:4],
        "LockedBundle": row.get("CloseSource") or row.get("OHLCVSource", "live"),
    }


def load_full_history(live_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if not HISTORICAL_BASE.exists():
        raise SystemExit(f"Missing historical site source: {HISTORICAL_BASE}")
    by_date: dict[str, dict[str, str]] = {}
    with HISTORICAL_BASE.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            by_date[row["Date"]] = row
    for row in live_rows:
        canonical = canonical_live_row(row)
        if canonical["Date"] and completed_close(canonical["GLI_Close"]) is not None:
            by_date[canonical["Date"]] = canonical
    return [by_date[d] for d in sorted(by_date)]


def build_history_records(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    previous: Decimal | None = None
    for row in rows:
        close = dec(row["GLI_Close"])
        change = close - previous if previous is not None else None
        pct = change / previous * Decimal(100) if change is not None and previous else None
        records.append({
            "date": row["Date"],
            "year": int(row["Date"][:4]),
            "open": number(row["GLI_Open"]),
            "high": number(row["GLI_High"]),
            "low": number(row["GLI_Low"]),
            "close": float(close),
            "volume": int(dec(row.get("TotalVolume", "0")).quantize(Decimal("1"), rounding=ROUND_HALF_UP)),
            "divisor": number(row.get("Divisor", "0")),
            "component_sum": number(row.get("ComponentSum", "0")),
            "roster": int(dec(row.get("RosterCount", "0"))),
            "change": None if change is None else float(change),
            "pct": None if pct is None else float(pct),
        })
        previous = close
    return records


def write_json(path: Path, payload: Any, *, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2 if pretty else None, separators=None if pretty else (",", ":")),
        encoding="utf-8",
    )


def write_history_data(records: list[dict[str, Any]]) -> None:
    write_json(REPORT_DATA / "gli_history.json", {
        "schema_version": 1,
        "base_date": BASE_DATE,
        "base_value": float(BASE_VALUE),
        "first_date": records[0]["date"],
        "last_date": records[-1]["date"],
        "rows": records,
    })


def write_history_page(records: list[dict[str, Any]]) -> None:
    years = sorted({r["year"] for r in records})
    year_buttons = '<button class="tab active" data-year="all">All</button>' + "".join(
        f'<button class="tab" data-year="{y}">{y}</button>' for y in years
    )
    body = f"""
<div class="page-head"><div><h1>Historical Values</h1><div class="muted">Complete daily history from {BASE_DATE} • {len(records):,} sessions</div></div></div>
<div class="source-note">Displayed price and index values use two decimals. Rankings and milestone calculations use the unrounded accepted close series.</div>
<div class="tabs" id="year-tabs">{year_buttons}</div>
<div class="table-wrap"><table aria-label="Great Lakes Index historical values">
<thead><tr><th>Date</th><th>Open</th><th>High</th><th>Low</th><th>Close</th><th>Point Change</th><th>% Change</th><th>Total Volume</th><th>Roster</th></tr></thead>
<tbody id="history-body"><tr><td colspan="9" class="empty">Loading history…</td></tr></tbody></table></div>
<script>
const nf2=new Intl.NumberFormat('en-US',{{minimumFractionDigits:2,maximumFractionDigits:2}});
const nfi=new Intl.NumberFormat('en-US',{{maximumFractionDigits:0}});
let historyRows=[]; let selectedYear='all';
function signed(v,suffix=''){{if(v===null||v===undefined)return '';return (v>0?'+':'')+nf2.format(v)+suffix;}}
function renderHistory(){{
 const rows=selectedYear==='all'?historyRows:historyRows.filter(r=>String(r.year)===selectedYear);
 document.getElementById('history-body').innerHTML=rows.map(r=>`<tr><td>${{r.date}}</td><td>${{nf2.format(r.open)}}</td><td>${{nf2.format(r.high)}}</td><td>${{nf2.format(r.low)}}</td><td><b>${{nf2.format(r.close)}}</b></td><td class="${{r.change>0?'positive':r.change<0?'negative':''}}">${{signed(r.change)}}</td><td class="${{r.pct>0?'positive':r.pct<0?'negative':''}}">${{signed(r.pct,'%')}}</td><td>${{nfi.format(r.volume)}}</td><td>${{r.roster||''}}</td></tr>`).join('');
}}
fetch('./data/gli_history.json',{{cache:'no-store'}}).then(r=>r.json()).then(d=>{{historyRows=d.rows;renderHistory();}}).catch(()=>{{document.getElementById('history-body').innerHTML='<tr><td colspan="9" class="empty">History data unavailable.</td></tr>';}});
document.getElementById('year-tabs').addEventListener('click',e=>{{if(!e.target.matches('.tab'))return;document.querySelectorAll('#year-tabs .tab').forEach(b=>b.classList.remove('active'));e.target.classList.add('active');selectedYear=e.target.dataset.year;renderHistory();}});
</script>
"""
    (REPORT / "history.html").write_text(page("GLI Historical Values", body, "history.html"), encoding="utf-8")


def top_rows(rows: list[dict[str, str]], key: str, reverse: bool, limit: int = 25) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda r: (dec(r[key]), r["date"]), reverse=reverse)[:limit]
    return [{
        "date": r["date"], "close": float(dec(r["close"])), "previous": float(dec(r["previous"])),
        "change": float(dec(r["change"])), "pct": float(dec(r["pct"])),
    } for r in ranked]


def write_market_moves(records: list[dict[str, Any]], exact_rows: list[dict[str, str]]) -> None:
    changes: list[dict[str, str]] = []
    previous: Decimal | None = None
    for row in exact_rows:
        close = dec(row["GLI_Close"])
        if previous is not None and previous != 0:
            change = close - previous
            pct = change / previous * Decimal(100)
            changes.append({"date": row["Date"], "year": row["Date"][:4], "close": str(close), "previous": str(previous), "change": str(change), "pct": str(pct)})
        previous = close
    buckets: dict[str, Any] = {}
    years = ["all"] + sorted({r["year"] for r in changes})
    for year in years:
        subset = changes if year == "all" else [r for r in changes if r["year"] == year]
        gains = [r for r in subset if dec(r["change"]) > 0]
        losses = [r for r in subset if dec(r["change"]) < 0]
        buckets[year] = {
            "point_gains": top_rows(gains, "change", True),
            "point_losses": top_rows(losses, "change", False),
            "percent_gains": top_rows(gains, "pct", True),
            "percent_losses": top_rows(losses, "pct", False),
        }
    write_json(REPORT_DATA / "market_moves.json", {"schema_version": 1, "years": years, "data": buckets})
    options = '<option value="all">All history</option>' + "".join(f'<option value="{y}">{y}</option>' for y in years if y != "all")
    body = f"""
<div class="page-head"><div><h1>Biggest Market Moves</h1><div class="muted">Largest daily changes in the Great Lakes Index</div></div><label>Period <select id="move-year">{options}</select></label></div>
<div class="tabs" id="move-tabs"><button class="tab active" data-view="point_gains">Point Gains</button><button class="tab" data-view="point_losses">Point Losses</button><button class="tab" data-view="percent_gains">Percent Gains</button><button class="tab" data-view="percent_losses">Percent Losses</button></div>
<div class="table-wrap"><table><thead><tr><th>Rank</th><th>Date</th><th>Previous Close</th><th>Close</th><th>Point Change</th><th>% Change</th></tr></thead><tbody id="moves-body"><tr><td colspan="6" class="empty">Loading rankings…</td></tr></tbody></table></div>
<script>
const nf2m=new Intl.NumberFormat('en-US',{{minimumFractionDigits:2,maximumFractionDigits:2}});let moveData=null;let moveView='point_gains';
function sm(v,s=''){{return (v>0?'+':'')+nf2m.format(v)+s;}}
function renderMoves(){{if(!moveData)return;const y=document.getElementById('move-year').value;const rows=moveData.data[y][moveView];document.getElementById('moves-body').innerHTML=rows.map((r,i)=>`<tr><td class="rank">${{i+1}}</td><td>${{r.date}}</td><td>${{nf2m.format(r.previous)}}</td><td><b>${{nf2m.format(r.close)}}</b></td><td class="${{r.change>0?'positive':'negative'}}">${{sm(r.change)}}</td><td class="${{r.pct>0?'positive':'negative'}}">${{sm(r.pct,'%')}}</td></tr>`).join('')||'<tr><td colspan="6" class="empty">No qualifying sessions.</td></tr>';}}
fetch('./data/market_moves.json',{{cache:'no-store'}}).then(r=>r.json()).then(d=>{{moveData=d;renderMoves();}});
document.getElementById('move-year').addEventListener('change',renderMoves);document.getElementById('move-tabs').addEventListener('click',e=>{{if(!e.target.matches('.tab'))return;document.querySelectorAll('#move-tabs .tab').forEach(b=>b.classList.remove('active'));e.target.classList.add('active');moveView=e.target.dataset.view;renderMoves();}});
</script>
"""
    (REPORT / "market-moves.html").write_text(page("GLI Biggest Market Moves", body, "market-moves.html"), encoding="utf-8")


def write_milestones(exact_rows: list[dict[str, str]]) -> None:
    closes = [(r["Date"], dec(r["GLI_Close"])) for r in exact_rows]
    maximum = max(v for _, v in closes)
    highest = int(maximum // Decimal(10)) * 10
    milestones: list[dict[str, Any]] = []
    previous_close: Decimal | None = None
    for threshold in range(100, highest + 1, 10):
        for i, (d, close) in enumerate(closes):
            if close > Decimal(threshold):
                prior = closes[i - 1][1] if i else None
                milestones.append({
                    "threshold": threshold, "date": d, "close": float(close),
                    "above": float(close - Decimal(threshold)),
                    "previous_close": None if prior is None else float(prior),
                })
                break
    write_json(REPORT_DATA / "milestones.json", {"schema_version": 1, "strictly_above": True, "milestones": milestones})
    cards = "".join(
        f'<article class="milestone"><div class="gli-k">First close above</div><div class="level">{m["threshold"]:,}</div><div><b>{m["date"]}</b></div><div class="muted">Closed at {m["close"]:,.2f} • {m["above"]:+.2f} above level</div></article>'
        for m in milestones
    )
    latest = milestones[-1]
    next_level = latest["threshold"] + 10
    body = f"""
<div class="page-head"><div><h1>Closing Milestones</h1><div class="muted">First daily close strictly above each 10-point level</div></div></div>
<div class="metric-cards"><div class="metric-card"><span class="gli-k">Milestones reached</span><strong>{len(milestones)}</strong></div><div class="metric-card"><span class="gli-k">Latest level</span><strong>{latest['threshold']}</strong><span class="muted">{latest['date']}</span></div><div class="metric-card"><span class="gli-k">Highest close</span><strong>{float(maximum):,.2f}</strong></div><div class="metric-card"><span class="gli-k">Next level</span><strong>{next_level}</strong></div></div>
<div class="milestone-grid">{cards}</div>
"""
    (REPORT / "milestones.html").write_text(page("GLI Closing Milestones", body, "milestones.html"), encoding="utf-8")


def load_constituents() -> list[dict[str, str]]:
    if not CONSTITUENTS.exists():
        return []
    with CONSTITUENTS.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def active_on(day: str, constituents: list[dict[str, str]]) -> set[str]:
    active: set[str] = set()
    for row in constituents:
        ticker = row.get("Ticker", "").strip().upper()
        start = row.get("StartDate", "").strip() or "0001-01-01"
        end = row.get("EndDate", "").strip() or "9999-12-31"
        if ticker and start <= day <= end:
            active.add(ticker)
    return active


def company_map() -> dict[str, str]:
    out: dict[str, str] = {}
    if COMPANY_CACHE.exists():
        with COMPANY_CACHE.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                ticker = row.get("Ticker", "").strip().upper()
                name = row.get("Company", "").strip()
                if ticker and name:
                    out[ticker] = name
    return out


def historical_company_name_map() -> dict[str, list[dict[str, str]]]:
    """Return date-aware company names keyed by ticker.

    This presentation metadata never changes component prices, weights, rosters,
    divisors, or accepted index history. ``company_names.csv`` remains the
    fallback for current/new live tickers not yet in the historical file.
    """
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    if not HISTORICAL_COMPANY_NAMES.exists():
        return {}
    with HISTORICAL_COMPANY_NAMES.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            ticker = row.get("Ticker", "").strip().upper()
            name = row.get("Company", "").strip()
            start = row.get("StartDate", "").strip() or "0001-01-01"
            end = row.get("EndDate", "").strip() or "9999-12-31"
            if ticker and name and start <= end:
                out[ticker].append({"start": start, "end": end, "name": name})
    for ranges in out.values():
        ranges.sort(key=lambda item: (item["start"], item["end"], item["name"]))
    return dict(out)


def _normalize_component_identity_label(value: str) -> str:
    """Normalize a presentation label for historical identity matching."""
    text = str(value or "").replace("\n", " ").strip().casefold().replace("&", " and ")
    text = re.sub(r"\b(name change)\s*:\s*", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def historical_component_symbol_map() -> dict[str, str]:
    """Return legacy Component History labels -> accepted historical tickers.

    The mapping is read from the dedicated ``ComponentHistoryLabels`` column
    in ``historical_company_names.csv``.  It intentionally does not use the
    broader identity alias field, so display symbols cannot be contaminated
    by name-change aliases or same-security lineage controls.
    """
    candidates: dict[str, set[str]] = defaultdict(set)
    if not HISTORICAL_COMPANY_NAMES.exists():
        return {}
    with HISTORICAL_COMPANY_NAMES.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            ticker = row.get("Ticker", "").strip().upper()
            labels = [part.strip() for part in row.get("ComponentHistoryLabels", "").split("|") if part.strip()]
            for label in labels:
                key = _normalize_component_identity_label(label)
                if ticker and key:
                    candidates[key].add(ticker)
    return {key: next(iter(tickers)) for key, tickers in candidates.items() if len(tickers) == 1}


def historical_symbol_for_label(label: str, symbol_map: dict[str, str] | None = None) -> str:
    symbol_map = symbol_map if symbol_map is not None else historical_component_symbol_map()
    return symbol_map.get(_normalize_component_identity_label(label), "")


def historical_component_identity_metadata() -> tuple[dict[str, list[dict[str, str]]], dict[str, str]]:
    """Return date-aware ticker identities and unambiguous legacy-label identities."""
    ticker_ranges: dict[str, list[dict[str, str]]] = defaultdict(list)
    label_candidates: dict[str, set[str]] = defaultdict(set)
    if not HISTORICAL_COMPANY_NAMES.exists():
        return {}, {}
    with HISTORICAL_COMPANY_NAMES.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            ticker = row.get("Ticker", "").strip().upper()
            identity = row.get("Identity", "").strip().upper() or ticker
            name = row.get("Company", "").strip()
            start = row.get("StartDate", "").strip() or "0001-01-01"
            end = row.get("EndDate", "").strip() or "9999-12-31"
            if not ticker:
                continue
            ticker_ranges[ticker].append({
                "start": start, "end": end, "identity": identity,
                "name": name, "source": row.get("Source", "").strip(),
            })
            labels = [name]
            labels.extend(part.strip() for part in row.get("Aliases", "").split("|") if part.strip())
            labels.extend(part.strip() for part in row.get("ComponentHistoryLabels", "").split("|") if part.strip())
            for label in labels:
                key = _normalize_component_identity_label(label)
                if key:
                    label_candidates[key].add(identity)
    for ranges in ticker_ranges.values():
        ranges.sort(key=lambda item: (item["start"], item["end"], item["identity"]))
    label_identity = {key: next(iter(ids)) for key, ids in label_candidates.items() if len(ids) == 1}
    return dict(ticker_ranges), label_identity


def _ticker_identity_meta_at(
    ticker: str,
    day: str,
    ticker_ranges: dict[str, list[dict[str, str]]],
    *,
    allow_nearest: bool = True,
) -> dict[str, str]:
    ticker = str(ticker or "").strip().upper()
    ranges = ticker_ranges.get(ticker, [])
    for item in ranges:
        if item["start"] <= day <= item["end"]:
            return {"symbol": ticker, **item}
    if allow_nearest and ranges:
        prior = [item for item in ranges if item["end"] < day]
        future = [item for item in ranges if item["start"] > day]
        if prior:
            item = max(prior, key=lambda x: x["end"])
            return {"symbol": ticker, **item}
        if future:
            item = min(future, key=lambda x: x["start"])
            return {"symbol": ticker, **item}
    return {"symbol": ticker, "identity": ticker, "name": "", "start": "", "end": "", "source": ""}


def historical_component_explicit_events() -> dict[str, list[dict[str, str]]]:
    """Read vetted Component History event annotations from the historical name master."""
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    if not HISTORICAL_COMPANY_NAMES.exists():
        return {}
    kind_map = {
        "NAME_CHANGE": "name_change",
        "NAME_AND_TICKER_CHANGE": "name_and_ticker_change",
        "TICKER_CHANGE": "ticker_change",
        "SECURITY_REPLACEMENT": "security_replacement",
        "SECURITY_REPLACEMENT_SAME_TICKER": "security_replacement_same_ticker",
    }
    with HISTORICAL_COMPANY_NAMES.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            raw_kind = row.get("EventType", "").strip().upper()
            if not raw_kind:
                continue
            day = row.get("EventDate", "").strip() or row.get("StartDate", "").strip()
            to_symbol = row.get("Ticker", "").strip().upper()
            to_name = row.get("Company", "").strip()
            from_symbol = row.get("PriorTicker", "").strip().upper()
            from_name = row.get("PriorCompany", "").strip()
            if not day or not to_symbol:
                continue
            out[day].append({
                "kind": kind_map.get(raw_kind, raw_kind.lower()),
                "from_symbol": from_symbol,
                "from_name": from_name,
                "to_symbol": to_symbol,
                "to_name": to_name,
                "detail": row.get("EventDetail", "").strip(),
                "source": "historical_company_names.csv",
            })
    return dict(out)


def _clean_component_history_snapshots(snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Clean legacy artifacts while preserving real security changes and naming continuity."""
    ticker_ranges, label_identity = historical_component_identity_metadata()
    legacy_symbol_map = historical_component_symbol_map()
    cleaned: list[dict[str, Any]] = []
    previous_day = "0001-01-01"

    def clean_label(value: str) -> str:
        return " ".join(str(value or "").replace("\n", " ").split())

    for raw in snapshots:
        snap = dict(raw)
        day = str(snap.get("date", ""))
        mode = snap.get("label_mode")
        components = [str(x) for x in snap.get("components", [])]
        added = [str(x) for x in snap.get("added", [])]
        removed = [str(x) for x in snap.get("removed", [])]

        # The legacy workbook placed NAME CHANGE markers inside the roster.
        # They are annotations, not constituents.  We infer the real event from
        # the removed/added security identities below.
        def is_name_change_marker(value: str) -> bool:
            return clean_label(value).upper().startswith("NAME CHANGE:")

        components = [x for x in components if not is_name_change_marker(x)]
        added = [x for x in added if not is_name_change_marker(x)]
        removed = [x for x in removed if not is_name_change_marker(x)]

        def meta(value: str, item_day: str) -> dict[str, str]:
            if mode == "ticker":
                symbol = clean_label(value).upper()
                found = _ticker_identity_meta_at(symbol, item_day, ticker_ranges, allow_nearest=True)
                return {
                    "value": value, "symbol": symbol,
                    "name": found.get("name", "") or symbol,
                    "identity": found.get("identity", "") or symbol,
                }
            label = clean_label(value)
            symbol = historical_symbol_for_label(label, legacy_symbol_map)
            if symbol:
                found = _ticker_identity_meta_at(symbol, item_day, ticker_ranges, allow_nearest=True)
                identity = found.get("identity", "") or symbol
            else:
                key = _normalize_component_identity_label(label)
                identity = label_identity.get(key, f"LABEL:{key}")
            return {"value": value, "symbol": symbol, "name": label, "identity": identity}

        add_meta = [meta(value, day) for value in added]
        rem_meta = [meta(value, previous_day or day) for value in removed]
        add_by_id: dict[str, list[int]] = defaultdict(list)
        rem_by_id: dict[str, list[int]] = defaultdict(list)
        for i, item in enumerate(add_meta):
            add_by_id[item["identity"]].append(i)
        for i, item in enumerate(rem_meta):
            rem_by_id[item["identity"]].append(i)

        consumed_add: set[int] = set()
        consumed_rem: set[int] = set()
        # Preserve events inferred on an earlier normalization pass.  The
        # Component History builder cleans the seed once before extending live
        # chronology and once after; without this carry, continuity events whose
        # false add/remove pair was already suppressed vanish on pass two.
        events: list[dict[str, str]] = [dict(e) for e in snap.get("events", [])]

        # Same underlying security: do not show false membership turnover.
        # Instead, explicitly describe the name/ticker continuity event.
        for identity in sorted(set(add_by_id) & set(rem_by_id)):
            for ai, ri in zip(add_by_id[identity], rem_by_id[identity]):
                a, r = add_meta[ai], rem_meta[ri]
                consumed_add.add(ai); consumed_rem.add(ri)
                same_symbol = bool(a["symbol"] and r["symbol"] and a["symbol"] == r["symbol"])
                same_name = _normalize_component_identity_label(a["name"]) == _normalize_component_identity_label(r["name"])
                if same_symbol and not same_name:
                    kind = "name_change"
                    detail = "Same accepted GLI security lineage and ticker; this is a company-name change, not membership turnover."
                elif not same_symbol and not same_name:
                    kind = "name_and_ticker_change"
                    detail = "Same accepted GLI security lineage; the company name and ticker changed without membership turnover."
                elif not same_symbol:
                    kind = "ticker_change"
                    detail = "Same accepted GLI security lineage; ticker changed without membership turnover."
                else:
                    continue
                events.append({
                    "kind": kind, "from_name": r["name"], "from_symbol": r["symbol"],
                    "to_name": a["name"], "to_symbol": a["symbol"], "detail": detail,
                    "source": "component identity continuity",
                })

        # Same ticker does NOT always mean same security.  If an unpaired
        # removal/addition reuses the same symbol, preserve both turnover badges
        # and call the security replacement out explicitly.
        for ri, r in enumerate(rem_meta):
            if ri in consumed_rem or not r["symbol"]:
                continue
            for ai, a in enumerate(add_meta):
                if ai in consumed_add or not a["symbol"]:
                    continue
                if r["symbol"] == a["symbol"] and r["identity"] != a["identity"]:
                    events.append({
                        "kind": "security_replacement_same_ticker",
                        "from_name": r["name"], "from_symbol": r["symbol"],
                        "to_name": a["name"], "to_symbol": a["symbol"],
                        "detail": "Ticker was reused, but the incoming component is a different security.",
                        "source": "component security identity",
                    })
                    break

        snap["components"] = components
        snap["added"] = [value for i, value in enumerate(added) if i not in consumed_add]
        snap["removed"] = [value for i, value in enumerate(removed) if i not in consumed_rem]
        snap["events"] = events
        snap["count"] = len(components)

        # Some legacy workbook notes called a predecessor ticker "deleted" even
        # when security-master review later established continuous identity.
        # Preserve the wording for provenance, but make the modern classification
        # explicit so the page does not contradict its own event badges.
        original_note = str(snap.get("note", "") or "").strip()
        continuity_kinds = {"name_change", "name_and_ticker_change", "ticker_change"}
        if (
            original_note.upper().startswith("STOCK")
            and removed
            and not snap["removed"]
            and events
            and all(e.get("kind") in continuity_kinds for e in events)
        ):
            snap["legacy_note"] = original_note
            snap["note"] = f"Legacy workbook note: {original_note} • Identity review: no membership turnover at this checkpoint."
        cleaned.append(snap)
        previous_day = day or previous_day
    return cleaned


def _apply_component_history_explicit_events(snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge vetted event notes and add event-only checkpoints when necessary."""
    explicit = historical_component_explicit_events()
    if not explicit:
        return snapshots
    by_date: dict[str, dict[str, Any]] = {str(s.get("date", "")): s for s in snapshots}

    def event_key(event: dict[str, str]) -> tuple[str, str, str]:
        return (
            event.get("from_symbol", "").upper(),
            event.get("to_symbol", "").upper(),
            event.get("kind", ""),
        )

    for day in sorted(explicit):
        snap = by_date.get(day)
        if snap is None:
            prior = [s for s in snapshots if str(s.get("date", "")) < day]
            if not prior:
                continue
            base = max(prior, key=lambda s: str(s.get("date", "")))
            components = list(base.get("components", []))
            snap = {
                "date": day,
                "title": f"The Great Lakes {len(components)}",
                "count": len(components),
                "components": components,
                "added": [], "removed": [], "note": "",
                "source": "historical_company_names.csv",
                "label_mode": base.get("label_mode", "ticker"),
                "events": [], "event_only": True,
            }
            snapshots.append(snap)
            by_date[day] = snap
        events = list(snap.get("events", []))
        for vetted in explicit[day]:
            match = None
            # Match first by from/to symbol regardless of provisional event kind.
            for event in events:
                if (event.get("from_symbol", "").upper(), event.get("to_symbol", "").upper()) == (
                    vetted.get("from_symbol", "").upper(), vetted.get("to_symbol", "").upper()
                ):
                    match = event
                    break
            if match is not None:
                match.update(vetted)
            else:
                events.append(dict(vetted))
        # Preserve insertion order but remove accidental exact duplicates.
        seen: set[tuple[str, str, str]] = set()
        unique = []
        for event in events:
            key = event_key(event)
            if key in seen:
                continue
            seen.add(key); unique.append(event)
        snap["events"] = unique
    return sorted(snapshots, key=lambda s: str(s.get("date", "")))


def accepted_cutoff() -> str:
    try:
        return json.loads(LIVE_ANCHOR.read_text(encoding="utf-8"))["accepted_through"]
    except Exception:
        return "9999-12-31"


def copy_weight_seed() -> dict[str, Any]:
    target = REPORT_DATA / "weights"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(WEIGHTS_BASE, target)
    years: dict[str, Any] = {}
    for path in sorted(target.glob("weights_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        years[str(payload["year"])] = {
            "file": f"weights/{path.name}",
            "dates": len(payload.get("dates", [])),
            "first_date": payload.get("dates", [None])[0] if payload.get("dates") else None,
            "last_date": payload.get("dates", [None])[-1] if payload.get("dates") else None,
            "status": "accepted component OHLCV",
        }
    return years


def build_live_weight_files(years: dict[str, Any], constituents: list[dict[str, str]]) -> None:
    if not PRICES.exists():
        return
    prices = pd.read_csv(PRICES, dtype=str).fillna("")
    rename = {}
    if "date" in prices.columns and "Date" not in prices.columns:
        rename["date"] = "Date"
    if "ticker" in prices.columns and "Ticker" not in prices.columns:
        rename["ticker"] = "Ticker"
    prices = prices.rename(columns=rename)
    if not {"Date", "Ticker", "Close"}.issubset(prices.columns):
        return
    cutoff = accepted_cutoff()
    prices = prices[prices["Date"] > cutoff].copy()
    if prices.empty:
        return
    for year, year_df in prices.groupby(prices["Date"].str[:4]):
        dates: list[str] = []
        snapshots: list[list[list[Any]]] = []
        for day, group in year_df.groupby("Date"):
            allowed = active_on(day, constituents)
            items: list[tuple[str, Decimal]] = []
            for _, row in group.iterrows():
                ticker = str(row["Ticker"]).strip().upper()
                if ticker not in allowed:
                    continue
                close = dec(row["Close"])
                if close > 0:
                    items.append((ticker, close))
            total = sum((v for _, v in items), Decimal(0))
            if not items or total == 0:
                continue
            weights = [[ticker, int((value / total * Decimal(1_000_000)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))] for ticker, value in sorted(items)]
            dates.append(day)
            snapshots.append(weights)
        if not dates:
            continue
        payload = {"schema_version": 1, "year": int(year), "unit": "parts_per_million_of_component_sum", "dates": dates, "snapshots": snapshots}
        path = REPORT_DATA / "weights" / f"weights_{year}.json"
        write_json(path, payload)
        years[year] = {"file": f"weights/{path.name}", "dates": len(dates), "first_date": dates[0], "last_date": dates[-1], "status": f"live sessions after accepted cutoff {cutoff}"}


def write_weights_page(constituents: list[dict[str, str]]) -> None:
    years = copy_weight_seed()
    build_live_weight_files(years, constituents)
    write_json(REPORT_DATA / "weights_manifest.json", {"schema_version": 1, "accepted_cutoff": accepted_cutoff(), "years": years}, pretty=True)
    body = """
<div class="page-head"><div><h1>Component Weights</h1><div class="muted">Daily price weights in the price-weighted Great Lakes Index</div></div></div>
<div class="source-note" id="weight-source">Historical weights use accepted component OHLCV rows. Company labels use the name effective on the selected date, with the current company-name cache as a fallback. For 2026, only live sessions after the accepted cutoff are displayed until an accepted 2026 component-level file is installed.</div>
<div class="panel"><div class="controls"><label>Year <select id="weight-year"></select></label><label>Date <input id="weight-date" type="date" readonly></label><input id="weight-slider" type="range" min="0" max="0" value="0" aria-label="Weight date"></div><div class="summary-grid"><div><div class="gli-k">As of</div><div class="gli-v" id="weight-asof">—</div></div><div><div class="gli-k">Components</div><div class="gli-v" id="weight-count">—</div></div><div><div class="gli-k">Largest weight</div><div class="gli-v" id="weight-largest">—</div></div><div><div class="gli-k">Total</div><div class="gli-v" id="weight-total">—</div></div><div><div class="gli-k">Dataset</div><div class="gli-v" id="weight-status" style="font-size:14px">—</div></div></div></div>
<div id="heatmap" class="heatmap"><div class="empty">Loading component weights…</div></div>
<script>
const companyNames=COMPANY_MAP_TOKEN;const historicalCompanyNames=HISTORICAL_COMPANY_NAMES_TOKEN;let manifest=null;let weightPayload=null;
function companyNameAt(symbol,day){const ranges=historicalCompanyNames[symbol]||[];for(const r of ranges){if(r.start<=day&&day<=r.end)return r.name;}return companyNames[symbol]||symbol;}
const yearSel=document.getElementById('weight-year'),slider=document.getElementById('weight-slider');
function renderHeat(index){if(!weightPayload)return;const items=weightPayload.snapshots[index].map(x=>({symbol:x[0],ppm:x[1]})).sort((a,b)=>b.ppm-a.ppm);const max=items[0]?.ppm||1;const total=items.reduce((s,x)=>s+x.ppm,0);const day=weightPayload.dates[index];document.getElementById('weight-date').value=day;document.getElementById('weight-asof').textContent=day;document.getElementById('weight-count').textContent=items.length;document.getElementById('weight-largest').textContent=items.length?`${items[0].symbol} ${(items[0].ppm/10000).toFixed(2)}%`:'—';document.getElementById('weight-total').textContent=(total/10000).toFixed(2)+'%';document.getElementById('weight-status').textContent=manifest.years[yearSel.value].status;document.getElementById('heatmap').innerHTML=items.map(x=>{const ratio=x.ppm/max;const alpha=.16+.78*ratio;const dark=alpha>.55;const name=companyNameAt(x.symbol,day);return `<div class="heat-cell" style="background:rgba(23,92,170,${alpha.toFixed(3)});color:${dark?'#fff':'#102a43'}" title="${x.symbol}: ${(x.ppm/10000).toFixed(4)}%"><div><div class="heat-symbol">${x.symbol}</div><div class="heat-company">${name}</div></div><div class="heat-weight">${(x.ppm/10000).toFixed(2)}%</div></div>`;}).join('');}
function loadYear(year){const meta=manifest.years[year];fetch('./data/'+meta.file,{cache:'no-store'}).then(r=>r.json()).then(d=>{weightPayload=d;slider.max=Math.max(0,d.dates.length-1);slider.value=d.dates.length-1;renderHeat(Number(slider.value));});}
fetch('./data/weights_manifest.json',{cache:'no-store'}).then(r=>r.json()).then(d=>{manifest=d;const years=Object.keys(d.years).sort((a,b)=>Number(a)-Number(b));yearSel.innerHTML=years.map(y=>`<option value="${y}">${y}</option>`).join('');yearSel.value=years[years.length-1];loadYear(yearSel.value);});
yearSel.addEventListener('change',()=>loadYear(yearSel.value));slider.addEventListener('input',()=>renderHeat(Number(slider.value)));
</script>
"""
    body = body.replace("COMPANY_MAP_TOKEN", json.dumps(company_map(), separators=(",", ":")))
    body = body.replace("HISTORICAL_COMPANY_NAMES_TOKEN", json.dumps(historical_company_name_map(), separators=(",", ":")))
    (REPORT / "weights.html").write_text(page("GLI Component Weights", body, "weights.html"), encoding="utf-8")


def extend_component_history(history_rows: list[dict[str, str]], constituents: list[dict[str, str]]) -> dict[str, Any]:
    payload = json.loads(ROSTER_BASE.read_text(encoding="utf-8"))
    snapshots = _clean_component_history_snapshots(list(payload["snapshots"]))
    previous = set()
    for snapshot in reversed(snapshots):
        if snapshot.get("label_mode") == "ticker":
            previous = set(snapshot["components"])
            break
    for day in [r["Date"] for r in history_rows if r["Date"] >= "2026-01-01"]:
        current = active_on(day, constituents)
        if current != previous:
            changed = current ^ previous
            notes = []
            for row in constituents:
                if row.get("Ticker", "").strip().upper() in changed and row.get("Notes", "").strip():
                    notes.append(row["Notes"].strip())
            snapshots.append({
                "date": day, "title": f"The Great Lakes {len(current)}", "count": len(current),
                "components": sorted(current), "added": sorted(current - previous), "removed": sorted(previous - current),
                "note": " • ".join(dict.fromkeys(notes)), "source": "constituents_great_lakes.csv", "label_mode": "ticker",
            })
            previous = current
    snapshots = _clean_component_history_snapshots(snapshots)
    snapshots = _apply_component_history_explicit_events(snapshots)
    legacy_symbol_map = historical_component_symbol_map()
    for snapshot in snapshots:
        if snapshot.get("label_mode") != "company_name":
            continue
        labels = list(snapshot.get("components", [])) + list(snapshot.get("added", [])) + list(snapshot.get("removed", []))
        symbol_map: dict[str, str] = {}
        for label in dict.fromkeys(str(x) for x in labels):
            symbol = historical_symbol_for_label(label, legacy_symbol_map)
            if symbol:
                symbol_map[label] = symbol
        snapshot["component_symbols"] = symbol_map
    payload["snapshots"] = snapshots
    # Keep the current-name cache for live/new symbols, but also publish the
    # date-aware historical name master so Component History can label each
    # ticker using the company name effective on that checkpoint date.
    payload["company_names"] = company_map()
    payload["historical_company_names"] = historical_company_name_map()
    return payload


def write_component_history(history_rows: list[dict[str, str]], constituents: list[dict[str, str]]) -> None:
    payload = extend_component_history(history_rows, constituents)
    write_json(REPORT_DATA / "component_history.json", payload)
    years = sorted({s["date"][:4] for s in payload["snapshots"]})
    year_buttons = '<button class="tab" data-year="all">All</button>' + "".join(f'<button class="tab" data-year="{y}">{y}</button>' for y in years)
    body = f"""
<div class="page-head"><div><h1>Component History</h1><div class="muted">Roster checkpoints modeled after GLI_Component_history.xlsx</div></div><input type="search" id="component-search" placeholder="Search company or ticker" aria-label="Search component history"></div>
<div class="source-note">Through May 1, 2014, company labels preserve the historical workbook and their historical ticker symbols are resolved from <code>site_data/historical_company_names.csv</code>. Name and ticker changes are called out separately from membership turnover. A reused ticker does not imply security continuity; documented same-ticker replacement events remain true removals/additions. Ticker-based checkpoints use the same historical-name file, with the current company-name cache only as a fallback; 2026 is extended from the committed constituent chronology.</div>
<div class="tabs" id="component-years">{year_buttons}</div><div id="component-timeline" class="timeline"><div class="empty">Loading component history…</div></div>
<script>
let componentData=null;let componentYear='{years[-1]}';let componentQuery='';
function esc(s){{return String(s??'').replace(/[&<>\"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c]));}}
function companyNameAt(symbol,day,allowNearest=false){{const key=String(symbol??'').toUpperCase();const ranges=(componentData?.historical_company_names||{{}})[key]||[];for(const r of ranges){{if(r.start<=day&&day<=r.end)return r.name;}}if(allowNearest&&ranges.length){{let prior=null,next=null;for(const r of ranges){{if(r.end<day&&(!prior||r.end>prior.end))prior=r;if(r.start>day&&(!next||r.start<next.start))next=r;}}if(prior)return prior.name;if(next)return next.name;}}return (componentData?.company_names||{{}})[key]||'';}}
function legacySymbol(s,label){{return (s.component_symbols||{{}})[String(label)]||'';}}
function componentSearchText(s,c){{if(s.label_mode==='ticker')return `${{c}} ${{companyNameAt(c,s.date,true)}}`;return `${{c}} ${{legacySymbol(s,c)}}`;}}
function changeBadge(kind,value,s){{const sign=kind==='add'?'+':'−';if(s.label_mode==='ticker'){{const name=companyNameAt(value,s.date,true);return `<span class="badge ${{kind}}">${{sign}} ${{esc(value)}}${{name?` — ${{esc(name)}}`:''}}</span>`;}}const symbol=legacySymbol(s,value);return `<span class="badge ${{kind}}">${{sign}} ${{esc(value)}}${{symbol?` (${{esc(symbol)}})`:''}}</span>`;}}
function eventLabel(kind){{return ({{name_change:'Name change',name_and_ticker_change:'Name + ticker change',ticker_change:'Ticker change',security_replacement:'Security replacement',security_replacement_same_ticker:'Security replacement · ticker reused'}})[kind]||'Component event';}}
function eventParty(name,symbol){{const n=String(name||'').trim();const t=String(symbol||'').trim();return n&&t?`${{n}} (${{t}})`:n||t;}}
function eventSearchText(e){{return `${{eventLabel(e.kind)}} ${{eventParty(e.from_name,e.from_symbol)}} ${{eventParty(e.to_name,e.to_symbol)}} ${{e.detail||''}}`;}}
function eventHtml(e){{const security=String(e.kind||'').startsWith('security_replacement');const from=eventParty(e.from_name,e.from_symbol);const to=eventParty(e.to_name,e.to_symbol);const arrow=from&&to?`${{esc(from)}} → ${{esc(to)}}`:esc(to||from);return `<div class="component-event ${{security?'security':''}}"><span class="event-label">${{esc(eventLabel(e.kind))}}:</span> ${{arrow}}${{e.detail?`<span class="event-detail">${{esc(e.detail)}}</span>`:''}}</div>`;}}
function renderComponents(){{if(!componentData)return;let rows=componentData.snapshots.filter(s=>componentYear==='all'||s.date.startsWith(componentYear));if(componentQuery)rows=rows.filter(s=>s.components.some(c=>componentSearchText(s,c).toLowerCase().includes(componentQuery))||(s.added||[]).some(c=>componentSearchText(s,c).toLowerCase().includes(componentQuery))||(s.removed||[]).some(c=>componentSearchText(s,c).toLowerCase().includes(componentQuery))||(s.events||[]).some(e=>eventSearchText(e).toLowerCase().includes(componentQuery)));rows=rows.slice().reverse();document.getElementById('component-timeline').innerHTML=rows.map(s=>{{const added=new Set(s.added||[]);const chips=s.components.map(c=>{{if(s.label_mode==='ticker'){{const name=companyNameAt(c,s.date);return `<div class="component ${{added.has(c)?'added':''}}"><div class="component-symbol">${{esc(c)}}</div>${{name?`<div class="component-name">${{esc(name)}}</div>`:''}}</div>`;}}const symbol=legacySymbol(s,c);return `<div class="component ${{added.has(c)?'added':''}}"><div class="component-symbol">${{esc(c)}}</div>${{symbol?`<div class="component-name">${{esc(symbol)}}</div>`:''}}</div>`;}}).join('');const adds=(s.added||[]).map(x=>changeBadge('add',x,s)).join('');const rems=(s.removed||[]).map(x=>changeBadge('remove',x,s)).join('');const events=(s.events||[]).map(eventHtml).join('');return `<article class="snapshot"><div class="snapshot-head"><div><div class="snapshot-date">${{s.date}}</div><div class="snapshot-title">${{esc(s.title)}} • ${{s.count}} components${{s.event_only?' • name/event checkpoint':''}}</div></div><div class="gli-pill">${{s.label_mode==='ticker'?'Symbols + historical names':'Company names + symbols'}}</div></div>${{s.note?`<div class="source-note">${{esc(s.note)}}</div>`:''}}${{events?`<div class="event-list">${{events}}</div>`:''}}${{adds||rems?`<div class="change-row">${{adds}}${{rems}}</div>`:''}}<div class="component-grid">${{chips}}</div></article>`;}}).join('')||'<div class="empty">No matching checkpoints.</div>';}}
fetch('./data/component_history.json',{{cache:'no-store'}}).then(r=>r.json()).then(d=>{{componentData=d;document.querySelector(`[data-year="${{componentYear}}"]`)?.classList.add('active');renderComponents();}});
document.getElementById('component-years').addEventListener('click',e=>{{if(!e.target.matches('.tab'))return;document.querySelectorAll('#component-years .tab').forEach(b=>b.classList.remove('active'));e.target.classList.add('active');componentYear=e.target.dataset.year;renderComponents();}});document.getElementById('component-search').addEventListener('input',e=>{{componentQuery=e.target.value.trim().toLowerCase();renderComponents();}});
</script>
"""
    (REPORT / "components.html").write_text(page("GLI Component History", body, "components.html"), encoding="utf-8")


def load_prices_optional() -> pd.DataFrame | None:
    if not PRICES.exists():
        return None
    df = pd.read_csv(PRICES)
    rename = {}
    if "date" in df.columns and "Date" not in df.columns:
        rename["date"] = "Date"
    if "ticker" in df.columns and "Ticker" not in df.columns:
        rename["ticker"] = "Ticker"
    df = df.rename(columns=rename)
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})
    if not {"Date", "Ticker", "Close"}.issubset(df.columns):
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def write_component_ohlcv(constituents: list[dict[str, str]]) -> None:
    df = load_prices_optional()
    if df is None or df.empty:
        return
    for col in ["Open", "High", "Low", "Volume", "Source"]:
        if col not in df.columns:
            df[col] = pd.NA
    df = df.sort_values(["Ticker", "Date"])
    df["PrevClose"] = df.groupby("Ticker")["Close"].shift(1)
    latest_date = df["Date"].max()
    day = latest_date.date().isoformat()
    allowed = active_on(day, constituents)
    latest = df[df["Date"] == latest_date].copy()
    latest["Ticker"] = latest["Ticker"].astype(str).str.upper().str.strip()
    latest = latest[latest["Ticker"].isin(allowed)]
    latest["Change"] = pd.to_numeric(latest["Close"], errors="coerce") - pd.to_numeric(latest["PrevClose"], errors="coerce")
    latest["Pct"] = latest["Change"] / pd.to_numeric(latest["PrevClose"], errors="coerce") * 100
    total = pd.to_numeric(latest["Close"], errors="coerce").sum()
    latest["Weight"] = pd.to_numeric(latest["Close"], errors="coerce") / total * 100 if total else pd.NA
    names = company_map()
    latest["Company"] = latest["Ticker"].map(names).fillna("")
    latest = latest.sort_values("Ticker")
    def f2(v: Any) -> str:
        try:
            return "" if pd.isna(v) else f"{float(v):,.2f}"
        except Exception:
            return ""
    body_rows = []
    for _, r in latest.iterrows():
        chg = r["Change"]
        pct = r["Pct"]
        cls = "positive" if pd.notna(chg) and chg > 0 else "negative" if pd.notna(chg) and chg < 0 else ""
        body_rows.append("<tr>" + "".join([
            f'<td>{html.escape(str(r["Company"]))}</td>', f'<td><b>{html.escape(str(r["Ticker"]))}</b></td>',
            f'<td>{f2(r["Open"])}</td>', f'<td>{f2(r["High"])}</td>', f'<td>{f2(r["Low"])}</td>', f'<td><b>{f2(r["Close"])}</b></td>',
            f'<td class="{cls}">{"" if pd.isna(chg) else ("+" if chg>0 else "")+f2(chg)}</td>',
            f'<td class="{cls}">{"" if pd.isna(pct) else ("+" if pct>0 else "")+f2(pct)+"%"}</td>',
            f'<td>{"" if pd.isna(r["Weight"]) else f2(r["Weight"])+"%"}</td>', f'<td>{"" if pd.isna(r["Volume"]) else fmt_int(r["Volume"])}</td>',
            f'<td>{day}</td>', f'<td>{html.escape(str(r["Source"]) if pd.notna(r["Source"]) else "")}</td>',
        ]) + "</tr>")
    body = f"""
<div class="page-head"><div><h1>Component OHLCV</h1><div class="muted">Latest snapshot per active component • Price = Close • Change vs prior close</div></div></div>
<div class="table-wrap"><table><thead><tr><th>Company</th><th>Symbol</th><th>Open</th><th>High</th><th>Low</th><th>Price</th><th>Change</th><th>% Change</th><th>Weight</th><th>Volume</th><th>Date</th><th>Source</th></tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>
"""
    (REPORT / "ohlcv.html").write_text(page("GLI Component OHLCV", body, "ohlcv.html"), encoding="utf-8")


def ticker_line(rows: list[dict[str, str]]) -> str:
    last = rows[-1]
    previous = rows[-2] if len(rows) > 1 else None
    close = dec(last["GLI_Close"])
    high, low = dec(last["GLI_High"]), dec(last["GLI_Low"])
    change = close - dec(previous["GLI_Close"]) if previous else Decimal(0)
    pct = change / dec(previous["GLI_Close"]) * Decimal(100) if previous and dec(previous["GLI_Close"]) else Decimal(0)
    arrow = "▲" if change > 0 else "▼" if change < 0 else "•"
    sign = "+" if change > 0 else ""
    return f"GLI {close:,.2f} {arrow}{sign}{change:,.2f} ({sign}{pct:.2f}%)  H {high:,.2f}  L {low:,.2f}  ({last['Date']})"


def ensure_css(html_text: str) -> str:
    if "--ink:#172033" in html_text:
        return html_text
    return html_text.replace("</head>", CSS_BLOCK + "\n</head>", 1) if "</head>" in html_text else CSS_BLOCK + html_text


def insert_after_body(html_text: str, block: str) -> str:
    match = re.search(r"<body[^>]*>", html_text, flags=re.I)
    if not match:
        return block + html_text
    return html_text[:match.end()] + "\n" + block + "\n" + html_text[match.end():]


def inject_home(full_rows: list[dict[str, str]]) -> None:
    """Write a clean home page from the latest completed-session history.

    The engine may emit an in-progress session with a zero close and may also
    leave an older injected shell in ``report/index.html``. Rebuilding the home
    page from scratch avoids stale duplicate blocks and ensures every displayed
    home-page value comes from the same completed row used by the ticker,
    history, rankings, milestones, and interactive chart.
    """
    if not full_rows:
        raise SystemExit("Cannot build the home page without completed GLI rows.")

    last = full_rows[-1]
    previous = full_rows[-2] if len(full_rows) > 1 else None
    close = dec(last["GLI_Close"])
    prev_close = dec(previous["GLI_Close"]) if previous else close
    change = close - prev_close if previous else Decimal(0)
    pct = change / prev_close * Decimal(100) if previous and prev_close else Decimal(0)

    accepted_through = "2026-08-04"
    if LIVE_ANCHOR.exists():
        try:
            anchor = json.loads(LIVE_ANCHOR.read_text(encoding="utf-8"))
            accepted_through = str(anchor.get("accepted_through") or accepted_through)
        except (OSError, json.JSONDecodeError, TypeError):
            pass

    ticker = f"""<!-- GLI_TICKER_V2 --><div id="gli-tickerbar"><div id="gltxt">Loading GLI…</div><div class="gli-pill">The Great Lakes Index (GLI)</div></div><script>fetch('ticker.txt',{{cache:'no-store'}}).then(r=>r.text()).then(t=>{{const s=t.trim();const cls=s.includes('▲')?'up':s.includes('▼')?'down':'flat';document.getElementById('gltxt').innerHTML=`<span class="${{cls}}">${{s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;')}}</span>`;}}).catch(()=>document.getElementById('gltxt').textContent='GLI ticker unavailable');</script>"""

    symbol = f"""<!-- GLI_SYMBOL_V2 --><section class="gli-symbolbox"><div style="display:flex;justify-content:space-between;align-items:flex-end;gap:12px;flex-wrap:wrap"><div><div class="gli-k">Symbol</div><div class="gli-v">GLI</div><div class="gli-mini">Price-weighted • Original base {BASE_VALUE:.2f} on {BASE_DATE}</div></div><div style="text-align:right"><div class="gli-k">As of</div><div class="gli-v" style="font-size:16px">{last['Date']}</div><div class="gli-mini">Change: {change:+.2f} ({pct:+.2f}%)</div></div></div><div style="margin-top:12px" class="gli-grid"><div><div class="gli-k">Open</div><div class="gli-v">{dec(last['GLI_Open']):,.2f}</div></div><div><div class="gli-k">High</div><div class="gli-v">{dec(last['GLI_High']):,.2f}</div></div><div><div class="gli-k">Low</div><div class="gli-v">{dec(last['GLI_Low']):,.2f}</div></div><div><div class="gli-k">Close</div><div class="gli-v">{close:,.2f}</div></div><div><div class="gli-k">Total Volume</div><div class="gli-v">{fmt_int(last['TotalVolume'])}</div></div></div></section>"""

    recent_rows = []
    previous_recent: Decimal | None = None
    # Calculate changes in chronological order, then display newest first.
    recent_calculated: list[tuple[dict[str, str], Decimal, Decimal]] = []
    for row in full_rows[-11:]:
        row_close = dec(row["GLI_Close"])
        row_change = row_close - previous_recent if previous_recent is not None else Decimal(0)
        row_pct = row_change / previous_recent * Decimal(100) if previous_recent else Decimal(0)
        recent_calculated.append((row, row_change, row_pct))
        previous_recent = row_close
    # Recalculate the first visible row against the session immediately before
    # the display window when one exists.
    if len(full_rows) > len(recent_calculated):
        first_row, _, _ = recent_calculated[0]
        prior_close = dec(full_rows[-len(recent_calculated)-1]["GLI_Close"])
        first_close = dec(first_row["GLI_Close"])
        first_change = first_close - prior_close
        first_pct = first_change / prior_close * Decimal(100) if prior_close else Decimal(0)
        recent_calculated[0] = (first_row, first_change, first_pct)

    for row, row_change, row_pct in reversed(recent_calculated):
        cls = "positive" if row_change > 0 else "negative" if row_change < 0 else ""
        recent_rows.append(
            "<tr>"
            f"<td>{html.escape(row['Date'])}</td>"
            f"<td>{dec(row['GLI_Open']):,.2f}</td>"
            f"<td>{dec(row['GLI_High']):,.2f}</td>"
            f"<td>{dec(row['GLI_Low']):,.2f}</td>"
            f"<td><b>{dec(row['GLI_Close']):,.2f}</b></td>"
            f"<td class=\"{cls}\">{row_change:+.2f}</td>"
            f"<td class=\"{cls}\">{row_pct:+.2f}%</td>"
            f"<td>{fmt_int(row.get('TotalVolume', '0'))}</td>"
            "</tr>"
        )

    chart = """<!-- GLI_INTERACTIVE_CHART_V5_DATE_INPUTS -->
<div class="chart-shell">
  <style>
  .gli-chart-date-controls{display:flex;align-items:end;gap:10px;flex-wrap:wrap;padding:10px 12px 4px}
  .gli-chart-date-controls label{display:grid;gap:4px;font-size:12px;font-weight:800;color:#344054}
  .gli-chart-date-controls input[type="date"]{min-width:154px;font:inherit;font-weight:650}
  .gli-chart-date-controls button{border:1px solid #175cd3;border-radius:9px;padding:8px 14px;background:#175cd3;color:#fff;font:inherit;font-weight:800;cursor:pointer;min-height:38px}
  .gli-chart-date-controls button:hover{background:#1849a9}
  .gli-chart-date-controls button:focus-visible{outline:3px solid #84adff;outline-offset:2px}
  .gli-chart-date-status{min-height:18px;flex-basis:100%;font-size:12px;color:#b42318}
  @media(max-width:600px){.gli-chart-date-controls{align-items:stretch;padding-left:6px;padding-right:6px}.gli-chart-date-controls label{flex:1 1 145px}.gli-chart-date-controls input[type="date"]{width:100%;min-width:0}.gli-chart-date-controls button{flex:1 1 100%}}
  </style>
  <div class="gli-chart-date-controls" aria-label="Candlestick chart date range">
    <label for="gli-date-start">Start date<input id="gli-date-start" type="date"></label>
    <label for="gli-date-end">End date<input id="gli-date-end" type="date"></label>
    <button id="gli-date-apply" type="button">Apply</button>
    <div class="gli-chart-date-status" id="gli-date-status" aria-live="polite"></div>
  </div>
  <div id="gli-candlestick" style="height:560px"><div class="chart-status">Loading interactive chart…</div></div>
  <div id="gli-chart-fallback" style="display:none"><img src="gli_close.png" alt="Great Lakes Index close chart" style="max-width:100%"></div>
</div>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script>
fetch('./data/gli_history.json',{cache:'no-store'}).then(r=>r.json()).then(d=>{
  const rows=d.rows;
  const trace={
    type:'candlestick',name:'GLI',x:rows.map(r=>r.date),
    open:rows.map(r=>r.open),high:rows.map(r=>r.high),low:rows.map(r=>r.low),close:rows.map(r=>r.close),
    increasing:{line:{color:'#087443'}},decreasing:{line:{color:'#b42318'}}
  };
  const layout={
    margin:{l:55,r:25,t:35,b:45},paper_bgcolor:'#fff',plot_bgcolor:'#fff',hovermode:'x unified',
    xaxis:{
      type:'date',rangeslider:{visible:false},
      rangeselector:{buttons:[
        {count:1,label:'1M',step:'month',stepmode:'backward'},
        {count:3,label:'3M',step:'month',stepmode:'backward'},
        {count:6,label:'6M',step:'month',stepmode:'backward'},
        {count:1,label:'YTD',step:'year',stepmode:'todate'},
        {count:1,label:'1Y',step:'year',stepmode:'backward'},
        {count:5,label:'5Y',step:'year',stepmode:'backward'},
        {step:'all',label:'All'}
      ]}
    },
    yaxis:{title:'Index Level',fixedrange:false},showlegend:false
  };
  const rowTimes=rows.map(r=>Date.parse(r.date));
  const dateStart=document.getElementById('gli-date-start');
  const dateEnd=document.getElementById('gli-date-end');
  const dateApply=document.getElementById('gli-date-apply');
  const dateStatus=document.getElementById('gli-date-status');
  const firstDate=rows[0]?.date||'';
  const lastDate=rows[rows.length-1]?.date||'';
  dateStart.min=firstDate;dateStart.max=lastDate;dateStart.value=firstDate;
  dateEnd.min=firstDate;dateEnd.max=lastDate;dateEnd.value=lastDate;
  function isoDay(value,fallback){
    if(value==null)return fallback;
    const text=String(value);
    const match=text.match(/^\\d{4}-\\d{2}-\\d{2}/);
    if(match)return match[0];
    const t=Date.parse(value);
    return Number.isFinite(t)?new Date(t).toISOString().slice(0,10):fallback;
  }
  function clampDay(value){
    let day=isoDay(value,'');
    if(!day)return '';
    if(firstDate&&day<firstDate)day=firstDate;
    if(lastDate&&day>lastDate)day=lastDate;
    return day;
  }
  function syncDateInputs(startValue,endValue){
    dateStart.value=clampDay(startValue==null?firstDate:startValue)||firstDate;
    dateEnd.value=clampDay(endValue==null?lastDate:endValue)||lastDate;
    dateStatus.textContent='';
  }
  function rescaleVisibleY(gd,startValue,endValue){
    let start=startValue==null?-Infinity:Date.parse(startValue);
    let end=endValue==null?Infinity:Date.parse(endValue);
    if(!Number.isFinite(start))start=-Infinity;
    if(!Number.isFinite(end))end=Infinity;
    if(start>end){const tmp=start;start=end;end=tmp;}
    let lo=Infinity,hi=-Infinity;
    for(let i=0;i<rows.length;i++){
      const t=rowTimes[i];
      if(!Number.isFinite(t)||t<start||t>end)continue;
      const low=Number(rows[i].low),high=Number(rows[i].high);
      if(Number.isFinite(low))lo=Math.min(lo,low);
      if(Number.isFinite(high))hi=Math.max(hi,high);
    }
    if(!Number.isFinite(lo)||!Number.isFinite(hi))return;
    const span=hi-lo;
    const pad=span>0?span*0.08:Math.max(Math.abs(hi)*0.02,1);
    Plotly.relayout(gd,{'yaxis.autorange':false,'yaxis.range':[lo-pad,hi+pad]});
  }
  function applyDateInputs(gd){
    const start=clampDay(dateStart.value);
    const end=clampDay(dateEnd.value);
    if(!start||!end){dateStatus.textContent='Enter both a start date and an end date.';return;}
    if(start>end){dateStatus.textContent='Start date must be on or before end date.';return;}
    if(start===end){dateStatus.textContent='Choose at least a two-day calendar range.';return;}
    dateStart.value=start;dateEnd.value=end;dateStatus.textContent='';
    Plotly.relayout(gd,{'xaxis.autorange':false,'xaxis.range':[start,end]});
    rescaleVisibleY(gd,start,end);
  }
  return Plotly.newPlot('gli-candlestick',[trace],layout,{responsive:true,displaylogo:false,scrollZoom:true}).then(gd=>{
    dateApply.addEventListener('click',()=>applyDateInputs(gd));
    [dateStart,dateEnd].forEach(input=>input.addEventListener('keydown',ev=>{if(ev.key==='Enter')applyDateInputs(gd);}));
    gd.on('plotly_relayout',ev=>{
      if(ev['xaxis.autorange']===true){
        syncDateInputs(firstDate,lastDate);rescaleVisibleY(gd,null,null);return;
      }
      let x0=ev['xaxis.range[0]'],x1=ev['xaxis.range[1]'];
      if(Array.isArray(ev['xaxis.range'])){x0=ev['xaxis.range'][0];x1=ev['xaxis.range'][1];}
      if(x0!==undefined||x1!==undefined){
        const current=gd.layout.xaxis&&gd.layout.xaxis.range;
        if(x0===undefined&&Array.isArray(current))x0=current[0];
        if(x1===undefined&&Array.isArray(current))x1=current[1];
        syncDateInputs(x0,x1);rescaleVisibleY(gd,x0,x1);
      }
    });
    return gd;
  });
}).catch(()=>{
  document.getElementById('gli-candlestick').style.display='none';
  document.getElementById('gli-chart-fallback').style.display='block';
});
</script>"""

    body = f"""
{ticker}
{symbol}
<div class="page-head"><div><h1>The Great Lakes Index (GLI)</h1><div class="muted">Price-weighted • Original base {BASE_VALUE:.2f} on {BASE_DATE}</div></div><a href="./history.html" style="font-weight:800;text-decoration:none">View full history →</a></div>
<div class="source-note">Accepted close and OHLCV chains are immutable through {accepted_through}. Later completed sessions roll forward live; unfinished sessions are withheld until a valid close is available.</div>
{chart}
<h2>Recent Levels</h2>
<div class="table-wrap"><table aria-label="Recent Great Lakes Index levels"><thead><tr><th>Date</th><th>Open</th><th>High</th><th>Low</th><th>Close</th><th>Point Change</th><th>% Change</th><th>Total Volume</th></tr></thead><tbody>{''.join(recent_rows)}</tbody></table></div>
"""
    (REPORT / "index.html").write_text(page("The Great Lakes Index (GLI)", body, "index.html"), encoding="utf-8")

def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    REPORT_DATA.mkdir(parents=True, exist_ok=True)
    live_rows = load_levels()
    full_rows = load_full_history(live_rows)
    records = build_history_records(full_rows)
    constituents = load_constituents()

    (REPORT / "ticker.txt").write_text(ticker_line(full_rows) + "\n", encoding="utf-8")
    write_history_data(records)
    write_history_page(records)
    write_market_moves(records, full_rows)
    write_milestones(full_rows)
    write_weights_page(constituents)
    write_component_history(full_rows, constituents)
    write_component_ohlcv(constituents)
    inject_home(full_rows)
    print(f"OK: built GLI site through {full_rows[-1]['Date']} with {len(full_rows):,} historical rows.")


if __name__ == "__main__":
    main()
