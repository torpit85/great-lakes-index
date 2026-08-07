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
.timeline{display:grid;gap:14px}.snapshot{border:1px solid var(--line);border-radius:14px;padding:16px;background:#fff}.snapshot-head{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;flex-wrap:wrap}.snapshot-date{font-size:20px;font-weight:900}.snapshot-title{color:var(--muted);font-weight:700}.change-row{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}.badge{border-radius:999px;padding:5px 9px;font-size:12px;font-weight:800}.badge.add{background:#ecfdf3;color:#027a48}.badge.remove{background:#fef3f2;color:#b42318}.component-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:7px;margin-top:12px}.component{border:1px solid #e2e7ef;border-radius:8px;padding:7px 9px;background:#f9fafb;font-size:13px}.component.added{border-color:#86d5ad;background:#ecfdf3;font-weight:750}.component-symbol{font-weight:850}.component-name{color:var(--muted);font-size:11px;margin-top:2px}
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
    snapshots = list(payload["snapshots"])
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
    payload["snapshots"] = snapshots
    payload["company_names"] = company_map()
    return payload


def write_component_history(history_rows: list[dict[str, str]], constituents: list[dict[str, str]]) -> None:
    payload = extend_component_history(history_rows, constituents)
    write_json(REPORT_DATA / "component_history.json", payload)
    years = sorted({s["date"][:4] for s in payload["snapshots"]})
    year_buttons = '<button class="tab" data-year="all">All</button>' + "".join(f'<button class="tab" data-year="{y}">{y}</button>' for y in years)
    body = f"""
<div class="page-head"><div><h1>Component History</h1><div class="muted">Roster checkpoints modeled after GLI_Component_history.xlsx</div></div><input type="search" id="component-search" placeholder="Search company or ticker" aria-label="Search component history"></div>
<div class="source-note">Through May 1, 2014, company labels preserve the historical workbook. Later checkpoints are derived from accepted component OHLCV rows and shown by ticker; 2026 is extended from the committed constituent chronology.</div>
<div class="tabs" id="component-years">{year_buttons}</div><div id="component-timeline" class="timeline"><div class="empty">Loading component history…</div></div>
<script>
let componentData=null;let componentYear='{years[-1]}';let componentQuery='';
function esc(s){{return String(s??'').replace(/[&<>\"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c]));}}
function renderComponents(){{if(!componentData)return;const names=componentData.company_names||{{}};let rows=componentData.snapshots.filter(s=>componentYear==='all'||s.date.startsWith(componentYear));if(componentQuery)rows=rows.filter(s=>s.components.some(c=>(c+' '+(names[c]||'')).toLowerCase().includes(componentQuery))||s.added.some(c=>c.toLowerCase().includes(componentQuery))||s.removed.some(c=>c.toLowerCase().includes(componentQuery)));rows=rows.slice().reverse();document.getElementById('component-timeline').innerHTML=rows.map(s=>{{const added=new Set(s.added||[]);const chips=s.components.map(c=>{{const name=s.label_mode==='ticker'?(names[c]||''):'';return `<div class="component ${{added.has(c)?'added':''}}"><div class="component-symbol">${{esc(c)}}</div>${{name?`<div class="component-name">${{esc(name)}}</div>`:''}}</div>`;}}).join('');const adds=(s.added||[]).map(x=>`<span class="badge add">+ ${{esc(x)}}</span>`).join('');const rems=(s.removed||[]).map(x=>`<span class="badge remove">− ${{esc(x)}}</span>`).join('');return `<article class="snapshot"><div class="snapshot-head"><div><div class="snapshot-date">${{s.date}}</div><div class="snapshot-title">${{esc(s.title)}} • ${{s.count}} components</div></div><div class="gli-pill">${{s.label_mode==='ticker'?'Symbols':'Company names'}}</div></div>${{s.note?`<div class="source-note">${{esc(s.note)}}</div>`:''}}${{adds||rems?`<div class="change-row">${{adds}}${{rems}}</div>`:''}}<div class="component-grid">${{chips}}</div></article>`;}}).join('')||'<div class="empty">No matching checkpoints.</div>';}}
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

    chart = """<!-- GLI_INTERACTIVE_CHART_V2 --><div class="chart-shell"><div id="gli-candlestick" style="height:560px"><div class="chart-status">Loading interactive chart…</div></div><div id="gli-chart-fallback" style="display:none"><img src="gli_close.png" alt="Great Lakes Index close chart" style="max-width:100%"></div></div><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script><script>fetch('./data/gli_history.json',{cache:'no-store'}).then(r=>r.json()).then(d=>{const rows=d.rows;const trace={type:'candlestick',name:'GLI',x:rows.map(r=>r.date),open:rows.map(r=>r.open),high:rows.map(r=>r.high),low:rows.map(r=>r.low),close:rows.map(r=>r.close),increasing:{line:{color:'#087443'}},decreasing:{line:{color:'#b42318'}}};const layout={margin:{l:55,r:25,t:35,b:45},paper_bgcolor:'#fff',plot_bgcolor:'#fff',hovermode:'x unified',xaxis:{type:'date',rangeslider:{visible:true},rangeselector:{buttons:[{count:1,label:'1M',step:'month',stepmode:'backward'},{count:3,label:'3M',step:'month',stepmode:'backward'},{count:6,label:'6M',step:'month',stepmode:'backward'},{count:1,label:'YTD',step:'year',stepmode:'todate'},{count:1,label:'1Y',step:'year',stepmode:'backward'},{count:5,label:'5Y',step:'year',stepmode:'backward'},{step:'all',label:'All'}]}},yaxis:{title:'Index Level',fixedrange:false},showlegend:false};return Plotly.newPlot('gli-candlestick',[trace],layout,{responsive:true,displaylogo:false,scrollZoom:true});}).catch(()=>{document.getElementById('gli-candlestick').style.display='none';document.getElementById('gli-chart-fallback').style.display='block';});</script>"""

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
