#!/usr/bin/env python3
"""Build and render Great Lakes Index daily market-breadth data.

Breadth uses the same membership-filtered clean-return pipeline as gli_feats.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

import gli_feats


def build(root: Path, site_data: Path, full_rows: list[dict[str, str]]) -> dict[str, Any]:
    idx = pd.DataFrame(full_rows).rename(
        columns={
            "GLI_Open": "Open",
            "GLI_High": "High",
            "GLI_Low": "Low",
            "GLI_Close": "Close",
            "TotalVolume": "Volume",
        }
    )
    for col in ["Open", "High", "Low", "Close", "Volume", "Divisor", "RosterCount"]:
        if col in idx.columns:
            idx[col] = pd.to_numeric(idx[col], errors="coerce")
    idx = idx.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

    raw = gli_feats._load_components(root, site_data)
    eligible, membership_audit = gli_feats._filter_components_to_membership(
        raw, root, site_data
    )
    transition_map = gli_feats._same_security_transition_map(root, site_data)
    comp = gli_feats._prepare_component_metrics(eligible, idx, transition_map)

    cc = comp.dropna(subset=["breadth_ret"]).copy()
    cc["adv"] = (cc["breadth_ret"] > 0).astype(int)
    cc["dec"] = (cc["breadth_ret"] < 0).astype(int)
    cc["flat"] = (cc["breadth_ret"] == 0).astype(int)

    breadth = (
        cc.groupby("Date", sort=True)
        .agg(
            advancing=("adv", "sum"),
            declining=("dec", "sum"),
            unchanged=("flat", "sum"),
            components=("Ticker", "size"),
        )
        .reset_index()
    )

    breadth["adv_pct"] = breadth["advancing"] / breadth["components"] * 100.0
    breadth["dec_pct"] = breadth["declining"] / breadth["components"] * 100.0
    breadth["unch_pct"] = breadth["unchanged"] / breadth["components"] * 100.0
    breadth["net"] = breadth["advancing"] - breadth["declining"]
    breadth["one_way_pct"] = breadth[["adv_pct", "dec_pct"]].max(axis=1)

    idx_returns = idx.set_index("Date")["Close"].pct_change() * 100.0
    breadth["gli_pct"] = breadth["Date"].map(idx_returns)

    if "RosterCount" in idx.columns:
        roster_map = idx.set_index("Date")["RosterCount"]
        breadth["roster"] = breadth["Date"].map(roster_map)
    else:
        breadth["roster"] = breadth["components"]

    breadth["roster"] = pd.to_numeric(breadth["roster"], errors="coerce")
    breadth["complete"] = (
        breadth["roster"].notna()
        & (breadth["roster"] > 0)
        & (breadth["components"] == breadth["roster"])
    )
    breadth["perfect"] = breadth["complete"] & (
        (breadth["advancing"] == breadth["roster"])
        | (breadth["declining"] == breadth["roster"])
    )

    rows: list[dict[str, Any]] = []
    for q in breadth.itertuples(index=False):
        direction = (
            "advancing"
            if q.advancing > q.declining
            else "declining"
            if q.declining > q.advancing
            else "even"
        )
        rows.append(
            {
                "date": str(q.Date),
                "year": int(str(q.Date)[:4]),
                "advancing": int(q.advancing),
                "declining": int(q.declining),
                "unchanged": int(q.unchanged),
                "components": int(q.components),
                "roster": None if pd.isna(q.roster) else int(q.roster),
                "adv_pct": float(q.adv_pct),
                "dec_pct": float(q.dec_pct),
                "unch_pct": float(q.unch_pct),
                "net": int(q.net),
                "one_way_pct": float(q.one_way_pct),
                "gli_pct": None if pd.isna(q.gli_pct) else float(q.gli_pct),
                "direction": direction,
                "complete": bool(q.complete),
                "perfect": bool(q.perfect),
            }
        )

    perfect_sessions = [r for r in rows if r["perfect"]]
    years = sorted({r["year"] for r in rows})

    return {
        "schema_version": 1,
        "generated_through": str(idx["Date"].iloc[-1]) if len(idx) else "",
        "breadth_through": rows[-1]["date"] if rows else "",
        "membership_filter": membership_audit,
        "definitions": [
            "Advancing, declining, and unchanged are based on the same membership-filtered clean component returns used by Feats & Records.",
            "Return chains require consecutive GLI sessions within a ticker tenure. Divisor-reset and explicit component-reset sessions use open-to-close movement so mechanical corporate-action or reconstitution changes do not masquerade as market breadth.",
            "Breadth Count is the number of components with a usable clean return on that session. Roster is the official active GLI component count.",
            "A Perfect Breadth session requires Breadth Count to equal Roster and every active component to move in the same direction.",
        ],
        "years": years,
        "latest": rows[-1] if rows else None,
        "perfect_sessions": perfect_sessions,
        "rows": rows,
    }


def render(payload: dict[str, Any]) -> str:
    latest = payload.get("latest") or {}
    perfect = payload.get("perfect_sessions") or []
    years = payload.get("years") or []

    year_options = '<option value="all">All history</option>' + "".join(
        f'<option value="{y}">{y}</option>' for y in years
    )

    perfect_rows = "".join(
        (
            f'<tr><td>{r["date"]}</td>'
            f'<td>{r["advancing"]}</td><td>{r["declining"]}</td>'
            f'<td>{r["unchanged"]}</td><td>{r["roster"]}</td>'
            f'<td>{"All advancing" if r["advancing"] else "All declining"}</td>'
            f'<td class="{"positive" if (r.get("gli_pct") or 0) > 0 else "negative" if (r.get("gli_pct") or 0) < 0 else ""}">'
            f'{(r.get("gli_pct") or 0):+.2f}%</td></tr>'
        )
        for r in perfect
    ) or '<tr><td colspan="7" class="empty">No perfect breadth sessions.</td></tr>'

    latest_line = (
        f'{latest.get("advancing", 0)}–{latest.get("declining", 0)}–'
        f'{latest.get("unchanged", 0)}'
    )
    latest_adv = float(latest.get("adv_pct") or 0)
    latest_net = int(latest.get("net") or 0)
    latest_date = latest.get("date", "")
    perfect_up = sum(1 for r in perfect if r["advancing"] > 0)
    perfect_down = sum(1 for r in perfect if r["declining"] > 0)

    definitions = "".join(f"<li>{d}</li>" for d in payload.get("definitions", []))

    return f"""
<div class="page-head">
  <div>
    <h1>Market Breadth</h1>
    <div class="muted">Daily advancing, declining, and unchanged GLI components</div>
  </div>
</div>

<div class="source-note">
  Breadth uses the same membership-filtered clean-return methodology as Feats &amp; Records.
  Perfect breadth additionally requires complete active-roster coverage.
</div>

<div class="metric-cards">
  <div class="metric-card"><span class="gli-k">Latest breadth</span><strong>{latest_line}</strong><span class="muted">{latest_date} • advancing–declining–unchanged</span></div>
  <div class="metric-card"><span class="gli-k">Latest advancing</span><strong>{latest_adv:.2f}%</strong><span class="muted">of usable breadth components</span></div>
  <div class="metric-card"><span class="gli-k">Latest net breadth</span><strong class="{"positive" if latest_net > 0 else "negative" if latest_net < 0 else ""}">{latest_net:+d}</strong><span class="muted">advancing minus declining</span></div>
  <div class="metric-card"><span class="gli-k">Perfect sessions</span><strong>{len(perfect)}</strong><span class="muted">{perfect_up} all-up • {perfect_down} all-down</span></div>
</div>

<h2>Daily Breadth History</h2>
<div class="controls">
  <label>Year <select id="breadth-year">{year_options}</select></label>
  <label>Extremity
    <select id="breadth-filter">
      <option value="all">All sessions</option>
      <option value="90">90%+ one-way breadth</option>
      <option value="95">95%+ one-way breadth</option>
      <option value="99">99%+ one-way breadth</option>
      <option value="perfect">Perfect breadth</option>
    </select>
  </label>
</div>

<div class="table-wrap">
<table aria-label="Great Lakes Index daily market breadth">
<thead><tr>
  <th>Date</th><th>Advancing</th><th>Declining</th><th>Unchanged</th>
  <th>Breadth Count</th><th>Roster</th><th>Adv %</th><th>Dec %</th>
  <th>Net Breadth</th><th>GLI % Change</th>
</tr></thead>
<tbody id="breadth-body"><tr><td colspan="10" class="empty">Loading breadth…</td></tr></tbody>
</table>
</div>

<h2 style="margin-top:22px">Perfect Breadth Sessions</h2>
<div class="table-wrap">
<table aria-label="Perfect Great Lakes Index breadth sessions">
<thead><tr><th>Date</th><th>Advancing</th><th>Declining</th><th>Unchanged</th><th>Roster</th><th>Direction</th><th>GLI % Change</th></tr></thead>
<tbody>{perfect_rows}</tbody>
</table>
</div>

<div class="panel" style="margin-top:18px">
  <h2>Methodology</h2>
  <ul class="muted" style="margin:0;padding-left:20px">{definitions}</ul>
</div>

<script>
const bPct=new Intl.NumberFormat('en-US',{{minimumFractionDigits:2,maximumFractionDigits:2}});
let breadthRows=[];

function signedBreadth(v,suffix=''){{
  if(v===null||v===undefined||Number.isNaN(Number(v)))return '';
  const n=Number(v);
  return (n>0?'+':'')+bPct.format(n)+suffix;
}}

function renderBreadth(){{
  const year=document.getElementById('breadth-year').value;
  const mode=document.getElementById('breadth-filter').value;
  let rows=breadthRows.filter(r=>year==='all'||String(r.year)===year);

  if(mode==='perfect')rows=rows.filter(r=>r.perfect);
  else if(mode!=='all')rows=rows.filter(r=>Number(r.one_way_pct)>=Number(mode));

  rows=rows.slice().reverse();
  document.getElementById('breadth-body').innerHTML=rows.map(r=>`
    <tr>
      <td>${{r.date}}</td>
      <td class="${{r.advancing>r.declining?'positive':''}}">${{r.advancing}}</td>
      <td class="${{r.declining>r.advancing?'negative':''}}">${{r.declining}}</td>
      <td>${{r.unchanged}}</td>
      <td>${{r.components}}</td>
      <td>${{r.roster??''}}</td>
      <td>${{bPct.format(r.adv_pct)}}%</td>
      <td>${{bPct.format(r.dec_pct)}}%</td>
      <td class="${{r.net>0?'positive':r.net<0?'negative':''}}">${{r.net>0?'+':''}}${{r.net}}</td>
      <td class="${{r.gli_pct>0?'positive':r.gli_pct<0?'negative':''}}">${{signedBreadth(r.gli_pct,'%')}}</td>
    </tr>`).join('') || '<tr><td colspan="10" class="empty">No qualifying sessions.</td></tr>';
}}

fetch('./data/breadth.json',{{cache:'no-store'}})
  .then(r=>r.json())
  .then(d=>{{breadthRows=d.rows||[];renderBreadth();}})
  .catch(()=>{{document.getElementById('breadth-body').innerHTML='<tr><td colspan="10" class="empty">Breadth data unavailable.</td></tr>';}});

document.getElementById('breadth-year').addEventListener('change',renderBreadth);
document.getElementById('breadth-filter').addEventListener('change',renderBreadth);
</script>
"""
