from __future__ import annotations

import csv, gzip, html, json, math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

HIST_GZ = "component_ohlcv_history_2005_2025.csv.gz"

# Corporate-action boundaries that are embedded in accepted component
# price lineage but are not represented by a same-day GLI divisor change.
#
# JCI 2007-10-03:
# Old Johnson Controls 3-for-1 split.  Accepted raw component history
# correctly changes from the pre-split price scale to the post-split
# price scale on this date.  Component performance metrics must use
# open-to-close for the reset session rather than interpreting the
# mechanical 3-for-1 price change as an investment return.
EXPLICIT_COMPONENT_RETURN_RESETS = {
    ("ACV", "2006-11-17"),
    ("JCI", "2007-10-03"),
}


def _fmt_num(v, decimals=2, sign=False, suffix=""):
    if v is None or not np.isfinite(float(v)):
        return "—"
    x=float(v); s=f"{x:,.{decimals}f}"
    if sign and x>0: s="+"+s
    return s+suffix

def _fmt_int(v):
    if v is None or not np.isfinite(float(v)): return "—"
    return f"{int(round(float(v))):,}"

def _rec(feat, holder, value, date="", detail="", ticker="", raw=None):
    return {"feat":feat,"holder":holder,"value":value,"date":date,"detail":detail,"ticker":ticker,"raw":raw}

def _period_label(v):
    return str(v)

def _name_maps(site_data: Path, root: Path):
    hist = defaultdict(list)
    hp=site_data/'historical_company_names.csv'
    if hp.exists():
        with hp.open(newline='',encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                t=(r.get('Ticker') or '').strip().upper(); n=(r.get('Company') or '').strip()
                if t and n:
                    hist[t].append(((r.get('StartDate') or '0001-01-01').strip(),(r.get('EndDate') or '9999-12-31').strip() or '9999-12-31',n))
    cur={}
    cp=root/'company_names.csv'
    if cp.exists():
        with cp.open(newline='',encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                t=(r.get('Ticker') or r.get('Symbol') or '').strip().upper()
                n=(r.get('Company') or r.get('Name') or '').strip()
                if t and n: cur[t]=n
    def name(ticker, day='9999-12-31'):
        ticker=str(ticker).upper()
        for s,e,n in hist.get(ticker,[]):
            if s<=day<=e: return n
        return cur.get(ticker,ticker)
    def label(ticker, day='9999-12-31'):
        n=name(ticker,day)
        return f"{ticker} — {n}" if n and n!=ticker else ticker
    return name,label


def _component_membership_ranges(root: Path, site_data: Path) -> dict[str, list[tuple[str, str]]]:
    """Return eligible GLI membership date ranges keyed by ticker.

    Historical ranges come from ``historical_company_names.csv`` through 2025.
    Live/current ranges come from ``constituents_great_lakes.csv`` beginning in
    2026.  The split prevents an open-ended historical name row from overriding
    a later 2026 roster removal.
    """
    ranges: dict[str, list[tuple[str, str]]] = defaultdict(list)
    hist_cutoff = "2025-12-31"

    hp = site_data / "historical_company_names.csv"
    if not hp.exists():
        raise SystemExit(f"Missing component membership source: {hp}")
    with hp.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"Ticker", "StartDate", "EndDate"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Historical membership source lacks columns: {sorted(missing)}")
        for row in reader:
            ticker = (row.get("Ticker") or "").strip().upper()
            if not ticker:
                continue
            start = (row.get("StartDate") or "0001-01-01").strip() or "0001-01-01"
            end = (row.get("EndDate") or hist_cutoff).strip() or hist_cutoff
            if start > hist_cutoff:
                continue
            end = min(end, hist_cutoff)
            if start <= end:
                ranges[ticker].append((start, end))

    cp = root / "constituents_great_lakes.csv"
    if not cp.exists():
        raise SystemExit(f"Missing live component membership source: {cp}")
    with cp.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"Ticker", "StartDate", "EndDate"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Live membership source lacks columns: {sorted(missing)}")
        for row in reader:
            ticker = (row.get("Ticker") or "").strip().upper()
            if not ticker:
                continue
            start = (row.get("StartDate") or "0001-01-01").strip() or "0001-01-01"
            end = (row.get("EndDate") or "9999-12-31").strip() or "9999-12-31"
            if end < "2026-01-01":
                continue
            start = max(start, "2026-01-01")
            if start <= end:
                ranges[ticker].append((start, end))

    for ticker in ranges:
        ranges[ticker].sort()
    return dict(ranges)


def _filter_components_to_membership(
    comp: pd.DataFrame,
    root: Path,
    site_data: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove component price rows outside the ticker's eligible GLI membership.

    The returned frame is the only component input used by the record book.
    This excludes pre-entry/post-removal quotes (including OTC trading) before
    any return, volume, weight, breadth, streak, or tenure metric is computed.
    """
    if comp.empty:
        return comp.copy(), {
            "raw_rows": 0,
            "eligible_rows": 0,
            "excluded_rows": 0,
            "excluded_tickers": [],
        }

    ranges = _component_membership_ranges(root, site_data)
    kept: list[pd.DataFrame] = []
    excluded_tickers: set[str] = set()
    excluded_rows = 0

    for ticker, group in comp.groupby("Ticker", sort=False):
        spans = ranges.get(str(ticker).upper(), [])
        if not spans:
            excluded_rows += len(group)
            excluded_tickers.add(str(ticker).upper())
            continue
        dates = group["Date"].astype(str)
        mask = pd.Series(False, index=group.index)
        for start, end in spans:
            mask |= dates.ge(start) & dates.le(end)
        if (~mask).any():
            excluded_rows += int((~mask).sum())
            excluded_tickers.add(str(ticker).upper())
        kept.append(group.loc[mask])

    if kept:
        filtered = pd.concat(kept, ignore_index=True)
        filtered = filtered.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    else:
        filtered = comp.iloc[0:0].copy()

    if filtered.empty:
        raise SystemExit("Membership filtering removed every component OHLCV row.")

    audit = {
        "raw_rows": int(len(comp)),
        "eligible_rows": int(len(filtered)),
        "excluded_rows": int(excluded_rows),
        "excluded_tickers": sorted(excluded_tickers),
    }
    return filtered, audit


def _load_components(root: Path, site_data: Path) -> pd.DataFrame:
    parts=[]
    hist=site_data/HIST_GZ
    if hist.exists():
        parts.append(pd.read_csv(hist,compression='gzip',usecols=['Date','Ticker','Open','High','Low','Close','Volume']))
    acc=site_data/'component_ohlcv'/'GLI_2026_component_ohlcv_used.csv'
    if acc.exists():
        parts.append(pd.read_csv(acc,usecols=['Date','Ticker','Open','High','Low','Close','Volume']))
    live=root/'gli_prices.csv'
    if live.exists():
        parts.append(pd.read_csv(live,usecols=['Date','Ticker','Open','High','Low','Close','Volume']))
    if not parts: return pd.DataFrame(columns=['Date','Ticker','Open','High','Low','Close','Volume'])
    df=pd.concat(parts,ignore_index=True)
    df['Ticker']=df['Ticker'].astype(str).str.upper().str.strip()
    for c in ['Open','High','Low','Close','Volume']: df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.dropna(subset=['Date','Ticker','Close']).copy()
    df=df[(df['Ticker']!='')&(df['Close']>0)]
    df=df.sort_values(['Date','Ticker']).drop_duplicates(['Date','Ticker'],keep='last').reset_index(drop=True)
    return df


def _streak_best(vals, predicate):
    best=cur=0; end=None
    for i,v in enumerate(vals):
        if predicate(v):
            cur+=1
            if cur>best: best=cur; end=i
        else: cur=0
    return best,end


def _index_feats(idx: pd.DataFrame):
    sections=[]
    idx=idx.copy().sort_values('Date').reset_index(drop=True)
    idx['prev_close']=idx['Close'].shift(1)
    idx['chg']=idx['Close']-idx['prev_close']; idx['pct']=idx['Close']/idx['prev_close']-1
    idx['gap']=idx['Open']-idx['prev_close']; idx['gap_pct']=idx['Open']/idx['prev_close']-1
    idx['oc']=idx['Close']-idx['Open']; idx['oc_pct']=idx['Close']/idx['Open']-1
    idx['range']=idx['High']-idx['Low']; idx['range_pct']=idx['range']/idx['Open']
    idx['low_recovery']=idx['Close']-idx['Low']; idx['high_reversal']=idx['Close']-idx['High']

    def rowrec(feat,row,value,detail=''):
        return _rec(feat,'GLI',value,row['Date'],detail)
    r=[]
    q=idx.loc[idx.Close.idxmax()]; r.append(rowrec('Highest closing value',q,_fmt_num(q.Close)))
    q=idx.loc[idx.High.idxmax()]; r.append(rowrec('Highest intraday value',q,_fmt_num(q.High)))
    q=idx.loc[idx.Close.idxmin()]; r.append(rowrec('Lowest closing value since inception',q,_fmt_num(q.Close)))
    q=idx.loc[idx.Low.idxmin()]; r.append(rowrec('Lowest intraday value since inception',q,_fmt_num(q.Low)))
    ath=idx['Close'].cummax(); isrec=idx['Close'].eq(ath)
    # consecutive record highs (base day included)
    best,end=_streak_best(isrec.tolist(),lambda x: bool(x))
    start=end-best+1 if end is not None else None
    r.append(_rec('Most consecutive record-high closes','GLI',f'{best} sessions',idx.iloc[end].Date if end is not None else '',f"{idx.iloc[start].Date} through {idx.iloc[end].Date}" if end is not None else ''))
    recyear=idx[isrec].assign(year=idx.loc[isrec,'Date'].str[:4]).groupby('year').size()
    if len(recyear):
        y=recyear.idxmax(); r.append(_rec('Most record-high closes in a calendar year','GLI',f'{int(recyear.max())} closes',y))
    recdates=idx.loc[isrec,'Date'].tolist()
    if len(recdates)>1:
        d=pd.to_datetime(recdates); gaps=pd.Series(d[1:].values-d[:-1].values)
        j=int(gaps.dt.days.idxmax()); r.append(_rec('Longest span without a new all-time closing high','GLI',f'{int(gaps.dt.days.iloc[j])} calendar days',recdates[j+1],f'Previous record: {recdates[j]}'))
    # fastest advances from any close to +threshold
    closes=idx['Close'].to_numpy(); dates=idx['Date'].to_numpy()
    for threshold in [10,25,50,100]:
        bestn=None; pair=None; j=0
        for i in range(len(closes)):
            if j<i+1: j=i+1
            while j<len(closes) and closes[j] < closes[i]+threshold: j+=1
            if j<len(closes):
                n=j-i
                if bestn is None or n<bestn: bestn=n; pair=(i,j)
        if pair:
            i,j=pair; r.append(_rec(f'Fastest {threshold}-point advance','GLI',f'{bestn} sessions',str(dates[j]),f'{dates[i]} ({closes[i]:.2f}) → {dates[j]} ({closes[j]:.2f})'))
    # century milestones
    maxc=int(idx.Close.max()//100)*100
    milestones=[]
    for th in range(100,maxc+1,100):
        hit=idx[idx.Close>=th]
        if len(hit): milestones.append((th,hit.iloc[0].Date))
    if len(milestones)>=2:
        gaps=[]
        for a,b in zip(milestones,milestones[1:]):
            i=idx.index[idx.Date==a[1]][0]; j=idx.index[idx.Date==b[1]][0]; gaps.append((j-i,a,b))
        mn=min(gaps,key=lambda x:x[0]); mx=max(gaps,key=lambda x:x[0])
        r.append(_rec('Shortest time between 100-point milestones','GLI',f'{mn[0]} sessions',mn[2][1],f'{mn[1][0]} → {mn[2][0]}'))
        r.append(_rec('Longest time between 100-point milestones','GLI',f'{mx[0]} sessions',mx[2][1],f'{mx[1][0]} → {mx[2][0]}'))
    sections.append(('Highs, Milestones & Progress',r))

    r=[]
    for feat,col,fn,fmt in [
        ('Largest intraday point range','range','max',lambda x:_fmt_num(x)),('Largest intraday percentage range','range_pct','max',lambda x:_fmt_num(x*100,2,suffix='%')),
        ('Largest gap up','gap','max',lambda x:_fmt_num(x,2,True)),('Largest gap down','gap','min',lambda x:_fmt_num(x,2,True)),
        ('Biggest open-to-close rally','oc','max',lambda x:_fmt_num(x,2,True)),('Biggest open-to-close decline','oc','min',lambda x:_fmt_num(x,2,True)),
        ('Largest low-to-close recovery','low_recovery','max',lambda x:_fmt_num(x)),('Largest high-to-close reversal','high_reversal','min',lambda x:_fmt_num(x,2,True))]:
        s=idx[col].dropna(); q=idx.loc[s.idxmax() if fn=='max' else s.idxmin()]; r.append(rowrec(feat,q,fmt(q[col])))
    s=idx[(idx.gap<0)&idx.pct.notna()];
    if len(s): q=s.loc[s.pct.idxmax()]; r.append(rowrec('Largest gain after opening lower',q,_fmt_num(q.pct*100,2,True,'%'),f'Opened {_fmt_num(q.gap_pct*100,2,True,"%")} vs prior close'))
    s=idx[(idx.gap>0)&idx.pct.notna()];
    if len(s): q=s.loc[s.pct.idxmin()]; r.append(rowrec('Largest loss after opening higher',q,_fmt_num(q.pct*100,2,True,'%'),f'Opened {_fmt_num(q.gap_pct*100,2,True,"%")} vs prior close'))
    s=idx[idx.pct.notna() & (idx.pct!=0)].copy();
    if len(s): q=s.loc[s.pct.abs().idxmin()]; r.append(rowrec('Closest finish to unchanged',q,_fmt_num(q.pct*100,4,True,'%'),f'{q.chg:+.4f} points'))
    s=idx[idx.pct.abs()<=0.001]
    if len(s): q=s.loc[s.range_pct.idxmax()]; r.append(rowrec('Largest intraday range on a nearly unchanged close',q,_fmt_num(q.range_pct*100,2,suffix='%'),'Close within ±0.10% of prior session'))
    sections.append(('Intraday & Session Feats',r))

    r=[]
    for n in [2,5,10]:
        ret=idx.Close/idx.Close.shift(n-1)-1
        for best,lab in [(True,'Best'),(False,'Worst')]:
            qidx=ret.idxmax() if best else ret.idxmin(); q=idx.loc[qidx]
            r.append(rowrec(f'{lab} {n}-session performance',q,_fmt_num(ret.loc[qidx]*100,2,True,'%'),f'Ending {q.Date}'))
    temp=idx.copy(); temp['dt']=pd.to_datetime(temp.Date)
    periods=[('week',temp.dt.dt.to_period('W-FRI')),('month',temp.dt.dt.to_period('M')),('quarter',temp.dt.dt.to_period('Q')),('calendar year',temp.dt.dt.to_period('Y'))]
    for label,per in periods:
        t=temp.assign(period=per.astype(str)).groupby('period').agg(first=('Close','first'),last=('Close','last'),date=('Date','last'),sessions=('Close','size'))
        if label=='week': t=t[t.sessions>=4]
        if label=='month': t=t[t.sessions>=15]
        t['ret']=t['last']/t['first']-1
        for best,lab in [(True,'Best'),(False,'Worst')]:
            q=t.loc[t.ret.idxmax() if best else t.ret.idxmin()]
            r.append(_rec(f'{lab} {label}','GLI',_fmt_num(q.ret*100,2,True,'%'),q.date,str(q.name)))
    sections.append(('Multi-Session Performance',r))

    # drawdowns
    r=[]; peak=idx.Close.cummax(); dd=idx.Close/peak-1
    q=idx.loc[dd.idxmin()]; r.append(rowrec('Largest decline from an all-time high',q,_fmt_num(dd.min()*100,2,True,'%')))
    for y,g in idx.assign(year=idx.Date.str[:4]).groupby('year'):
        d=g.Close/g.Close.cummax()-1
        if len(d):
            pass
    annual=[]
    for y,g in idx.assign(year=idx.Date.str[:4]).groupby('year'):
        d=g.Close/g.Close.cummax()-1; annual.append((d.min(),y,g.loc[d.idxmin()].Date))
    if annual:
        v,y,d=min(annual); r.append(_rec('Largest calendar-year drawdown','GLI',_fmt_num(v*100,2,True,'%'),d,y))
    # longest drawdown = sessions below prior ATH until recovered
    peakval=-np.inf; starti=None; bestlen=0; bestpair=None
    for i,row in idx.iterrows():
        if row.Close>=peakval:
            if starti is not None:
                ln=i-starti
                if ln>bestlen: bestlen=ln; bestpair=(starti,i)
                starti=None
            peakval=max(peakval,row.Close)
        elif starti is None: starti=i-1
    if starti is not None:
        ln=len(idx)-1-starti
        if ln>bestlen: bestlen=ln; bestpair=(starti,len(idx)-1)
    if bestpair:
        a,b=bestpair; r.append(_rec('Longest drawdown','GLI',f'{bestlen} sessions',idx.iloc[b].Date,f'{idx.iloc[a].Date} through {idx.iloc[b].Date}'))
    for th in [0.10,0.20,0.30]:
        # A recovery episode starts only after the index falls at least the
        # threshold from the running peak. Track the true trough, then measure
        # the number of GLI sessions needed to regain that same peak.
        best=None
        peak_i=0; peak_v=float(idx.Close.iloc[0])
        in_episode=False; episode_peak_i=None; episode_peak_v=None; trough_i=None; trough_v=None
        for i in range(1,len(idx)):
            close=float(idx.Close.iloc[i])
            if not in_episode:
                if close>=peak_v:
                    peak_i=i; peak_v=close
                elif close<=peak_v*(1-th):
                    in_episode=True; episode_peak_i=peak_i; episode_peak_v=peak_v
                    trough_i=i; trough_v=close
            else:
                if close<trough_v:
                    trough_i=i; trough_v=close
                if close>=episode_peak_v:
                    n=i-trough_i
                    cand=(n,episode_peak_i,trough_i,i)
                    if best is None or n<best[0]: best=cand
                    in_episode=False; peak_i=i; peak_v=close
        if best:
            n,i,t,e=best
            r.append(_rec(f'Fastest recovery from a {int(th*100)}% decline','GLI',f'{n} sessions',idx.loc[e].Date,f'Trough {idx.loc[t].Date}; prior peak {idx.loc[i].Date}'))
    sections.append(('Drawdowns & Recoveries',r))
    return sections


def _prepare_component_metrics(comp: pd.DataFrame, idx: pd.DataFrame):
    comp=comp.copy().sort_values(['Ticker','Date']).reset_index(drop=True)
    session_map={d:i for i,d in enumerate(idx.Date)}
    comp['session_i']=comp.Date.map(session_map)
    # Quotes outside the completed GLI session calendar must not enter any feat.
    comp=comp[comp.session_i.notna()].copy().reset_index(drop=True)

    # A ticker tenure is continuous only across consecutive GLI sessions.  Since
    # component rows are membership-filtered before this function, a removal and
    # later re-entry necessarily creates a session gap and therefore a new tenure.
    tenure_break=comp.Ticker.ne(comp.Ticker.shift())|comp.session_i.ne(comp.session_i.shift()+1)
    comp['_tenure_id']=tenure_break.cumsum()
    tenure_grp=comp.groupby('_tenure_id',sort=False)

    comp['prev_close']=tenure_grp.Close.shift(1)
    comp['prev_session_i']=tenure_grp.session_i.shift(1)
    comp['continuous']=comp.session_i.eq(comp.prev_session_i+1)
    comp['cc_ret']=np.where(comp.continuous,comp.Close/comp.prev_close-1,np.nan)
    comp['oc_ret']=np.where(comp.Open>0,comp.Close/comp.Open-1,np.nan)
    comp['intraday_range_pct']=np.where(comp.Open>0,(comp.High-comp.Low)/comp.Open,np.nan)
    comp['rvol20']=tenure_grp.Volume.transform(lambda s:s/s.shift(1).rolling(20,min_periods=20).mean())
    comp['prev_volume']=tenure_grp.Volume.shift(1)
    comp['volume_mult_prev']=np.where(comp.continuous & (comp.prev_volume>0),comp.Volume/comp.prev_volume,np.nan)
    div_map=idx.set_index('Date')['Divisor'] if 'Divisor' in idx.columns else pd.Series(dtype=float)
    if len(div_map):
        div_changed=div_map.ne(div_map.shift(1))
        comp['stable_divisor']=~comp.Date.map(div_changed).fillna(False)
    else:
        comp['stable_divisor']=True
    # Raw historical closes can jump mechanically on split/divisor-reset dates.
    #
    # Most such boundaries are identified by a GLI divisor change.  A small
    # number of accepted component lineages contain an explicit corporate-action
    # boundary without a same-day divisor change; those are listed above in
    # EXPLICIT_COMPONENT_RETURN_RESETS.
    #
    # For either kind of reset, use that session's open-to-close move instead
    # of the mechanical close-to-close price discontinuity.
    explicit_component_reset = pd.Series(
        [
            (ticker, day) in EXPLICIT_COMPONENT_RETURN_RESETS
            for ticker, day in zip(comp.Ticker, comp.Date)
        ],
        index=comp.index,
        dtype=bool,
    )
    comp['explicit_component_reset']=explicit_component_reset
    reset_ret=comp.oc_ret.where(comp.oc_ret.notna(),comp.cc_ret)
    comp['clean_ret']=np.where(
        comp.continuous,
        np.where(
            comp.stable_divisor & ~comp.explicit_component_reset,
            comp.cc_ret,
            reset_ret,
        ),
        np.nan,
    )
    factor=(1+comp.clean_ret).where(comp.clean_ret.notna(),1.0)
    comp['perf_index']=factor.groupby(comp._tenure_id).cumprod()
    first_close=comp.groupby('_tenure_id').Close.transform('first')
    comp['adj_price']=first_close*comp.perf_index
    return comp



def _identity_section(site_data:Path,label):
    p=site_data/'historical_company_names.csv';r=[]
    if not p.exists():return [('Identity & Corporate History',r)]
    df=pd.read_csv(p,dtype=str).fillna('')
    events=df[df.EventType!=''] if 'EventType' in df.columns else pd.DataFrame()
    if len(events):
        tc=events[events.EventType.str.contains('ticker',case=False,na=False)].groupby('Identity').size()
        if len(tc):i=tc.idxmax();row=df[df.Identity==i].iloc[-1];n=int(tc.max());r.append(_rec('Most ticker changes while represented in GLI history',i,f'{n} change' if n==1 else f'{n} changes'))
        nc=events[events.EventType.str.contains('name',case=False,na=False)].groupby('Identity').size()
        if len(nc):i=nc.idxmax();n=int(nc.max());r.append(_rec('Most company-name changes while represented in GLI history',i,f'{n} change' if n==1 else f'{n} changes'))
    # symbols per identity
    sy=df.groupby('Identity').Ticker.nunique();
    if len(sy):i=sy.idxmax();r.append(_rec('Most separate historical symbols associated with one corporate lineage',i,f'{int(sy.max())} symbols'))
    return [('Identity & Corporate History',r)]



def _best_true_component_run(df: pd.DataFrame, cond: pd.Series):
    if df.empty: return None
    cond=cond.fillna(False).astype(bool)
    ticker_break=df['Ticker'].ne(df['Ticker'].shift(1))
    session_break=df['session_i'].ne(df['session_i'].shift(1)+1)
    run_break=(~cond)|ticker_break|session_break
    run_id=run_break.cumsum()
    x=df.loc[cond,['Ticker','Date']].copy()
    if x.empty:return None
    x['run_id']=run_id[cond].to_numpy()
    g=x.groupby('run_id',sort=False).agg(Ticker=('Ticker','first'),start=('Date','first'),end=('Date','last'),sessions=('Date','size'))
    q=g.loc[g.sessions.idxmax()]
    return int(q.sessions),q.Ticker,q.start,q.end


def _component_sections(comp, idx, label):
    sections=[]; rank_tickers=[]
    def rr(feat,q,val,detail=''):
        t=str(q.Ticker);rank_tickers.append(t);return _rec(feat,label(t,str(q.Date)),val,str(q.Date),detail,t)
    r=[];s=comp[comp.Open>0]
    q=s.loc[s.oc_ret.idxmax()];r.append(rr('Largest single-session percentage gain',q,_fmt_num(q.oc_ret*100,2,True,'%'),'Open-to-close; avoids split/re-entry discontinuities'))
    q=s.loc[s.oc_ret.idxmin()];r.append(rr('Largest single-session percentage decline',q,_fmt_num(q.oc_ret*100,2,True,'%'),'Open-to-close; avoids split/re-entry discontinuities'))
    q=s.loc[s.intraday_range_pct.idxmax()];r.append(rr('Largest intraday percentage range',q,_fmt_num(q.intraday_range_pct*100,2,suffix='%')))
    grp=comp.groupby('Ticker',sort=False)
    tenure_grp=comp.groupby('_tenure_id',sort=False)
    for n in [2,5,10]:
        prior_close=tenure_grp.adj_price.shift(n-1);prior_si=tenure_grp.session_i.shift(n-1)
        # Component price records depend on an uninterrupted active ticker tenure,
        # not on GLI divisor changes caused by unrelated roster events.
        valid=(comp.session_i-prior_si==n-1)
        vals=(comp.adj_price/prior_close-1).where(valid)
        good=vals.dropna()
        if len(good):
            for best,txt in [(True,'Best'),(False,'Worst')]:
                qi=good.idxmax() if best else good.idxmin();q=comp.loc[qi]
                r.append(rr(f'{txt} {n}-session performance',q,_fmt_num(vals.loc[qi]*100,2,True,'%')))
    c=comp.copy();c['dt']=pd.to_datetime(c.Date)
    period_specs=[
        ('week',c.dt.dt.to_period('W-FRI'),4),
        ('month',c.dt.dt.to_period('M'),15),
        ('quarter',c.dt.dt.to_period('Q'),45),
        ('calendar year',c.dt.dt.to_period('Y'),180),
    ]
    for pname,per,min_sessions in period_specs:
        cp=c.assign(period=per.astype(str)).copy()
        x=cp.groupby(['_tenure_id','Ticker','period'],sort=False).agg(first=('adj_price','first'),last=('adj_price','last'),date=('Date','last'),sessions=('adj_price','size')).reset_index()
        # Avoid letting a handful of sessions immediately before removal or
        # after entry compete against substantially complete calendar periods.
        x=x[x.sessions>=min_sessions]
        x['ret']=x['last']/x['first']-1
        if len(x):
            for best,txt in [(True,'Best'),(False,'Worst')]:
                q=x.loc[x.ret.idxmax() if best else x.ret.idxmin()]
                obj=type('Q',(),{'Ticker':q.Ticker,'Date':q.date})
                r.append(rr(f'{txt} {pname}',obj,_fmt_num(q.ret*100,2,True,'%'),q.period))
    sections.append(('Price Performance',r))

    r=[];comp['weight']=comp.Close/comp.groupby('Date').Close.transform('sum')
    q=comp.loc[comp.weight.idxmax()];r.append(rr('Highest component weight ever',q,_fmt_num(q.weight*100,2,suffix='%')))
    posw=comp[comp.weight>0];q=posw.loc[posw.weight.idxmin()];r.append(rr('Lowest nonzero component weight ever',q,_fmt_num(q.weight*100,4,suffix='%')))
    comp['prev_weight']=grp.weight.shift(1);comp['weight_delta']=(comp.weight-comp.prev_weight).where(comp.continuous)
    z=comp.weight_delta.dropna()
    if len(z):
        q=comp.loc[z.idxmax()];r.append(rr('Largest one-session increase in weight',q,_fmt_num(q.weight_delta*100,2,True,' pp')))
        q=comp.loc[z.idxmin()];r.append(rr('Largest one-session decrease in weight',q,_fmt_num(q.weight_delta*100,2,True,' pp')))
    top=comp.loc[comp.groupby('Date').weight.idxmax(),['Date','Ticker']].sort_values('Date').reset_index(drop=True)
    vc=top.Ticker.value_counts();t=vc.idxmax();d=top[top.Ticker==t].Date.iloc[-1];r.append(_rec('Most sessions as the GLI’s largest-weighted component',label(t,d),f'{int(vc.max()):,} sessions',d,ticker=t));rank_tickers.append(t)
    same=top.Ticker.eq(top.Ticker.shift());rid=(~same).cumsum();runs=top.groupby(rid).agg(Ticker=('Ticker','first'),start=('Date','first'),end=('Date','last'),sessions=('Date','size'));q=runs.loc[runs.sessions.idxmax()];r.append(_rec('Longest consecutive run as the GLI’s largest-weighted component',label(q.Ticker,q.end),f'{int(q.sessions)} sessions',q.end,f'{q.start} through {q.end}',q.Ticker));rank_tickers.append(q.Ticker)
    # largest #1/#2 weight gap, vectorized using rank within date
    ranked=comp.sort_values(['Date','weight'],ascending=[True,False]).copy();ranked['rn']=ranked.groupby('Date').cumcount();two=ranked[ranked.rn<2].pivot(index='Date',columns='rn',values='weight').dropna();gap=two[0]-two[1];d=gap.idxmax();q=ranked[(ranked.Date==d)&(ranked.rn==0)].iloc[0];runner=ranked[(ranked.Date==d)&(ranked.rn==1)].iloc[0];r.append(rr('Largest weight gap between #1 and #2 components',q,_fmt_num(gap.loc[d]*100,2,suffix=' pp'),f'Runner-up: {label(runner.Ticker,d)}'))
    div=dict(zip(idx.Date,idx.Divisor));comp['divisor']=comp.Date.map(div);comp['contrib']=((comp.Close-comp.prev_close)/comp.divisor).where(comp.continuous & comp.stable_divisor & (comp.divisor>0));cs=comp.dropna(subset=['contrib'])
    if len(cs):
        q=cs.loc[cs.contrib.idxmax()];r.append(rr('Largest positive one-session contribution to the GLI',q,_fmt_num(q.contrib,2,True,' points')))
        q=cs.loc[cs.contrib.idxmin()];r.append(rr('Largest negative one-session contribution to the GLI',q,_fmt_num(q.contrib,2,True,' points')))
        yr=cs.assign(year=cs.Date.str[:4]).groupby(['Ticker','year']).contrib.sum().reset_index()
        q=yr.loc[yr.contrib.idxmax()];obj=type('Q',(),{'Ticker':q.Ticker,'Date':q.year});r.append(rr('Largest cumulative positive contribution in a calendar year',obj,_fmt_num(q.contrib,2,True,' points'),q.year))
        q=yr.loc[yr.contrib.idxmin()];obj=type('Q',(),{'Ticker':q.Ticker,'Date':q.year});r.append(rr('Largest cumulative negative contribution in a calendar year',obj,_fmt_num(q.contrib,2,True,' points'),q.year))
        tot=cs.groupby('Ticker').contrib.sum();t=tot.idxmax();r.append(_rec('Largest cumulative contribution over all GLI sessions',label(t),_fmt_num(tot.max(),2,True,' points'),ticker=t));rank_tickers.append(t)
        pos=cs[cs.contrib>0];neg=cs[cs.contrib<0]
        if len(pos):
            winners=pos.loc[pos.groupby('Date').contrib.idxmax()];vc=winners.Ticker.value_counts();t=vc.idxmax();r.append(_rec('Most sessions as the largest positive contributor',label(t),f'{int(vc.max()):,} sessions',ticker=t));rank_tickers.append(t)
        if len(neg):
            losers=neg.loc[neg.groupby('Date').contrib.idxmin()];vc=losers.Ticker.value_counts();t=vc.idxmax();r.append(_rec('Most sessions as the largest negative contributor',label(t),f'{int(vc.max()):,} sessions',ticker=t));rank_tickers.append(t)
    sections.append(('Contribution & Weight',r));return sections,rank_tickers


def _streak_sections(comp,idx,label):
    r=[];chg=idx.Close.diff()
    for feat,pred in [('Longest winning streak',lambda x:x>0),('Longest losing streak',lambda x:x<0),('Longest streak without a declining session',lambda x:x>=0),('Longest streak without an advancing session',lambda x:x<=0)]:
        best,end=_streak_best(chg.iloc[1:].tolist(),pred);end=end+1 if end is not None else None
        if end is not None:r.append(_rec(feat,'GLI',f'{best} sessions',idx.iloc[end].Date,f'{idx.iloc[end-best+1].Date} through {idx.iloc[end].Date}'))
    rec=idx.Close.eq(idx.Close.cummax());best,end=_streak_best(rec.tolist(),lambda x:bool(x));
    if end is not None:r.append(_rec('Longest streak of record-high closes','GLI',f'{best} sessions',idx.iloc[end].Date))
    absr=idx.Close.pct_change().abs()
    for pct in [.01,.02]:
        for without in [False,True]:
            pred=(lambda x,p=pct:x<p) if without else (lambda x,p=pct:x>=p);best,end=_streak_best(absr.iloc[1:].tolist(),pred);end=end+1 if end is not None else None
            if end is not None:r.append(_rec(f'Longest streak {"without a " if without else "of "}±{int(pct*100)}% {"move" if without else "sessions"}','GLI',f'{best} sessions',idx.iloc[end].Date))
    temp=idx.copy();temp['dt']=pd.to_datetime(temp.Date)
    for unit,per in [('weeks',temp.dt.dt.to_period('W-FRI')),('months',temp.dt.dt.to_period('M'))]:
        x=temp.assign(period=per.astype(str)).groupby('period').agg(first=('Close','first'),last=('Close','last'),date=('Date','last'));ret=x['last']/x['first']-1
        for positive,word in [(True,'positive'),(False,'negative')]:
            best,end=_streak_best(ret.tolist(),(lambda z:z>0) if positive else (lambda z:z<0))
            if end is not None:r.append(_rec(f'Longest streak of {word} {unit}','GLI',f'{best} {unit}',x.iloc[end].date,f'{x.index[end-best+1]} through {x.index[end]}'))
    sections=[('GLI Streaks',r)]
    cr=[]
    conds=[('Longest winning streak',comp.clean_ret>0),('Longest losing streak',comp.clean_ret<0),('Longest streak without a decline',comp.clean_ret>=0),('Longest streak without an advance',comp.clean_ret<=0)]
    idxret=idx.set_index('Date').Close.pct_change();rel=comp.clean_ret-comp.Date.map(idxret)
    conds += [('Longest streak outperforming the GLI',rel>0),('Longest streak underperforming the GLI',rel<0)]
    tenure_break=comp.Ticker.ne(comp.Ticker.shift())|comp.session_i.ne(comp.session_i.shift()+1);tenure_id=tenure_break.cumsum();cumhigh=comp.groupby(tenure_id).adj_price.cummax();conds.append(('Longest streak of new GLI-tenure highs',comp.adj_price>=cumhigh))
    for feat,cond in conds:
        z=_best_true_component_run(comp,cond)
        if z: n,t,start,end=z;cr.append(_rec(feat,label(t,end),f'{n} sessions',end,f'{start} through {end}',t))
    sections.append(('Component Streaks',cr));return sections


def _volume_sections(comp,idx,label):
    sections=[];r=[];v=idx[idx.Volume>0].copy()
    q=v.loc[v.Volume.idxmax()];r.append(_rec('Highest aggregate component volume in one session','GLI',_fmt_int(q.Volume),q.Date));q=v.loc[v.Volume.idxmin()];r.append(_rec('Lowest valid aggregate component volume in one session','GLI',_fmt_int(q.Volume),q.Date));v['dt']=pd.to_datetime(v.Date)
    for pname,per,minn in [('week',v.dt.dt.to_period('W-FRI'),4),('month',v.dt.dt.to_period('M'),15),('calendar year',v.dt.dt.to_period('Y'),100)]:
        x=v.assign(period=per.astype(str)).groupby('period').agg(volume=('Volume','sum'),sessions=('Volume','size'),date=('Date','last'))
        if pname!='calendar year':x=x[x.sessions>=minn]
        else:x=x[(x.sessions>=minn)&(~x.index.str.startswith('2005'))&(~x.index.str.startswith(idx.Date.iloc[-1][:4]))]
        if len(x):
            q=x.loc[x.volume.idxmax()];r.append(_rec(f'Highest-volume {pname}','GLI',_fmt_int(q.volume),q.date,str(q.name)));q=x.loc[x.volume.idxmin()];r.append(_rec(f'Lowest-volume {pname}','GLI',_fmt_int(q.volume),q.date,str(q.name)))
    for n in [5,20]:roll=v.Volume.rolling(n).sum();q=v.loc[roll.idxmax()];r.append(_rec(f'Highest {n}-session aggregate volume','GLI',_fmt_int(roll.max()),q.Date))
    v['vdiff']=v.Volume.diff();v['vmult']=v.Volume/v.Volume.shift(1);v['rvol20']=v.Volume/v.Volume.shift(1).rolling(20).mean()
    for feat,col,fn,fmt in [('Largest one-session increase in aggregate volume','vdiff','max',_fmt_int),('Largest one-session decrease in aggregate volume','vdiff','min',_fmt_int),('Largest volume multiple versus previous session','vmult','max',lambda z:f'{z:.2f}×'),('Highest aggregate relative volume versus prior 20-session average','rvol20','max',lambda z:f'{z:.2f}×'),('Lowest aggregate relative volume versus prior 20-session average','rvol20','min',lambda z:f'{z:.2f}×')]:
        z=v[col].dropna();q=v.loc[z.idxmax() if fn=='max' else z.idxmin()];r.append(_rec(feat,'GLI',fmt(q[col]),q.Date))
    delta=v.Close.diff();s=v[delta>0];q=s.loc[s.Volume.idxmax()];r.append(_rec('Highest-volume advancing session','GLI',_fmt_int(q.Volume),q.Date));s=v[delta<0];q=s.loc[s.Volume.idxmax()];r.append(_rec('Highest-volume declining session','GLI',_fmt_int(q.Volume),q.Date));s=v[v.Close.pct_change().abs()<=.001]
    if len(s):q=s.loc[s.Volume.idxmax()];r.append(_rec('Highest-volume session ending nearly unchanged','GLI',_fmt_int(q.Volume),q.Date,'Close within ±0.10%'))
    s=v[v.Close.eq(v.Close.cummax())];q=s.loc[s.Volume.idxmax()];r.append(_rec('Highest volume on an all-time-high close','GLI',_fmt_int(q.Volume),q.Date));sections.append(('GLI Aggregate Volume',r))
    r=[];avg=v.Volume.shift(1).rolling(20).mean()
    for feat,series,pred in [('Longest streak of above-average volume',v.Volume/avg,lambda x:x>1),('Longest streak of below-average volume',v.Volume/avg,lambda x:x<1),('Longest streak of rising aggregate volume',v.Volume.diff(),lambda x:x>0),('Longest streak of falling aggregate volume',v.Volume.diff(),lambda x:x<0)]:
        best,end=_streak_best(series.tolist(),lambda x:pd.notna(x) and pred(x));
        if end is not None:r.append(_rec(feat,'GLI',f'{best} sessions',v.iloc[end].Date,f'{v.iloc[end-best+1].Date} through {v.iloc[end].Date}'))
    sections.append(('GLI Volume Streaks',r))
    r=[];c=comp[comp.Volume>0].copy();q=c.loc[c.Volume.idxmax()];r.append(_rec('Highest single-session component volume',label(q.Ticker,q.Date),_fmt_int(q.Volume),q.Date,ticker=q.Ticker));q=c.loc[c.Volume.idxmin()];r.append(_rec('Lowest valid single-session component volume',label(q.Ticker,q.Date),_fmt_int(q.Volume),q.Date,ticker=q.Ticker));c['dt']=pd.to_datetime(c.Date)
    for pname,per in [('week',c.dt.dt.to_period('W-FRI')),('month',c.dt.dt.to_period('M')),('calendar year',c.dt.dt.to_period('Y'))]:
        x=c.assign(period=per.astype(str)).groupby(['Ticker','period']).agg(volume=('Volume','sum'),avg=('Volume','mean'),date=('Date','last'),sessions=('Volume','size')).reset_index();
        if pname=='week':x=x[x.sessions>=4]
        if pname=='month':x=x[x.sessions>=15]
        q=x.loc[x.volume.idxmax()];r.append(_rec(f'Highest-volume component {pname}',label(q.Ticker,q.date),_fmt_int(q.volume),q.date,q.period,q.Ticker))
        if pname=='calendar year':q=x.loc[x.avg.idxmax()];r.append(_rec('Highest average daily component volume in a calendar year',label(q.Ticker,q.date),_fmt_int(q.avg),q.date,q.period,q.Ticker))
    z=c.volume_mult_prev.dropna();q=c.loc[z.idxmax()];r.append(_rec('Largest one-session component volume multiple vs prior session',label(q.Ticker,q.Date),f'{q.volume_mult_prev:.2f}×',q.Date,ticker=q.Ticker));z=c.rvol20.dropna();q=c.loc[z.idxmax()];r.append(_rec('Highest component relative volume versus prior 20-session average',label(q.Ticker,q.Date),f'{q.rvol20:.2f}×',q.Date,ticker=q.Ticker));q=c.loc[z.idxmin()];r.append(_rec('Lowest component relative volume versus prior 20-session average',label(q.Ticker,q.Date),f'{q.rvol20:.2f}×',q.Date,ticker=q.Ticker))
    c['volshare']=c.Volume/c.groupby('Date').Volume.transform('sum');q=c.loc[c.volshare.idxmax()];r.append(_rec('Largest share of total GLI component volume in one session',label(q.Ticker,q.Date),_fmt_num(q.volshare*100,2,suffix='%'),q.Date,ticker=q.Ticker));top=c.loc[c.groupby('Date').Volume.idxmax(),['Date','Ticker']].sort_values('Date').reset_index(drop=True);vc=top.Ticker.value_counts();t=vc.idxmax();r.append(_rec('Most sessions as the GLI’s highest-volume component',label(t),f'{int(vc.max()):,} sessions',ticker=t));rid=top.Ticker.ne(top.Ticker.shift()).cumsum();runs=top.groupby(rid).agg(Ticker=('Ticker','first'),start=('Date','first'),end=('Date','last'),sessions=('Date','size'));q=runs.loc[runs.sessions.idxmax()];r.append(_rec('Longest streak as the GLI’s highest-volume component',label(q.Ticker,q.end),f'{int(q.sessions)} sessions',q.end,f'{q.start} through {q.end}',q.Ticker));top['year']=top.Date.str[:4];z=top.groupby(['Ticker','year']).size();k=z.idxmax();r.append(_rec('Most #1-volume finishes in a calendar year',label(k[0],k[1]),f'{int(z.max())} sessions',k[1],ticker=k[0]));sections.append(('Component Volume',r))
    r=[];s=comp[comp.rvol20.notna()&(comp.Open>0)]
    for mult in [2,3,5]:
        x=s[s.rvol20>=mult]
        if len(x):q=x.loc[x.oc_ret.idxmax()];r.append(_rec(f'Largest component gain on at least {mult}× normal volume',label(q.Ticker,q.Date),_fmt_num(q.oc_ret*100,2,True,'%'),q.Date,f'RVOL {q.rvol20:.2f}×',q.Ticker));q=x.loc[x.oc_ret.idxmin()];r.append(_rec(f'Largest component decline on at least {mult}× normal volume',label(q.Ticker,q.Date),_fmt_num(q.oc_ret*100,2,True,'%'),q.Date,f'RVOL {q.rvol20:.2f}×',q.Ticker))
    for feat,cond in [('Longest streak with both price and volume rising',(comp.clean_ret>0)&(comp.Volume>comp.prev_volume)),('Longest streak with both price and volume falling',(comp.clean_ret<0)&(comp.Volume<comp.prev_volume))]:
        z=_best_true_component_run(comp,cond)
        if z:n,t,start,end=z;r.append(_rec(feat,label(t,end),f'{n} sessions',end,f'{start} through {end}',t))
    sections.append(('Price + Volume Feats',r));return sections

def _breadth_and_rare(comp,idx,label):
    cc=comp.dropna(subset=['clean_ret']).copy()
    cc['adv']=(cc.clean_ret>0).astype(int);cc['dec']=(cc.clean_ret<0).astype(int);cc['flat']=(cc.clean_ret==0).astype(int)
    b=cc.groupby('Date').agg(adv=('adv','sum'),dec=('dec','sum'),flat=('flat','sum'),n=('Ticker','size')).reset_index();b['advp']=b.adv/b.n;b['decp']=b.dec/b.n
    idxret=idx.set_index('Date').Close.pct_change();b['idxret']=b.Date.map(idxret)
    r=[]
    for feat,col in [('Most advancing components in one session','adv'),('Most declining components in one session','dec')]:q=b.loc[b[col].idxmax()];r.append(_rec(feat,'GLI',f'{int(q[col])} components',q.Date))
    for feat,col in [('Highest percentage of components advancing','advp'),('Highest percentage of components declining','decp')]:q=b.loc[b[col].idxmax()];r.append(_rec(feat,'GLI',_fmt_num(q[col]*100,2,suffix='%'),q.Date))
    s=b[b.advp==1]
    if len(s):q=s.iloc[0];r.append(_rec('Most unanimous advancing session','GLI',f'{int(q.adv)} of {int(q.n)} advanced',q.Date))
    s=b[b.decp==1]
    if len(s):q=s.iloc[0];r.append(_rec('Most unanimous declining session','GLI',f'{int(q.dec)} of {int(q.n)} declined',q.Date))
    s=b[(b.idxret>0)&(b.dec>b.adv)]
    if len(s):q=s.loc[s.idxret.idxmax()];r.append(_rec('Largest GLI gain with a majority of components declining','GLI',_fmt_num(q.idxret*100,2,True,'%'),q.Date,f'{int(q.adv)} up / {int(q.dec)} down'))
    s=b[(b.idxret<0)&(b.adv>b.dec)]
    if len(s):q=s.loc[s.idxret.idxmin()];r.append(_rec('Largest GLI decline with a majority of components advancing','GLI',_fmt_num(q.idxret*100,2,True,'%'),q.Date,f'{int(q.adv)} up / {int(q.dec)} down'))
    y=b.assign(year=b.Date.str[:4]).groupby('year').agg(adv_sessions=('idxret',lambda s:int((s>0).sum())),decl_sessions=('idxret',lambda s:int((s<0).sum())),sessions=('idxret','size'));y['adv_pct']=y.adv_sessions/y.sessions
    q=y.loc[y.adv_sessions.idxmax()];r.append(_rec('Most advancing GLI sessions in a calendar year','GLI',f'{int(q.adv_sessions)} sessions',str(q.name)));q=y.loc[y.decl_sessions.idxmax()];r.append(_rec('Most declining GLI sessions in a calendar year','GLI',f'{int(q.decl_sessions)} sessions',str(q.name)));q=y.loc[y.adv_pct.idxmax()];r.append(_rec('Highest percentage of advancing sessions in a calendar year','GLI',_fmt_num(q.adv_pct*100,2,suffix='%'),str(q.name)));q=y.loc[y.adv_pct.idxmin()];r.append(_rec('Lowest percentage of advancing sessions in a calendar year','GLI',_fmt_num(q.adv_pct*100,2,suffix='%'),str(q.name)))
    breadth=[('Breadth',r)]

    r=[]
    s=b[(b.idxret>0)&(b.dec>b.adv)]
    if len(s):q=s.loc[s.idxret.idxmax()];r.append(_rec('GLI advances despite a majority of components declining','GLI',_fmt_num(q.idxret*100,2,True,'%'),q.Date,f'{int(q.adv)} up / {int(q.dec)} down'))
    s=b[(b.idxret<0)&(b.adv>b.dec)]
    if len(s):q=s.loc[s.idxret.idxmin()];r.append(_rec('GLI declines despite a majority of components advancing','GLI',_fmt_num(q.idxret*100,2,True,'%'),q.Date,f'{int(q.adv)} up / {int(q.dec)} down'))
    div=idx.set_index('Date').Divisor;cc['div']=cc.Date.map(div);cc['contrib']=((cc.Close-cc.prev_close)/cc['div']).where(cc.stable_divisor)
    # daily leader table; contribution leaders exist only on non-reset sessions
    gi=cc.groupby('Date')
    gain_idx=gi.oc_ret.idxmax();decl_idx=gi.oc_ret.idxmin();vol_idx=gi.Volume.idxmax()
    contrib_cc=cc.dropna(subset=['contrib'])
    cgi=contrib_cc.groupby('Date')
    pos_idx=cgi.contrib.idxmax();neg_idx=cgi.contrib.idxmin()
    leader=pd.DataFrame({
        'gain':cc.loc[gain_idx].set_index('Date').Ticker,
        'decl':cc.loc[decl_idx].set_index('Date').Ticker,
        'pos':contrib_cc.loc[pos_idx].set_index('Date').Ticker,
        'neg':contrib_cc.loc[neg_idx].set_index('Date').Ticker,
        'vol':cc.loc[vol_idx].set_index('Date').Ticker,
    })
    for feat,a,z in [('Biggest gainer and biggest positive contributor on the same session','gain','pos'),('Biggest decliner and biggest negative contributor on the same session','decl','neg'),('Highest-volume component and biggest gainer on the same session','vol','gain'),('Highest-volume component and biggest decliner on the same session','vol','decl')]:
        m=leader[a].eq(leader[z]);n=int(m.sum())
        if n:
            d=leader.index[m][-1];t=leader.loc[d,a];r.append(_rec(feat,label(t,d),f'{n} sessions',d,'Most recent qualifying session',t))
    # >50% of daily move
    move=idx.set_index('Date').Close.diff();posrows=contrib_cc.loc[pos_idx].set_index('Date');negrows=contrib_cc.loc[neg_idx].set_index('Date')
    ps=(posrows.contrib/move).dropna();ps=ps[(move.reindex(ps.index)>0)&(ps>0.5)]
    if len(ps):d=ps.idxmax();t=posrows.loc[d].Ticker;r.append(_rec('One component accounts for more than 50% of the GLI’s daily advance',label(t,d),_fmt_num(ps.loc[d]*100,2,suffix='%'),d,ticker=t))
    ns=(negrows.contrib/move).dropna();ns=ns[(move.reindex(ns.index)<0)&(ns>0.5)]
    if len(ns):d=ns.idxmax();t=negrows.loc[d].Ticker;r.append(_rec('One component accounts for more than 50% of the GLI’s daily decline',label(t,d),_fmt_num(ns.loc[d]*100,2,suffix='%'),d,ticker=t))
    # perfect periods (vectorized counts; avoids Python lambdas across thousands of groups)
    c=cc.copy();c['dt']=pd.to_datetime(c.Date);c['week']=c.dt.dt.to_period('W-FRI').astype(str);c['month']=c.dt.dt.to_period('M').astype(str);c['posflag']=(c.clean_ret>0).astype(int);c['negflag']=(c.clean_ret<0).astype(int);c['absmove']=c.clean_ret.abs()
    period_tables={}
    for period in ['week','month']:
        period_tables[period]=c.groupby(['Ticker',period],sort=False).agg(sessions=('clean_ret','size'),pos_count=('posflag','sum'),neg_count=('negflag','sum'),magnitude=('absmove','sum'),date=('Date','max')).reset_index()
    for period,desc,pos,min_sessions in [('week','Perfect five-session week',True,5),('week','Perfect negative five-session week',False,5),('month','Perfect calendar month',True,15),('month','Perfect negative calendar month',False,15)]:
        x=period_tables[period].copy();x=x[(x.sessions==5) if period=='week' else (x.sessions>=min_sessions)];x=x[(x.pos_count==x.sessions) if pos else (x.neg_count==x.sessions)]
        if len(x):q=x.loc[x.magnitude.idxmax()];r.append(_rec(desc,label(q.Ticker,q.date),f'{int(q.sessions)} straight {"advancing" if pos else "declining"} sessions',q.date,getattr(q,period),q.Ticker))
    # consecutive double-digit close-to-close moves
    prev_ret=cc.groupby('Ticker').clean_ret.shift(1);prev_si=cc.groupby('Ticker').session_i.shift(1);valid=cc.continuous & (cc.session_i-prev_si==1) & prev_ret.notna()
    cases=[('Back-to-back double-digit percentage gains',(prev_ret>=.10)&(cc.clean_ret>=.10)),('Back-to-back double-digit percentage declines',(prev_ret<=-.10)&(cc.clean_ret<=-.10)),('Back-to-back opposite-direction double-digit moves',(prev_ret.abs()>=.10)&(cc.clean_ret.abs()>=.10)&(prev_ret*cc.clean_ret<0))]
    for desc,cond in cases:
        m=valid&cond
        if m.any():
            score=(prev_ret.abs()+cc.clean_ret.abs()).where(m);qi=score.idxmax();q=cc.loc[qi];r.append(_rec(desc,label(q.Ticker,q.Date),f'{prev_ret.loc[qi]*100:+.1f}% then {q.clean_ret*100:+.1f}%',q.Date,ticker=q.Ticker))
    return breadth,[('Rare Feats',r)]

def _get_section(categories, cat_id, title):
    cat=next(c for c in categories if c['id']==cat_id)
    sec=next((s for s in cat['sections'] if s['title']==title),None)
    if sec is None:
        sec={'title':title,'records':[]};cat['sections'].append(sec)
    return sec['records']


def _replace_record(records, feat, rec):
    for i,r in enumerate(records):
        if r.get('feat')==feat:
            records[i]=rec;return
    records.append(rec)


def _tenure_frame(comp: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    wanted=['Date','Ticker','Open','High','Low','Close','adj_price','clean_ret','Volume','session_i','cc_ret','oc_ret','rvol20','prev_volume','continuous','stable_divisor','weight','contrib']
    cols=[x for x in wanted if x in comp.columns]
    c=comp[cols].sort_values(['Ticker','session_i']).copy()
    br=c.Ticker.ne(c.Ticker.shift())|c.session_i.ne(c.session_i.shift()+1)
    c['tenure_id']=br.cumsum()
    t=c.groupby('tenure_id',sort=False).agg(
        Ticker=('Ticker','first'),start=('Date','first'),end=('Date','last'),
        start_i=('session_i','first'),end_i=('session_i','last'),sessions=('Date','size'),
        entry_close=('Close','first'),exit_close=('Close','last'),
        total_volume=('Volume','sum'),avg_volume=('Volume','mean')
    ).reset_index()
    return c,t


def _normalized_membership_states(root: Path, idx: pd.DataFrame) -> list[tuple[str, dict[str, str]]]:
    """Return normalized Component History membership states keyed by identity.

    This deliberately reuses the Component History builder's accepted
    normalization layer.  That layer distinguishes true membership turnover
    from name/ticker continuity and from same-ticker security replacement.
    Keeping Feats on the same chronology prevents quote gaps or the broader
    historical name table from inventing departures/returns.
    """
    if idx.empty:
        return []
    try:
        import importlib
        site_builder = importlib.import_module('gli_site_build')
    except Exception as exc:
        raise SystemExit(f'Unable to import GLI site builder for membership chronology: {exc}') from exc

    required = [
        'extend_component_history',
        'historical_component_identity_metadata',
        '_ticker_identity_meta_at',
        '_normalize_component_identity_label',
    ]
    missing = [name for name in required if not hasattr(site_builder, name)]
    if missing:
        raise SystemExit(f'GLI site builder lacks normalized membership helpers: {missing}')

    cp = root / 'constituents_great_lakes.csv'
    if not cp.exists():
        raise SystemExit(f'Missing live component membership source: {cp}')
    with cp.open(newline='', encoding='utf-8-sig') as stream:
        constituents = list(csv.DictReader(stream))

    # Component History only needs the GLI session dates when extending 2026.
    history_rows = [{'Date': str(day)} for day in idx.Date.astype(str)]
    payload = site_builder.extend_component_history(history_rows, constituents)
    snapshots = sorted(payload.get('snapshots', []), key=lambda s: str(s.get('date', '')))
    ticker_ranges, label_identity = site_builder.historical_component_identity_metadata()

    states: list[tuple[str, dict[str, str]]] = []
    for snapshot in snapshots:
        day = str(snapshot.get('date', ''))
        if not day:
            continue
        mode = snapshot.get('label_mode')
        symbol_map = snapshot.get('component_symbols') or {}
        current: dict[str, str] = {}
        for raw in snapshot.get('components', []):
            label_text = str(raw).strip()
            if not label_text or label_text.upper().startswith('NAME CHANGE:'):
                continue
            if mode == 'ticker':
                symbol = label_text.upper()
            else:
                symbol = str(symbol_map.get(label_text, '')).strip().upper()

            if symbol:
                meta = site_builder._ticker_identity_meta_at(
                    symbol, day, ticker_ranges, allow_nearest=True
                )
                identity = str(meta.get('identity') or symbol).strip().upper()
            else:
                key = site_builder._normalize_component_identity_label(label_text)
                identity = str(label_identity.get(key, f'LABEL:{key}')).strip().upper()

            if identity:
                # Match Component History's own identity enumeration behavior:
                # one represented security/ticker per accepted identity at a
                # checkpoint.  Continuity events can therefore change ticker
                # without ending the identity's GLI tenure.
                current[identity] = symbol or identity
        states.append((day, current))
    return states


def _membership_tenure_frames(
    root: Path,
    site_data: Path,
    idx: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build identity-level and ticker-level GLI tenure frames from Component History."""
    identity_cols = [
        'tenure_id','Identity','Ticker','start_ticker','end_ticker',
        'start','end','start_i','end_i','sessions'
    ]
    ticker_cols = ['tenure_id','Ticker','start','end','start_i','end_i','sessions']
    if idx.empty:
        return pd.DataFrame(columns=identity_cols), pd.DataFrame(columns=ticker_cols)

    states = _normalized_membership_states(root, idx)
    if not states:
        raise SystemExit('Normalized Component History produced no membership states.')

    dates = idx.Date.astype(str).tolist()
    states = [(d, m) for d, m in states if d <= dates[-1]]
    if not states:
        raise SystemExit('Normalized Component History has no checkpoint on/before the GLI history.')

    by_identity: dict[str, list[tuple[int, str]]] = defaultdict(list)
    by_ticker: dict[str, list[int]] = defaultdict(list)
    current: dict[str, str] = {}
    j = 0
    for i, day in enumerate(dates):
        while j < len(states) and states[j][0] <= day:
            current = states[j][1]
            j += 1
        for identity, ticker in current.items():
            # Accepted 2005 base reconstruction supersedes the legacy Component
            # History workbook's "Original 25" label for CF Industries.  CF was
            # a deferred addition effective 2005-08-11, not an opening
            # constituent on the corrected 2005-08-01 GLI base date.
            if (identity == 'CF' or ticker == 'CF') and day < '2005-08-11':
                continue
            by_identity[identity].append((i, ticker))
            if ticker:
                by_ticker[ticker].append(i)

    identity_rows = []
    tenure_id = 0
    for identity, obs in sorted(by_identity.items()):
        if not obs:
            continue
        split_at = [k for k in range(1, len(obs)) if obs[k][0] != obs[k-1][0] + 1]
        starts = [0] + split_at
        ends = split_at + [len(obs)]
        for a0, b0 in zip(starts, ends):
            block = obs[a0:b0]
            if not block:
                continue
            tenure_id += 1
            a = int(block[0][0]); b = int(block[-1][0])
            start_ticker = str(block[0][1]).upper()
            end_ticker = str(block[-1][1]).upper()
            identity_rows.append({
                'tenure_id': tenure_id,
                'Identity': identity,
                # Backward-compatible display ticker: use the ticker at the end
                # of the accepted identity tenure.
                'Ticker': end_ticker,
                'start_ticker': start_ticker,
                'end_ticker': end_ticker,
                'start': dates[a], 'end': dates[b],
                'start_i': a, 'end_i': b, 'sessions': b-a+1,
            })

    ticker_rows = []
    ticker_tid = 0
    for ticker, members in sorted(by_ticker.items()):
        arr = np.asarray(members, dtype=int)
        if not len(arr):
            continue
        breaks = np.flatnonzero(np.diff(arr) > 1) + 1
        for block in np.split(arr, breaks):
            if not len(block):
                continue
            ticker_tid += 1
            a = int(block[0]); b = int(block[-1])
            ticker_rows.append({
                'tenure_id': ticker_tid, 'Ticker': ticker,
                'start': dates[a], 'end': dates[b],
                'start_i': a, 'end_i': b, 'sessions': int(len(block)),
            })

    return (
        pd.DataFrame(identity_rows, columns=identity_cols),
        pd.DataFrame(ticker_rows, columns=ticker_cols),
    )


def _membership_tenure_frame(root: Path, site_data: Path, idx: pd.DataFrame) -> pd.DataFrame:
    """Return true component/security-identity GLI tenures."""
    return _membership_tenure_frames(root, site_data, idx)[0]


def _membership_ticker_tenure_frame(root: Path, site_data: Path, idx: pd.DataFrame) -> pd.DataFrame:
    """Return continuous membership spans under an individual ticker symbol."""
    return _membership_tenure_frames(root, site_data, idx)[1]

def _membership_boundary_rows(
    comp: pd.DataFrame,
    tenures: pd.DataFrame,
    boundary: str,
) -> pd.DataFrame:
    """Return component rows that exactly coincide with membership boundaries."""
    if comp.empty or tenures.empty:
        return comp.iloc[0:0].copy()
    if boundary not in {'start','end'}:
        raise ValueError("boundary must be 'start' or 'end'")
    key=f'{boundary}_i'
    ticker_col = f'{boundary}_ticker' if f'{boundary}_ticker' in tenures.columns else 'Ticker'
    right=tenures[['tenure_id',ticker_col,key]].rename(
        columns={'tenure_id':'membership_tenure_id',ticker_col:'Ticker',key:'session_i'}
    )
    return comp.merge(right,on=['Ticker','session_i'],how='inner')


def _active_roster_tickers(root: Path, day: str) -> set[str]:
    p=root/'constituents_great_lakes.csv'
    if not p.exists():
        return set()
    active=set()
    with p.open(newline='',encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            ticker=(row.get('Ticker') or '').strip().upper()
            if not ticker:
                continue
            start=(row.get('StartDate') or '0001-01-01').strip() or '0001-01-01'
            end=(row.get('EndDate') or '9999-12-31').strip() or '9999-12-31'
            if start<=day<=end:
                active.add(ticker)
    return active


def _current_company_names(root: Path) -> dict[str,str]:
    p=root/'company_names.csv';names={}
    if not p.exists():
        return names
    with p.open(newline='',encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            ticker=(row.get('Ticker') or row.get('Symbol') or '').strip().upper()
            company=(row.get('Company') or row.get('Name') or '').strip()
            if ticker and company:
                names[ticker]=company
    return names


def _identity_longest_records(
    root: Path,
    site_data: Path,
    idx: pd.DataFrame,
    label,
    membership_tenures: pd.DataFrame,
) -> list[dict[str,Any]]:
    """Compute identity/ticker/name longevity through the current GLI session.

    ``historical_company_names.csv`` is an archive through 2025, so a row that
    ends at that archive boundary is extended only when the same ticker remains
    a live member and the current company name still matches.  Ticker tenure is
    taken directly from date-effective membership and therefore does not depend
    on component-price coverage.
    """
    p=site_data/'historical_company_names.csv'
    if not p.exists() or idx.empty:
        return []
    last=str(idx.Date.iloc[-1])
    h=pd.read_csv(p,dtype=str).fillna('')
    if h.empty:
        return []
    active=_active_roster_tickers(root,last)
    current_names=_current_company_names(root)

    def norm_name(v: Any) -> str:
        # Company caches sometimes omit punctuation or a trailing legal suffix
        # (for example, "American Electric Power Company" vs
        # "American Electric Power Company, Inc.").  Ignore those cosmetic
        # differences without treating a substantive renamed company as equal.
        import re
        words=re.findall(r'[a-z0-9]+',str(v).casefold().replace('&',' and '))
        suffixes={'inc','incorporated','corp','corporation','co','company','plc','llc','lp','ltd','limited'}
        while words and words[-1] in suffixes:
            words.pop()
        return ' '.join(words)

    h['s']=pd.to_datetime(h['StartDate'],errors='coerce')
    raw_end=h['EndDate'].replace('',last)
    h['name_effective_end']=raw_end
    h['identity_effective_end']=raw_end
    # Corporate identity continues through a 2025 archive cutoff when the same
    # ticker is still an eligible member.  Company-name tenure is more strict:
    # extend it only when the current cached name still matches (allowing only
    # cosmetic punctuation/legal-suffix differences).
    for i,row in h.iterrows():
        ticker=str(row.get('Ticker','')).strip().upper()
        end=str(row.get('EndDate','')).strip()
        company=str(row.get('Company','')).strip()
        current=current_names.get(ticker,'')
        archival_cutoff=(not end or end=='2025-12-31')
        if ticker in active and archival_cutoff:
            h.at[i,'identity_effective_end']=last
            if current and norm_name(current)==norm_name(company):
                h.at[i,'name_effective_end']=last
    h['e']=pd.to_datetime(h['name_effective_end'].replace('',last),errors='coerce').fillna(pd.Timestamp(last))
    h=h[h.s.notna()].copy()
    h['days']=(h.e-h.s).dt.days

    records=[]
    # Corporate lineage: the normalized Component History identity tenure is
    # authoritative.  True absences break continuity; ticker/name continuity does not.
    if not membership_tenures.empty:
        mt=membership_tenures.copy()
        mt['days']=(pd.to_datetime(mt.end)-pd.to_datetime(mt.start)).dt.days
        best=int(mt.days.max());ties=mt[mt.days==best].sort_values(['Identity','end_ticker'])
        if len(ties)==1:
            q=ties.iloc[0];holder=str(q.Identity);detail=f'Since {q.start}';date=q.end
        else:
            holder='Multiple';detail=f"Since {ties.start.iloc[0]}; " + ', '.join(ties.end_ticker.tolist());date=ties.end.iloc[0]
        records.append(_rec('Longest continuously represented corporate lineage',holder,f'{best:,} calendar days',date,detail))

    # One ticker: use ticker-specific roster spans, so a ticker change ends this
    # record even when the underlying corporate/security identity continues.
    ticker_tenures=_membership_ticker_tenure_frame(root,site_data,idx)
    if not ticker_tenures.empty:
        tt=ticker_tenures.copy();tt['days']=(pd.to_datetime(tt.end)-pd.to_datetime(tt.start)).dt.days
        best=int(tt.days.max());ties=tt[tt.days==best].sort_values('Ticker')
        if len(ties)==1:
            q=ties.iloc[0];holder=label(q.Ticker,q.end);detail=f'Since {q.start}';ticker=q.Ticker;date=q.end
        else:
            holder='Multiple';detail=f"Since {ties.start.iloc[0]}; " + ', '.join(ties.Ticker.tolist());ticker='';date=ties.end.iloc[0]
        records.append(_rec('Longest tenure under one ticker',holder,f'{best:,} calendar days',date,detail,ticker))

    # One company name: extend a historical cutoff row only when the current name matches.
    if not h.empty:
        best=int(h.days.max());ties=h[h.days==best].sort_values(['Company','Ticker'])
        if len(ties)==1:
            q=ties.iloc[0];holder=q.Company;detail=f'Since {q.StartDate}';ticker=q.Ticker;date=str(q.e.date())
        else:
            holder='Multiple';detail=f"Since {ties.StartDate.iloc[0]}; " + ', '.join(ties.Company.tolist());ticker='';date=str(ties.e.iloc[0].date())
        records.append(_rec('Longest tenure under one company name',holder,f'{best:,} calendar days',date,detail,ticker))
    return records


def _augment_categories(categories, comp, idx, label, root, site_data):
    c,tenures=_tenure_frame(comp)
    membership_tenures=_membership_tenure_frame(root,site_data,idx)
    idx_last=idx.Date.iloc[-1]

    # Component price feats tied to a continuous tenure.
    pr=_get_section(categories,'component','Price Performance')
    best_entry=None; best_rally=None; worst_dd=None; fastest_double=None; first_double=None; max_doublings=None
    recoveries={.20:None,.30:None,.50:None}
    for tid,g in c.groupby('tenure_id',sort=False):
        g=g.sort_values('session_i').reset_index(drop=True)
        t=g.Ticker.iloc[0]; entry=float(g.adj_price.iloc[0]); arr=g.adj_price.to_numpy(float)
        if entry>0:
            ratios=arr/entry; j=int(np.nanargmax(ratios)); cand=(ratios[j]-1,t,g.Date.iloc[j],g.Date.iloc[0],ratios[j],tid)
            if best_entry is None or cand[0]>best_entry[0]: best_entry=cand
            hit=np.flatnonzero(ratios>=2)
            if len(hit):
                j=int(hit[0]); elapsed=j; cand2=(elapsed,t,g.Date.iloc[j],g.Date.iloc[0],tid)
                if fastest_double is None or cand2[0]<fastest_double[0] or (cand2[0]==fastest_double[0] and cand2[2]<fastest_double[2]): fastest_double=cand2
                cand3=(g.Date.iloc[j],t,g.Date.iloc[0],elapsed,tid)
                if first_double is None or cand3[0]<first_double[0]: first_double=cand3
            levels=int(math.floor(math.log2(float(np.nanmax(ratios))))) if np.nanmax(ratios)>=2 else 0
            cand4=(levels,t,g.Date.iloc[int(np.nanargmax(ratios))],g.Date.iloc[0],float(np.nanmax(ratios)),tid)
            if max_doublings is None or cand4[0]>max_doublings[0]: max_doublings=cand4
        runmin=np.minimum.accumulate(arr); rally=arr/runmin-1;j=int(np.nanargmax(rally)); low_i=int(np.argmin(arr[:j+1]));cand=(rally[j],t,g.Date.iloc[j],g.Date.iloc[low_i],tid)
        if best_rally is None or cand[0]>best_rally[0]:best_rally=cand
        runmax=np.maximum.accumulate(arr);dd=arr/runmax-1;j=int(np.nanargmin(dd));peak_i=int(np.argmax(arr[:j+1]));cand=(dd[j],t,g.Date.iloc[j],g.Date.iloc[peak_i],tid)
        if worst_dd is None or cand[0]<worst_dd[0]:worst_dd=cand
        # Recovery from first threshold breach to reclaiming the governing peak.
        for th in recoveries:
            peak=arr[0]; peak_i=0; trigger=None; target=None
            for i in range(1,len(arr)):
                if trigger is None:
                    if arr[i]>=peak:
                        peak=arr[i];peak_i=i
                    elif arr[i]/peak-1<=-th:
                        trigger=i;target=peak;target_i=peak_i
                else:
                    if arr[i]>=target:
                        elapsed=i-trigger;cand=(elapsed,t,g.Date.iloc[i],g.Date.iloc[trigger],g.Date.iloc[target_i],tid)
                        if recoveries[th] is None or elapsed<recoveries[th][0]:recoveries[th]=cand
                        peak=arr[i];peak_i=i;trigger=None;target=None
    if best_entry:
        gain,t,d,start,mult,_=best_entry
        pr += [_rec('Largest gain from GLI entry price',label(t,d),_fmt_num(gain*100,2,True,'%'),d,f'Entry {start}',t),
               _rec('Largest multiple of GLI entry price',label(t,d),f'{mult:.2f}×',d,f'Entry {start}',t)]
    if fastest_double:
        n,t,d,start,_=fastest_double;pr.append(_rec('Fastest doubling from GLI entry price',label(t,d),f'{n} sessions',d,f'Entry {start}',t))
    if best_rally:
        v,t,d,low,_=best_rally;pr.append(_rec('Largest rally from a GLI-tenure low',label(t,d),_fmt_num(v*100,2,True,'%'),d,f'Tenure low {low}',t))
    if worst_dd:
        v,t,d,peak,_=worst_dd;pr.append(_rec('Largest drawdown while in the index',label(t,d),_fmt_num(v*100,2,True,'%'),d,f'Peak {peak}',t))
    for th,z in recoveries.items():
        if z:
            n,t,d,trigger,peak,_=z;pr.append(_rec(f'Fastest recovery from a {int(th*100)}% component drawdown',label(t,d),f'{n} sessions',d,f'Threshold hit {trigger}; recovered prior peak from {peak}',t))

    # Contribution extras.
    wr=_get_section(categories,'component','Contribution & Weight')
    if 'weight' not in comp:
        comp['weight']=comp.Close/comp.groupby('Date').Close.transform('sum')
    if 'contrib' not in comp:
        div=idx.set_index('Date').Divisor;comp['divisor']=comp.Date.map(div);comp['contrib']=((comp.Close-comp.prev_close)/comp.divisor).where(comp.continuous & comp.stable_divisor & (comp.divisor>0))
    cs=comp.loc[comp.contrib.notna(),['Date','Ticker','contrib']].copy();pos=cs[cs.contrib>0];neg=cs[cs.contrib<0]
    if len(pos):
        winners=pos.loc[pos.groupby('Date').contrib.idxmax()].copy();winners['year']=winners.Date.str[:4];z=winners.groupby(['Ticker','year']).size();k=z.idxmax();wr.append(_rec('Most #1 positive-contributor finishes in a calendar year',label(k[0],k[1]),f'{int(z.max())} sessions',k[1],ticker=k[0]))
    if len(neg):
        losers=neg.loc[neg.groupby('Date').contrib.idxmin()].copy();losers['year']=losers.Date.str[:4];z=losers.groupby(['Ticker','year']).size();k=z.idxmax();wr.append(_rec('Most #1 negative-contributor finishes in a calendar year',label(k[0],k[1]),f'{int(z.max())} sessions',k[1],ticker=k[0]))
    idx_move=idx.set_index('Date').Close.diff(); idx_pct=idx.set_index('Date').Close.pct_change()
    shares=[]
    for d,g in cs.groupby('Date'):
        mv=idx_move.get(d)
        pc=idx_pct.get(d)
        if pd.isna(mv) or pd.isna(pc) or abs(pc)<.0025 or mv==0: continue
        q=g.loc[g.contrib.abs().idxmax()]; shares.append((abs(q.contrib/mv),d,q.Ticker,q.contrib,mv))
    if shares:
        z=max(shares,key=lambda x:x[0]);wr.append(_rec('Highest percentage of a single day’s GLI move attributable to one component',label(z[2],z[1]),_fmt_num(z[0]*100,2,suffix='%'),z[1],f'{z[3]:+.2f} contribution points; sessions with |GLI move| ≥ 0.25%',z[2]))
    # annual share, positive GLI years only
    cs['year']=cs.Date.str[:4];cy=cs.groupby(['Ticker','year']).contrib.sum().reset_index();iy=idx.assign(year=idx.Date.str[:4]).groupby('year').agg(first=('Close','first'),last=('Close','last'));iy['gain']=iy['last']-iy['first']
    vals=[]
    for _,q in cy.iterrows():
        gain=iy.loc[q.year,'gain'] if q.year in iy.index else np.nan
        if pd.notna(gain) and gain>0 and q.contrib>0: vals.append((q.contrib/gain,q.Ticker,q.year,q.contrib,gain))
    if vals:
        z=max(vals,key=lambda x:x[0]);wr.append(_rec('Highest percentage of a calendar year’s GLI gain attributable to one component',label(z[1],z[2]),_fmt_num(z[0]*100,2,suffix='%'),z[2],f'{z[3]:+.2f} component points vs {z[4]:+.2f} GLI points',z[1]))

    # Additional streaks.
    sr=_get_section(categories,'streaks','GLI Streaks')
    ma=idx.Close.rolling(20,min_periods=20).mean()
    for feat,cond in [('Longest streak above the 20-session moving average',idx.Close>ma),('Longest streak below the 20-session moving average',idx.Close<ma)]:
        best,end=_streak_best(cond.tolist(),lambda x:bool(x));
        if end is not None and best:sr.append(_rec(feat,'GLI',f'{best} sessions',idx.iloc[end].Date,f'{idx.iloc[end-best+1].Date} through {idx.iloc[end].Date}'))
    cr=_get_section(categories,'streaks','Component Streaks')
    cp=comp[['Date','Ticker','Close']].copy();cp['dt']=pd.to_datetime(cp.Date)
    for unit,pers in [('weeks',cp.dt.dt.to_period('W-FRI')),('months',cp.dt.dt.to_period('M'))]:
        key='period';x=cp.assign(period=pers).groupby(['Ticker','period']).agg(first=('Close','first'),last=('Close','last'),date=('Date','last')).reset_index();x['ret']=x['last']/x['first']-1;x['ord']=x['period'].apply(lambda p:p.ordinal);x=x.sort_values(['Ticker','ord']).reset_index(drop=True);x['session_i']=x['ord'];x['Date']=x['date']
        for positive,word in [(True,'positive'),(False,'negative')]:
            z=_best_true_component_run(x,(x.ret>0) if positive else (x.ret<0))
            if z:n,t,start,end=z;cr.append(_rec(f'Longest streak of {word} {unit}',label(t,end),f'{n} {unit}',end,f'{start} through {end}',t))
    if len(pos):
        w=pos.loc[pos.groupby('Date').contrib.idxmax(),['Date','Ticker']].sort_values('Date').copy();w['session_i']=w.Date.map(dict(zip(idx.Date,idx.index)));z=_best_true_component_run(w,pd.Series(True,index=w.index));
        if z:n,t,start,end=z;cr.append(_rec('Longest streak as the day’s biggest positive contributor',label(t,end),f'{n} sessions',end,f'{start} through {end}',t))
    if len(neg):
        w=neg.loc[neg.groupby('Date').contrib.idxmin(),['Date','Ticker']].sort_values('Date').copy();w['session_i']=w.Date.map(dict(zip(idx.Date,idx.index)));z=_best_true_component_run(w,pd.Series(True,index=w.index));
        if z:n,t,start,end=z;cr.append(_rec('Longest streak as the day’s biggest negative contributor',label(t,end),f'{n} sessions',end,f'{start} through {end}',t))
    top=comp.loc[comp.groupby('Date').weight.idxmax(),['Date','Ticker']].sort_values('Date').copy();top['session_i']=top.Date.map(dict(zip(idx.Date,idx.index)));z=_best_true_component_run(top,pd.Series(True,index=top.index));
    if z:n,t,start,end=z;cr.append(_rec('Longest streak as the GLI’s largest-weighted component',label(t,end),f'{n} sessions',end,f'{start} through {end}',t))

    # Volume additions.
    vr=_get_section(categories,'volume','GLI Aggregate Volume')
    prior252=idx.Close.shift(1).rolling(252,min_periods=200).max();new52=idx.Close>prior252;s=idx[new52 & (idx.Volume>0)]
    if len(s):q=s.loc[s.Volume.idxmax()];vr.append(_rec('Highest volume on a new 52-week high','GLI',_fmt_int(q.Volume),q.Date))
    cv=_get_section(categories,'volume','Component Volume')
    tq=tenures.loc[tenures.total_volume.idxmax()];cv.append(_rec('Most shares traded during one continuous GLI tenure',label(tq.Ticker,tq.end),_fmt_int(tq.total_volume),tq.end,f'{tq.start} through {tq.end}',tq.Ticker))
    totals=comp.groupby('Ticker').Volume.sum();t=totals.idxmax();cv.append(_rec('Most shares traded across all GLI tenures',label(t),_fmt_int(totals.max()),ticker=t))
    tq=tenures.loc[tenures.avg_volume.idxmax()];cv.append(_rec('Highest average daily volume during one continuous GLI tenure',label(tq.Ticker,tq.end),_fmt_int(tq.avg_volume),tq.end,f'{tq.start} through {tq.end}',tq.Ticker))
    comp['volume_increase']=(comp.Volume-comp.prev_volume).where(comp.continuous);z=comp.volume_increase.dropna();q=comp.loc[z.idxmax()];cv.append(_rec('Largest one-session increase in component volume',label(q.Ticker,q.Date),_fmt_int(q.volume_increase),q.Date,ticker=q.Ticker))
    # Entry/final/return volume uses actual membership boundaries.  A quote gap
    # cannot masquerade as an index entry, removal, or return.
    membership_first_rows=_membership_boundary_rows(c,membership_tenures,'start')
    if len(membership_first_rows):
        q=membership_first_rows.loc[membership_first_rows.Volume.idxmax()];cv.append(_rec('Highest-volume first session after joining',label(q.Ticker,q.Date),_fmt_int(q.Volume),q.Date,ticker=q.Ticker))
    completed=membership_tenures[membership_tenures.end!=idx_last]
    completed_rows=_membership_boundary_rows(c,completed,'end')
    if len(completed_rows):
        q=completed_rows.loc[completed_rows.Volume.idxmax()];cv.append(_rec('Highest-volume final session before removal',label(q.Ticker,q.Date),_fmt_int(q.Volume),q.Date,ticker=q.Ticker))
    membership_counts=membership_tenures.groupby('Identity').cumcount() if len(membership_tenures) else pd.Series(dtype=int)
    returned_membership=membership_tenures.loc[membership_counts>0] if len(membership_tenures) else membership_tenures
    return_rows=_membership_boundary_rows(c,returned_membership,'start')
    if len(return_rows):
        q=return_rows.loc[return_rows.Volume.idxmax()];cv.append(_rec('Highest-volume session following a return to the GLI',label(q.Ticker,q.Date),_fmt_int(q.Volume),q.Date,ticker=q.Ticker))
    # component volume streaks
    for feat,cond in [('Longest above-average-volume streak',comp.rvol20>1),('Longest streak of increasing component volume',comp.Volume>comp.prev_volume)]:
        z=_best_true_component_run(comp,cond & comp.continuous)
        if z:n,t,start,end=z;cv.append(_rec(feat,label(t,end),f'{n} sessions',end,f'{start} through {end}',t))
    pv=_get_section(categories,'volume','Price + Volume Feats')
    for feat,cond in [('Longest winning streak accompanied by above-average volume',(comp.clean_ret>0)&(comp.rvol20>1)),('Longest losing streak accompanied by above-average volume',(comp.clean_ret<0)&(comp.rvol20>1))]:
        z=_best_true_component_run(comp,cond)
        if z:n,t,start,end=z;pv.append(_rec(feat,label(t,end),f'{n} sessions',end,f'{start} through {end}',t))
    tenure_high=c.groupby('tenure_id').adj_price.cummax();tenure_low=c.groupby('tenure_id').adj_price.cummin()
    s=c[(c.adj_price>=tenure_high)&c.rvol20.notna()];
    if len(s):q=s.loc[s.rvol20.idxmax()];pv.append(_rec('Highest relative volume on a new GLI-tenure high',label(q.Ticker,q.Date),f'{q.rvol20:.2f}×',q.Date,ticker=q.Ticker))
    s=c[(c.adj_price<=tenure_low)&c.rvol20.notna()];
    if len(s):q=s.loc[s.rvol20.idxmax()];pv.append(_rec('Highest relative volume on a new GLI-tenure low',label(q.Ticker,q.Date),f'{q.rvol20:.2f}×',q.Date,ticker=q.Ticker))

    # Membership fixes/additions.  Membership facts come from roster intervals,
    # never from whether a component quote happens to be present on a session.
    tr=_get_section(categories,'membership','Tenure')
    continuous_original=membership_tenures[(membership_tenures.start==idx.Date.iloc[0])&(membership_tenures.end==idx_last)]
    tickers=sorted(continuous_original.end_ticker.tolist())
    _replace_record(tr,'Original components still active',_rec('Original components still active','Multiple',f'{len(tickers)} components',idx_last,', '.join(tickers)))
    dr=_get_section(categories,'membership','Departures & Returns')
    returns=[]
    for identity,g in membership_tenures.sort_values(['Identity','start_i']).groupby('Identity'):
        rows=list(g.itertuples(index=False))
        for a,b in zip(rows,rows[1:]):returns.append((identity,a,b))
    month_stats=[];return_gain=[]
    for identity,a,b in returns:
        # Use only a price segment that starts on the true roster return date;
        # a later quote gap must not create a synthetic return period.
        t=b.start_ticker
        price_match=tenures[(tenures.Ticker==t)&(tenures.start_i==b.start_i)]
        if price_match.empty:
            continue
        price_tid=price_match.iloc[0].tenure_id
        g=c[c.tenure_id==price_tid].sort_values('session_i').reset_index(drop=True)
        if len(g)>=21 and int(g.session_i.iloc[20])-int(g.session_i.iloc[0])==20:
            month_stats.append((g.adj_price.iloc[20]/g.adj_price.iloc[0]-1,t,g.Date.iloc[20],g.Date.iloc[0]))
        if len(g):
            j=g.adj_price.idxmax();mx=float(g.adj_price.max()/g.adj_price.iloc[0]-1);return_gain.append((mx,t,g.loc[j,'Date'],g.Date.iloc[0]))
    if month_stats:
        z=max(month_stats);dr.append(_rec('Best first month after returning',label(z[1],z[2]),_fmt_num(z[0]*100,2,True,'%'),z[2],f'Returned {z[3]}; first 21 sessions',z[1]))
    if return_gain:
        z=max(return_gain);dr.append(_rec('Largest gain from return-date price',label(z[1],z[2]),_fmt_num(z[0]*100,2,True,'%'),z[2],f'Returned {z[3]}',z[1]))
    ir=_get_section(categories,'membership','Identity & Corporate History')
    for rec in _identity_longest_records(root,site_data,idx,label,membership_tenures):
        _replace_record(ir,rec['feat'],rec)

    # Rare GLI feats and component entry/exit feats.
    rare=_get_section(categories,'rare','Rare Feats')
    prev=idx.Close.shift(1);gap=idx.Open/prev-1;ath=idx.Close.eq(idx.Close.cummax());s=idx[ath & (gap<=-.01)]
    if len(s):q=s.loc[(s.Close/s.Open-1).idxmax()];rare.append(_rec('Record-high close following a sharply negative open','GLI',_fmt_num((q.Close/q.Open-1)*100,2,True,'%'),q.Date,f'Opened {gap.loc[q.name]*100:+.2f}% vs prior close'))
    s=idx[ath]
    if len(s):q=s.loc[(s.Close-s.Low).idxmax()];rare.append(_rec('Largest intraday comeback ending at a record high','GLI',_fmt_num(q.Close-q.Low,2,suffix=' points'),q.Date))
    if len(s):q=s.loc[s.Volume.idxmax()];rare.append(_rec('Highest-volume record-high session','GLI',_fmt_int(q.Volume),q.Date))
    both=idx.Close.eq(idx.Close.cummax()) & idx.High.eq(idx.High.cummax());best,end=_streak_best(both.tolist(),lambda x:bool(x));
    if end is not None:rare.append(_rec('Most consecutive sessions setting both closing and intraday highs','GLI',f'{best} sessions',idx.iloc[end].Date,f'{idx.iloc[end-best+1].Date} through {idx.iloc[end].Date}'))
    # breadth reversal
    b=comp.loc[comp.clean_ret.notna(),['Date','Ticker','clean_ret']].assign(adv=lambda x:(x.clean_ret>0).astype(int)).groupby('Date').agg(adv=('adv','sum'),n=('Ticker','size')).reset_index();b['advp']=b.adv/b.n;b['swing']=b.advp.diff().abs();
    if b.swing.notna().any():q=b.loc[b.swing.idxmax()];prevrow=b.loc[q.name-1];rare.append(_rec('Largest breadth reversal from one session to the next','GLI',_fmt_num(q.swing*100,2,suffix=' percentage points'),q.Date,f'{prevrow.advp*100:.1f}% advancing → {q.advp*100:.1f}%'))
    # top five prior weights all opposite index
    comp['prev_weight']=comp.groupby('Ticker').weight.shift(1);valid=comp.loc[comp.continuous & comp.prev_weight.notna(),['Date','Ticker','clean_ret','prev_weight']].copy();valid['rank_prev']=valid.groupby('Date').prev_weight.rank(method='first',ascending=False);top5=valid[valid.rank_prev<=5];z=top5.groupby('Date').agg(n=('Ticker','size'),all_down=('clean_ret',lambda x:bool((x<0).all())),all_up=('clean_ret',lambda x:bool((x>0).all()))).reset_index();z['idxret']=z.Date.map(idx.set_index('Date').Close.pct_change())
    s=z[(z.n==5)&z.all_down&(z.idxret>0)]
    if len(s):q=s.loc[s.idxret.idxmax()];rare.append(_rec('GLI advances while all five largest-weighted components decline','GLI',_fmt_num(q.idxret*100,2,True,'%'),q.Date))
    s=z[(z.n==5)&z.all_up&(z.idxret<0)]
    if len(s):q=s.loc[s.idxret.idxmin()];rare.append(_rec('GLI declines while all five largest-weighted components advance','GLI',_fmt_num(q.idxret*100,2,True,'%'),q.Date))
    # Entry/final component feats are anchored to actual roster boundaries.
    firsts=_membership_boundary_rows(c,membership_tenures,'start')
    if len(firsts):
        q=firsts.loc[firsts.oc_ret.idxmax()];rare.append(_rec('Best first session after entering the GLI',label(q.Ticker,q.Date),_fmt_num(q.oc_ret*100,2,True,'%'),q.Date,ticker=q.Ticker))
        q=firsts.loc[firsts.oc_ret.idxmin()];rare.append(_rec('Worst first session after entering the GLI',label(q.Ticker,q.Date),_fmt_num(q.oc_ret*100,2,True,'%'),q.Date,ticker=q.Ticker))
    first_month=[]
    for m in membership_tenures.itertuples(index=False):
        price_match=tenures[(tenures.Ticker==m.start_ticker)&(tenures.start_i==m.start_i)]
        if price_match.empty:
            continue
        tid=price_match.iloc[0].tenure_id
        g=c[c.tenure_id==tid].sort_values('session_i').reset_index(drop=True)
        if len(g)>=21 and int(g.session_i.iloc[20])-int(g.session_i.iloc[0])==20:
            first_month.append((g.adj_price.iloc[20]/g.adj_price.iloc[0]-1,g.Ticker.iloc[0],g.Date.iloc[20],g.Date.iloc[0]))
    if first_month:
        z=max(first_month);rare.append(_rec('Best first month as a GLI component',label(z[1],z[2]),_fmt_num(z[0]*100,2,True,'%'),z[2],f'Entry {z[3]}; first 21 sessions',z[1]));z=min(first_month);rare.append(_rec('Worst first month as a GLI component',label(z[1],z[2]),_fmt_num(z[0]*100,2,True,'%'),z[2],f'Entry {z[3]}; first 21 sessions',z[1]))
    finals=_membership_boundary_rows(c,completed,'end')
    finals=finals[finals.oc_ret>0] if len(finals) else finals
    if len(finals):
        q=finals.loc[finals.oc_ret.idxmax()];rare.append(_rec('Largest gain on the final session before removal',label(q.Ticker,q.Date),_fmt_num(q.oc_ret*100,2,True,'%'),q.Date,ticker=q.Ticker))
    if first_double:
        d,t,start,n,_=first_double;rare.append(_rec('First component to double while in the GLI',label(t,d),'2.00× entry price',d,f'Entry {start}; {n} sessions',t))
    if fastest_double:
        n,t,d,start,_=fastest_double;rare.append(_rec('Fastest component to double',label(t,d),f'{n} sessions',d,f'Entry {start}',t))
    if max_doublings and max_doublings[0]>0:
        levels,t,d,start,mult,_=max_doublings;rare.append(_rec('Component with the most separate doublings during one tenure',label(t,d),f'{levels} doubling levels',d,f'Entry {start}; peak {mult:.2f}× entry',t))

    # Make the >50% rare-feat entries count-based, not denominator-sensitive maxima.
    for direction in ['advance','decline']:
        feat=f'One component accounts for more than 50% of the GLI’s daily {direction}'
        qualifying=[]
        for d,g in cs.groupby('Date'):
            mv=idx_move.get(d)
            if pd.isna(mv) or mv==0:continue
            if direction=='advance' and mv>0:
                q=g.loc[g.contrib.idxmax()];share=q.contrib/mv
                if share>.5:qualifying.append((d,q.Ticker,share))
            elif direction=='decline' and mv<0:
                q=g.loc[g.contrib.idxmin()];share=q.contrib/mv
                if share>.5:qualifying.append((d,q.Ticker,share))
        if qualifying:
            d,t,share=qualifying[-1];_replace_record(rare,feat,_rec(feat,label(t,d),f'{len(qualifying)} sessions',d,f'Most recent: {share*100:.1f}% of the index move',t))

    # Rebuild meta records after all additions.
    meta=_get_section(categories,'rare','Record-Book Meta Feats');meta.clear()
    all_records=[]
    for cat in categories:
        for sec in cat['sections']:
            if sec['title']=='Record-Book Meta Feats':continue
            for rec in sec['records']:
                all_records.append((cat['id'],rec))
    counts=Counter(rec.get('ticker') for _,rec in all_records if rec.get('ticker'))
    if counts:
        t,n=counts.most_common(1)[0];meta.append(_rec('Company holding the most current all-time records',label(t),f'{n} records',ticker=t))
    catsets=defaultdict(set);dates=defaultdict(list)
    for cid,rec in all_records:
        t=rec.get('ticker');d=rec.get('date','')
        if t:catsets[t].add(cid)
        if t and len(str(d))>=4 and str(d)[:4].isdigit():dates[t].append(str(d))
    if catsets:
        t=max(catsets,key=lambda x:(len(catsets[x]),counts[x]));meta.append(_rec('Company appearing in the most feat categories',label(t),f'{len(catsets[t])} categories',detail=f'{counts[t]} total record appearances',ticker=t))
    if dates:
        t=max(dates,key=lambda x:len({d[:4] for d in dates[x]}));yrs=sorted({d[:4] for d in dates[t]});meta.append(_rec('Company with records spanning the most different years',label(t),f'{len(yrs)} years',detail=f'{yrs[0]}–{yrs[-1]}',ticker=t))
        full_dates={t:[d for d in ds if len(d)>=10 and d[4:5]=='-' and d[7:8]=='-'] for t,ds in dates.items()}
        spans={t:(max(pd.to_datetime(ds))-min(pd.to_datetime(ds))).days for t,ds in full_dates.items() if len(ds)>=2}
        if spans:
            t=max(spans,key=spans.get);ds=sorted(full_dates[t]);meta.append(_rec('Company with the widest span between its oldest and newest record',label(t),f'{spans[t]:,} calendar days',ds[-1],f'Oldest record {ds[0]}',t))
    year_counts=Counter(str(rec.get('date',''))[:4] for _,rec in all_records if len(str(rec.get('date','')))>=4 and str(rec.get('date',''))[:4].isdigit())
    if year_counts:
        y,n=year_counts.most_common(1)[0];meta.append(_rec('Most record-book feats dated to one calendar year','GLI record book',f'{n} records',y))



def _membership_sections(comp,idx,label,root:Path,site_data:Path):
    # Membership/return facts use the same normalized Component History identity
    # chronology as the Component History page.  Component OHLCV is consulted
    # only for records that explicitly require a boundary-day quote.
    c,_price_tenures=_tenure_frame(comp);rdf=_membership_tenure_frame(root,site_data,idx);r=[];last=idx.Date.iloc[-1]
    if rdf.empty:
        return [('Tenure',r),('Departures & Returns',[])],rdf

    def latest_ticker(identity: str) -> str:
        g=rdf[rdf.Identity==identity].sort_values('end_i')
        return str(g.iloc[-1].end_ticker) if len(g) else str(identity)

    # Longest continuous identity tenure (tie-aware).
    best=int(rdf.sessions.max());ties=rdf[rdf.sessions==best].sort_values(['Identity','end_ticker'])
    if len(ties)==1:
        q=ties.iloc[0];r.append(_rec('Longest continuous GLI tenure',label(q.end_ticker,q.end),f'{best:,} sessions',q.end,f'{q.start} through {q.end}',q.end_ticker))
    else:
        r.append(_rec('Longest continuous GLI tenure','Multiple',f'{best:,} sessions',ties.end.iloc[0],f"{ties.start.iloc[0]} through {ties.end.iloc[0]}; " + ', '.join(ties.end_ticker.tolist())))

    # Most total represented sessions across all true identity tenures.
    totals=rdf.groupby('Identity').sessions.sum();best_total=int(totals.max());ids=sorted(totals[totals==best_total].index.tolist())
    tickers=sorted(dict.fromkeys(latest_ticker(i) for i in ids))
    if len(ids)==1:
        t=tickers[0];r.append(_rec('Most total sessions as a GLI component',label(t),f'{best_total:,} sessions',ticker=t))
    else:
        r.append(_rec('Most total sessions as a GLI component','Multiple',f'{best_total:,} sessions',detail=', '.join(tickers)))

    complete=rdf[rdf.end!=last]
    if len(complete):
        best_short=int(complete.sessions.min());ties=complete[complete.sessions==best_short].sort_values(['end','Identity'])
        q=ties.iloc[0]
        detail=f'{q.start} through {q.end}' if len(ties)==1 else f'{q.start} through {q.end}; tied: ' + ', '.join(ties.end_ticker.tolist())
        holder=label(q.end_ticker,q.end) if len(ties)==1 else 'Multiple'
        r.append(_rec('Shortest completed GLI tenure',holder,f'{best_short} sessions',q.end,detail,q.end_ticker if len(ties)==1 else ''))

    active=rdf[rdf.end==last]
    if len(active):
        oldest_i=int(active.start_i.min());ties=active[active.start_i==oldest_i].sort_values(['Identity','end_ticker']);best_sessions=int(ties.sessions.max())
        if len(ties)==1:
            q=ties.iloc[0];r.append(_rec('Oldest continuously active GLI component',label(q.end_ticker,q.end),f'{int(q.sessions):,} sessions',q.start,f'Active through {q.end}',q.end_ticker))
        else:
            r.append(_rec('Oldest continuously active GLI component','Multiple',f'{best_sessions:,} sessions',ties.start.iloc[0],f'Active through {last}; ' + ', '.join(ties.end_ticker.tolist())))

    originals=rdf[(rdf.start==idx.Date.iloc[0])&(rdf.end==last)];ticks=sorted(originals.end_ticker.tolist())
    r.append(_rec('Original components still active','Multiple',f'{len(ticks)} components',last,', '.join(ticks)))

    year_sets=defaultdict(set)
    for row in rdf.itertuples(index=False):
        for y in idx.iloc[int(row.start_i):int(row.end_i)+1].Date.astype(str).str[:4].unique():
            year_sets[row.Identity].add(str(y))
    if year_sets:
        max_years=max(len(v) for v in year_sets.values());ids=sorted(i for i,v in year_sets.items() if len(v)==max_years)
        yt=sorted(dict.fromkeys(latest_ticker(i) for i in ids))
        if len(ids)==1:r.append(_rec('Most calendar years represented in the index',label(yt[0]),f'{max_years} years',ticker=yt[0]))
        else:
            detail=', '.join(yt) if len(yt)<=12 else f'{len(ids)} components tied'
            r.append(_rec('Most calendar years represented in the index','Multiple',f'{max_years} years',detail=detail))
    sections=[('Tenure',r)]

    r=[];counts=rdf.groupby('Identity').size();ret=counts[counts>1]
    if len(ret):
        max_tenures=int(ret.max());ids=sorted(ret[ret==max_tenures].index.tolist());tt=sorted(dict.fromkeys(latest_ticker(i) for i in ids))
        if len(ids)==1:r.append(_rec('Most separate GLI tenures',label(tt[0]),f'{max_tenures} tenures',ticker=tt[0]))
        else:r.append(_rec('Most separate GLI tenures','Multiple',f'{max_tenures} tenures',detail=', '.join(tt)))
        returns=[]
        for identity,g in rdf.sort_values(['Identity','start_i']).groupby('Identity'):
            rows=list(g.itertuples(index=False))
            for a,b in zip(rows,rows[1:]):returns.append((int(b.start_i-a.end_i-1),identity,a,b))
        returns.sort(key=lambda x:x[3].start_i)
        if returns:
            x=returns[0];t=x[3].start_ticker;r.append(_rec('First component to leave and later return',label(t,x[3].start),f'{x[0]} sessions absent',x[3].start,f'Previous tenure ended {x[2].end}',t))
            latest=max(z[3].start_i for z in returns);latest_rows=[z for z in returns if z[3].start_i==latest]
            x=sorted(latest_rows,key=lambda z:z[3].start_ticker)[0];t=x[3].start_ticker;holder=label(t,x[3].start) if len(latest_rows)==1 else 'Multiple';detail='' if len(latest_rows)==1 else ', '.join(sorted(z[3].start_ticker for z in latest_rows))
            r.append(_rec('Most recent returning component',holder,f'{x[0]} sessions absent',x[3].start,detail,t if len(latest_rows)==1 else ''))
            x=max(returns,key=lambda z:z[0]);t=x[3].start_ticker;r.append(_rec('Longest absence before returning',label(t,x[3].start),f'{x[0]:,} sessions',x[3].start,f'Absent after {x[2].end}',t))
            x=min(returns,key=lambda z:z[0]);t=x[3].start_ticker;r.append(_rec('Shortest absence before returning',label(t,x[3].start),f'{x[0]} sessions',x[3].start,ticker=t))
            abs_tot=defaultdict(int)
            for x in returns:abs_tot[x[1]]+=x[0]
            identity=max(abs_tot,key=abs_tot.get);t=latest_ticker(identity);r.append(_rec('Most total sessions spent outside the index between active tenures',label(t),f'{abs_tot[identity]:,} sessions',ticker=t))

            returned=rdf.loc[rdf.groupby('Identity').cumcount()>0]
            firsts=_membership_boundary_rows(c,returned,'start')
            if len(firsts):
                z=firsts.oc_ret.dropna()
                if len(z):
                    q=firsts.loc[z.idxmax()];r.append(_rec('Best first session after returning',label(q.Ticker,q.Date),_fmt_num(q.oc_ret*100,2,True,'%'),q.Date,ticker=q.Ticker))
                    q=firsts.loc[z.idxmin()];r.append(_rec('Worst first session after returning',label(q.Ticker,q.Date),_fmt_num(q.oc_ret*100,2,True,'%'),q.Date,ticker=q.Ticker))
    sections.append(('Departures & Returns',r));return sections,rdf

def _price_start_ok(value: Any, min_price: float | None) -> bool:
    """Return whether a Price Performance candidate clears the start-price floor."""
    if min_price is None:
        return True
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(value) and value > min_price)


def _filtered_price_performance_records(
    comp: pd.DataFrame,
    idx: pd.DataFrame,
    label,
    min_price: float,
) -> list[dict[str, Any]]:
    """Recompute Price Performance records with a strict starting-price floor.

    The filter is deliberately scoped to Component Feats -> Price Performance.
    It tests the price at the start/reference point of the measured feat:
    session Open for single-session records; starting Close for rolling/calendar
    returns; entry Close for entry/doubling records; tenure-low Close for rallies;
    and governing-peak Close for drawdown/recovery records.
    """
    records: list[dict[str, Any]] = []

    def rr(feat, q, val, detail='', start_price=None):
        ticker = str(q.Ticker)
        raw={'start_price':float(start_price),'min_price':float(min_price)} if start_price is not None else None
        return _rec(feat, label(ticker, str(q.Date)), val, str(q.Date), detail, ticker, raw)

    # Single-session records: test that session's Open.
    s = comp[(comp.Open > 0) & (comp.Open > min_price)]
    if len(s):
        q=s.loc[s.oc_ret.idxmax()];records.append(rr('Largest single-session percentage gain',q,_fmt_num(q.oc_ret*100,2,True,'%'),'Open-to-close; avoids split/re-entry discontinuities',q.Open))
        q=s.loc[s.oc_ret.idxmin()];records.append(rr('Largest single-session percentage decline',q,_fmt_num(q.oc_ret*100,2,True,'%'),'Open-to-close; avoids split/re-entry discontinuities',q.Open))
        q=s.loc[s.intraday_range_pct.idxmax()];records.append(rr('Largest intraday percentage range',q,_fmt_num(q.intraday_range_pct*100,2,suffix='%'),start_price=q.Open))

    # Rolling multi-session records: test the actual Close at the start of the window.
    tenure_grp=comp.groupby('_tenure_id',sort=False)
    for n in [2,5,10]:
        prior_adj=tenure_grp.adj_price.shift(n-1)
        prior_si=tenure_grp.session_i.shift(n-1)
        prior_close=tenure_grp.Close.shift(n-1)
        valid=(comp.session_i-prior_si==n-1) & (prior_close > min_price)
        vals=(comp.adj_price/prior_adj-1).where(valid)
        good=vals.dropna()
        if len(good):
            for best,txt in [(True,'Best'),(False,'Worst')]:
                qi=good.idxmax() if best else good.idxmin();q=comp.loc[qi]
                records.append(rr(f'{txt} {n}-session performance',q,_fmt_num(vals.loc[qi]*100,2,True,'%'),start_price=prior_close.loc[qi]))

    # Calendar-period records: test the first actual Close in the qualifying tenure-period.
    c=comp.copy();c['dt']=pd.to_datetime(c.Date)
    period_specs=[
        ('week',c.dt.dt.to_period('W-FRI'),4),
        ('month',c.dt.dt.to_period('M'),15),
        ('quarter',c.dt.dt.to_period('Q'),45),
        ('calendar year',c.dt.dt.to_period('Y'),180),
    ]
    for pname,per,min_sessions in period_specs:
        cp=c.assign(period=per.astype(str)).copy()
        x=cp.groupby(['_tenure_id','Ticker','period'],sort=False).agg(
            first=('adj_price','first'), last=('adj_price','last'),
            start_close=('Close','first'), date=('Date','last'),
            sessions=('adj_price','size')
        ).reset_index()
        x=x[(x.sessions>=min_sessions) & (x.start_close > min_price)]
        x['ret']=x['last']/x['first']-1
        if len(x):
            for best,txt in [(True,'Best'),(False,'Worst')]:
                q=x.loc[x.ret.idxmax() if best else x.ret.idxmin()]
                obj=type('Q',(),{'Ticker':q.Ticker,'Date':q.date})
                records.append(rr(f'{txt} {pname}',obj,_fmt_num(q.ret*100,2,True,'%'),q.period,q.start_close))

    # Continuous-tenure records.
    c,_=_tenure_frame(comp)
    best_entry=None; best_rally=None; worst_dd=None; fastest_double=None
    recoveries={.20:None,.30:None,.50:None}
    for tid,g in c.groupby('tenure_id',sort=False):
        g=g.sort_values('session_i').reset_index(drop=True)
        ticker=g.Ticker.iloc[0]
        arr=g.adj_price.to_numpy(float)
        raw_close=g.Close.to_numpy(float)
        if not len(arr):
            continue

        entry=float(raw_close[0])
        if _price_start_ok(entry,min_price):
            ratios=arr/arr[0]
            j=int(np.nanargmax(ratios))
            cand=(ratios[j]-1,ticker,g.Date.iloc[j],g.Date.iloc[0],ratios[j],tid)
            if best_entry is None or cand[0]>best_entry[0]: best_entry=cand
            hit=np.flatnonzero(ratios>=2)
            if len(hit):
                j=int(hit[0]);elapsed=j;cand2=(elapsed,ticker,g.Date.iloc[j],g.Date.iloc[0],tid)
                if fastest_double is None or cand2[0]<fastest_double[0] or (cand2[0]==fastest_double[0] and cand2[2]<fastest_double[2]): fastest_double=cand2

        runmin=np.minimum.accumulate(arr)
        rally=arr/runmin-1
        j=int(np.nanargmax(rally));low_i=int(np.argmin(arr[:j+1]))
        if _price_start_ok(raw_close[low_i],min_price):
            cand=(rally[j],ticker,g.Date.iloc[j],g.Date.iloc[low_i],tid)
            if best_rally is None or cand[0]>best_rally[0]:best_rally=cand

        runmax=np.maximum.accumulate(arr)
        dd=arr/runmax-1
        j=int(np.nanargmin(dd));peak_i=int(np.argmax(arr[:j+1]))
        if _price_start_ok(raw_close[peak_i],min_price):
            cand=(dd[j],ticker,g.Date.iloc[j],g.Date.iloc[peak_i],tid)
            if worst_dd is None or cand[0]<worst_dd[0]:worst_dd=cand

        # Recovery threshold is governed by the prior peak; apply the floor there.
        for th in recoveries:
            peak=arr[0];peak_i=0;trigger=None;target=None;target_i=None
            for i in range(1,len(arr)):
                if trigger is None:
                    if arr[i]>=peak:
                        peak=arr[i];peak_i=i
                    elif arr[i]/peak-1<=-th:
                        trigger=i;target=peak;target_i=peak_i
                else:
                    if arr[i]>=target:
                        if target_i is not None and _price_start_ok(raw_close[target_i],min_price):
                            elapsed=i-trigger;cand=(elapsed,ticker,g.Date.iloc[i],g.Date.iloc[trigger],g.Date.iloc[target_i],tid)
                            if recoveries[th] is None or elapsed<recoveries[th][0]:recoveries[th]=cand
                        peak=arr[i];peak_i=i;trigger=None;target=None;target_i=None

    if best_entry:
        gain,t,d,start,mult,tid=best_entry
        entry_row=c[(c.tenure_id==tid) & (c.Date==start)].iloc[0] if ((c.tenure_id==tid) & (c.Date==start)).any() else None
        start_price=float(entry_row.Close) if entry_row is not None else None
        raw={'start_price':start_price,'min_price':float(min_price)}
        records += [_rec('Largest gain from GLI entry price',label(t,d),_fmt_num(gain*100,2,True,'%'),d,f'Entry {start}',t,raw),
                    _rec('Largest multiple of GLI entry price',label(t,d),f'{mult:.2f}×',d,f'Entry {start}',t,raw)]
    if fastest_double:
        n,t,d,start,tid=fastest_double;sp=float(c[(c.tenure_id==tid)&(c.Date==start)].iloc[0].Close);records.append(_rec('Fastest doubling from GLI entry price',label(t,d),f'{n} sessions',d,f'Entry {start}',t,{'start_price':sp,'min_price':float(min_price)}))
    if best_rally:
        v,t,d,low,tid=best_rally;sp=float(c[(c.tenure_id==tid)&(c.Date==low)].iloc[0].Close);records.append(_rec('Largest rally from a GLI-tenure low',label(t,d),_fmt_num(v*100,2,True,'%'),d,f'Tenure low {low}',t,{'start_price':sp,'min_price':float(min_price)}))
    if worst_dd:
        v,t,d,peak,tid=worst_dd;sp=float(c[(c.tenure_id==tid)&(c.Date==peak)].iloc[0].Close);records.append(_rec('Largest drawdown while in the index',label(t,d),_fmt_num(v*100,2,True,'%'),d,f'Peak {peak}',t,{'start_price':sp,'min_price':float(min_price)}))
    for th,z in recoveries.items():
        if z:
            n,t,d,trigger,peak,tid=z;sp=float(c[(c.tenure_id==tid)&(c.Date==peak)].iloc[0].Close);records.append(_rec(f'Fastest recovery from a {int(th*100)}% component drawdown',label(t,d),f'{n} sessions',d,f'Threshold hit {trigger}; recovered prior peak from {peak}',t,{'start_price':sp,'min_price':float(min_price)}))
    return records

def build(root:Path,site_data:Path,full_rows:list[dict[str,str]]):
    name,label=_name_maps(site_data,root)
    idx=pd.DataFrame(full_rows).rename(columns={'GLI_Open':'Open','GLI_High':'High','GLI_Low':'Low','GLI_Close':'Close','TotalVolume':'Volume'})
    for col in ['Open','High','Low','Close','Volume','Divisor']:
        idx[col]=pd.to_numeric(idx[col],errors='coerce')
    idx=idx.dropna(subset=['Date','Close']).sort_values('Date').reset_index(drop=True)
    raw_comp=_load_components(root,site_data)
    eligible_comp,membership_audit=_filter_components_to_membership(raw_comp,root,site_data)
    comp=_prepare_component_metrics(eligible_comp,idx)
    cats=[]
    index_sections=_index_feats(idx);breadth,rare=_breadth_and_rare(comp,idx,label);index_sections.extend(breadth)
    cats.append({'id':'index','title':'Index Feats','sections':[{'title':t,'records':r} for t,r in index_sections]})
    component_sections,_=_component_sections(comp,idx,label);cats.append({'id':'component','title':'Component Feats','sections':[{'title':t,'records':r} for t,r in component_sections]})
    streak_sections=_streak_sections(comp,idx,label);cats.append({'id':'streaks','title':'Streaks','sections':[{'title':t,'records':r} for t,r in streak_sections]})
    volume_sections=_volume_sections(comp,idx,label);cats.append({'id':'volume','title':'Volume & Trading Activity','sections':[{'title':t,'records':r} for t,r in volume_sections]})
    membership,_=_membership_sections(comp,idx,label,root,site_data);membership.extend(_identity_section(site_data,label));cats.append({'id':'membership','title':'Membership & Longevity','sections':[{'title':t,'records':r} for t,r in membership]})
    cats.append({'id':'rare','title':'Rare Feats','sections':[{'title':t,'records':r} for t,r in rare] + [{'title':'Record-Book Meta Feats','records':[]}]})
    _augment_categories(cats,comp,idx,label,root,site_data)
    price_performance_filters={
        'gt1':_filtered_price_performance_records(comp,idx,label,1.0),
        'gt5':_filtered_price_performance_records(comp,idx,label,5.0),
    }
    return {
        'schema_version':2,'generated_through':idx.Date.iloc[-1],'component_data_through':comp.Date.max() if len(comp) else '',
        'component_membership_filter':membership_audit,
        'price_performance_filters':price_performance_filters,
        'definitions':[
            'Index daily point/percentage gains and losses are intentionally excluded; those remain on Market Moves.',
            'All component-derived feats use only dates when the ticker was an eligible GLI component: historical membership through 2025 comes from historical_company_names.csv and 2026 membership comes from constituents_great_lakes.csv. Pre-entry and post-removal quotes, including OTC trading, are excluded.',
            'Membership and longevity records use the same normalized Component History chronology as the Component History page, projected onto the GLI session calendar. True removals/returns break a tenure; name/ticker continuity does not; missing component-price rows cannot create a departure or return.',
            'Component single-session gain/loss records use open-to-close returns so splits, ticker re-entries, and overnight identity discontinuities do not masquerade as trading-session feats.',
            'Component return chains require consecutive GLI trading sessions within the same ticker tenure. On GLI divisor-reset sessions, the return chain uses open-to-close movement to neutralize mechanical split/reconstitution discontinuities; contribution records exclude those reset sessions.',
            'Multi-session component price records use uninterrupted active-ticker sessions and cannot cross a removal/re-entry boundary. Calendar week/month/quarter/year rankings require at least 4/15/45/180 active sessions respectively so very short entry/removal fragments do not compete as full periods.',
            'The Component Feats → Price Performance minimum-price dropdown is scoped only to that section. All is the default; >$1 and >$5 require the feat starting/reference price to be strictly above the selected floor (session Open for single-session records; starting Close for rolling/calendar returns; entry Close for entry/doubling records; tenure-low Close for rallies; governing-peak Close for drawdown/recovery records).',
            'Relative volume (RVOL) compares the current session with the prior 20 active sessions for that component; GLI RVOL uses the prior 20 index sessions.',
            'Raw share-volume records use the accepted reported share counts and are therefore sensitive to stock splits and long-run changes in shares outstanding; RVOL records provide a within-era comparison.',
            'Lowest week/month volume records require at least 4/15 sessions respectively. Partial 2005 and the current partial calendar year are excluded from lowest annual-volume comparisons.',
            'Sector/state-specific feats are reserved until a historical sector/state metadata table is available.'
        ],'categories':cats
    }

# Final optimized membership scan.

def render(payload:dict)->str:
    def render_rows(records):
        rows=''.join(f'<tr><td>{html.escape(str(r["feat"]))}</td><td><b>{html.escape(str(r.get("holder","") or "—"))}</b></td><td>{html.escape(str(r.get("value","") or "—"))}</td><td>{html.escape(str(r.get("date","") or "—"))}</td><td class="muted feat-detail">{html.escape(str(r.get("detail","") or ""))}</td></tr>' for r in records)
        return rows or '<tr><td colspan="5" class="empty">No qualifying record is currently available.</td></tr>'

    tabs=''.join(f'<button class="tab {"active" if i==0 else ""}" data-feat-tab="{html.escape(c["id"])}">{html.escape(c["title"])}</button>' for i,c in enumerate(payload['categories']))
    panels=[]
    price_row_html={}
    price_variants=payload.get('price_performance_filters',{})
    for i,c in enumerate(payload['categories']):
        secs=[]
        for sec in c['sections']:
            rows=render_rows(sec['records'])
            is_price=(c['id']=='component' and sec['title']=='Price Performance')
            if is_price:
                price_row_html['all']=rows
                price_row_html['gt1']=render_rows(price_variants.get('gt1',[]))
                price_row_html['gt5']=render_rows(price_variants.get('gt5',[]))
                controls=(
                    '<div class="controls" style="margin-top:0">'
                    '<label>Minimum starting price <select id="price-performance-min">'
                    '<option value="all" selected>All</option>'
                    '<option value="gt1">&gt;$1</option>'
                    '<option value="gt5">&gt;$5</option>'
                    '</select></label></div>'
                )
                tbody=f'<tbody id="price-performance-body">{rows}</tbody>'
            else:
                controls=''
                tbody=f'<tbody>{rows}</tbody>'
            secs.append(f'<section class="panel feat-section"><h2>{html.escape(sec["title"])}</h2>{controls}<div class="table-wrap"><table><thead><tr><th>Feat</th><th>Record holder</th><th>Record</th><th>Date / Period</th><th>Details</th></tr></thead>{tbody}</table></div></section>')
        panels.append(f'<div class="feat-panel" data-feat-panel="{html.escape(c["id"])}" style="display:{"block" if i==0 else "none"}">{"".join(secs)}</div>')
    defs=''.join(f'<li>{html.escape(x)}</li>' for x in payload['definitions'])
    price_rows_json=json.dumps(price_row_html,separators=(',',':'),ensure_ascii=False).replace('</','<\\/')
    return f'''<div class="page-head"><div><h1>GLI Feats & Records</h1><div class="muted">Index, component, streak, volume, membership and rare-feat record book • through {html.escape(payload['generated_through'])}</div></div><a href="./market-moves.html" style="font-weight:800;text-decoration:none">Daily gain/loss records → Market Moves</a></div>
<div class="source-note"><b>Record-book definitions</b><ul style="margin:7px 0 0 18px;padding:0">{defs}</ul></div>
<div class="controls"><input id="feat-search" type="search" placeholder="Search feats, companies, symbols…" style="min-width:min(420px,100%)"></div>
<div class="tabs" id="feat-tabs">{tabs}</div>
{''.join(panels)}
<script>
const fs=document.getElementById('feat-search');let activeFeat='index';
const priceRows={price_rows_json};
const priceSelect=document.getElementById('price-performance-min');
function featApply(){{const q=(fs.value||'').trim().toLowerCase();document.querySelectorAll('.feat-panel').forEach(p=>p.style.display=p.dataset.featPanel===activeFeat?'block':'none');document.querySelectorAll(`.feat-panel[data-feat-panel="${{activeFeat}}"] .feat-section`).forEach(sec=>{{let any=false;sec.querySelectorAll('tbody tr').forEach(tr=>{{const ok=!q||tr.textContent.toLowerCase().includes(q);tr.style.display=ok?'':'none';if(ok)any=true;}});sec.style.display=any?'':'none';}});}}
function pricePerformanceApply(){{if(!priceSelect)return;const body=document.getElementById('price-performance-body');if(!body)return;body.innerHTML=priceRows[priceSelect.value]||priceRows.all||'';featApply();}}
document.getElementById('feat-tabs').addEventListener('click',e=>{{if(!e.target.matches('[data-feat-tab]'))return;document.querySelectorAll('[data-feat-tab]').forEach(x=>x.classList.remove('active'));e.target.classList.add('active');activeFeat=e.target.dataset.featTab;featApply();}});fs.addEventListener('input',featApply);if(priceSelect)priceSelect.addEventListener('change',pricePerformanceApply);
</script>'''
# --- Optimized record scans (override the reference implementations above). ---
