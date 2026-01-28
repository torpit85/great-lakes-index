#!/usr/bin/env python3
"""
GLI Site Builder (UPDATED: Total Volume)

This version is drop-in: you can replace your existing gli_site_build.py with it.

What it does:
- Reads gli_levels.csv (expects TotalVolume column if you updated the engine)
- Writes report/ticker.txt
- Writes report/history.html (Date, OHLC, TotalVolume)
- Injects a ticker bar + symbol box (includes Total Volume) + history link into report/index.html

Then your cron can rsync report/ -> docs/ and push.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

BASE_DATE = "2024-12-31"
BASE_VALUE = 100.00

ROOT = Path(__file__).resolve().parent
LEVELS = ROOT / "gli_levels.csv"
PRICES = ROOT / "gli_prices.csv"  # optional: component OHLCV history
CONSTITUENTS = ROOT / "constituents_great_lakes.csv"  # optional: ticker list (may include names)
COMPANY_CACHE = ROOT / "company_names.csv"  # cached ticker -> company name (from yfinance)
REPORT = ROOT / "report"

TICKER_HTML_MARK = "<!-- GLI_TICKER_BAR -->"
SYMBOL_BOX_MARK = "<!-- GLI_SYMBOL_BOX -->"
HISTORY_LINK_MARK = "<!-- GLI_HISTORY_LINK -->"
NAV_BAR_MARK = "<!-- GLI_NAV_BAR -->"

CSS_BLOCK = """
<style>
#gli-tickerbar{
  font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
  font-weight: 800;
  font-size: 18px;
  letter-spacing: 0.2px;
  padding: 12px 16px;
  margin: 0 0 14px 0;
  border: 1px solid #ddd;
  border-radius: 12px;
  background: linear-gradient(180deg, #fafafa, #f0f0f0);
  box-shadow: 0 1px 2px rgba(0,0,0,0.05);
  display:flex;
  justify-content:space-between;
  align-items:center;
  gap:16px;
  flex-wrap:wrap;
}
#gltxt .up{ color:#0a7a2f; }
#gltxt .down{ color:#b00020; }
#gltxt .flat{ color:#444; }

.gli-pill{
  font-size: 12px;
  font-weight: 800;
  border:1px solid #ddd;
  border-radius: 999px;
  padding: 6px 10px;
  background:#fff;
}

.gli-symbolbox{
  font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
  border: 1px solid #ddd;
  border-radius: 14px;
  padding: 14px 16px;
  margin: 0 0 18px 0;
  background:#fff;
}
.gli-grid{
  display:grid;
  grid-template-columns: repeat(5, minmax(140px, 1fr));
  gap: 10px 18px;
}
@media (max-width: 920px){
  .gli-grid{ grid-template-columns: repeat(2, minmax(140px, 1fr)); }
}

.gli-k{ color:#666; font-size: 12px; font-weight:800; text-transform:uppercase; letter-spacing:0.6px; }
.gli-v{ font-size: 18px; font-weight: 900; margin-top:2px; }

.gli-mini{ color:#666; font-size: 13px; margin-top:6px; line-height:1.25; }

.gli-linkbar{
  font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
  margin: 8px 0 18px 0;
}
.gli-linkbar a{ text-decoration:none; font-weight:800; }

.gli-nav{
  font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
  margin: 0 0 14px 0;
  padding: 10px 12px;
  border: 1px solid #ddd;
  border-radius: 12px;
  background: #f6f6f6;
}
.gli-nav a{ text-decoration:none; font-weight:800; margin-right:14px; }
.gli-nav a:hover{ text-decoration:underline; }

table{ border-collapse: collapse; width:100%; }
th,td{ border:1px solid #ddd; padding:8px; text-align:right; }
th:first-child,td:first-child{ text-align:left; }
</style>
"""

def fmt_int(x) -> str:
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return "0"

def load_levels() -> pd.DataFrame:
    if not LEVELS.exists():
        raise SystemExit("gli_levels.csv not found. Run the GLI engine first.")
    df = pd.read_csv(LEVELS)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    if "TotalVolume" not in df.columns:
        # Backwards compatible: if engine not updated, treat as zero.
        df["TotalVolume"] = 0
    return df
def load_prices_optional() -> pd.DataFrame | None:
    """Load gli_prices.csv if present. Expected to contain per-ticker daily OHLCV.

    We only need: Date, Ticker, Close (and ideally Open/High/Low/Adj Close/Volume/Source).
    """
    if not PRICES.exists():
        return None
    df = pd.read_csv(PRICES)
    # Normalize common column spellings
    if "date" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"date": "Date"})
    if "ticker" in df.columns and "Ticker" not in df.columns:
        df = df.rename(columns={"ticker": "Ticker"})
    df["Date"] = pd.to_datetime(df["Date"])
    if "Ticker" not in df.columns:
        raise SystemExit("gli_prices.csv exists but has no 'Ticker' column.")
    # Prefer Close; fall back to Adj Close if necessary
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})
    if "Close" not in df.columns:
        raise SystemExit("gli_prices.csv exists but has no 'Close' (or 'Adj Close') column.")
    return df



def load_company_map_optional() -> dict[str, str] | None:
    """Try to load a mapping of ticker/symbol -> company name from CONSTITUENTS.

    Your constituents_great_lakes.csv currently has Ticker,Active,Notes, so this will
    usually return None unless you later add a name column.
    """
    if not CONSTITUENTS.exists():
        return None
    try:
        c = pd.read_csv(CONSTITUENTS)
    except Exception:
        return None

    # Symbol column
    sym_col = None
    for cand in ["Symbol", "Ticker", "symbol", "ticker"]:
        if cand in c.columns:
            sym_col = cand
            break
    if sym_col is None:
        return None

    # Company/name column
    name_col = None
    for cand in ["Company", "Name", "Company Name", "company", "name", "company_name", "companyName"]:
        if cand in c.columns:
            name_col = cand
            break
    if name_col is None:
        return None

    out: dict[str, str] = {}
    for _, row in c[[sym_col, name_col]].dropna().iterrows():
        sym = str(row[sym_col]).strip()
        nm = str(row[name_col]).strip()
        if sym and nm:
            out[sym] = nm
    return out if out else None


def load_company_cache() -> dict[str, str]:
    """Load cached company names from COMPANY_CACHE if present."""
    if not COMPANY_CACHE.exists():
        return {}
    try:
        df = pd.read_csv(COMPANY_CACHE)
        if "Ticker" not in df.columns or "Company" not in df.columns:
            return {}
        out: dict[str, str] = {}
        for _, r in df[["Ticker", "Company"]].dropna().iterrows():
            out[str(r["Ticker"]).strip()] = str(r["Company"]).strip()
        return out
    except Exception:
        return {}


def save_company_cache(cache: dict[str, str]) -> None:
    """Persist company cache to disk."""
    if not cache:
        return
    try:
        df = pd.DataFrame(sorted(cache.items()), columns=["Ticker", "Company"])
        df.to_csv(COMPANY_CACHE, index=False)
    except Exception:
        pass


def fetch_company_names_yfinance(tickers: list[str], existing: dict[str, str] | None = None) -> dict[str, str]:
    """Best-effort company name lookup via yfinance (shortName/longName).

    Uses `existing` to avoid refetching. Never raises.
    """
    existing = existing or {}
    missing = [t for t in tickers if t and t not in existing]
    if not missing:
        return existing

    try:
        import yfinance as yf
    except Exception:
        return existing

    for t in missing:
        try:
            info = yf.Ticker(t).info or {}
            name = info.get("shortName") or info.get("longName") or info.get("displayName") or ""
            name = str(name).strip()
            if name:
                existing[t] = name
        except Exception:
            continue

    return existing
def compute_latest_changes(prices: pd.DataFrame) -> pd.DataFrame:
    """Return per-ticker latest price + day-over-day change + percent change."""
    df = prices.copy()
    df = df.sort_values(["Ticker", "Date"])
    df["PrevClose"] = df.groupby("Ticker")["Close"].shift(1)
    latest_date = df["Date"].max()
    latest = df[df["Date"] == latest_date].copy()
    latest["Price"] = latest["Close"]
    latest["Change"] = latest["Close"] - latest["PrevClose"]
    latest["PctChange"] = (latest["Change"] / latest["PrevClose"]) * 100
    return latest[["Ticker", "Date", "Price", "Change", "PctChange"]].copy()

def enhance_ohlcv_html_with_price_change() -> None:
    """Add Price and % Change columns to report/ohlcv.html (Component OHLCV page).

    - Price is taken from the 'Close' column in the table (or computed via gli_prices.csv).
    - % Change is computed vs the previous close per ticker using gli_prices.csv (preferred).
      If gli_prices.csv is missing, % Change will be left blank.
    """
    ohlcv_path = REPORT / "ohlcv.html"
    if not ohlcv_path.exists():
        # Engine may not have produced it; nothing to do.
        return

    html = ohlcv_path.read_text(encoding="utf-8")

    # Read the first table on the page
    try:
        tables = pd.read_html(html)
        if not tables:
            return
        t = tables[0].copy()
    except Exception:
        # If parsing fails, don't break the site build.
        return

    # Normalize expected columns
    if "Ticker" not in t.columns:
        # Sometimes it might be called 'Symbol'
        if "Symbol" in t.columns:
            t = t.rename(columns={"Symbol": "Ticker"})
        else:
            return

    # Price: prefer explicit Close, else Adj Close
    if "Close" in t.columns:
        t["Price"] = t["Close"]
    elif "Adj Close" in t.columns:
        t["Price"] = t["Adj Close"]
    else:
        t["Price"] = ""

    # Percent change: best-effort using gli_prices.csv
    pct_map = {}
    chg_map = {}
    prices = load_prices_optional()
    if prices is not None:
        latest = compute_latest_changes(prices)
        pct_map = dict(zip(latest["Ticker"], latest["PctChange"]))
        chg_map = dict(zip(latest["Ticker"], latest["Change"]))

    t["% Change"] = t["Ticker"].map(lambda x: pct_map.get(x, float("nan")))
    t["Change"] = t["Ticker"].map(lambda x: chg_map.get(x, float("nan")))

    # Format display
    def _fmt_price(x):
        try:
            return f"{float(x):,.2f}"
        except Exception:
            return ""

    def _fmt_pct(x):
        try:
            if pd.isna(x):
                return ""
            return f"{float(x):+.2f}%"
        except Exception:
            return ""

    def _fmt_chg(x):
        try:
            if pd.isna(x):
                return ""
            return f"{float(x):+.2f}"
        except Exception:
            return ""

    t["Price"] = t["Price"].map(_fmt_price)
    t["% Change"] = t["% Change"].map(_fmt_pct)
    t["Change"] = t["Change"].map(_fmt_chg)

    # Reorder: put after Ticker
    cols = list(t.columns)
    # Remove if already present in weird spot
    for c in ["Price", "% Change", "Change"]:
        if c in cols:
            cols.remove(c)

    # Place right after Ticker
    if "Ticker" in cols:
        i = cols.index("Ticker") + 1
        cols = cols[:i] + ["Price", "% Change", "Change"] + cols[i:]
    else:
        cols = ["Ticker", "Price", "% Change", "Change"] + cols

    t = t[cols]

    table_html = t.to_html(index=False, escape=True)

    # Wrap in our standard style + nav for consistency
    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Component OHLCV</title>
  {CSS_BLOCK}
</head>
<body style="margin:24px;">
  <div class="gli-nav">
    <a href="./index.html"><b>Home</b></a>
    <a href="./history.html">Historical Values</a>
    <a href="./ohlcv.html">Component OHLCV</a>
  </div>
  <h1 style="margin:0; font-family:system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;">Component OHLCV</h1>
  <div class="gli-mini">Latest snapshot per ticker • Price = Close • % Change vs prior close</div>
  <div style="margin-top:14px;">{table_html}</div>
</body>
</html>
"""

    ohlcv_path.write_text(page, encoding="utf-8")


def build_component_ohlcv_from_prices() -> bool:
    """Build report/ohlcv.html directly from gli_prices.csv with Company + Price + % Change.

    Column order:
      Company, Symbol, Open, High, Low, Price, Change, % Change, Date, Source

    Company name lookup order:
      1) constituents_great_lakes.csv if it contains a name column (often it won't)
      2) company_names.csv cache
      3) yfinance metadata (shortName/longName), cached to company_names.csv
    """
    prices = load_prices_optional()
    if prices is None:
        return False

    df = prices.copy()
    if "Date" not in df.columns or "Ticker" not in df.columns:
        return False

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"])

    # Ensure standard columns exist so the table is stable
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Source"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Previous close for day-over-day change
    df["PrevClose"] = df.groupby("Ticker")["Close"].shift(1)

    latest_date = df["Date"].max()
    latest = df[df["Date"] == latest_date].copy()

    # Price + change fields (Price = Close)
    latest["Price"] = latest["Close"]
    latest["Change"] = latest["Close"] - latest["PrevClose"]
    latest["% Change"] = (latest["Change"] / latest["PrevClose"]) * 100

    # Company + Symbol
    company_map = load_company_map_optional()
    if not company_map:
        cache = load_company_cache()
        tickers = [str(t).strip() for t in latest["Ticker"].tolist()]
        cache = fetch_company_names_yfinance(tickers, existing=cache)
        save_company_cache(cache)
        company_map = cache if cache else None

    latest["Symbol"] = latest["Ticker"]
    latest["Company"] = latest["Ticker"].map(company_map).fillna("") if company_map else ""

    # Formatting helpers
    def f2(x):
        try:
            if pd.isna(x):
                return ""
            return f"{float(x):,.2f}"
        except Exception:
            return ""

    def f0(x):
        try:
            if pd.isna(x):
                return ""
            return f"{int(float(x)):,}"
        except Exception:
            return ""

    def fpct(x):
        try:
            if pd.isna(x):
                return ""
            return f"{float(x):+.2f}%"
        except Exception:
            return ""

    def fchg(x):
        try:
            if pd.isna(x):
                return ""
            return f"{float(x):+.2f}"
        except Exception:
            return ""

    # Format
    latest["Date"] = pd.to_datetime(latest["Date"]).dt.date.astype(str)
    for c in ["Open", "High", "Low", "Price"]:
        latest[c] = latest[c].map(f2)
    latest["Change"] = latest["Change"].map(fchg)
    latest["% Change"] = latest["% Change"].map(fpct)
    latest["Source"] = latest["Source"].fillna("").astype(str)

    # Final column order
    cols = ["Company", "Symbol", "Open", "High", "Low", "Price", "Change", "% Change", "Date", "Source"]
    for col in cols:
        if col not in latest.columns:
            latest[col] = ""
    latest = latest[cols]

    table_html = latest.to_html(index=False, escape=True)

    REPORT.mkdir(parents=True, exist_ok=True)
    out_path = REPORT / "ohlcv.html"
    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Component OHLCV</title>
  {CSS_BLOCK}
</head>
<body style="margin:24px;">
  <div class="gli-nav">
    <a href="./index.html"><b>Home</b></a>
    <a href="./history.html">Historical Values</a>
    <a href="./ohlcv.html">Component OHLCV</a>
  </div>
  <h1 style="margin:0; font-family:system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;">Component OHLCV</h1>
  <div class="gli-mini">Latest snapshot per ticker • Price = Close • % Change vs prior close</div>
  <div style="margin-top:14px;">{table_html}</div>
</body>
</html>
"""
    out_path.write_text(page, encoding="utf-8")
    return True
def build_ticker_line(df: pd.DataFrame) -> str:
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else None

    close = float(last["GLI_Close"])
    high  = float(last["GLI_High"])
    low   = float(last["GLI_Low"])
    date  = last["Date"].date().isoformat()

    chg = 0.0
    pct = 0.0
    arrow = "•"
    if prev is not None:
        prev_close = float(prev["GLI_Close"])
        chg = close - prev_close
        pct = (chg / prev_close) * 100 if prev_close else 0.0
        arrow = "▲" if chg > 0 else ("▼" if chg < 0 else "•")

    sign = "+" if chg > 0 else ""
    return f"GLI {close:,.2f} {arrow}{sign}{chg:,.2f} ({sign}{pct:.2f}%)  H {high:,.2f}  L {low:,.2f}  ({date})"

def write_ticker_txt(df: pd.DataFrame) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "ticker.txt").write_text(build_ticker_line(df) + "\n", encoding="utf-8")

def write_history_html(df: pd.DataFrame) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)

    out = df.copy()
    out["Date"] = out["Date"].dt.date.astype(str)

    keep = ["Date", "GLI_Open", "GLI_High", "GLI_Low", "GLI_Close", "TotalVolume"]
    out = out[keep].copy()

    for c in ["GLI_Open", "GLI_High", "GLI_Low", "GLI_Close"]:
        out[c] = out[c].map(lambda x: f"{float(x):,.2f}")
    out["TotalVolume"] = out["TotalVolume"].map(fmt_int)

    table = out.to_html(index=False, escape=True)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>GLI History</title>
  {CSS_BLOCK}
</head>
<body style="margin:24px;">
  <div class="gli-linkbar"><a href="./index.html">← Back to GLI</a></div>
  <h1 style="margin:0; font-family:system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;">GLI History</h1>
  <div class="gli-mini">Price-weighted • Base {BASE_VALUE:.2f} on {BASE_DATE}</div>
  <div style="margin-top:14px;">{table}</div>
</body>
</html>
"""
    (REPORT / "history.html").write_text(html, encoding="utf-8")

def ensure_css_in_head(html: str) -> str:
    if CSS_BLOCK.strip() in html:
        return html
    if "</head>" in html:
        return html.replace("</head>", CSS_BLOCK + "\n</head>", 1)
    return CSS_BLOCK + "\n" + html

def insert_after_body_open(html: str, block: str) -> str:
    if "<body" not in html:
        return block + "\n" + html
    i = html.find("<body")
    j = html.find(">", i)
    if j == -1:
        return block + "\n" + html
    return html[:j+1] + "\n" + block + "\n" + html[j+1:]

def inject_blocks_into_index(df: pd.DataFrame) -> None:
    index_path = REPORT / "index.html"
    if not index_path.exists():
        raise SystemExit("report/index.html not found. Run the GLI engine first with --report-dir report.")

    html = index_path.read_text(encoding="utf-8")
    html = ensure_css_in_head(html)

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else None

    opn = float(last["GLI_Open"])
    high = float(last["GLI_High"])
    low = float(last["GLI_Low"])
    close = float(last["GLI_Close"])
    date = last["Date"].date().isoformat()
    total_vol = float(last["TotalVolume"])

    chg = 0.0
    pct = 0.0
    if prev is not None:
        prev_close = float(prev["GLI_Close"])
        chg = close - prev_close
        pct = (chg / prev_close) * 100 if prev_close else 0.0

    ticker_block = f"""{TICKER_HTML_MARK}
<div id="gli-tickerbar">
  <div id="gltxt">Loading GLI…</div>
  <div class="gli-pill">The Great Lakes Index (GLI)</div>
</div>
<script>
fetch("ticker.txt", {{ cache: "no-store" }})
  .then(r => r.text())
  .then(t => {{
    const s = t.trim();
    let cls = "flat";
    if (s.includes("▲")) cls = "up";
    else if (s.includes("▼")) cls = "down";
    document.getElementById("gltxt").innerHTML =
      "<span class='" + cls + "'>" +
      s.replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;") +
      "</span>";
  }})
  .catch(() => {{
    document.getElementById("gltxt").textContent = "GLI ticker unavailable";
  }});
</script>
"""

    nav_block = f"""{NAV_BAR_MARK}
<div class="gli-nav">
  <a href="./index.html"><b>Home</b></a>
  <a href="./history.html">Historical Values</a>
  <a href="./ohlcv.html">Component OHLCV</a>
</div>
"""

    symbol_box = f"""{SYMBOL_BOX_MARK}
<div class="gli-symbolbox">
  <div style="display:flex; justify-content:space-between; align-items:flex-end; gap:12px; flex-wrap:wrap;">
    <div>
      <div class="gli-k">Symbol</div>
      <div class="gli-v">GLI</div>
      <div class="gli-mini">Price-weighted • Base {BASE_VALUE:.2f} on {BASE_DATE}</div>
    </div>
    <div style="text-align:right;">
      <div class="gli-k">As of</div>
      <div class="gli-v" style="font-size:16px;">{date}</div>
      <div class="gli-mini">Change: {chg:+.2f} ({pct:+.2f}%)</div>
    </div>
  </div>
  <div style="margin-top:12px;" class="gli-grid">
    <div><div class="gli-k">Open</div><div class="gli-v">{opn:,.2f}</div></div>
    <div><div class="gli-k">High</div><div class="gli-v">{high:,.2f}</div></div>
    <div><div class="gli-k">Low</div><div class="gli-v">{low:,.2f}</div></div>
    <div><div class="gli-k">Close</div><div class="gli-v">{close:,.2f}</div></div>
    <div><div class="gli-k">Total Volume</div><div class="gli-v">{fmt_int(total_vol)}</div></div>
  </div>
</div>
"""

    history_link = f"""{HISTORY_LINK_MARK}
<div class="gli-linkbar"><a href="./history.html">View full history →</a></div>
"""

    # Insert near top of <body> in a nice order (history link, symbol box, ticker bar)
    if HISTORY_LINK_MARK not in html:
        html = insert_after_body_open(html, history_link)
    if SYMBOL_BOX_MARK not in html:
        html = insert_after_body_open(html, symbol_box)
    if TICKER_HTML_MARK not in html:
        html = insert_after_body_open(html, ticker_block)
    if NAV_BAR_MARK not in html:
        html = insert_after_body_open(html, nav_block)

    index_path.write_text(html, encoding="utf-8")

def main() -> None:
    df = load_levels()
    REPORT.mkdir(parents=True, exist_ok=True)
    write_ticker_txt(df)
    write_history_html(df)
    inject_blocks_into_index(df)
    # Prefer rebuilding Component OHLCV directly from gli_prices.csv (reliable Price/% Change)
    if not build_component_ohlcv_from_prices():
        # Fallback: try patching an existing report/ohlcv.html if prices file missing
        enhance_ohlcv_html_with_price_change()
    print("OK: ticker.txt + history.html + enhanced index.html + enhanced ohlcv.html.")

if __name__ == "__main__":
    main()
