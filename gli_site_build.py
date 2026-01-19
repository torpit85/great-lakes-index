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
    print("OK: ticker.txt + history.html + enhanced index.html (includes Total Volume).")

if __name__ == "__main__":
    main()
