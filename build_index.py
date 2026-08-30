#!/usr/bin/env python3
"""
build_index.py
--------------
Builds a custom NSE textile-sector index and writes docs/data/textile_index.json,
which the web viewer (docs/index.html) reads directly. No file uploads.

Run locally:  python3 -m pip install -r requirements.txt ; python3 build_index.py
In CI:        run automatically every weekday after market close (see workflow).

What it produces (one JSON), for BOTH weightings x MULTIPLE timeframes:
    data.capmcap.{1h,1D,1W,1M}   and   data.equal.{1h,1D,1W,1M}
so the chart can switch weighting and timeframe with no rebuild.

Method — a "fixed basket that rebalances monthly" (this is close to how real
indexes work; see README):
    * On each rebalance date we split a notional pot across the constituents by
      target weight (equal, or capped market-cap), buying a fixed number of
      "units" of each. Between rebalances those units are held, so each name's
      weight drifts with its price exactly like a real market-cap index.
    * Index level = sum(units_i * price_i). Continuous across rebalances (no
      artificial jumps). New listings join at the next rebalance; dead names
      drop out. Missing days are bridged by forward-fill.
    * Index CLOSE is sound at every timeframe. Intraday/daily candle WICKS are
      approximations (constituents don't peak at the same instant); weekly and
      monthly candles are resampled from the daily close series, so THEIR wicks
      are sound.
"""

import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit("Missing dependency. Run:  python3 -m pip install -r requirements.txt")

# ----------------------------------------------------------------------------- 
# CONSTANTS — tune these
# -----------------------------------------------------------------------------
INDEX_NAME  = "TEXINDEX"
BASE_VALUE  = 1000.0        # arbitrary starting level, like Nifty's 1000
WEIGHT_CAP  = 0.10          # single-name cap for the market-cap mode
REBALANCE   = "M"           # rebalance weights monthly
DAILY_LOOKBACK  = "5y"      # history for daily/weekly/monthly
HOURLY_LOOKBACK = "730d"    # yfinance limit for 60-minute data (~2y)
DMA_SHORT, DMA_LONG = 20, 50
BIGMOVE_WIN, BIGMOVE_SIGMA = 20, 2.0
MIN_ROWS = 60
OUT_JSON = "docs/data/textile_index.json"

# Constituents — bare NSE symbols (".NS" appended for yfinance). Grouped so you
# can prune/swap. The script prints anything it can't download so you can fix it.
TICKERS = [
    # Home textiles / made-ups
    "WELSPUNLIV", "TRIDENT", "ICIL", "HIMATSEIDE", "GHCLTEXTIL",
    # Cotton spinning / yarn
    "VTL", "NITINSPIN", "NAHARSPING", "AMBIKCO", "SPORTKING",
    "SANGAMIND", "RSWM", "BANSWRAS", "SUTLEJTEX", "PRECOT",
    "NAHARINDUS", "SURYALAXMI",
    # Apparel / garments / retail
    "PAGEIND", "KPRMILL", "GOKEX", "SPAL", "PGIL",
    "KITEX", "LUXIND", "DOLLAR", "RUPA", "KKCL",
    "MONTECARLO", "SIYSIL", "ARVINDFASN", "CANTABIL",
    # Branded lifestyle (recent demergers)
    "ABFRL", "ABLBL", "RAYMONDLSL",
    # Fabric / processing / diversified textile
    "ARVIND", "RAYMOND", "BOMDYEING", "ALOKINDS", "DONEAR",
    "JINDWORLD", "FILATEX", "SANATHAN", "GANECOS",
    # Technical textiles
    "GARFIBRES", "SARLAPOLY",
    # Man-made fibre / synthetics
    "CENTENKA", "INDORAMA", "MAYURUNIQ", "VISHAL", "MAFATIND",
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def cap_weights(w, cap):
    w = w.astype(float).copy()
    for _ in range(100):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = ~over
        pool = w[under].sum()
        if pool <= 0:
            break
        w[under] += excess * (w[under] / pool)
    return w / w.sum()


def get_shares(yf_symbol):
    try:
        fi = yf.Ticker(yf_symbol).fast_info
        for key in ("shares", "sharesOutstanding", "shares_outstanding"):
            try:
                val = fi[key]
            except (KeyError, TypeError):
                val = getattr(fi, key, None)
            if val:
                return float(val)
    except Exception:
        pass
    return None


def target_weights(active, ref_px, shares, weighting, cap):
    if weighting == "equal" or shares is None:
        w = pd.Series(1.0, index=active)
        return w / w.sum()
    mcap = {s: shares[s] * ref_px[s] for s in active
            if shares.get(s) and s in ref_px and pd.notna(ref_px[s])}
    if not mcap:
        w = pd.Series(1.0, index=active)
        return w / w.sum()
    w = pd.Series(mcap)
    w = cap_weights(w / w.sum(), cap)
    return w  # names without market cap are simply excluded this period


def build_basket_ohlc(close, open_, high, low, shares, weighting, cap, base):
    """Fixed-basket, monthly-rebalanced index O/H/L/C DataFrame."""
    close = close.sort_index().ffill(limit=5)
    open_ = open_.reindex_like(close).ffill(limit=5)
    high  = high.reindex_like(close).ffill(limit=5)
    low   = low.reindex_like(close).ffill(limit=5)

    idx = close.index
    periods = pd.PeriodIndex(idx, freq=REBALANCE)
    is_rebal = np.r_[True, periods[1:] != periods[:-1]]

    units = None
    O = np.full(len(idx), np.nan)
    H = np.full(len(idx), np.nan)
    L = np.full(len(idx), np.nan)
    C = np.full(len(idx), np.nan)

    for i in range(len(idx)):
        if is_rebal[i] or units is None:
            ref_level = base if (i == 0 or np.isnan(C[i - 1])) else C[i - 1]
            pxref = close.iloc[i - 1] if i > 0 else close.iloc[i]
            active = [s for s in close.columns
                      if pd.notna(pxref.get(s)) and pd.notna(close.iloc[i].get(s))]
            if not active:
                continue
            w = target_weights(active, pxref.to_dict(), shares, weighting, cap)
            units = (w * ref_level) / pxref[w.index]
        u = units
        C[i] = float((u * close.iloc[i].reindex(u.index)).sum())
        O[i] = float((u * open_.iloc[i].reindex(u.index)).sum())
        H[i] = float((u * high.iloc[i].reindex(u.index)).sum())
        L[i] = float((u * low.iloc[i].reindex(u.index)).sum())

    df = pd.DataFrame({"o": O, "h": H, "l": L, "c": C}, index=idx).dropna()
    # keep candles coherent
    df["h"] = df[["h", "o", "c"]].max(axis=1)
    df["l"] = df[["l", "o", "c"]].min(axis=1)
    return df.round(2)


def resample_ohlc_from_close(close_series, rule):
    s = close_series.dropna()
    out = pd.DataFrame({
        "o": s.resample(rule).first(),
        "h": s.resample(rule).max(),
        "l": s.resample(rule).min(),
        "c": s.resample(rule).last(),
    }).dropna()
    return out.round(2)


def big_move_flags(close_series):
    r = close_series.pct_change()
    sd = r.rolling(BIGMOVE_WIN).std()
    return (r.abs() > BIGMOVE_SIGMA * sd).fillna(False)


def daily_extras(close_panel, ret_panel):
    """Breadth (% above DMAs) and advance/decline for the daily timeframe."""
    n = close_panel.notna().sum(axis=1).replace(0, np.nan)
    br20 = ((close_panel > close_panel.rolling(DMA_SHORT).mean()).sum(axis=1) / n * 100).round(1)
    br50 = ((close_panel > close_panel.rolling(DMA_LONG).mean()).sum(axis=1) / n * 100).round(1)
    adv = (ret_panel > 0).sum(axis=1)
    dec = (ret_panel < 0).sum(axis=1)
    return br20, br50, adv, dec


def daily_records(df, br20, br50, adv, dec):
    big = big_move_flags(df["c"])
    recs = []
    for t, row in df.iterrows():
        recs.append({
            "t": t.strftime("%Y-%m-%d"),
            "o": row["o"], "h": row["h"], "l": row["l"], "c": row["c"],
            "br20": _num(br20.get(t)), "br50": _num(br50.get(t)),
            "adv": _int(adv.get(t)), "dec": _int(dec.get(t)),
            "big": int(bool(big.get(t, False))),
        })
    return recs


def period_records(df, breadth_daily=None):
    big = big_move_flags(df["c"])
    recs = []
    for t, row in df.iterrows():
        rec = {"t": t.strftime("%Y-%m-%d"),
               "o": row["o"], "h": row["h"], "l": row["l"], "c": row["c"],
               "big": int(bool(big.get(t, False)))}
        if breadth_daily is not None:
            # breadth as of the period end (last available daily reading <= t)
            upto = breadth_daily.loc[:t]
            if len(upto):
                rec["br50"] = _num(upto.iloc[-1])
        recs.append(rec)
    return recs


def hourly_records(df):
    return [{"t": int(t.timestamp()), "o": r["o"], "h": r["h"], "l": r["l"], "c": r["c"]}
            for t, r in df.iterrows()]


def _num(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def _int(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else int(x)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def download(interval, period, symbols):
    raw = yf.download(symbols, period=period, interval=interval, auto_adjust=True,
                      group_by="column", threads=True, progress=False)
    if raw.empty:
        return None
    return {f: raw[f].copy() for f in ("Open", "High", "Low", "Close")}


def main():
    yf_symbols = [t + ".NS" for t in TICKERS]
    disp = {t + ".NS": t for t in TICKERS}

    print(f"Downloading daily ({DAILY_LOOKBACK}) for {len(yf_symbols)} tickers...")
    d = download("1d", DAILY_LOOKBACK, yf_symbols)
    if d is None:
        sys.exit("No daily data returned.")

    close_d = d["Close"]
    good, dropped = [], []
    for s in yf_symbols:
        if s in close_d.columns and close_d[s].notna().sum() >= MIN_ROWS:
            good.append(s)
        else:
            dropped.append(disp.get(s, s))
    # Retry dropped tickers one-by-one — bulk downloads sometimes return a few
    # empty due to Yahoo throttling; individual fetches usually recover them.
    if dropped:
        print(f"  Retrying {len(dropped)} dropped ticker(s) individually...")
        still_dropped = []
        for name in list(dropped):
            s = name + ".NS"
            try:
                r = yf.download(s, period=DAILY_LOOKBACK, interval="1d",
                                auto_adjust=True, progress=False)
                if not r.empty and r["Close"].notna().sum() >= MIN_ROWS:
                    for f in ("Open", "High", "Low", "Close"):
                        col = r[f]
                        d[f][s] = col.iloc[:, 0] if hasattr(col, "columns") else col
                    good.append(s)
                    print(f"    recovered {name}")
                else:
                    still_dropped.append(name)
            except Exception:
                still_dropped.append(name)
        dropped = still_dropped

    if dropped:
        print("  ! Still skipped (genuinely no data): " + ", ".join(sorted(dropped)))
    if not good:
        sys.exit("No usable tickers.")
    print(f"  {len(good)} constituents usable.")

    good = [s for s in yf_symbols if s in good]  # keep original order
    for k in d:
        d[k] = d[k][good]

    print("Fetching shares outstanding for market-cap weights...")
    shares = {s: get_shares(s) for s in good}
    missing = [disp[s] for s in good if not shares.get(s)]
    if missing:
        print("  ! No market cap (excluded from cap weighting): " + ", ".join(missing))

    print(f"Downloading hourly ({HOURLY_LOOKBACK})...")
    h = download("60m", HOURLY_LOOKBACK, good)

    payload = {}
    for weighting in ("capmcap", "equal"):
        print(f"Building '{weighting}' ...")
        daily = build_basket_ohlc(d["Close"], d["Open"], d["High"], d["Low"],
                                  shares, weighting, WEIGHT_CAP, BASE_VALUE)
        ret_panel = d["Close"].pct_change()
        br20, br50, adv, dec = daily_extras(d["Close"], ret_panel)

        weekly = resample_ohlc_from_close(daily["c"], "W-FRI")
        monthly = resample_ohlc_from_close(daily["c"], "ME")

        tf = {
            "1D": daily_records(daily, br20, br50, adv, dec),
            "1W": period_records(weekly, breadth_daily=br50),
            "1M": period_records(monthly, breadth_daily=br50),
        }
        if h is not None:
            hourly = build_basket_ohlc(h["Close"], h["Open"], h["High"], h["Low"],
                                       shares, weighting, WEIGHT_CAP, BASE_VALUE)
            tf["1h"] = hourly_records(hourly)
        payload[weighting] = tf

    out = {
        "name": INDEX_NAME,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "base_value": BASE_VALUE,
        "cap": WEIGHT_CAP,
        "constituents_used": [disp[s] for s in good],
        "constituents_dropped": sorted(dropped),
        "timeframes": ["1h", "1D", "1W", "1M"],
        "data": payload,
    }
    import os
    os.makedirs("docs/data", exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, separators=(",", ":"))

    last = payload["capmcap"]["1D"][-1]
    print(f"\nDone. {len(good)} constituents. Latest {last['t']}: {last['c']} "
          f"(breadth {last.get('br50')}% > 50DMA). Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
