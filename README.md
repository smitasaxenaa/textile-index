# TEXINDEX — custom NSE textile-sector index

A self-updating index of ~50 NSE textile stocks. A GitHub Action rebuilds the
data every weekday after market close; GitHub Pages serves an interactive chart
(candles + breadth) with switches for **timeframe** (1H / 1D / 1W / 1M),
**weighting** (market-cap / equal), and **theme** (dark / light).

## How it works
- `build_index.py` downloads constituent prices via `yfinance`, builds the index
  with a fixed-basket, monthly-rebalanced method (see the docstring), and writes
  `docs/data/textile_index.json` for both weightings across all timeframes.
- `docs/index.html` reads that JSON directly and renders it — no uploads.
- `.github/workflows/build.yml` runs the builder on a weekday cron and commits
  the refreshed JSON, so Pages updates itself.

## Run locally
```
python3 -m pip install -r requirements.txt
python3 build_index.py          # writes docs/data/textile_index.json
```
Then open `docs/index.html` (or serve the `docs/` folder).

## Method vs a real index (e.g. Nifty 50)
Same core idea — weight stocks by size, blend their moves, express the result
against an arbitrary base value (Nifty uses 1000; so do we). Differences: Nifty
uses **free-float** market cap (only publicly tradable shares) and a **divisor**
adjusted continuously for corporate actions, is computed in real time, and
rebalances membership semi-annually under strict rules. Ours uses total shares
from yfinance, a monthly rebalance, adjusted prices instead of a divisor, and a
hand-picked constituent list. Close values are exact; 1H/1D candle wicks are
approximate; 1W/1M wicks are exact (resampled from daily closes).

## Tuning
Edit the constants at the top of `build_index.py`: `TICKERS`, `WEIGHT_CAP`,
`BASE_VALUE`, `REBALANCE`, lookbacks, DMA windows, big-move sensitivity.
