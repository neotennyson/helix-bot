"""Fetch hourly OHLCV for a liquid NSE universe via yfinance.

The broker adapters only serve ~7 weeks of intraday history, so intraday
research was previously impossible to validate. Yahoo serves ~2 years of 1h
bars for NSE tickers, which is enough to walk-forward an hourly strategy.

Read-only and research-only: no broker order path is touched.
"""

import json
import sys
import time
from pathlib import Path

import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE = ROOT / "config" / "equity_universe_wide.json"
OUTPUT_DIR = ROOT / "data" / "equity_hourly"
PAUSE_SECONDS = 0.7
INTERVAL = "1h"
PERIOD = "2y"


def main() -> int:
    manifest = json.loads(UNIVERSE.read_text(encoding="utf-8"))
    symbols = sys.argv[1:] or [entry["symbol"] for entry in manifest["equities"]]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_out = []
    for symbol in symbols:
        ticker = f"{symbol}.NS"
        target = OUTPUT_DIR / f"{symbol}_1h.csv"
        try:
            frame = yf.Ticker(ticker).history(period=PERIOD, interval=INTERVAL, auto_adjust=True)
        except Exception as error:  # network or symbol failures must not abort the batch
            manifest_out.append({"symbol": symbol, "status": "failed", "error": f"{type(error).__name__}: {error}"})
            continue
        if frame is None or frame.empty:
            manifest_out.append({"symbol": symbol, "status": "empty"})
            print(f"{symbol:12s} EMPTY", flush=True)
            continue
        frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
        rows = []
        for stamp, row in frame.iterrows():
            rows.append(
                f"{stamp.isoformat()},{symbol},{row['Open']:.4f},{row['High']:.4f},{row['Low']:.4f},{row['Close']:.4f},{int(row['Volume']) if row['Volume'] == row['Volume'] else 0},"
            )
        header = "timestamp,symbol,open,high,low,close,volume,expiry"
        target.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
        manifest_out.append(
            {
                "symbol": symbol,
                "status": "fetched",
                "bars": len(rows),
                "first": str(frame.index[0]),
                "last": str(frame.index[-1]),
            }
        )
        print(f"{symbol:12s} {len(rows):5d} bars  {str(frame.index[0])[:19]} -> {str(frame.index[-1])[:19]}", flush=True)
        time.sleep(PAUSE_SECONDS)
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest_out, indent=2), encoding="utf-8")
    good = sum(1 for item in manifest_out if item["status"] == "fetched")
    bad = [item for item in manifest_out if item["status"] != "fetched"]
    print(f"fetched={good} issues={len(bad)}")
    for item in bad:
        print("  ", item["symbol"], item["status"], item.get("error", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())