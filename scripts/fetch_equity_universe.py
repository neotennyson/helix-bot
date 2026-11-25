"""Fetch daily NSE equity history for the configured universe (read-only).

Usage: python3 scripts/fetch_equity_universe.py [start_date] [end_date]
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helix.broker_data import HistoricalDataError, fetch_dhan_daily_history

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE = ROOT / "config" / (sys.argv[3] if len(sys.argv) > 3 else "equity_universe.json")
OUTPUT_DIR = ROOT / "data" / "equity"
PAUSE_SECONDS = 1.5


def main() -> int:
    start_date = sys.argv[1] if len(sys.argv) > 1 else "2019-01-01"
    end_date = sys.argv[2] if len(sys.argv) > 2 else "2026-09-24"
    manifest = json.loads(UNIVERSE.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for entry in manifest["equities"]:
        symbol = entry["symbol"]
        target = OUTPUT_DIR / f"{symbol}_daily.csv"
        if target.exists() and target.stat().st_size > 500:
            summary.append({"symbol": symbol, "status": "cached", "path": str(target.relative_to(ROOT))})
            continue
        try:
            bars = fetch_dhan_daily_history(
                entry["security_id"],
                start_date,
                end_date,
                exchange_segment=entry.get("exchange_segment", "NSE_EQ"),
                instrument=entry.get("instrument", "EQUITY"),
                include_open_interest=False,
                output=target,
            )
        except HistoricalDataError as error:
            summary.append({"symbol": symbol, "status": "failed", "error": str(error)})
            continue
        summary.append({
            "symbol": symbol,
            "status": "fetched",
            "bars": len(bars),
            "first": bars[0].timestamp.date().isoformat(),
            "last": bars[-1].timestamp.date().isoformat(),
            "path": str(target.relative_to(ROOT)),
        })
        print(f"{symbol:12s} {len(bars):5d} bars  {bars[0].timestamp.date()} -> {bars[-1].timestamp.date()}")
        time.sleep(PAUSE_SECONDS)
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fetched = sum(1 for item in summary if item["status"] in {"fetched", "cached"})
    failed = [item for item in summary if item["status"] == "failed"]
    print(f"ready={fetched} failed={len(failed)}")
    for item in failed:
        print("  failed:", item["symbol"], item.get("error"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())