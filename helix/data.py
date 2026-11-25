import csv
import math
import random
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from .models import Bar


IST = ZoneInfo("Asia/Kolkata")


def parse_timestamp(value: str) -> datetime:
    text = value.strip()
    if not text:
        raise ValueError("timestamp is empty")
    if text.replace(".", "", 1).isdigit():
        timestamp = datetime.fromtimestamp(float(text), tz=ZoneInfo("UTC"))
    else:
        normalized = text.replace("Z", "+00:00")
        timestamp = datetime.fromisoformat(normalized)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=IST)
    return timestamp.astimezone(IST)


def parse_date(value: str) -> date:
    return date.fromisoformat(value.strip())


def parse_time(value: str) -> time:
    return time.fromisoformat(value.strip())


def _value(row: dict[str, str], *names: str) -> str:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    raise ValueError(f"missing one of columns: {', '.join(names)}")


def load_bars(path: str | Path, symbol: str | None = None) -> list[Bar]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(str(source))
    bars: list[Bar] = []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        for row_number, row in enumerate(reader, start=2):
            try:
                timestamp_columns = ("timestamp", "datetime", "date_time")
                if any(row.get(column) for column in timestamp_columns):
                    timestamp_value = _value(row, *timestamp_columns)
                else:
                    timestamp_value = f"{_value(row, 'date')} {_value(row, 'time')}"
                timestamp = parse_timestamp(timestamp_value)
                row_symbol = row.get("symbol", symbol or "NIFTY").strip() or (symbol or "NIFTY")
                if symbol is not None and row_symbol != symbol:
                    continue
                expiry = row.get("expiry") or None
                bar = Bar(
                    timestamp=timestamp,
                    symbol=row_symbol,
                    open=float(_value(row, "open", "o")),
                    high=float(_value(row, "high", "h")),
                    low=float(_value(row, "low", "l")),
                    close=float(_value(row, "close", "c")),
                    volume=int(float(row.get("volume", 0) or 0)),
                    expiry=expiry,
                )
                bar.validate()
                bars.append(bar)
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid row {row_number}: {error}") from error
    bars.sort(key=lambda item: item.timestamp)
    if not bars:
        raise ValueError("no bars matched the requested symbol")
    for previous, current in zip(bars, bars[1:]):
        if current.timestamp == previous.timestamp:
            raise ValueError("duplicate timestamp in bar data")
    return bars


def save_bars(path: str | Path, bars: list[Bar]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = ["timestamp", "symbol", "open", "high", "low", "close", "volume", "expiry"]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for bar in bars:
            writer.writerow(
                {
                    "timestamp": bar.timestamp.isoformat(),
                    "symbol": bar.symbol,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "expiry": bar.expiry or "",
                }
            )


def download_yfinance_proxy(
    ticker: str,
    output: str | Path,
    symbol: str | None = None,
    period: str = "60d",
    interval: str = "15m",
) -> list[Bar]:
    import yfinance as yf

    raw = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False, group_by="column")
    if raw is None or raw.empty:
        raise ValueError("the data provider returned no rows")
    if getattr(raw.columns, "nlevels", 1) > 1:
        if ticker in raw.columns.get_level_values(0):
            raw = raw.xs(ticker, axis=1, level=0)
        else:
            raw = raw.xs(ticker, axis=1, level=1)
    rows: list[Bar] = []
    row_symbol = symbol or ticker.replace("^", "")
    for timestamp, row in raw.iterrows():
        close = row.get("Close")
        if close is None or not math.isfinite(float(close)):
            continue
        parsed_timestamp = timestamp.to_pydatetime()
        if parsed_timestamp.tzinfo is None:
            parsed_timestamp = parsed_timestamp.replace(tzinfo=IST)
        bar = Bar(
            timestamp=parsed_timestamp.astimezone(IST),
            symbol=row_symbol,
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(close),
            volume=int(float(row.get("Volume", 0) or 0)),
        )
        bar.validate()
        rows.append(bar)
    rows.sort(key=lambda item: item.timestamp)
    save_bars(output, rows)
    return rows


def filter_session(bars: list[Bar], start: str = "09:15", end: str = "15:30") -> list[Bar]:
    start_time = parse_time(start)
    end_time = parse_time(end)
    return [bar for bar in bars if start_time <= bar.timestamp.timetz().replace(tzinfo=None) <= end_time]


def generate_demo_bars(
    symbol: str = "NIFTY",
    days: int = 90,
    interval_minutes: int = 15,
    seed: int = 17,
    start_date: date | None = None,
) -> list[Bar]:
    if days < 30:
        raise ValueError("days must be at least 30")
    if interval_minutes <= 0 or 24 * 60 % interval_minutes != 0:
        raise ValueError("interval_minutes must divide a day")
    generator = random.Random(seed)
    current_date = start_date or date(2024, 1, 1)
    price = 22000.0
    bars: list[Bar] = []
    for day_index in range(days):
        while current_date.weekday() >= 5:
            current_date += timedelta(days=1)
        phase = day_index % 30
        trend = 0.0008 if 8 <= phase <= 18 else (-0.0004 if 22 <= phase <= 27 else 0.00005)
        volatility = 0.0012 if 8 <= phase <= 18 else 0.0007
        open_price = price
        for slot in range(int(9.5 * 60 / interval_minutes), int(15.5 * 60 / interval_minutes) + 1):
            timestamp = datetime.combine(current_date, time(0, 0), tzinfo=IST) + timedelta(minutes=slot * interval_minutes)
            move = generator.gauss(trend, volatility)
            open_value = price
            close_value = max(1.0, open_value * (1.0 + move))
            spread = abs(generator.gauss(0.0007, 0.0002))
            high_value = max(open_value, close_value) * (1.0 + spread)
            low_value = min(open_value, close_value) * (1.0 - spread)
            volume = int(max(100, generator.gauss(900, 180)))
            bars.append(
                Bar(
                    timestamp=timestamp,
                    symbol=symbol,
                    open=round(open_value, 2),
                    high=round(high_value, 2),
                    low=round(low_value, 2),
                    close=round(close_value, 2),
                    volume=volume,
                )
            )
            price = close_value
        current_date += timedelta(days=1)
    return bars


def infer_interval_minutes(bars: list[Bar]) -> int:
    if len(bars) < 2:
        return 0
    differences = [
        (current.timestamp - previous.timestamp).total_seconds() / 60.0
        for previous, current in zip(bars, bars[1:])
        if current.timestamp.date() == previous.timestamp.date()
    ]
    if not differences:
        return 0
    return int(round(sorted(differences)[len(differences) // 2]))


def is_market_bar(bar: Bar) -> bool:
    current = bar.timestamp.timetz().replace(tzinfo=None)
    return time(9, 15) <= current <= time(15, 30)


def finite_or_zero(value: float) -> float:
    return value if math.isfinite(value) else 0.0
