import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from tokens import get_authorized_get, post_dhan_historical

from .data import IST, parse_timestamp, save_bars
from .models import Bar


class HistoricalDataError(RuntimeError):
    pass


def _as_timestamp(value: Any) -> datetime:
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric > 10_000_000_000:
            numeric /= 1000.0
        return datetime.fromtimestamp(numeric, tz=timezone.utc).astimezone(IST)
    return parse_timestamp(str(value))


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise HistoricalDataError("historical response contains a non-numeric price") from error


def _field(item: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in item and item[name] is not None:
            return item[name]
    raise HistoricalDataError("historical candle is missing a required field")


def _candle_to_bar(item: Any, expiry: str | None) -> Bar:
    if isinstance(item, dict):
        timestamp = _field(item, "timestamp", "time", "datetime", "date")
        open_value = _field(item, "open", "o")
        high_value = _field(item, "high", "h")
        low_value = _field(item, "low", "l")
        close_value = _field(item, "close", "c")
        volume = item.get("volume", item.get("v", item.get("total_traded_volume", 0)))
    elif isinstance(item, (list, tuple)) and len(item) >= 5:
        timestamp, open_value, high_value, low_value, close_value = item[:5]
        volume = item[5] if len(item) > 5 else 0
    else:
        raise HistoricalDataError("historical response contains an invalid candle")
    bar = Bar(
        timestamp=_as_timestamp(timestamp),
        symbol="",
        open=_number(open_value),
        high=_number(high_value),
        low=_number(low_value),
        close=_number(close_value),
        volume=int(_number(volume)),
        expiry=expiry,
    )
    bar.validate()
    return bar


def _candle_rows(payload: Any) -> list[Any]:
    if not isinstance(payload, dict):
        raise HistoricalDataError("historical response must be an object")
    status = str(payload.get("status", "success")).upper()
    if status not in {"SUCCESS", "OK", "TRUE"}:
        raise HistoricalDataError("historical data service returned an unsuccessful status")
    root = payload.get("payload")
    if not isinstance(root, dict):
        root = payload.get("data")
    if isinstance(root, list):
        return root
    if not isinstance(root, dict):
        root = payload
    candles = root.get("candles")
    if isinstance(candles, list):
        return candles
    if isinstance(candles, tuple):
        return list(candles)
    fields = ("timestamp", "open", "high", "low", "close")
    if all(isinstance(root.get(field), list) for field in fields):
        length = min(len(root[field]) for field in fields)
        volume_values = root.get("volume")
        rows = []
        for index in range(length):
            row = {field: root[field][index] for field in fields}
            if isinstance(volume_values, list) and index < len(volume_values):
                row["volume"] = volume_values[index]
            rows.append(row)
        return rows
    raise HistoricalDataError("historical response does not contain candle rows")


def _parse_payload(payload: Any, symbol: str, expiry: str | None, strict: bool = True) -> list[Bar]:
    bars = []
    for item in _candle_rows(payload):
        try:
            bar = _candle_to_bar(item, expiry)
        except (HistoricalDataError, ValueError):
            if strict:
                raise
            continue
        bars.append(Bar(
            timestamp=bar.timestamp,
            symbol=symbol,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
            expiry=bar.expiry,
        ))
    bars.sort(key=lambda item: item.timestamp)
    if not bars:
        raise HistoricalDataError("historical response contained no candles")
    deduplicated: list[Bar] = []
    for bar in bars:
        if deduplicated and deduplicated[-1].timestamp == bar.timestamp:
            if strict:
                raise HistoricalDataError("historical response contains duplicate timestamps")
            continue
        deduplicated.append(bar)
    return deduplicated


def fetch_groww_history(
    symbol: str,
    start_time: str,
    end_time: str,
    interval_minutes: int = 15,
    exchange: str = "NSE",
    segment: str = "FNO",
    expiry: str | None = None,
    output: str | Path | None = None,
) -> list[Bar]:
    query = urlencode({
        "exchange": exchange,
        "segment": segment,
        "groww_symbol": symbol,
        "start_time": start_time,
        "end_time": end_time,
        "candle_interval": f"{interval_minutes}minute",
    })
    url = f"https://api.groww.in/v1/historical/candles?{query}"
    raw = get_authorized_get(url, "groww", {"X-API-VERSION": "1.0"})
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HistoricalDataError("Groww returned invalid JSON") from error
    bars = _parse_payload(payload, symbol, expiry)
    if output is not None:
        save_bars(output, bars)
    return bars


def fetch_dhan_history(
    security_id: str,
    start_time: str,
    end_time: str,
    interval_minutes: int = 15,
    exchange_segment: str = "NSE_FNO",
    instrument: str = "FUTURE",
    include_open_interest: bool = True,
    expiry: str | None = None,
    output: str | Path | None = None,
) -> list[Bar]:
    payload = {
        "securityId": str(security_id),
        "exchangeSegment": exchange_segment,
        "instrument": instrument,
        "interval": str(interval_minutes),
        "fromDate": start_time,
        "toDate": end_time,
        "oi": bool(include_open_interest),
    }
    raw = post_dhan_historical("https://api.dhan.co/v2/charts/intraday", payload)
    try:
        response = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HistoricalDataError("Dhan returned invalid JSON") from error
    bars = _parse_payload(response, str(security_id), expiry)
    if output is not None:
        save_bars(output, bars)
    return bars


def fetch_dhan_daily_history(
    security_id: str,
    start_date: str,
    end_date: str,
    exchange_segment: str = "NSE_FNO",
    instrument: str = "FUTIDX",
    include_open_interest: bool = True,
    expiry: str | None = None,
    output: str | Path | None = None,
) -> list[Bar]:
    payload = {
        "securityId": str(security_id),
        "exchangeSegment": exchange_segment,
        "instrument": instrument,
        "fromDate": start_date,
        "toDate": end_date,
        "oi": bool(include_open_interest),
    }
    raw = post_dhan_historical("https://api.dhan.co/v2/charts/historical", payload)
    try:
        response = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HistoricalDataError("Dhan returned invalid JSON") from error
    bars = _parse_payload(response, str(security_id), expiry, strict=False)
    if output is not None:
        save_bars(output, bars)
    return bars
