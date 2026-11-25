"""Multi-instrument universe scan and equal-weight portfolio aggregation.

Symbols are scanned independently with the same engine and cost model used by
single-symbol research. Portfolio metrics are built from an equal-weight blend of
each sleeve's capital base, so a portfolio number here means "capital split
equally across the included symbols", not a single shared account.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backtest import BacktestEngine, calculate_metrics
from .config import BacktestConfig
from .data import load_bars
from .models import Bar, EquityPoint, Trade


@dataclass(frozen=True)
class UniverseMember:
    symbol: str
    security_id: str
    path: Path
    exchange_segment: str = "NSE_EQ"
    instrument: str = "EQUITY"


def load_universe(manifest_path: str | Path, data_dir: str | Path) -> list[UniverseMember]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    root = Path(data_dir)
    members: list[UniverseMember] = []
    for entry in manifest["equities"]:
        symbol = entry["symbol"]
        path = root / f"{symbol}_daily.csv"
        if not path.exists():
            continue
        members.append(
            UniverseMember(
                symbol=symbol,
                security_id=str(entry["security_id"]),
                path=path,
                exchange_segment=entry.get("exchange_segment", "NSE_EQ"),
                instrument=entry.get("instrument", "EQUITY"),
            )
        )
    if not members:
        raise ValueError("universe manifest did not resolve to any data files")
    return members


def scan_symbol(
    template: dict[str, Any],
    member: UniverseMember,
    mode: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = json.loads(json.dumps(template))
    payload["symbol"] = member.security_id
    payload["strategy_mode"] = mode
    for key, value in (overrides or {}).items():
        if key in {"strategy", "risk", "costs"} and isinstance(value, dict):
            payload.setdefault(key, {})
            payload[key].update(value)
        else:
            payload[key] = value
    config = BacktestConfig.from_dict(payload)
    bars = load_bars(member.path)
    result = BacktestEngine(config).run(bars)
    metrics = result.metrics
    return {
        "symbol": member.symbol,
        "security_id": member.security_id,
        "strategy_mode": mode,
        "profile": overrides.get("label", "base") if isinstance(overrides, dict) else "base",
        "bars": len(bars),
        "first_bar": bars[0].timestamp.isoformat() if bars else None,
        "last_bar": bars[-1].timestamp.isoformat() if bars else None,
        "metrics": metrics,
        "equity_curve": result.equity_curve,
        "trades": result.trades,
        "initial_capital": result.initial_capital,
    }


def _daily_returns(curve: list[EquityPoint], initial_capital: float) -> dict[Any, float]:
    by_day: dict[Any, float] = {}
    for point in curve:
        by_day[point.timestamp.date()] = point.equity
    returns: dict[Any, float] = {}
    previous = initial_capital
    for day, value in by_day.items():
        returns[day] = value / previous - 1.0 if previous > 0 else 0.0
        previous = value
    return returns


def aggregate_portfolio(sleeves: list[dict[str, Any]], bar_minutes: int = 1440) -> dict[str, Any]:
    """Blend sleeve equity curves with equal weight on the union of timestamps."""
    if not sleeves:
        return {"status": "empty_portfolio"}
    if any(float(sleeve["initial_capital"]) <= 0 for sleeve in sleeves):
        return {"status": "empty_portfolio"}
    timestamp_equity: dict[Any, list[float]] = {}
    timestamp_price: dict[Any, list[float]] = {}
    for sleeve in sleeves:
        base = float(sleeve["initial_capital"])
        for point in sleeve["equity_curve"]:
            timestamp_equity.setdefault(point.timestamp, []).append(point.equity / base)
            timestamp_price.setdefault(point.timestamp, []).append(point.price)
    # Weights are renormalised over the sleeves that actually have data at each
    # timestamp, so a late listing does not read as a portfolio loss.
    equity_by_timestamp: dict[Any, float] = {}
    price_by_timestamp: dict[Any, float] = {}
    contributors_by_timestamp: dict[Any, int] = {}
    for timestamp, values in timestamp_equity.items():
        contributors_by_timestamp[timestamp] = len(values)
        equity_by_timestamp[timestamp] = sum(values) / len(values)
        price_by_timestamp[timestamp] = sum(timestamp_price[timestamp]) / len(timestamp_price[timestamp])
    curve: list[EquityPoint] = []
    peak = 0.0
    for timestamp in sorted(equity_by_timestamp):
        equity = equity_by_timestamp[timestamp]
        price = price_by_timestamp[timestamp]
        peak = max(peak, equity)
        drawdown = equity / peak - 1.0 if peak else 0.0
        curve.append(EquityPoint(timestamp, equity, drawdown, price, 0))
    trades: list[Trade] = []
    for sleeve in sleeves:
        for trade in sleeve["trades"]:
            trades.append(trade)
    metrics = calculate_metrics(curve, trades, 1.0, bar_minutes)
    returns_by_symbol = {
        sleeve["symbol"]: _daily_returns(sleeve["equity_curve"], float(sleeve["initial_capital"]))
        for sleeve in sleeves
    }
    correlations = _correlation_matrix(returns_by_symbol)
    metrics["average_pairwise_correlation"] = _average_off_diagonal(correlations)
    metrics["sleeves"] = len(sleeves)
    metrics["min_contributing_sleeves"] = min(contributors_by_timestamp.values()) if contributors_by_timestamp else 0
    metrics["max_contributing_sleeves"] = max(contributors_by_timestamp.values()) if contributors_by_timestamp else 0
    metrics["worst_symbol_cagr"] = min((float(s["metrics"].get("cagr", 0.0)) for s in sleeves), default=0.0)
    metrics["profitable_symbol_share"] = (
        sum(1 for sleeve in sleeves if float(sleeve["metrics"].get("total_return", 0.0)) > 0.0) / len(sleeves)
        if sleeves
        else 0.0
    )
    return {
        "status": "ok",
        "weighting": "equal_weight_renormalised_over_available_sleeves",
        "metrics": metrics,
        "correlation_matrix": correlations,
        "weight_by_symbol": {symbol: 1.0 / len(sleeves) for symbol in sorted({s["symbol"] for s in sleeves})},
        "equity_curve": curve,
        "trades": trades,
    }


def _correlation_matrix(returns_by_symbol: dict[str, dict[Any, float]]) -> dict[str, dict[str, float]]:
    symbols = sorted(returns_by_symbol)
    matrix: dict[str, dict[str, float]] = {symbol: {} for symbol in symbols}
    for left in symbols:
        for right in symbols:
            if left == right:
                matrix[left][right] = 1.0
                continue
            days = sorted(set(returns_by_symbol[left]) & set(returns_by_symbol[right]))
            left_series = [returns_by_symbol[left][day] for day in days]
            right_series = [returns_by_symbol[right][day] for day in days]
            matrix[left][right] = _pearson(left_series, right_series)
    return matrix


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) < 3:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    covariance = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_var = sum((a - left_mean) ** 2 for a in left)
    right_var = sum((b - right_mean) ** 2 for b in right)
    if left_var <= 0 or right_var <= 0:
        return 0.0
    return covariance / ((left_var ** 0.5) * (right_var ** 0.5))


def _average_off_diagonal(matrix: dict[str, dict[str, float]]) -> float:
    values: list[float] = []
    symbols = sorted(matrix)
    for index, left in enumerate(symbols):
        for right in symbols[index + 1 :]:
            values.append(matrix[left][right])
    return sum(values) / len(values) if values else 0.0


def rank_rows(scans: list[dict[str, Any]], min_trades: int = 20, min_sharpe: float = 0.5) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scan in scans:
        metrics = scan["metrics"]
        trades = int(metrics.get("trades", 0) or 0)
        sharpe = float(metrics.get("sharpe", 0.0) or 0.0)
        cagr = float(metrics.get("cagr", 0.0) or 0.0)
        profit_factor = metrics.get("profit_factor")
        max_drawdown = float(metrics.get("max_drawdown", 0.0) or 0.0)
        qualifies = bool(
            trades >= min_trades
            and sharpe >= min_sharpe
            and cagr > 0.06
            and (profit_factor is None or float(profit_factor) >= 1.10)
            and max_drawdown > -0.15
        )
        rows.append(
            {
                "symbol": scan["symbol"],
                "strategy_mode": scan["strategy_mode"],
                "profile": scan.get("profile", "base"),
                "total_return": metrics.get("total_return"),
                "cagr": cagr,
                "sharpe": sharpe,
                "sortino": metrics.get("sortino"),
                "max_drawdown": max_drawdown,
                "profit_factor": profit_factor,
                "trades": trades,
                "win_rate": metrics.get("win_rate"),
                "exposure_fraction": metrics.get("exposure_fraction"),
                "benchmark_return": metrics.get("benchmark_return"),
                "qualifies": qualifies,
            }
        )
    rows.sort(key=lambda row: (row["qualifies"], row["sharpe"], row["cagr"]), reverse=True)
    return rows


def strip_curves(scans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in scan.items() if key not in {"equity_curve", "trades"}}
        for scan in scans
    ]


def bars_for(member: UniverseMember) -> list[Bar]:
    return load_bars(member.path)
