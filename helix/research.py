from dataclasses import asdict, replace
from datetime import time
from typing import Any

from .backtest import BacktestEngine
from .config import BacktestConfig
from .data import infer_interval_minutes
from .models import Bar
from .optimize import WalkForwardOptimizer


def inspect_data(bars: list[Bar], symbol: str, expected_interval: int = 15) -> dict[str, Any]:
    if not bars:
        return {"status": "fail", "issues": ["no bars"]}
    issues: list[str] = []
    warnings: list[str] = []
    timestamps = [bar.timestamp for bar in bars]
    duplicate_count = len(timestamps) - len(set(timestamps))
    if duplicate_count:
        issues.append(f"{duplicate_count} duplicate timestamps")
    interval = infer_interval_minutes(bars)
    daily = expected_interval >= 1440 or interval == 0
    if expected_interval and interval and interval != expected_interval:
        warnings.append(f"median interval is {interval} minutes, expected {expected_interval}")
    out_of_session = 0
    gaps = 0
    if not daily:
        for bar in bars:
            current = bar.timestamp.timetz().replace(tzinfo=None)
            if not (time(9, 15) <= current <= time(15, 30)):
                out_of_session += 1
        for previous, current in zip(bars, bars[1:]):
            if previous.timestamp.date() == current.timestamp.date():
                minutes = (current.timestamp - previous.timestamp).total_seconds() / 60.0
                if expected_interval and minutes > expected_interval * 1.5:
                    gaps += 1
    if out_of_session:
        warnings.append(f"{out_of_session} bars are outside the regular session")
    if gaps:
        warnings.append(f"{gaps} intraday gaps exceed 1.5x the expected interval")
    if all(bar.volume == 0 for bar in bars):
        warnings.append("volume is zero for every bar")
    if not any(bar.expiry for bar in bars):
        warnings.append("expiry metadata is missing")
    span_days = (timestamps[-1].date() - timestamps[0].date()).days
    if span_days < 120:
        warnings.append("history is shorter than 120 calendar days")
    status = "fail" if issues else "warning" if warnings else "pass"
    return {
        "status": status,
        "symbol": symbol,
        "bars": len(bars),
        "calendar_days": span_days,
        "median_interval_minutes": 1440 if daily and interval == 0 else interval,
        "duplicate_timestamps": duplicate_count,
        "out_of_session_bars": out_of_session,
        "intraday_gaps": gaps,
        "issues": issues,
        "warnings": warnings,
    }


def rolling_window_results(config: BacktestConfig, bars: list[Bar], strategy, window_bars: int = 510) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for start in range(0, max(0, len(bars) - window_bars + 1), window_bars):
        window = bars[start:start + window_bars]
        if len(window) < window_bars:
            break
        metrics = BacktestEngine(replace(config, strategy=strategy)).run(window).metrics
        results.append({
            "start": window[0].timestamp.isoformat(),
            "end": window[-1].timestamp.isoformat(),
            "total_return": metrics.get("total_return"),
            "profit_factor": metrics.get("profit_factor"),
            "max_drawdown": metrics.get("max_drawdown"),
            "trades": metrics.get("trades"),
        })
    return results


def run_research(config: BacktestConfig, bars: list[Bar], data_source: str = "unspecified") -> dict[str, Any]:
    quality = inspect_data(bars, config.symbol, config.bar_minutes)
    optimizer = WalkForwardOptimizer(config)
    validation = optimizer.run(bars)
    selected_parameters = validation["selected_parameters"]
    strategy_values = dict(asdict(config.strategy))
    strategy_values.update(selected_parameters)
    selected_strategy = type(config.strategy)(**strategy_values)
    window_bars = 510 if config.bar_minutes >= 1440 else 250
    rolling_windows = rolling_window_results(config, bars, selected_strategy, window_bars)
    positive_rolling_windows = sum(1 for item in rolling_windows if (item.get("total_return") or 0.0) > 0.0)
    stress_specs = {
        "base": {"slippage_bps": config.costs.slippage_bps, "cost_multiplier": 1.0},
        "adverse": {"slippage_bps": config.costs.slippage_bps * 2.0, "cost_multiplier": 1.5},
        "severe": {"slippage_bps": config.costs.slippage_bps * 3.0, "cost_multiplier": 2.0},
    }
    stress_results: dict[str, Any] = {}
    for name, spec in stress_specs.items():
        stressed_costs = replace(
            config.costs,
            slippage_bps=float(spec["slippage_bps"]),
            brokerage_pct=config.costs.brokerage_pct * float(spec["cost_multiplier"]),
            exchange_txn_pct=config.costs.exchange_txn_pct * float(spec["cost_multiplier"]),
            stt_sell_pct=config.costs.stt_sell_pct * float(spec["cost_multiplier"]),
            stamp_buy_pct=config.costs.stamp_buy_pct * float(spec["cost_multiplier"]),
            sebi_pct=config.costs.sebi_pct * float(spec["cost_multiplier"]),
        )
        stressed_config = replace(config, strategy=selected_strategy, costs=stressed_costs)
        result = BacktestEngine(stressed_config).run(bars)
        stress_results[name] = {
            "metrics": result.metrics,
            "trades": len(result.trades),
        }
    holdout_metrics = validation.get("holdout_metrics", {})
    severe_metrics = stress_results.get("severe", {}).get("metrics", {})
    minimum_positive_windows = max(3, int(len(rolling_windows) * 0.8))
    cash_hurdle = config.risk.cash_hurdle_annual
    holdout_excess_cagr = holdout_metrics.get("cagr", -1.0) - cash_hurdle
    paper_review_candidate = (
        holdout_metrics.get("total_return", -1.0) > 0.0
        and holdout_excess_cagr > 0.0
        and holdout_metrics.get("sharpe", -1.0) >= 0.5
        and holdout_metrics.get("max_drawdown", -1.0) > -0.15
        and holdout_metrics.get("profit_factor") is not None
        and holdout_metrics.get("profit_factor", 0.0) >= 1.10
        and validation.get("positive_validation_folds", 0) >= max(1, (len(validation.get("folds", [])) + 1) // 2)
        and positive_rolling_windows >= minimum_positive_windows
        and severe_metrics.get("total_return", -1.0) > 0.0
    )
    return {
        "data_source": data_source,
        "research_only": True,
        "quality": quality,
        "validation": validation,
        "selected_strategy": asdict(selected_strategy),
        "rolling_windows": rolling_windows,
        "positive_rolling_windows": positive_rolling_windows,
        "cash_hurdle_annual": cash_hurdle,
        "holdout_excess_cagr": holdout_excess_cagr,
        "cost_stress": stress_results,
        "paper_review_candidate": paper_review_candidate,
        "readiness": {
            "data_is_marked_research_only": True,
            "profit_is_guaranteed": False,
            "paper_review_candidate": paper_review_candidate,
            "cash_hurdle_annual": cash_hurdle,
            "minimum_sharpe": 0.5,
            "minimum_positive_rolling_windows": minimum_positive_windows,
            "next_gate": "complete a forward paper-trading observation period and recheck the live contract multiplier, expiry, and broker costs",
        },
    }
