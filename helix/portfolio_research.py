"""Portfolio-level walk-forward research across the equity universe.

The earlier universe scan ranked sleeves on full history and then reported
holdout numbers, which leaks the holdout into selection. This module picks
sleeves using data strictly before each test window, so the reported portfolio
return is genuinely out-of-sample.
"""

from dataclasses import dataclass
from typing import Any

from .backtest import calculate_metrics
from .config import BacktestConfig
from .models import EquityPoint
from .universe import UniverseMember, aggregate_portfolio, scan_symbol


@dataclass(frozen=True)
class Sleeve:
    symbol: str
    strategy_mode: str
    profile: str


def split_windows(bars_by_symbol: dict[str, list], folds: int = 4) -> list[tuple[tuple[Any, Any], tuple[Any, Any]]]:
    """Build expanding train/test windows that never overlap.

    Windows are cut on calendar timestamps derived from the reference series, so
    a train window always ends exactly where its test window starts. A symbol
    only needs enough bars inside its own slice to be selectable.
    """
    usable = [bars for bars in bars_by_symbol.values() if bars]
    if not usable:
        raise ValueError("no bars available for window construction")
    reference = max(usable, key=len)
    total = len(reference)
    if total < folds * 60:
        raise ValueError("not enough history for walk-forward windows")
    development = int(total * 0.85)
    holdout_start = reference[development].timestamp
    fold_span = development // folds
    windows: list[tuple[tuple[Any, Any], tuple[Any, Any]]] = []
    for index in range(folds):
        train_start = reference[index * fold_span].timestamp
        train_end = reference[(index + 1) * fold_span].timestamp
        test_end = reference[min(development, (index + 2) * fold_span)].timestamp
        # The final fold has no future development data left to test on.
        if train_end <= train_start or test_end <= train_end:
            continue
        windows.append(((train_start, train_end), (train_end, test_end)))
    return windows


def slice_bars(bars: list, bounds: tuple[Any, Any]) -> list:
    start, end = bounds
    return [bar for bar in bars if start <= bar.timestamp < end]


def select_sleeves(
    template: dict[str, Any],
    members: list[UniverseMember],
    modes: dict[str, list[dict[str, Any]]],
    bars_by_symbol: dict[str, list],
    train_bounds: tuple[Any, Any],
    top_k: int,
    min_train_trades: int = 8,
) -> tuple[list[Sleeve], list[dict[str, Any]]]:
    scored: list[tuple[float, Sleeve, dict[str, Any]]] = []
    for member in members:
        bars = bars_by_symbol.get(member.symbol) or []
        window = slice_bars(bars, train_bounds)
        if len(window) < 250:
            continue
        for mode, profiles in modes.items():
            for profile in profiles:
                overrides = {key: value for key, value in profile.items() if key != "label"}
                label = str(profile.get("label", "base"))
                try:
                    scan = scan_symbol(template, member, mode, overrides)
                except (ValueError, RuntimeError):
                    continue
                scan_metrics = scan["metrics"]
                trades = int(scan_metrics.get("trades", 0) or 0)
                sharpe = float(scan_metrics.get("sharpe", 0.0) or 0.0)
                returns = float(scan_metrics.get("total_return", 0.0) or 0.0)
                if trades < min_train_trades:
                    continue
                # train-window only statistics: recompute from the sliced curve
                train_curve = [p for p in scan["equity_curve"] if train_bounds[0] <= p.timestamp < train_bounds[1]]
                if len(train_curve) < 60:
                    continue
                first, last = train_curve[0].equity, train_curve[-1].equity
                if first <= 0:
                    continue
                score = last / first - 1.0
                scored.append((score, Sleeve(member.symbol, mode, label), {"train_return": score, "sharpe": sharpe, "trades": trades}))
    scored.sort(key=lambda item: item[0], reverse=True)
    chosen = [item[1] for item in scored[:top_k]]
    return chosen, [{"symbol": s.symbol, "strategy_mode": s.strategy_mode, "profile": s.profile, **m} for _, s, m in scored[:top_k]]


def run_portfolio_walk_forward(
    template: dict[str, Any],
    members: list[UniverseMember],
    modes: dict[str, list[dict[str, Any]]],
    bars_by_symbol: dict[str, list],
    top_k: int = 25,
    folds: int = 4,
    bar_minutes: int = 1440,
) -> dict[str, Any]:
    windows = split_windows(bars_by_symbol, folds)
    fold_curves: list[tuple[Any, list[Any]]] = []
    per_fold: list[dict[str, Any]] = []
    all_trades: list[Any] = []
    for index, (train_bounds, test_bounds) in enumerate(windows):
        sleeves, train_scores = select_sleeves(template, members, modes, bars_by_symbol, train_bounds, top_k)
        if not sleeves:
            continue
        scans = []
        for sleeve in sleeves:
            member = next(m for m in members if m.symbol == sleeve.symbol)
            profile = next(
                p for p in modes[sleeve.strategy_mode] if str(p.get("label", "base")) == sleeve.profile
            )
            overrides = {key: value for key, value in profile.items() if key != "label"}
            try:
                scan = scan_symbol(template, member, sleeve.strategy_mode, overrides)
            except (ValueError, RuntimeError):
                continue
            test_curve = [p for p in scan["equity_curve"] if test_bounds[0] <= p.timestamp < test_bounds[1]]
            if len(test_curve) < 20:
                continue
            base = test_curve[0].equity
            scan["equity_curve"] = [type(p)(p.timestamp, 1.0 + (p.equity - base) / base, 0.0, p.price, 0) for p in test_curve]
            scan["initial_capital"] = 1.0
            scan["trades"] = [t for t in scan["trades"] if test_bounds[0] <= t.exit_time < test_bounds[1]]
            scans.append(scan)
        if not scans:
            continue
        portfolio = aggregate_portfolio(scans, bar_minutes)
        metrics = portfolio["metrics"]
        all_trades.extend(portfolio["trades"])
        # Folds are sequential, not concurrent, so their curves concatenate in
        # time order and compound from the previous fold's final equity. They
        # must never be averaged by date.
        fold_curve = [p for p in portfolio["equity_curve"] if test_bounds[0] <= p.timestamp < test_bounds[1]]
        if fold_curve:
            fold_curves.append((test_bounds[0], fold_curve))
        per_fold.append({
            "fold": index,
            "train_window": [train_bounds[0].isoformat(), train_bounds[1].isoformat()],
            "test_window": [test_bounds[0].isoformat(), test_bounds[1].isoformat()],
            "sleeves_selected": [{"symbol": s.symbol, "strategy_mode": s.strategy_mode, "profile": s.profile} for s in sleeves],
            "test_metrics": {
                key: metrics.get(key)
                for key in ("cagr", "sharpe", "sortino", "max_drawdown", "profit_factor", "trades", "total_return")
            },
            "average_pairwise_correlation": metrics.get("average_pairwise_correlation"),
        })
    if not per_fold:
        return {"status": "insufficient_data", "reason": "no fold produced a testable portfolio"}
    stitched: list[Any] = []
    equity = 1.0
    peak = 1.0
    for _start, curve in sorted(fold_curves, key=lambda item: item[0]):
        # Folds are sequential, so each curve is normalised to 1.0 at its own
        # start and must be rescaled onto the running equity before appending.
        base = curve[0].equity
        for point in curve:
            equity *= (point.equity / base) if base > 0 else 1.0
            base = point.equity
            peak = max(peak, equity)
            drawdown = equity / peak - 1.0 if peak else 0.0
            stitched.append(EquityPoint(point.timestamp, equity, drawdown, point.price, 0))
    stitched_metrics = calculate_metrics(stitched, all_trades, 1.0, bar_minutes)
    cash_hurdle = float((template.get("risk") or {}).get("cash_hurdle_annual", 0.06))
    return {
        "status": "ok",
        "method": "expanding_train_then_unseen_test_per_fold",
        "selection_leakage": "sleeves chosen from train window only; test window never informs selection",
        "top_k": top_k,
        "universe_size": len(members),
        "modes": {mode: [p.get("label", "base") for p in profiles] for mode, profiles in modes.items()},
        "folds": per_fold,
        "stitched_metrics": {
            key: stitched_metrics.get(key)
            for key in ("cagr", "sharpe", "sortino", "max_drawdown", "profit_factor", "trades", "total_return", "win_rate")
        },
        "cash_hurdle_annual": cash_hurdle,
        "excess_cagr_over_cash": float(stitched_metrics.get("cagr", 0.0)) - cash_hurdle,
        "promotion": {
            "passed": bool(
                float(stitched_metrics.get("cagr", 0.0)) > cash_hurdle
                and float(stitched_metrics.get("sharpe", -1.0)) >= 0.5
                and float(stitched_metrics.get("max_drawdown", -1.0)) > -0.15
                and float(stitched_metrics.get("profit_factor") or 0.0) >= 1.10
            ),
            "rules": {
                "excess_cagr_over_cash": cash_hurdle,
                "minimum_sharpe": 0.5,
                "minimum_max_drawdown": -0.15,
                "minimum_profit_factor": 1.10,
            },
        },
    }
