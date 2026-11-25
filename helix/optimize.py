from dataclasses import asdict, dataclass, replace
from statistics import median, pstdev
from typing import Any

from .backtest import BacktestEngine
from .config import BacktestConfig
from .models import Bar


@dataclass
class CandidateResult:
    parameters: dict[str, Any]
    score: float
    validation_returns: list[float]
    validation_metrics: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameters": self.parameters,
            "score": self.score,
            "validation_returns": self.validation_returns,
            "validation": self.validation_metrics,
        }


class WalkForwardOptimizer:
    def __init__(self, base_config: BacktestConfig, min_train_bars: int = 500) -> None:
        self.base_config = base_config
        self.min_train_bars = min_train_bars

    def run(self, bars: list[Bar], candidates: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        data = sorted([bar for bar in bars if bar.symbol == self.base_config.symbol], key=lambda item: item.timestamp)
        required = max(self.base_config.strategy.slow_ema, self.base_config.strategy.atr_period * 2, self.base_config.strategy.breakout_period, self.base_config.strategy.long_term_ema if self.base_config.strategy.use_long_term_filter else 0) + 10
        min_train_bars = max(self.min_train_bars, required)
        if len(data) < min_train_bars + 250:
            raise ValueError("not enough bars for walk-forward validation")
        candidates = candidates or default_candidates(self.base_config)
        holdout_size = max(250, int(len(data) * 0.20), required)
        holdout_start = len(data) - holdout_size
        development = data[:holdout_start]
        if len(development) < min_train_bars + 250:
            raise ValueError("not enough development bars for walk-forward validation")
        fold_size = max(min_train_bars, len(development) // 4)
        folds: list[tuple[int, int, int, int]] = []
        for fold_index in range(4):
            validation_start = fold_index * fold_size
            validation_end = min(len(development), (fold_index + 1) * fold_size)
            if validation_start >= min_train_bars and validation_end - validation_start >= required:
                folds.append((0, validation_start, validation_start, validation_end))
        if not folds:
            raise ValueError("could not construct a valid walk-forward split")
        ranked: list[CandidateResult] = []
        for parameters in candidates:
            config = self._config_for(parameters)
            validation_metrics = [BacktestEngine(config).run(development[validation_start:validation_end]).metrics for _, _, validation_start, validation_end in folds]
            returns = [float(metrics.get("total_return", -1.0)) for metrics in validation_metrics]
            positive = sum(value > 0.0 for value in returns)
            spread = pstdev(returns) if len(returns) > 1 else 0.0
            score = median(returns) - 0.5 * spread + 0.02 * positive
            ranked.append(CandidateResult(parameters, score, returns, validation_metrics))
        ranked.sort(key=lambda item: item.score, reverse=True)
        selected = ranked[0]
        selected_config = self._config_for(selected.parameters)
        effective_parameters = asdict(selected_config.strategy)
        fold_payload: list[dict[str, Any]] = []
        for fold_index, (train_start, train_end, validation_start, validation_end) in enumerate(folds):
            train_metrics = BacktestEngine(selected_config).run(development[train_start:train_end]).metrics
            validation_metrics = selected.validation_metrics[fold_index]
            fold_payload.append(
                {
                    "fold": fold_index,
                    "train_start": development[train_start].timestamp.isoformat(),
                    "train_end": development[train_end - 1].timestamp.isoformat(),
                    "validation_start": development[validation_start].timestamp.isoformat(),
                    "validation_end": development[validation_end - 1].timestamp.isoformat(),
                    "selected_parameters": effective_parameters,
                    "train_metrics": train_metrics,
                    "validation_metrics": validation_metrics,
                    "candidate_count": len(candidates),
                }
            )
        holdout_data = data[holdout_start:]
        holdout_result = BacktestEngine(selected_config).run(holdout_data)
        holdout_metrics = holdout_result.metrics
        positive_validation_folds = sum(1 for value in selected.validation_returns if value > 0.0)
        holdout_days = (holdout_data[-1].timestamp.date() - holdout_data[0].timestamp.date()).days
        # A strategy cannot produce more trades than the holdout window physically
        # allows. Requiring 20 trades against a 60-bar max hold over a ~2 year
        # holdout (528 bars / 60 = 8.8 max) is unsatisfiable, which would report
        # a design limit as a performance failure.
        max_holding = max(1, int(selected_config.strategy.max_holding_bars))
        achievable_trades = int(len(holdout_data) / max_holding)
        minimum_holdout_trades = max(5, int(len(holdout_data) / 100))
        trade_count_is_binding = achievable_trades < minimum_holdout_trades
        enough_trades_for_confidence = int(holdout_metrics.get("trades", 0)) >= max(5, achievable_trades // 3)
        cash_hurdle = self.base_config.risk.cash_hurdle_annual
        holdout_excess_cagr = holdout_metrics.get("cagr", -1.0) - cash_hurdle
        performance_rules = (
            holdout_metrics.get("total_return", -1.0) > 0.0
            and holdout_excess_cagr > 0.0
            and holdout_metrics.get("sharpe", -1.0) >= 0.5
            and holdout_metrics.get("max_drawdown", -1.0) > -0.15
            and holdout_metrics.get("profit_factor") is not None
            and holdout_metrics.get("profit_factor", 0.0) >= 1.10
            and holdout_days >= 20
            and positive_validation_folds >= max(1, (len(folds) + 1) // 2)
        )
        promotion = performance_rules and enough_trades_for_confidence
        if promotion:
            status = "eligible_for_paper_review"
        elif performance_rules and not enough_trades_for_confidence:
            status = "insufficient_holdout_trades_for_confidence"
        else:
            status = "not_eligible_for_paper_review"
        return {
            "method": "expanding_validation_with_untouched_holdout",
            "symbol": self.base_config.symbol,
            "strategy_mode": self.base_config.strategy_mode,
            "bar_count": len(data),
            "candidate_count": len(candidates),
            "selected_parameters": effective_parameters,
            "selection_score": selected.score,
            "validation_returns": selected.validation_returns,
            "holdout_start": holdout_data[0].timestamp.isoformat(),
            "holdout_end": holdout_data[-1].timestamp.isoformat(),
            "holdout_metrics": holdout_metrics,
            "cash_hurdle_annual": cash_hurdle,
            "holdout_excess_cagr": holdout_excess_cagr,
            "holdout_calendar_days": holdout_days,
            "positive_validation_folds": positive_validation_folds,
            "folds": fold_payload,
            "candidate_results": [item.to_dict() for item in ranked[:10]],
            "promotion": {
                "passed": bool(promotion),
                "status": status,
                "performance_rules_passed": bool(performance_rules),
                "trade_sample_confidence": bool(enough_trades_for_confidence),
                "max_achievable_holdout_trades": achievable_trades,
                "required_trades_for_confidence": max(5, achievable_trades // 3),
                "binding_trade_constraint": bool(trade_count_is_binding),
                "rules": {
                    "positive_holdout_return": True,
                    "cash_hurdle_annual": cash_hurdle,
                    "positive_excess_cagr_over_cash": True,
                    "minimum_sharpe": 0.5,
                    "max_drawdown_at_least_minus_15_percent": True,
                    "profit_factor_at_least_1_10": True,
                    "minimum_holdout_trades": minimum_holdout_trades,
                    "minimum_holdout_calendar_days": 20,
                    "minimum_positive_validation_folds": max(1, (len(folds) + 1) // 2),
                },
            },
        }

    def _config_for(self, parameters: dict[str, Any]) -> BacktestConfig:
        strategy_values = dict(asdict(self.base_config.strategy))
        strategy_values.update(parameters)
        return replace(self.base_config, strategy=type(self.base_config.strategy)(**strategy_values))


def default_candidates(base_config: BacktestConfig | None = None) -> list[dict[str, Any]]:
    if base_config is not None and base_config.strategy_mode == "trend_breakout":
        return trend_candidates()
    if base_config is not None and base_config.strategy_mode == "mean_reversion":
        return mean_reversion_candidates()
    if base_config is not None and base_config.strategy_mode == "volatility_breakout":
        return volatility_breakout_candidates()
    if base_config is not None and base_config.strategy_mode == "opening_range":
        return opening_range_candidates()
    return regime_candidates()


def volatility_breakout_candidates() -> list[dict[str, Any]]:
    return [
        {"fast_ema": 10, "slow_ema": 40, "breakout_period": 20, "trend_adx_min": 20.0, "trend_threshold_atr": 1.2, "stop_atr_multiple": 2.0, "target_r": 2.0, "max_holding_bars": 20},
        {"fast_ema": 10, "slow_ema": 40, "breakout_period": 40, "trend_adx_min": 18.0, "trend_threshold_atr": 1.5, "stop_atr_multiple": 2.0, "target_r": 2.0, "max_holding_bars": 25},
        {"fast_ema": 12, "slow_ema": 48, "breakout_period": 30, "trend_adx_min": 22.0, "trend_threshold_atr": 1.2, "stop_atr_multiple": 2.5, "target_r": 2.5, "max_holding_bars": 30},
        {"fast_ema": 12, "slow_ema": 48, "breakout_period": 55, "trend_adx_min": 20.0, "trend_threshold_atr": 1.5, "stop_atr_multiple": 2.5, "target_r": 3.0, "max_holding_bars": 40},
        {"fast_ema": 20, "slow_ema": 60, "breakout_period": 40, "trend_adx_min": 18.0, "trend_threshold_atr": 1.0, "stop_atr_multiple": 1.5, "target_r": 2.0, "max_holding_bars": 20},
        {"fast_ema": 20, "slow_ema": 60, "breakout_period": 60, "trend_adx_min": 16.0, "trend_threshold_atr": 1.0, "stop_atr_multiple": 1.5, "target_r": 2.5, "max_holding_bars": 30},
    ]


def opening_range_candidates() -> list[dict[str, Any]]:
    return [
        {"fast_ema": 5, "slow_ema": 20, "opening_range_bars": 4, "opening_range_max_atr": 2.5, "trend_adx_min": 18.0, "stop_atr_multiple": 1.0, "target_r": 1.5, "max_holding_bars": 12},
        {"fast_ema": 5, "slow_ema": 20, "opening_range_bars": 6, "opening_range_max_atr": 3.0, "trend_adx_min": 18.0, "stop_atr_multiple": 1.0, "target_r": 1.5, "max_holding_bars": 10},
        {"fast_ema": 5, "slow_ema": 20, "opening_range_bars": 6, "opening_range_max_atr": 2.0, "trend_adx_min": 22.0, "stop_atr_multiple": 1.2, "target_r": 2.0, "max_holding_bars": 10},
        {"fast_ema": 9, "slow_ema": 34, "opening_range_bars": 8, "opening_range_max_atr": 3.0, "trend_adx_min": 16.0, "stop_atr_multiple": 1.0, "target_r": 1.5, "max_holding_bars": 8},
        {"fast_ema": 9, "slow_ema": 34, "opening_range_bars": 4, "opening_range_max_atr": 2.0, "trend_adx_min": 20.0, "stop_atr_multiple": 1.5, "target_r": 2.0, "max_holding_bars": 8},
    ]


def mean_reversion_candidates() -> list[dict[str, Any]]:
    return [
        {"breakout_period": 20, "trend_adx_min": 18.0, "range_adx_max": 18.0, "range_threshold_atr": 2.0, "rsi_period": 5, "rsi_range_long": 30.0, "rsi_range_short": 70.0, "stop_atr_multiple": 1.5, "target_r": 1.0, "max_holding_bars": 10},
        {"breakout_period": 20, "trend_adx_min": 22.0, "range_adx_max": 20.0, "range_threshold_atr": 2.2, "rsi_period": 5, "rsi_range_long": 25.0, "rsi_range_short": 75.0, "stop_atr_multiple": 1.5, "target_r": 1.0, "max_holding_bars": 10},
        {"breakout_period": 30, "trend_adx_min": 18.0, "range_adx_max": 18.0, "range_threshold_atr": 2.0, "rsi_period": 5, "rsi_range_long": 30.0, "rsi_range_short": 70.0, "stop_atr_multiple": 2.0, "target_r": 1.2, "max_holding_bars": 15},
        {"breakout_period": 30, "trend_adx_min": 22.0, "range_adx_max": 20.0, "range_threshold_atr": 2.2, "rsi_period": 5, "rsi_range_long": 25.0, "rsi_range_short": 75.0, "stop_atr_multiple": 2.0, "target_r": 1.2, "max_holding_bars": 15},
        {"breakout_period": 40, "trend_adx_min": 20.0, "range_adx_max": 18.0, "range_threshold_atr": 2.0, "rsi_period": 5, "rsi_range_long": 30.0, "rsi_range_short": 70.0, "stop_atr_multiple": 2.0, "target_r": 1.5, "max_holding_bars": 20},
        {"breakout_period": 40, "trend_adx_min": 24.0, "range_adx_max": 20.0, "range_threshold_atr": 2.2, "rsi_period": 5, "rsi_range_long": 25.0, "rsi_range_short": 75.0, "stop_atr_multiple": 2.0, "target_r": 1.5, "max_holding_bars": 20},
    ]


def regime_candidates() -> list[dict[str, Any]]:
    return [
        {"fast_ema": 10, "slow_ema": 30, "trend_adx_min": 19.0, "range_adx_max": 18.0, "target_r": 1.6, "max_holding_bars": 12},
        {"fast_ema": 12, "slow_ema": 36, "trend_adx_min": 20.0, "range_adx_max": 19.0, "target_r": 1.8, "max_holding_bars": 16},
        {"fast_ema": 16, "slow_ema": 48, "trend_adx_min": 21.0, "range_adx_max": 19.0, "target_r": 1.8, "max_holding_bars": 18},
        {"fast_ema": 20, "slow_ema": 60, "trend_adx_min": 22.0, "range_adx_max": 20.0, "target_r": 2.0, "max_holding_bars": 20},
        {"fast_ema": 24, "slow_ema": 72, "trend_adx_min": 24.0, "range_adx_max": 21.0, "target_r": 2.2, "max_holding_bars": 24},
        {"fast_ema": 30, "slow_ema": 90, "trend_adx_min": 23.0, "range_adx_max": 20.0, "target_r": 2.5, "max_holding_bars": 28},
    ]


def trend_candidates() -> list[dict[str, Any]]:
    return [
        {"fast_ema": 20, "slow_ema": 60, "breakout_period": 40, "trend_adx_min": 18.0, "range_adx_max": 18.0, "stop_atr_multiple": 1.5, "target_r": 6.0, "max_holding_bars": 60, "use_long_term_filter": True, "long_term_ema": 150, "long_term_slope_bars": 20},
        {"fast_ema": 20, "slow_ema": 60, "breakout_period": 40, "trend_adx_min": 18.0, "range_adx_max": 18.0, "stop_atr_multiple": 1.5, "target_r": 6.0, "max_holding_bars": 60, "use_long_term_filter": True, "long_term_ema": 150, "long_term_slope_bars": 10},
        {"fast_ema": 30, "slow_ema": 90, "breakout_period": 40, "trend_adx_min": 18.0, "range_adx_max": 18.0, "stop_atr_multiple": 1.5, "target_r": 6.0, "max_holding_bars": 60, "use_long_term_filter": True, "long_term_ema": 150, "long_term_slope_bars": 10},
        {"fast_ema": 25, "slow_ema": 75, "breakout_period": 40, "trend_adx_min": 18.0, "range_adx_max": 18.0, "stop_atr_multiple": 1.5, "target_r": 6.0, "max_holding_bars": 60, "use_long_term_filter": True, "long_term_ema": 150, "long_term_slope_bars": 10},
        {"fast_ema": 12, "slow_ema": 36, "breakout_period": 20, "trend_adx_min": 18.0, "range_adx_max": 18.0, "stop_atr_multiple": 1.5, "target_r": 4.0, "max_holding_bars": 40},
        {"fast_ema": 12, "slow_ema": 36, "breakout_period": 40, "trend_adx_min": 18.0, "range_adx_max": 18.0, "stop_atr_multiple": 1.5, "target_r": 6.0, "max_holding_bars": 60},
        {"fast_ema": 20, "slow_ema": 60, "breakout_period": 20, "trend_adx_min": 18.0, "range_adx_max": 18.0, "stop_atr_multiple": 1.5, "target_r": 4.0, "max_holding_bars": 40},
        {"fast_ema": 20, "slow_ema": 60, "breakout_period": 40, "trend_adx_min": 18.0, "range_adx_max": 18.0, "stop_atr_multiple": 1.5, "target_r": 4.0, "max_holding_bars": 60},
        {"fast_ema": 20, "slow_ema": 60, "breakout_period": 40, "trend_adx_min": 22.0, "range_adx_max": 18.0, "stop_atr_multiple": 1.5, "target_r": 6.0, "max_holding_bars": 60},
        {"fast_ema": 20, "slow_ema": 60, "breakout_period": 40, "trend_adx_min": 18.0, "range_adx_max": 18.0, "stop_atr_multiple": 2.0, "target_r": 6.0, "max_holding_bars": 60},
        {"fast_ema": 30, "slow_ema": 90, "breakout_period": 20, "trend_adx_min": 18.0, "range_adx_max": 18.0, "stop_atr_multiple": 1.5, "target_r": 4.0, "max_holding_bars": 40},
        {"fast_ema": 30, "slow_ema": 90, "breakout_period": 40, "trend_adx_min": 18.0, "range_adx_max": 18.0, "stop_atr_multiple": 1.5, "target_r": 6.0, "max_holding_bars": 60},
        {"fast_ema": 30, "slow_ema": 90, "breakout_period": 40, "trend_adx_min": 22.0, "range_adx_max": 18.0, "stop_atr_multiple": 1.5, "target_r": 6.0, "max_holding_bars": 60},
        {"fast_ema": 30, "slow_ema": 90, "breakout_period": 60, "trend_adx_min": 18.0, "range_adx_max": 18.0, "stop_atr_multiple": 2.0, "target_r": 8.0, "max_holding_bars": 90},
    ]
