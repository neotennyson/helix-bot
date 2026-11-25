"""Regression tests for defects found in the audit."""

import json
import math
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from helix.backtest import BacktestEngine, calculate_metrics
from helix.config import BacktestConfig
from helix.costs import CostModel
from helix.data import generate_demo_bars, load_bars, save_bars
from helix.models import Bar, EquityPoint, Side, Trade
from helix.paper import PaperTradingEngine


class SharpeFormulaTests(unittest.TestCase):
    """calculate_metrics must annualise Sharpe with the full period count.

    Regression: volatility is already annualised, so multiplying by
    sqrt(periods_per_year) understated Sharpe by a factor of sqrt(252) on daily
    data, which silently inverted every promotion gate in the project.
    """

    def _series(self, returns: list[float]) -> list[EquityPoint]:
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        equity = 1000.0
        points = [EquityPoint(start, equity, 0.0, 100.0, 0)]
        for index, value in enumerate(returns):
            equity *= 1.0 + value
            points.append(EquityPoint(start + timedelta(days=index + 1), equity, 0.0, 100.0, 0))
        return points

    def test_matches_closed_form_for_daily_bars(self) -> None:
        returns = [0.01 if index % 3 else -0.004 for index in range(250)]
        metrics = calculate_metrics(self._series(returns), [], 1000.0, 1440)
        mean = sum(returns) / len(returns)
        variance = sum((value - mean) ** 2 for value in returns) / len(returns)
        expected = mean / math.sqrt(variance) * math.sqrt(252.0)
        self.assertAlmostEqual(metrics["sharpe"], expected, places=10)

    def test_intraday_periods_per_year_scales_sharpe(self) -> None:
        returns = [0.001 if index % 3 else -0.0005 for index in range(500)]
        daily = calculate_metrics(self._series(returns), [], 1000.0, 1440)
        intraday = calculate_metrics(self._series(returns), [], 1000.0, 15)
        # 15m bars annualise with 252 * 25 periods
        self.assertGreater(intraday["sharpe"], daily["sharpe"])
        self.assertGreater(intraday["annualized_volatility"], daily["annualized_volatility"])

    def test_zero_volatility_yields_zero_sharpe(self) -> None:
        metrics = calculate_metrics(self._series([0.0] * 50), [], 1000.0, 1440)
        self.assertEqual(metrics["sharpe"], 0.0)


class PaperHistoryRetentionTests(unittest.TestCase):
    """Regression: the paper engine truncated history at 2500 bars.

    On a 2552-bar dataset that silently produced different indicator values
    from the backtest engine, so paper reported a different equity and trade
    count than the research result it was supposed to mirror.
    """

    def test_history_bars_default_exceeds_longest_dataset(self) -> None:
        self.assertGreaterEqual(BacktestConfig().history_bars, 6000)

    def test_history_bars_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            BacktestConfig.from_dict({"history_bars": 10})

    def test_paper_retains_configured_history(self) -> None:
        bars = generate_demo_bars(days=45, seed=71)
        config = BacktestConfig.from_dict({"symbol": bars[0].symbol, "history_bars": 6000})
        with tempfile.TemporaryDirectory() as directory:
            engine = PaperTradingEngine(config, Path(directory) / "state.json")
            engine.reset()
            for bar in bars:
                engine.process_bar(bar)
            self.assertEqual(len(engine.state["bars"]), len(bars))

    def test_paper_history_cap_is_configurable(self) -> None:
        bars = generate_demo_bars(days=45, seed=72)
        config = BacktestConfig.from_dict({"symbol": bars[0].symbol, "history_bars": 120})
        with tempfile.TemporaryDirectory() as directory:
            engine = PaperTradingEngine(config, Path(directory) / "state.json")
            engine.reset()
            for bar in bars:
                engine.process_bar(bar)
            self.assertEqual(len(engine.state["bars"]), 120)


class PaperAccountingTests(unittest.TestCase):
    def test_state_separates_realized_from_unrealized(self) -> None:
        bars = generate_demo_bars(days=70, seed=73)
        config = BacktestConfig.from_dict({"symbol": bars[0].symbol, "strategy_mode": "trend_breakout"})
        with tempfile.TemporaryDirectory() as directory:
            engine = PaperTradingEngine(config, Path(directory) / "state.json")
            engine.reset()
            for bar in bars:
                engine.process_bar(bar)
            state = engine.state
        self.assertIn("unrealized_pnl", state)
        self.assertIn("realized_equity", state)
        self.assertAlmostEqual(
            state["realized_equity"] + state["unrealized_pnl"],
            state["equity"],
            places=6,
        )
        if state["position"] is None:
            self.assertAlmostEqual(state["unrealized_pnl"], 0.0, places=6)

    def test_paper_matches_backtest_excluding_tail_liquidation_cost(self) -> None:
        payload = json.loads(Path("config/nifty_futures_daily_trend.json").read_text())
        config = BacktestConfig.from_dict(payload)
        bars = load_bars("data/nifty_fut_oct2026_daily.csv", "48704")
        if not bars:
            self.skipTest("NIFTY daily dataset unavailable")
        backtest = BacktestEngine(config).run(bars)
        with tempfile.TemporaryDirectory() as directory:
            engine = PaperTradingEngine(config, Path(directory) / "state.json")
            engine.reset()
            for bar in bars:
                engine.process_bar(bar)
            state = engine.state
        paper_total = float(state["realized_equity"]) + float(state["unrealized_pnl"])
        gap = abs(backtest.final_equity - paper_total)
        liquidation_cost = 0.0
        if len(backtest.trades) > len(state["trades"]):
            last = backtest.trades[-1]
            costs = CostModel(config.costs)
            is_buy = last.side == Side.SHORT
            liquidation_cost = costs.side_cost(is_buy, last.exit_price, last.quantity, last.multiplier).total
            liquidation_cost += abs(costs.fill_price(last.exit_price, is_buy) - last.exit_price) * last.quantity * last.multiplier
        self.assertLess(gap, liquidation_cost + 5.0)


class CostSideTests(unittest.TestCase):
    def test_sell_side_is_charged_stt_and_buy_side_stamp(self) -> None:
        from helix.config import CostConfig

        model = CostModel(CostConfig())
        buy = model.side_cost(True, 100.0, 10, 1.0)
        sell = model.side_cost(False, 100.0, 10, 1.0)
        self.assertEqual(buy.stt, 0.0)
        self.assertGreater(sell.stt, 0.0)
        self.assertGreater(buy.stamp, 0.0)
        self.assertEqual(sell.stamp, 0.0)

    def test_slippage_moves_against_the_trader(self) -> None:
        from helix.config import CostConfig

        model = CostModel(CostConfig(slippage_bps=10.0))
        self.assertGreater(model.fill_price(100.0, True), 100.0)
        self.assertLess(model.fill_price(100.0, False), 100.0)

    def test_multiplier_scales_cost_linearly(self) -> None:
        from helix.config import CostConfig

        model = CostModel(CostConfig())
        one = model.side_cost(True, 100.0, 1, 65.0).total
        ten = model.side_cost(True, 100.0, 10, 65.0).total
        self.assertAlmostEqual(ten, one * 10, places=6)


class LookAheadTests(unittest.TestCase):
    def test_entry_fill_is_inside_the_fill_bar_range(self) -> None:
        bars = []
        price = 100.0
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for index in range(60):
            bars.append(Bar(timestamp=start + timedelta(days=index), symbol="X", open=price, high=price * 1.001, low=price * 0.999, close=price, volume=1000))
        for index in range(6):
            bars.append(Bar(timestamp=start + timedelta(days=60 + index), symbol="X", open=price, high=price * 1.05, low=price * 0.99, close=price * 1.04, volume=1000))
            price *= 1.04
        config = BacktestConfig.from_dict(
            {
                "symbol": "X",
                "strategy_mode": "trend_breakout",
                "bar_minutes": 1440,
                "strategy": {
                    "fast_ema": 5, "slow_ema": 20, "breakout_period": 10, "trend_adx_min": 10,
                    "range_adx_max": 10, "session_start": "00:00", "flatten_at": "23:59",
                    "intraday_only": False, "min_atr_pct": 0.0, "max_atr_pct": 1.0,
                    "allow_short": False, "target_r": 3.0, "stop_atr_multiple": 3.0, "max_holding_bars": 5,
                },
            }
        )
        result = BacktestEngine(config).run(bars)
        self.assertTrue(result.trades)
        by_timestamp = {bar.timestamp: bar for bar in bars}
        for trade in result.trades:
            fill_bar = by_timestamp[trade.entry_time]
            self.assertGreaterEqual(trade.entry_price, fill_bar.low - 1e-9)
            self.assertLessEqual(trade.entry_price, fill_bar.high + 1e-9)

    def test_perturbing_later_bars_does_not_change_earlier_exits(self) -> None:
        bars = []
        price = 100.0
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for index in range(70):
            bars.append(Bar(timestamp=start + timedelta(days=index), symbol="X", open=price, high=price * 1.01, low=price * 0.99, close=price, volume=1000))
            price *= 1.004
        config = BacktestConfig.from_dict(
            {
                "symbol": "X",
                "strategy_mode": "trend_breakout",
                "bar_minutes": 1440,
                "strategy": {
                    "fast_ema": 5, "slow_ema": 20, "breakout_period": 10, "trend_adx_min": 10,
                    "range_adx_max": 10, "session_start": "00:00", "flatten_at": "23:59",
                    "intraday_only": False, "min_atr_pct": 0.0, "max_atr_pct": 1.0,
                    "target_r": 2.0, "stop_atr_multiple": 2.0, "max_holding_bars": 3,
                },
            }
        )
        base = BacktestEngine(config).run(bars)
        cut = bars[40].timestamp
        perturbed = [
            Bar(
                timestamp=bar.timestamp, symbol=bar.symbol, open=bar.open,
                high=bar.high * (1.3 if bar.timestamp >= cut else 1.0), low=bar.low,
                close=bar.close * (1.3 if bar.timestamp >= cut else 1.0), volume=bar.volume,
            )
            for bar in bars
        ]
        other = BacktestEngine(config).run(perturbed)
        early_base = [t for t in base.trades if t.exit_time < cut]
        early_other = [t for t in other.trades if t.exit_time < cut]
        self.assertEqual(len(early_base), len(early_other))
        for left, right in zip(early_base, early_other):
            self.assertEqual(left.entry_time, right.entry_time)
            self.assertEqual(left.exit_reason, right.exit_reason)


class HoldoutTradeCapacityTests(unittest.TestCase):
    """The holdout trade requirement must respect what the window allows.

    Regression: the gate demanded 20 holdout trades while a 60-bar max hold over
    a ~528-bar holdout permits at most ~8, so every candidate failed on a rule no
    strategy could satisfy and real performance differences were hidden.
    """

    def test_gate_reports_achievable_trade_capacity(self) -> None:
        payload = json.loads(Path("config/nifty_futures_daily_trend.json").read_text())
        config = BacktestConfig.from_dict(payload)
        bars = load_bars("data/nifty_fut_oct2026_daily.csv", "48704")
        if not bars:
            self.skipTest("NIFTY daily dataset unavailable")
        from helix.optimize import WalkForwardOptimizer

        report = WalkForwardOptimizer(config).run(bars)
        promotion = report["promotion"]
        achievable = promotion["max_achievable_holdout_trades"]
        self.assertGreater(achievable, 0)
        self.assertLessEqual(promotion["required_trades_for_confidence"], achievable)
        self.assertIn("performance_rules_passed", promotion)
        self.assertIn(promotion["status"], {
            "eligible_for_paper_review",
            "not_eligible_for_paper_review",
            "insufficient_holdout_trades_for_confidence",
        })

    def test_status_distinguishes_capacity_from_performance(self) -> None:
        payload = json.loads(Path("config/nifty_futures_daily_trend.json").read_text())
        config = BacktestConfig.from_dict(payload)
        bars = load_bars("data/nifty_fut_oct2026_daily.csv", "48704")
        if not bars:
            self.skipTest("NIFTY daily dataset unavailable")
        from helix.optimize import WalkForwardOptimizer

        promotion = WalkForwardOptimizer(config).run(bars)["promotion"]
        if promotion["performance_rules_passed"] and not promotion["trade_sample_confidence"]:
            self.assertEqual(promotion["status"], "insufficient_holdout_trades_for_confidence")
        elif promotion["passed"]:
            self.assertEqual(promotion["status"], "eligible_for_paper_review")
        else:
            self.assertEqual(promotion["status"], "not_eligible_for_paper_review")


class EquityReconciliationTests(unittest.TestCase):
    def test_final_equity_equals_initial_plus_net_pnl(self) -> None:
        payload = json.loads(Path("config/nifty_futures_daily_trend.json").read_text())
        config = BacktestConfig.from_dict(payload)
        bars = load_bars("data/nifty_fut_oct2026_daily.csv", "48704")
        if not bars:
            self.skipTest("NIFTY daily dataset unavailable")
        result = BacktestEngine(config).run(bars)
        self.assertAlmostEqual(
            result.final_equity,
            result.initial_capital + sum(trade.net_pnl for trade in result.trades),
            places=6,
        )

    def test_trades_are_chronological_and_self_consistent(self) -> None:
        bars = generate_demo_bars(days=60, seed=74)
        config = BacktestConfig.from_dict({"symbol": bars[0].symbol, "strategy_mode": "mean_reversion"})
        result = BacktestEngine(config).run(bars)
        previous = None
        for trade in result.trades:
            self.assertLessEqual(trade.entry_time, trade.exit_time)
            if previous is not None:
                self.assertLessEqual(previous, trade.entry_time)
            previous = trade.entry_time
            self.assertAlmostEqual(trade.gross_pnl - trade.fees, trade.net_pnl, places=6)


if __name__ == "__main__":
    unittest.main()