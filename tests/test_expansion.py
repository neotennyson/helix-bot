import json
import tempfile
import unittest
from pathlib import Path

from helix.config import BacktestConfig
from helix.data import generate_demo_bars
from helix.opening_range_strategy import OpeningRangeBreakoutStrategy
from helix.options import (
    OptionConfig,
    normal_cdf,
    option_delta,
    option_price,
    run_option_research,
    strike_for_delta,
)
from helix.models import Bar
from helix.session import SessionWindow


def _clone(bar, symbol: str | None = None, hour: int | None = None, minute: int | None = None) -> Bar:
    timestamp = bar.timestamp
    if hour is not None:
        timestamp = timestamp.replace(hour=hour, minute=minute if minute is not None else 0)
    return Bar(
        timestamp=timestamp,
        symbol=symbol if symbol is not None else bar.symbol,
        open=bar.open,
        high=bar.high,
        low=bar.low,
        close=bar.close,
        volume=bar.volume,
        expiry=bar.expiry,
    )


def _with_time(bar, hour: int, minute: int) -> Bar:
    return _clone(bar, hour=hour, minute=minute)
from helix.universe import aggregate_portfolio, load_universe, rank_rows, scan_symbol
from helix.volatility_breakout_strategy import VolatilityBreakoutStrategy


class SessionWindowTests(unittest.TestCase):
    def test_daily_bar_ignores_clock_gate(self) -> None:
        config = BacktestConfig.from_dict({"strategy": {"session_start": "09:15", "flatten_at": "15:30"}}).strategy
        window = SessionWindow.build(config, "Asia/Kolkata")
        daily = generate_demo_bars(days=30, seed=5)
        # daily-equivalent bar: midnight timestamp
        midnight = _with_time(daily[-1], hour=0, minute=0)
        self.assertTrue(window.is_daily_bar(midnight))
        self.assertTrue(window.allows_entry(midnight))

    def test_intraday_bar_respects_session(self) -> None:
        config = BacktestConfig.from_dict(
            {"strategy": {"session_start": "09:30", "flatten_at": "15:05", "intraday_only": True}}
        ).strategy
        window = SessionWindow.build(config, "Asia/Kolkata")
        bars = generate_demo_bars(days=30, seed=6)
        early = _with_time(bars[0], hour=9, minute=0)
        self.assertFalse(window.allows_entry(early))


class NewStrategyTests(unittest.TestCase):
    def test_volatility_breakout_reconciles(self) -> None:
        from helix.backtest import BacktestEngine

        bars = generate_demo_bars(days=80, seed=31)
        config = BacktestConfig.from_dict({"symbol": "NIFTY", "strategy_mode": "volatility_breakout"})
        result = BacktestEngine(config).run(bars)
        expected = config.risk.initial_capital + sum(trade.net_pnl for trade in result.trades)
        self.assertAlmostEqual(result.final_equity, expected, places=6)

    def test_opening_range_flattens_at_session_end(self) -> None:
        bars = generate_demo_bars(days=30, seed=32)
        config = BacktestConfig.from_dict(
            {
                "symbol": "NIFTY",
                "strategy_mode": "opening_range",
                "strategy": {"session_start": "09:15", "flatten_at": "15:30"},
            }
        )
        strategy = OpeningRangeBreakoutStrategy(config.strategy)
        last = bars[-1]
        before = _with_time(last, hour=15, minute=29)
        at_close = _with_time(last, hour=15, minute=30)
        self.assertFalse(strategy.should_flatten(before))
        self.assertTrue(strategy.should_flatten(at_close))

    def test_volatility_breakout_confidence_gate_blocks_weak_signals(self) -> None:
        bars = generate_demo_bars(days=80, seed=33)
        strategy = VolatilityBreakoutStrategy(BacktestConfig().strategy)
        for index in range(70, len(bars)):
            signal = strategy.signal(bars[: index + 1])
            if signal is not None and signal.side.value in {"LONG", "SHORT"}:
                self.assertGreaterEqual(signal.confidence, BacktestConfig().strategy.confidence_threshold)
                return
        self.skipTest("no directional signal produced on the synthetic series")


class OptionPricingTests(unittest.TestCase):
    def test_put_call_parity(self) -> None:
        call = option_price(24000, 24000, 30, 0.15, 0.06, "CE")
        put = option_price(24000, 24000, 30, 0.15, 0.06, "PE")
        expected = 24000 - 24000 * pow(2.718281828, -0.06 * 30 / 365.0)
        self.assertAlmostEqual(call - put, expected, places=4)

    def test_expired_option_is_intrinsic(self) -> None:
        self.assertAlmostEqual(option_price(25000, 24000, 0, 0.15, 0.06, "CE"), 1000.0, places=6)
        self.assertAlmostEqual(option_price(23000, 24000, 0, 0.15, 0.06, "PE"), 1000.0, places=6)

    def test_strike_for_delta_inverts(self) -> None:
        for right in ("CE", "PE"):
            strike = strike_for_delta(24000, 21, 0.14, 0.25, right)
            self.assertAlmostEqual(abs(option_delta(24000, strike, 21, 0.14, 0.065, right)), 0.25, places=3)

    def test_normal_cdf_bounds(self) -> None:
        self.assertAlmostEqual(normal_cdf(0.0), 0.5, places=6)
        self.assertGreater(normal_cdf(3.0), 0.99)


class OptionResearchTests(unittest.TestCase):
    def test_report_is_flagged_synthetic_and_never_promotes(self) -> None:
        bars = generate_demo_bars(days=260, seed=41)
        report = run_option_research(OptionConfig(initial_capital=10_000_000.0), bars)
        self.assertEqual(report["real_option_quotes"], False)
        self.assertFalse(report["promotion"]["passed"])
        self.assertFalse(report["paper_review_candidate"])
        self.assertIn("disclaimer", report)

    def test_capital_limit_blocks_oversized_legs(self) -> None:
        bars = generate_demo_bars(days=260, seed=42)
        tiny = run_option_research(OptionConfig(initial_capital=10_000.0), bars)
        self.assertEqual(tiny["metrics"]["trades"], 0)
        self.assertGreater(tiny["skipped_signals_for_capital_limit"], 0)


class UniverseTests(unittest.TestCase):
    def _fixture(self) -> tuple[Path, BacktestConfig, list]:
        directory = Path(tempfile.mkdtemp())
        bars = generate_demo_bars(days=260, seed=51)
        path_one = directory / "AAA_daily.csv"
        path_two = directory / "BBB_daily.csv"
        from helix.data import save_bars

        save_bars(path_one, [self._retag(bar, "1") for bar in bars])
        save_bars(path_two, [self._retag(bar, "2") for bar in bars])
        manifest = directory / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "equities": [
                        {"symbol": "AAA", "security_id": "1"},
                        {"symbol": "BBB", "security_id": "2"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        return manifest, directory, bars

    @staticmethod
    def _retag(bar, symbol: str):
        return _clone(bar, symbol=symbol)

    def test_load_scan_and_aggregate(self) -> None:
        manifest, directory, _ = self._fixture()
        members = load_universe(manifest, directory)
        self.assertEqual([member.symbol for member in members], ["AAA", "BBB"])
        template = json.loads(Path("config/equity_daily_template.json").read_text(encoding="utf-8"))
        template["bar_minutes"] = 1440
        template["strategy"]["session_start"] = "00:00"
        template["strategy"]["flatten_at"] = "23:59"
        scans = [scan_symbol(template, member, "volatility_breakout") for member in members]
        self.assertEqual(len(scans), 2)
        rows = rank_rows(scans)
        self.assertEqual(len(rows), 2)
        portfolio = aggregate_portfolio(scans, 1440)
        self.assertEqual(portfolio["status"], "ok")
        total_return = float(portfolio["metrics"]["total_return"])
        self.assertGreater(total_return, -1.0)
        self.assertLess(total_return, 100.0)
        self.assertEqual(portfolio["metrics"]["sleeves"], 2)
        self.assertEqual(portfolio["metrics"]["min_contributing_sleeves"], 2)


if __name__ == "__main__":
    unittest.main()