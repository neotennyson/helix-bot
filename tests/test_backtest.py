import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from helix.backtest import BacktestEngine
from helix.config import BacktestConfig
from helix.data import generate_demo_bars, load_bars, save_bars
from helix.paper import PaperTradingEngine


class BacktestTests(unittest.TestCase):
    def test_backtest_has_causal_entries_and_reconciles_pnl(self) -> None:
        bars = generate_demo_bars(days=70, seed=9)
        config = BacktestConfig.from_dict({"symbol": "NIFTY", "bar_minutes": 15})
        result = BacktestEngine(config).run(bars)
        signal_times = {signal.timestamp for signal in result.signals}
        for trade in result.trades:
            self.assertNotEqual(trade.exit_time, datetime.min)
            self.assertGreaterEqual(trade.exit_time, trade.entry_time)
            self.assertTrue(any(trade.entry_time > signal_time for signal_time in signal_times))
        expected = config.risk.initial_capital + sum(trade.net_pnl for trade in result.trades)
        self.assertAlmostEqual(result.final_equity, expected, places=6)

    def test_csv_round_trip(self) -> None:
        bars = generate_demo_bars(days=30, seed=4)[:80]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bars.csv"
            save_bars(path, bars)
            loaded = load_bars(path, "NIFTY")
        self.assertEqual(len(loaded), len(bars))
        self.assertEqual(loaded[0].timestamp, bars[0].timestamp)
        self.assertEqual(loaded[-1].close, bars[-1].close)

    def test_paper_engine_is_paper_only(self) -> None:
        bars = generate_demo_bars(days=70, seed=9)
        config = BacktestConfig.from_dict({"symbol": "NIFTY"})
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            engine = PaperTradingEngine(config, path)
            engine.initialize_history(bars)
            engine.process_bar(bars[-1])
            self.assertEqual(engine.state["mode"], "PAPER_ONLY")
            self.assertEqual(engine.state["broker"], "none")
            self.assertEqual(engine.state["last_timestamp"], bars[-1].timestamp.isoformat())


if __name__ == "__main__":
    unittest.main()
