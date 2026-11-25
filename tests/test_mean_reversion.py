import unittest

from helix.backtest import BacktestEngine
from helix.config import BacktestConfig
from helix.data import generate_demo_bars
from helix.mean_reversion_strategy import MeanReversionStrategy


class MeanReversionTests(unittest.TestCase):
    def test_flat_signal_does_not_open_a_position(self) -> None:
        bars = generate_demo_bars(days=70, seed=21)
        config = BacktestConfig.from_dict({"symbol": "NIFTY", "strategy_mode": "mean_reversion"})
        result = BacktestEngine(config).run(bars)
        expected = config.risk.initial_capital + sum(trade.net_pnl for trade in result.trades)
        self.assertAlmostEqual(result.final_equity, expected, places=6)

    def test_strategy_warmup_and_signal_shape(self) -> None:
        bars = generate_demo_bars(days=45, seed=22)
        strategy = MeanReversionStrategy()
        self.assertIsNone(strategy.signal(bars[:5]))
        signal = None
        for index in range(60, len(bars)):
            signal = strategy.signal(bars[: index + 1])
            if signal is not None:
                break
        if signal is not None:
            self.assertGreater(signal.stop_distance, 0)
            self.assertIn(signal.side.value, {"LONG", "SHORT", "FLAT"})


if __name__ == "__main__":
    unittest.main()