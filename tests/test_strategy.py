import unittest

from helix.data import generate_demo_bars
from helix.models import Side
from helix.strategy import RegimeStrategy


class StrategyTests(unittest.TestCase):
    def test_strategy_never_signals_before_warmup(self) -> None:
        bars = generate_demo_bars(days=30)
        strategy = RegimeStrategy()
        self.assertIsNone(strategy.signal(bars[:5]))

    def test_signal_side_is_valid(self) -> None:
        bars = generate_demo_bars(days=45, seed=3)
        strategy = RegimeStrategy()
        signal = None
        for index in range(70, len(bars)):
            signal = strategy.signal(bars[: index + 1])
            if signal is not None:
                break
        if signal is not None:
            self.assertIn(signal.side, {Side.LONG, Side.SHORT})
            self.assertGreater(signal.stop_distance, 0)
            self.assertGreaterEqual(signal.confidence, 0.56)


if __name__ == "__main__":
    unittest.main()
