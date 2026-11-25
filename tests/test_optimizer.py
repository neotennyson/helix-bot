import unittest

from helix.optimize import WalkForwardOptimizer
from helix.config import BacktestConfig
from helix.data import generate_demo_bars


class OptimizerTests(unittest.TestCase):
    def test_walk_forward_returns_untouched_holdout(self) -> None:
        bars = generate_demo_bars(days=90, seed=7)
        config = BacktestConfig.from_dict({"symbol": "NIFTY"})
        report = WalkForwardOptimizer(config, min_train_bars=500).run(bars)
        self.assertIn("promotion", report)
        self.assertIn("holdout_metrics", report)
        self.assertGreaterEqual(len(report["folds"]), 1)
        self.assertNotEqual(report["holdout_start"], report["holdout_end"])


if __name__ == "__main__":
    unittest.main()
