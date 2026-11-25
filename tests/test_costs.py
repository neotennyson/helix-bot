import unittest

from helix.config import BacktestConfig, CostConfig
from helix.costs import CostModel


class CostTests(unittest.TestCase):
    def test_sell_side_includes_stt_and_buy_side_does_not(self) -> None:
        model = CostModel(CostConfig())
        buy = model.side_cost(True, 100.0, 1, 1.0)
        sell = model.side_cost(False, 100.0, 1, 1.0)
        self.assertEqual(buy.stt, 0.0)
        self.assertGreater(sell.stt, 0.0)
        self.assertGreater(sell.total, buy.total)

    def test_slippage_direction(self) -> None:
        model = CostModel(CostConfig(slippage_bps=10.0))
        self.assertGreater(model.fill_price(100.0, True), 100.0)
        self.assertLess(model.fill_price(100.0, False), 100.0)


if __name__ == "__main__":
    unittest.main()
