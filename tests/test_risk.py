import unittest
from datetime import datetime, timedelta

from helix.config import RiskConfig
from helix.risk import RiskManager


class RiskTests(unittest.TestCase):
    def test_position_size_respects_risk_and_notional_caps(self) -> None:
        config = RiskConfig(initial_capital=1_000_000.0, risk_per_trade_pct=0.005, max_notional_pct=1.0, max_margin_pct=0.30, margin_pct=0.15, max_quantity=1000)
        manager = RiskManager(config)
        sizing = manager.size_position(1_000_000.0, 20_000.0, 100.0)
        self.assertGreaterEqual(sizing.quantity, 1)
        self.assertLessEqual(sizing.notional, 1_000_000.0)

    def test_daily_loss_halts_new_entries(self) -> None:
        config = RiskConfig(initial_capital=100_000.0, max_daily_loss_pct=0.02)
        manager = RiskManager(config)
        now = datetime.now()
        manager.start_day(now, 100_000.0)
        manager.mark(now, 97_000.0)
        self.assertFalse(manager.can_enter(now, 97_000.0, 100.0, 2.0, 1))
    def test_consecutive_loss_halt_resets_on_new_day(self) -> None:
        config = RiskConfig(initial_capital=100_000.0, max_consecutive_losses=1, max_daily_loss_pct=1.0)
        manager = RiskManager(config)
        first_day = datetime(2026, 1, 1, 10)
        manager.start_day(first_day, 100_000.0)
        manager.register_exit(-100.0, 1)
        self.assertFalse(manager.can_enter(first_day, 99_000.0, 100.0, 2.0, 2))
        second_day = first_day + timedelta(days=1)
        self.assertTrue(manager.can_enter(second_day, 99_000.0, 100.0, 2.0, 4))


if __name__ == "__main__":
    unittest.main()
