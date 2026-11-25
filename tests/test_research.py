import unittest

from helix.config import BacktestConfig
from helix.data import generate_demo_bars
from helix.research import inspect_data, run_research


class ResearchTests(unittest.TestCase):
    def test_data_quality_and_stress_report(self) -> None:
        bars = generate_demo_bars(days=90, seed=12)
        config = BacktestConfig.from_dict({"symbol": "NIFTY"})
        quality = inspect_data(bars, "NIFTY")
        report = run_research(config, bars, data_source="synthetic")
        self.assertIn(quality["status"], {"pass", "warning"})
        self.assertIn("base", report["cost_stress"])
        self.assertIn("adverse", report["cost_stress"])
        self.assertIn("severe", report["cost_stress"])
        self.assertFalse(report["readiness"]["profit_is_guaranteed"])


if __name__ == "__main__":
    unittest.main()
