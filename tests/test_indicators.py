import unittest

from helix.indicators import adx, atr, ema, rsi
from helix.models import Bar
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


class IndicatorTests(unittest.TestCase):
    def setUp(self) -> None:
        timestamp = datetime(2024, 1, 1, 9, 15, tzinfo=ZoneInfo("Asia/Kolkata"))
        self.bars = [
            Bar(timestamp=timestamp + timedelta(minutes=15 * index), symbol="NIFTY", open=100 + index, high=102 + index, low=99 + index, close=101 + index, volume=1000)
            for index in range(80)
        ]

    def test_ema_and_rsi_warmup(self) -> None:
        values = [bar.close for bar in self.bars]
        fast = ema(values, 10)
        rsi_values = rsi(values, 14)
        self.assertIsNone(fast[8])
        self.assertIsNotNone(fast[9])
        self.assertIsNone(rsi_values[13])
        self.assertIsNotNone(rsi_values[14])

    def test_atr_and_adx_are_positive_for_trend(self) -> None:
        atr_values = atr(self.bars, 14)
        adx_values = adx(self.bars, 14)
        self.assertGreater(atr_values[-1], 0)
        self.assertGreater(adx_values[-1], 0)


if __name__ == "__main__":
    unittest.main()
