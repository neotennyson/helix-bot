import unittest

from helix.broker_data import _parse_payload


class BrokerDataTests(unittest.TestCase):
    def test_groww_candle_array_is_parsed(self) -> None:
        payload = {
            "status": "SUCCESS",
            "payload": {
                "candles": [[1704065700, 100.0, 102.0, 99.0, 101.0, 1000, 10]],
            },
        }
        bars = _parse_payload(payload, "NSE-NIFTY", "2024-01-04")
        self.assertEqual(len(bars), 1)
        self.assertEqual(bars[0].close, 101.0)
        self.assertEqual(bars[0].expiry, "2024-01-04")

    def test_dhan_column_payload_is_parsed(self) -> None:
        payload = {
            "status": "success",
            "data": {
                "timestamp": [1704065700],
                "open": [100.0],
                "high": [102.0],
                "low": [99.0],
                "close": [101.0],
                "volume": [1000],
            },
        }
        bars = _parse_payload(payload, "security-1", None)
        self.assertEqual(len(bars), 1)
        self.assertEqual(bars[0].symbol, "security-1")
        self.assertEqual(bars[0].volume, 1000)
    def test_daily_parser_can_skip_malformed_rows(self) -> None:
        payload = {
            "timestamp": [1704065700, 1704070200],
            "open": [100.0, 100.0],
            "high": [102.0, 99.0],
            "low": [99.0, 101.0],
            "close": [101.0, 100.0],
            "volume": [1000, 1000],
        }
        bars = _parse_payload(payload, "security-1", None, strict=False)
        self.assertEqual(len(bars), 1)
        self.assertEqual(bars[0].timestamp.isoformat(), "2024-01-01T05:05:00+05:30")


if __name__ == "__main__":
    unittest.main()
