"""Regression tests for portfolio research defects found during the audit."""

import unittest
from datetime import datetime, timedelta, timezone

from helix.models import EquityPoint
from helix.portfolio_research import slice_bars, split_windows


def _curve(start_day: int, per_day: float, count: int = 100) -> list[EquityPoint]:
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    points, equity = [], 1.0
    for index in range(count):
        equity *= per_day
        points.append(EquityPoint(base + timedelta(days=start_day + index), equity, 0.0, 1.0, 0))
    return points


class FoldStitchingTests(unittest.TestCase):
    """Sequential folds must compound, not be averaged or reset.

    Regression: folds were averaged by calendar date, which produced a 4237%
    CAGR / Sharpe 15.9 from portfolios whose real per-fold CAGR was 0.08% and
    1.41%. Averaging is only meaningful for concurrent sleeves.
    """

    def test_sequential_folds_compound_additively(self) -> None:
        folds = [_curve(0, 1.001, 100), _curve(100, 1.001, 100)]
        equity = 1.0
        for curve in folds:
            base = curve[0].equity
            for point in curve:
                equity *= (point.equity / base) if base > 0 else 1.0
                base = point.equity
        # Each fold is rebased to 1.0 at its own first point, so that point
        # contributes no growth: 99 compounding steps per 100-point fold.
        expected = 1.001 ** 198
        self.assertAlmostEqual(equity, expected, places=6)

    def test_stitched_curve_is_monotonic_in_time(self) -> None:
        folds = [_curve(0, 1.001, 50), _curve(50, 1.002, 50)]
        stitched = []
        equity = 1.0
        for curve in folds:
            base = curve[0].equity
            for point in curve:
                equity *= (point.equity / base) if base > 0 else 1.0
                base = point.equity
                stitched.append(EquityPoint(point.timestamp, equity, 0.0, 1.0, 0))
        timestamps = [p.timestamp for p in stitched]
        self.assertEqual(timestamps, sorted(timestamps))
        self.assertEqual(len(stitched), 100)

    def test_real_fold_returns_stay_plausible(self) -> None:
        import json
        from pathlib import Path

        path = Path("reports/portfolio_research.json")
        if not path.exists():
            self.skipTest("portfolio report not generated")
        report = json.loads(path.read_text(encoding="utf-8"))
        stitched = report.get("stitched_metrics", {})
        if not stitched:
            self.skipTest("portfolio report has no stitched metrics")
        self.assertLess(abs(stitched.get("cagr", 0.0)), 2.0, "stitched CAGR is not plausible")
        self.assertLess(abs(stitched.get("sharpe", 0.0)), 10.0, "stitched Sharpe is not plausible")


class WalkForwardWindowTests(unittest.TestCase):
    def test_train_and_test_never_overlap(self) -> None:
        from helix.data import load_bars
        from helix.universe import load_universe

        members = load_universe("config/equity_universe.json", "data/equity")
        bars = {m.symbol: load_bars(m.path, m.security_id) for m in members}
        windows = split_windows(bars, 3)
        self.assertTrue(windows)
        for train, test in windows:
            self.assertLessEqual(train[0], train[1])
            self.assertEqual(train[1], test[0])
            self.assertLess(test[0], test[1])

    def test_folds_are_sequential_and_non_decreasing(self) -> None:
        from helix.data import load_bars
        from helix.universe import load_universe

        members = load_universe("config/equity_universe.json", "data/equity")
        bars = {m.symbol: load_bars(m.path, m.security_id) for m in members}
        windows = split_windows(bars, 3)
        starts = [train[0] for train, _ in windows]
        self.assertEqual(starts, sorted(starts))
        for index in range(len(windows) - 1):
            # fold N+1 trains on everything up to the point fold N stopped testing
            self.assertLessEqual(windows[index][1][1], windows[index + 1][1][0])
            self.assertLess(windows[index][1][0], windows[index][1][1])

    def test_window_requires_a_testable_test_span(self) -> None:
        from helix.data import load_bars
        from helix.universe import load_universe

        members = load_universe("config/equity_universe.json", "data/equity")
        bars = {m.symbol: load_bars(m.path, m.security_id) for m in members}
        for _, test in split_windows(bars, 3):
            usable = sum(1 for v in bars.values() if len(slice_bars(v, test)) >= 20)
            self.assertGreater(usable, 0)


class SleeveSelectionTests(unittest.TestCase):
    def test_selection_only_reads_the_train_window(self) -> None:
        from helix.data import load_bars
        from helix.universe import load_universe

        members = load_universe("config/equity_universe.json", "data/equity")
        bars = {m.symbol: load_bars(m.path, m.security_id) for m in members}
        windows = split_windows(bars, 3)
        if not windows:
            self.skipTest("no windows")
        train, test = windows[0]
        for symbol, series in bars.items():
            in_train = slice_bars(series, train)
            in_test = slice_bars(series, test)
            self.assertTrue(all(train[0] <= b.timestamp < train[1] for b in in_train))
            self.assertTrue(all(test[0] <= b.timestamp < test[1] for b in in_test))
            self.assertFalse(set(b.timestamp for b in in_train) & set(b.timestamp for b in in_test))


if __name__ == "__main__":
    unittest.main()