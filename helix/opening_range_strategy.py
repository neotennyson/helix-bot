from math import isfinite
from zoneinfo import ZoneInfo

from .config import StrategyConfig
from .session import SessionWindow
from .indicators import adx, atr, ema, rsi
from .models import Bar, Regime, Side, Signal


class OpeningRangeBreakoutStrategy:
    """Intraday opening-range breakout on a fixed-bar chart.

    Builds a range from the first `opening_range_bars` bars of the session,
    then trades a close beyond that range when ATR expansion and trend
    direction agree. Flattening happens through `should_flatten`, so the
    existing position-expiry path handles the exit.
    """

    def __init__(self, config: StrategyConfig | None = None, timezone: str = "Asia/Kolkata") -> None:
        self.config = config or StrategyConfig()
        self.timezone = ZoneInfo(timezone)
        self.session = SessionWindow.build(self.config, timezone)
        self._opening_range_bars = max(2, int(self.config.opening_range_bars))
        self._last_adx = 0.0
        self._last_rsi = 0.0

    def signal(self, bars: list[Bar]) -> Signal | None:
        warmup = max(self._opening_range_bars + 3, self.config.atr_period * 2, self.config.adx_period * 2) + 5
        if len(bars) < warmup:
            return None
        current = bars[-1]
        if not self.session.allows_entry(current):
            return None
        range_bars = self._current_session(bars)
        if len(range_bars) < self._opening_range_bars + 1:
            return None
        opening = range_bars[: self._opening_range_bars]
        range_high = max(bar.high for bar in opening)
        range_low = min(bar.low for bar in opening)
        range_height = max(range_high - range_low, 1e-9)
        atr_values = atr(bars, self.config.atr_period)
        adx_values = adx(bars, self.config.adx_period)
        closes = [bar.close for bar in bars]
        fast_values = ema(closes, self.config.fast_ema)
        slow_values = ema(closes, self.config.slow_ema)
        rsi_values = rsi(closes, self.config.rsi_period)
        current_atr = atr_values[-1]
        current_adx = adx_values[-1]
        fast_ema = fast_values[-1]
        slow_ema = slow_values[-1]
        current_rsi = rsi_values[-1]
        if any(value is None for value in (current_atr, current_adx, fast_ema, slow_ema, current_rsi)):
            return None
        if current_atr <= 0 or not all(
            isfinite(float(value)) for value in (current_atr, current_adx, fast_ema, slow_ema, current_rsi)
        ):
            return None
        atr_pct = current_atr / current.close
        if atr_pct < self.config.min_atr_pct or atr_pct > self.config.max_atr_pct:
            return None
        self._last_adx = float(current_adx)
        self._last_rsi = float(current_rsi)
        if range_height <= self.config.opening_range_max_atr * float(current_atr):
            return None
        upward_stack = float(fast_ema) >= float(slow_ema)
        if current.close > range_high and float(current_adx) >= self.config.trend_adx_min and upward_stack:
            strength = min(1.0, (current.close - range_high) / max(float(current_atr), 1e-9))
            confidence = 0.56 + 0.22 * strength
            if confidence < self.config.confidence_threshold:
                return None
            return self._make_signal(current, Side.LONG, min(0.95, confidence), "orb_long", current_atr, range_low)
        if (
            self.config.allow_short
            and current.close < range_low
            and float(current_adx) >= self.config.trend_adx_min
            and not upward_stack
        ):
            strength = min(1.0, (range_low - current.close) / max(float(current_atr), 1e-9))
            confidence = 0.56 + 0.22 * strength
            if confidence < self.config.confidence_threshold:
                return None
            return self._make_signal(current, Side.SHORT, min(0.95, confidence), "orb_short", current_atr, range_high)
        return self._make_signal(current, Side.FLAT, 0.5, "inside_opening_range", current_atr, range_low)

    def should_flatten(self, bar: Bar) -> bool:
        return self.session.should_flatten(bar)

    def _current_session(self, bars: list[Bar]) -> list[Bar]:
        latest = bars[-1].timestamp.astimezone(self.timezone).date()
        session: list[Bar] = []
        for bar in reversed(bars):
            if bar.timestamp.astimezone(self.timezone).date() != latest:
                break
            session.append(bar)
        return list(reversed(session))

    def _make_signal(self, bar: Bar, side: Side, confidence: float, reason: str, current_atr: float, invalidation: float) -> Signal:
        stop_distance = max(current_atr * self.config.stop_atr_multiple, abs(bar.close - invalidation) * 0.5)
        return Signal(
            timestamp=bar.timestamp,
            symbol=bar.symbol,
            side=side,
            confidence=confidence,
            regime=Regime.TREND,
            reason=reason,
            atr=current_atr,
            stop_distance=stop_distance,
            target_r=self.config.target_r,
            metadata={
                "price": bar.close,
                "adx": round(self._last_adx, 4),
                "rsi": round(self._last_rsi, 4),
            },
        )