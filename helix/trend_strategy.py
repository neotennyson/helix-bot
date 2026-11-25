from datetime import time
from math import isfinite
from zoneinfo import ZoneInfo

from .config import StrategyConfig
from .session import SessionWindow
from .indicators import adx, atr, ema, safe_ratio
from .models import Bar, Regime, Side, Signal


class TrendBreakoutStrategy:
    def __init__(self, config: StrategyConfig | None = None, timezone: str = "Asia/Kolkata") -> None:
        self.config = config or StrategyConfig()
        self.timezone = ZoneInfo(timezone)
        self.session = SessionWindow.build(self.config, timezone)
        self._last_adx = 0.0

    def signal(self, bars: list[Bar]) -> Signal | None:
        warmup = max(self.config.slow_ema, self.config.breakout_period, self.config.adx_period * 2, self.config.long_term_ema + self.config.long_term_slope_bars if self.config.use_long_term_filter else 0) + 5
        if len(bars) < warmup:
            return None
        current = bars[-1]
        if not self.session.allows_entry(current):
            return None
        closes = [bar.close for bar in bars]
        highs = [bar.high for bar in bars]
        lows = [bar.low for bar in bars]
        fast_values = ema(closes, self.config.fast_ema)
        slow_values = ema(closes, self.config.slow_ema)
        long_values = ema(closes, self.config.long_term_ema) if self.config.use_long_term_filter else [None] * len(closes)
        atr_values = atr(bars, self.config.atr_period)
        adx_values = adx(bars, self.config.adx_period)
        fast = fast_values[-1]
        slow = slow_values[-1]
        long_term = long_values[-1]
        if not self.config.use_long_term_filter:
            long_term = current.close
        long_slope = 0.0
        if self.config.use_long_term_filter:
            previous_long = long_values[-self.config.long_term_slope_bars - 1]
            if previous_long is None:
                return None
            long_slope = long_term - previous_long
        previous_fast = fast_values[-2]
        previous_slow = slow_values[-2]
        current_atr = atr_values[-1]
        current_adx = adx_values[-1]
        if any(value is None for value in (fast, slow, long_term, previous_fast, previous_slow, current_atr, current_adx)):
            return None
        if current_atr <= 0 or not all(isfinite(float(value)) for value in (fast, slow, current_atr, current_adx)):
            return None
        self._last_adx = float(current_adx)
        atr_pct = safe_ratio(current_atr, current.close)
        if atr_pct < self.config.min_atr_pct or atr_pct > self.config.max_atr_pct:
            return None
        average_volume = safe_ratio(sum(bar.volume for bar in bars[-20:]), 20.0)
        if average_volume > 0 and safe_ratio(current.volume, average_volume) < self.config.min_volume_ratio:
            return None
        period = self.config.breakout_period
        prior_high = max(highs[-period - 1:-1])
        prior_low = min(lows[-period - 1:-1])
        long_alignment = not self.config.use_long_term_filter or (fast > slow > long_term and current.close > long_term and long_slope > 0)
        short_alignment = not self.config.use_long_term_filter or (fast < slow < long_term and current.close < long_term and long_slope < 0)
        long_regime = fast > slow and current.close > slow and current_adx >= self.config.trend_adx_min and long_alignment
        short_regime = self.config.allow_short and fast < slow and current.close < fast and current_adx >= self.config.trend_adx_min and short_alignment
        long_cross = previous_fast <= previous_slow and fast > slow
        short_cross = previous_fast >= previous_slow and fast < slow
        long_breakout = current.close > prior_high
        short_breakout = self.config.allow_short and current.close < prior_low
        if self.config.use_long_term_filter and self.config.allow_short and fast < slow and current.close < long_term and long_slope < 0:
            return self._make_signal(current, Side.SHORT, 0.58, "long_term_flip_short", current_atr)
        if self.config.use_long_term_filter and fast > slow and current.close > long_term and long_slope > 0:
            return self._make_signal(current, Side.LONG, 0.58, "long_term_flip_long", current_atr)
        if long_regime and (long_cross or long_breakout):
            confidence = min(0.95, 0.62 + min(current_adx / 250.0, 0.2) + (0.08 if long_breakout else 0.0))
            return self._make_signal(current, Side.LONG, confidence, "trend_breakout_long" if long_breakout else "ema_cross_long", current_atr)
        if short_regime and (short_cross or short_breakout):
            confidence = min(0.95, 0.62 + min(current_adx / 250.0, 0.2) + (0.08 if short_breakout else 0.0))
            return self._make_signal(current, Side.SHORT, confidence, "trend_breakout_short" if short_breakout else "ema_cross_short", current_atr)
        if short_regime:
            return self._make_signal(current, Side.SHORT, 0.58, "trend_flip_short", current_atr)
        if long_regime:
            return self._make_signal(current, Side.LONG, 0.58, "trend_flip_long", current_atr)
        return None

    def should_flatten(self, bar: Bar) -> bool:
        return self.session.should_flatten(bar)

    def _make_signal(self, bar: Bar, side: Side, confidence: float, reason: str, current_atr: float) -> Signal | None:
        if confidence < self.config.confidence_threshold:
            return None
        stop_distance = max(current_atr * self.config.stop_atr_multiple, bar.close * 0.0005)
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
            metadata={"price": bar.close, "adx": round(self._last_adx, 4)},
        )
