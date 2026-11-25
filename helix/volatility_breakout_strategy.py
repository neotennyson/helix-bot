from math import isfinite
from zoneinfo import ZoneInfo

from .config import StrategyConfig
from .session import SessionWindow
from .indicators import adx, atr, ema, rsi
from .models import Bar, Regime, Side, Signal


class VolatilityBreakoutStrategy:
    """ATR/Donchian hybrid breakout with a volatility-regime gate.

    Enters when price clears a channel built from recent bar extremes, backed by
    trend strength (ADX) and a directional EMA stack. Returns FLAT while price
    sits inside the channel so an open position is released at the mean.
    """

    def __init__(self, config: StrategyConfig | None = None, timezone: str = "Asia/Kolkata") -> None:
        self.config = config or StrategyConfig()
        self.timezone = ZoneInfo(timezone)
        self.session = SessionWindow.build(self.config, timezone)
        self._last_adx = 0.0
        self._last_rsi = 0.0

    def signal(self, bars: list[Bar]) -> Signal | None:
        period = self.config.breakout_period
        warmup = max(period + 2, self.config.slow_ema, self.config.atr_period * 2, self.config.adx_period * 2) + 5
        if len(bars) < warmup:
            return None
        current = bars[-1]
        if not self.session.allows_entry(current):
            return None
        window = bars[-(period + 1) : -1]
        upper = max(bar.high for bar in window)
        lower = min(bar.low for bar in window)
        closes = [bar.close for bar in bars]
        atr_values = atr(bars, self.config.atr_period)
        adx_values = adx(bars, self.config.adx_period)
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
        channel_width = max(upper - lower, 1e-9)
        wide_channel = channel_width >= self.config.trend_threshold_atr * current_atr
        upward_stack = float(fast_ema) > float(slow_ema)
        if current.close > upper and wide_channel and float(current_adx) >= self.config.trend_adx_min and upward_stack:
            strength = min(1.0, (current.close - upper) / max(float(current_atr), 1e-9))
            confidence = 0.55 + 0.2 * strength
            confidence += 0.05 if float(current_rsi) >= 55.0 else -0.05
            if confidence < self.config.confidence_threshold:
                return None
            return self._make_signal(current, Side.LONG, min(0.95, confidence), "volatility_breakout_long", current_atr)
        if (
            self.config.allow_short
            and current.close < lower
            and wide_channel
            and float(current_adx) >= self.config.trend_adx_min
            and not upward_stack
        ):
            strength = min(1.0, (lower - current.close) / max(float(current_atr), 1e-9))
            confidence = 0.55 + 0.2 * strength
            confidence += 0.05 if float(current_rsi) <= 45.0 else -0.05
            if confidence < self.config.confidence_threshold:
                return None
            return self._make_signal(current, Side.SHORT, min(0.95, confidence), "volatility_breakout_short", current_atr)
        reason = "mid_channel_flat" if current.close <= (upper + lower) / 2.0 else "wait_breakout"
        return self._make_signal(current, Side.FLAT, 0.5, reason, current_atr)

    def should_flatten(self, bar: Bar) -> bool:
        return self.session.should_flatten(bar)

    def _make_signal(self, bar: Bar, side: Side, confidence: float, reason: str, current_atr: float) -> Signal:
        stop_distance = max(current_atr * self.config.stop_atr_multiple, bar.close * 0.0015)
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