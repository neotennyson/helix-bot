from math import isfinite
from zoneinfo import ZoneInfo

from .config import StrategyConfig
from .session import SessionWindow
from .indicators import adx, atr, rolling_std, rsi, sma
from .models import Bar, Regime, Side, Signal


class MeanReversionStrategy:
    def __init__(self, config: StrategyConfig | None = None, timezone: str = "Asia/Kolkata") -> None:
        self.config = config or StrategyConfig()
        self.timezone = ZoneInfo(timezone)
        self.session = SessionWindow.build(self.config, timezone)
        self._last_adx = 0.0
        self._last_rsi = 0.0

    def signal(self, bars: list[Bar]) -> Signal | None:
        period = self.config.breakout_period
        warmup = max(period, self.config.atr_period * 2, self.config.adx_period * 2) + 5
        if len(bars) < warmup:
            return None
        current = bars[-1]
        if not self.session.allows_entry(current):
            return None
        closes = [bar.close for bar in bars]
        means = sma(closes, period)
        deviations = rolling_std(closes, period)
        rsi_values = rsi(closes, self.config.rsi_period)
        atr_values = atr(bars, self.config.atr_period)
        adx_values = adx(bars, self.config.adx_period)
        mean_value = means[-1]
        deviation = deviations[-1]
        current_rsi = rsi_values[-1]
        current_atr = atr_values[-1]
        current_adx = adx_values[-1]
        if any(value is None for value in (mean_value, deviation, current_rsi, current_atr, current_adx)):
            return None
        if current_atr <= 0 or not all(isfinite(float(value)) for value in (mean_value, deviation, current_rsi, current_atr, current_adx)):
            return None
        self._last_rsi = float(current_rsi)
        self._last_adx = float(current_adx)
        if current_adx >= self.config.trend_adx_min:
            return None
        lower = mean_value - self.config.range_threshold_atr * deviation
        upper = mean_value + self.config.range_threshold_atr * deviation
        if current.close < lower and current_rsi <= self.config.rsi_range_long:
            distance = abs(lower - current.close) / max(current_atr, 1e-9)
            return self._make_signal(current, Side.LONG, min(0.9, 0.58 + distance * 0.08), "range_reversion_long", current_atr)
        if self.config.allow_short and current.close > upper and current_rsi >= self.config.rsi_range_short:
            distance = abs(current.close - upper) / max(current_atr, 1e-9)
            return self._make_signal(current, Side.SHORT, min(0.9, 0.58 + distance * 0.08), "range_reversion_short", current_atr)
        if current.close >= mean_value:
            return self._make_signal(current, Side.FLAT, 0.5, "mean_exit_or_wait", current_atr)
        return self._make_signal(current, Side.FLAT, 0.5, "mean_exit_or_wait", current_atr)

    def should_flatten(self, bar: Bar) -> bool:
        return self.session.should_flatten(bar)

    def _make_signal(self, bar: Bar, side: Side, confidence: float, reason: str, current_atr: float) -> Signal:
        stop_distance = max(current_atr * self.config.stop_atr_multiple, bar.close * 0.0005)
        return Signal(
            timestamp=bar.timestamp,
            symbol=bar.symbol,
            side=side,
            confidence=confidence,
            regime=Regime.RANGE,
            reason=reason,
            atr=current_atr,
            stop_distance=stop_distance,
            target_r=self.config.target_r,
            metadata={"price": bar.close, "adx": round(self._last_adx, 4), "rsi": round(self._last_rsi, 4)},
        )