from dataclasses import dataclass
from datetime import time
from math import isfinite
from zoneinfo import ZoneInfo

from .config import StrategyConfig
from .data import parse_time
from .indicators import adx, atr, ema, rsi, safe_ratio, slope
from .models import Bar, Regime, Side, Signal


@dataclass
class IndicatorState:
    fast: float
    slow: float
    atr: float
    rsi: float
    adx: float
    volume_ratio: float
    fast_slope: float
    slow_slope: float


class RegimeStrategy:
    def __init__(self, config: StrategyConfig | None = None, timezone: str = "Asia/Kolkata") -> None:
        self.config = config or StrategyConfig()
        self.timezone = ZoneInfo(timezone)
        self._session_start = parse_time(self.config.session_start)
        self._flatten_at = parse_time(self.config.flatten_at)
        self._expiry_entry_cutoff = parse_time(self.config.expiry_entry_cutoff)
        self._expiry_flatten_at = parse_time(self.config.expiry_flatten_at)
        self._last_rsi = 0.0
        self._last_adx = 0.0
        self._last_daily_bias = None

    def signal(self, bars: list[Bar]) -> Signal | None:
        if len(bars) < max(self.config.slow_ema, self.config.atr_period, self.config.adx_period * 2) + 5:
            return None
        current = bars[-1]
        local_time = current.timestamp.astimezone(self.timezone).time().replace(tzinfo=None)
        if local_time < self._session_start or (self.config.intraday_only and local_time >= self._flatten_at):
            return None
        if current.expiry and local_time >= self._expiry_entry_cutoff:
            return None
        closes = [bar.close for bar in bars]
        fast_values = ema(closes, self.config.fast_ema)
        slow_values = ema(closes, self.config.slow_ema)
        atr_values = atr(bars, self.config.atr_period)
        rsi_values = rsi(closes, self.config.rsi_period)
        adx_values = adx(bars, self.config.adx_period)
        fast = fast_values[-1]
        slow = slow_values[-1]
        current_atr = atr_values[-1]
        current_rsi = rsi_values[-1]
        current_adx = adx_values[-1]
        if any(value is None for value in (fast, slow, current_atr, current_rsi, current_adx)):
            return None
        if current_atr <= 0 or not all(isfinite(float(value)) for value in (fast, slow, current_atr, current_rsi, current_adx)):
            return None
        average_volume = safe_ratio(sum(bar.volume for bar in bars[-20:]), 20.0)
        volume_ratio = safe_ratio(current.volume, average_volume)
        if average_volume > 0 and volume_ratio < self.config.min_volume_ratio:
            return None
        atr_pct = safe_ratio(current_atr, current.close)
        if atr_pct < self.config.min_atr_pct or atr_pct > self.config.max_atr_pct:
            return None
        fast_slope_value = slope(fast_values, min(5, self.config.fast_ema - 1))
        slow_slope_value = slope(slow_values, min(10, self.config.slow_ema - 1))
        if fast_slope_value is None or slow_slope_value is None:
            return None
        self._last_rsi = float(current_rsi)
        self._last_adx = float(current_adx)
        self._last_daily_bias = self._daily_bias(bars, current)
        spread_atr = abs(fast - slow) / current_atr
        trend = current_adx >= self.config.trend_adx_min and spread_atr >= self.config.trend_threshold_atr
        if trend:
            if fast > slow and current.close > fast and current_rsi >= self.config.rsi_trend_low and current_rsi <= self.config.rsi_trend_high and fast_slope_value > 0 and (self._last_daily_bias is None or self._last_daily_bias >= 0):
                confidence = min(0.95, 0.58 + spread_atr * 0.18 + min(current_adx / 100.0, 0.2))
                return self._make_signal(current, Side.LONG, confidence, Regime.TREND, "ema_trend_long", current_atr)
            if self.config.allow_short and fast < slow and current.close < fast and current_rsi >= 100.0 - self.config.rsi_trend_high and current_rsi <= 100.0 - self.config.rsi_trend_low and fast_slope_value < 0 and (self._last_daily_bias is None or self._last_daily_bias <= 0):
                confidence = min(0.95, 0.58 + spread_atr * 0.18 + min(current_adx / 100.0, 0.2))
                return self._make_signal(current, Side.SHORT, confidence, Regime.TREND, "ema_trend_short", current_atr)
            return None
        if current_adx > self.config.range_adx_max:
            return None
        distance_atr = (current.close - slow) / current_atr
        if distance_atr <= -self.config.range_threshold_atr and current_rsi <= self.config.rsi_range_long:
            confidence = min(0.9, 0.56 + abs(distance_atr - -self.config.range_threshold_atr) * 0.2)
            return self._make_signal(current, Side.LONG, confidence, Regime.RANGE, "range_reversion_long", current_atr)
        if self.config.allow_short and distance_atr >= self.config.range_threshold_atr and current_rsi >= self.config.rsi_range_short:
            confidence = min(0.9, 0.56 + abs(distance_atr - self.config.range_threshold_atr) * 0.2)
            return self._make_signal(current, Side.SHORT, confidence, Regime.RANGE, "range_reversion_short", current_atr)
        return None

    def should_flatten(self, bar: Bar) -> bool:
        local_time = bar.timestamp.astimezone(self.timezone).time().replace(tzinfo=None)
        if bar.expiry and local_time >= self._expiry_flatten_at:
            return True
        return self.config.intraday_only and local_time >= self._flatten_at

    def _daily_bias(self, bars: list[Bar], current: Bar) -> int | None:
        if not self.config.use_daily_bias:
            return None
        daily_closes: dict[object, float] = {}
        for bar in bars:
            if bar.timestamp.date() < current.timestamp.date():
                daily_closes[bar.timestamp.date()] = bar.close
        values = [daily_closes[key] for key in sorted(daily_closes)]
        if len(values) < self.config.daily_slow_ema:
            return None
        fast_values = ema(values, self.config.daily_fast_ema)
        slow_values = ema(values, self.config.daily_slow_ema)
        fast = fast_values[-1]
        slow = slow_values[-1]
        if fast is None or slow is None:
            return None
        if fast > slow:
            return 1
        if fast < slow:
            return -1
        return 0

    def _make_signal(self, bar: Bar, side: Side, confidence: float, regime: Regime, reason: str, current_atr: float) -> Signal | None:
        if confidence < self.config.confidence_threshold:
            return None
        stop_distance = max(
            current_atr * self.config.stop_atr_multiple,
            bar.close * 0.0005,
        )
        return Signal(
            timestamp=bar.timestamp,
            symbol=bar.symbol,
            side=side,
            confidence=confidence,
            regime=regime,
            reason=reason,
            atr=current_atr,
            stop_distance=stop_distance,
            target_r=self.config.target_r,
            metadata={
                "price": bar.close,
                "adx": round(self._last_adx, 4),
                "rsi": round(self._last_rsi, 4),
                "daily_bias": self._last_daily_bias,
            },
        )

    def prepare_signal_metadata(self, bars: list[Bar]) -> None:
        closes = [bar.close for bar in bars]
        rsi_values = rsi(closes, self.config.rsi_period)
        adx_values = adx(bars, self.config.adx_period)
        self._last_rsi = float(rsi_values[-1] or 0.0)
        self._last_adx = float(adx_values[-1] or 0.0)
