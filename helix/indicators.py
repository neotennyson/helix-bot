import math
from collections.abc import Sequence

from .models import Bar


def ema(values: Sequence[float], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("period must be positive")
    result: list[float | None] = [None] * len(values)
    if len(values) < period:
        return result
    seed = sum(values[:period]) / period
    result[period - 1] = seed
    multiplier = 2.0 / (period + 1.0)
    current = seed
    for index in range(period, len(values)):
        current = (values[index] - current) * multiplier + current
        result[index] = current
    return result


def sma(values: Sequence[float], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    if period <= 0 or len(values) < period:
        return result
    current = sum(values[:period]) / period
    result[period - 1] = current
    for index in range(period, len(values)):
        current = (current * (period - 1) + values[index]) / period
        result[index] = current
    return result


def rolling_std(values: Sequence[float], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    if period <= 1 or len(values) < period:
        return result
    for index in range(period - 1, len(values)):
        window = values[index - period + 1:index + 1]
        mean_value = sum(window) / period
        variance = sum((value - mean_value) ** 2 for value in window) / period
        result[index] = math.sqrt(variance)
    return result


def true_range(bars: Sequence[Bar]) -> list[float]:
    if not bars:
        return []
    result = [bars[0].high - bars[0].low]
    for previous, current in zip(bars, bars[1:]):
        result.append(
            max(
                current.high - current.low,
                abs(current.high - previous.close),
                abs(current.low - previous.close),
            )
        )
    return result


def atr(bars: Sequence[Bar], period: int) -> list[float | None]:
    ranges = true_range(bars)
    result: list[float | None] = [None] * len(bars)
    if len(bars) < period:
        return result
    current = sum(ranges[:period]) / period
    result[period - 1] = current
    for index in range(period, len(bars)):
        current = ((current * (period - 1)) + ranges[index]) / period
        result[index] = current
    return result


def rsi(values: Sequence[float], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    if len(values) <= period:
        return result
    gains = [max(values[index] - values[index - 1], 0.0) for index in range(1, len(values))]
    losses = [max(values[index - 1] - values[index], 0.0) for index in range(1, len(values))]
    average_gain = sum(gains[:period]) / period
    average_loss = sum(losses[:period]) / period
    result[period] = rsi_value(average_gain, average_loss)
    for index in range(period + 1, len(values)):
        average_gain = ((average_gain * (period - 1)) + gains[index - 1]) / period
        average_loss = ((average_loss * (period - 1)) + losses[index - 1]) / period
        result[index] = rsi_value(average_gain, average_loss)
    return result


def rsi_value(average_gain: float, average_loss: float) -> float:
    if average_loss == 0:
        return 100.0 if average_gain > 0 else 50.0
    relative_strength = average_gain / average_loss
    return 100.0 - (100.0 / (1.0 + relative_strength))


def adx(bars: Sequence[Bar], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(bars)
    if len(bars) < period * 2:
        return result
    ranges = true_range(bars)
    plus_dm: list[float] = [0.0]
    minus_dm: list[float] = [0.0]
    for previous, current in zip(bars, bars[1:]):
        up_move = current.high - previous.high
        down_move = previous.low - current.low
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)
    smoothed_range = sum(ranges[1:period + 1])
    smoothed_plus = sum(plus_dm[1:period + 1])
    smoothed_minus = sum(minus_dm[1:period + 1])
    dx_values: list[float | None] = [None] * len(bars)
    for index in range(period, len(bars)):
        if index > period:
            smoothed_range = smoothed_range - (smoothed_range / period) + ranges[index]
            smoothed_plus = smoothed_plus - (smoothed_plus / period) + plus_dm[index]
            smoothed_minus = smoothed_minus - (smoothed_minus / period) + minus_dm[index]
        if smoothed_range <= 0:
            dx_values[index] = 0.0
            continue
        plus_di = 100.0 * smoothed_plus / smoothed_range
        minus_di = 100.0 * smoothed_minus / smoothed_range
        denominator = plus_di + minus_di
        dx_values[index] = 0.0 if denominator == 0 else 100.0 * abs(plus_di - minus_di) / denominator
    valid = [value for value in dx_values if value is not None]
    if len(valid) < period:
        return result
    current = sum(valid[:period]) / period
    first_index = 2 * period - 1
    if first_index >= len(result):
        return result
    result[first_index] = current
    for index in range(first_index + 1, len(bars)):
        value = dx_values[index]
        if value is None:
            continue
        current = ((current * (period - 1)) + value) / period
        result[index] = current
    return result


def slope(values: Sequence[float | None], lookback: int) -> float | None:
    if len(values) < lookback + 1:
        return None
    current = values[-1]
    previous = values[-lookback - 1]
    if current is None or previous is None:
        return None
    return (current - previous) / lookback


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0 or not math.isfinite(denominator):
        return 0.0
    return numerator / denominator
