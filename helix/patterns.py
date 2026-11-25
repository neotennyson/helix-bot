"""Pattern-based double-bottom / double-top trading for NSE equity.

This is the first family in this project built from price *structure* rather than
indicators. Momentum, mean reversion and breakouts all read smoothed series;
a double bottom reads the raw path of price: a decline, a rally, a retest that
holds, and a break of the neckline.

The published NSE backtest behind this design reported +22.70% against a Nifty
50 benchmark of -0.07% over two years, profit factor 1.20 on a 33% win rate, and
two structural findings this module is built around:

  * payoff asymmetry carries the edge. A 15% target against a 5% stop needs to
    win only a third of the time, so the winning trades must be allowed to run
    rather than being trimmed at the first sign of a bounce
  * turnover is the whole game. At a 20-day maximum hold the strategy turns over
    roughly a dozen times a year per name, which is what lets it survive a ~0.23%
    Indian delivery round trip that kills high-frequency signals

Detection is deliberately simple and geometric, and every input is a bar at or
before the decision date.
"""

from dataclasses import dataclass, field
from typing import Any

from .backtest import calculate_metrics
from .config import CostConfig
from .costs import CostModel
from .models import Bar, EquityPoint, Side, Signal, Trade

DELIVERY_COSTS = CostConfig(
    slippage_bps=5.0, brokerage_per_order=20.0, brokerage_pct=0.001,
    exchange_txn_pct=0.0000297, stt_sell_pct=0.001, stamp_buy_pct=0.00015,
    sebi_pct=0.000001, gst_pct=0.18,
)


@dataclass
class PatternConfig:
    initial_capital: float = 5_000_000.0
    scan_bars: int = 60              # how far back to look for the structure
    min_bottom_tolerance: float = 0.04   # bottom2 must be within 4% of bottom1
    max_bottom_tolerance: float = 0.20
    min_separation: int = 4          # bars between bottom and neckline
    max_separation: int = 40
    min_neckline_break_pct: float = 0.005   # close must clear the neckline by 0.5%
    require_volume: bool = True      # volume at bottom2 >= volume at bottom1
    target_pct: float = 0.15
    stop_pct: float = 0.05
    max_holding_bars: int = 20
    allow_long: bool = True
    allow_short: bool = False
    top_n: int = 10
    max_weight: float = 0.10
    cash_hurdle_annual: float = 0.06
    costs: CostConfig = field(default_factory=lambda: DELIVERY_COSTS)


def _is_bottom(bars: list[Bar], index: int) -> bool:
    """A swing low: lower than the bars on either side."""
    if index <= 0 or index >= len(bars) - 1:
        return False
    return bars[index].low < bars[index - 1].low and bars[index].low <= bars[index + 1].low


def _is_top(bars: list[Bar], index: int) -> bool:
    """A swing high: higher than the bars on either side."""
    if index <= 0 or index >= len(bars) - 1:
        return False
    return bars[index].high > bars[index - 1].high and bars[index].high >= bars[index + 1].high


def detect_w_curve(bars: list[Bar], config: PatternConfig) -> tuple[float, int, int] | None:
    """Return (neckline, bottom1_index, bottom2_index) when a double bottom just broke."""
    n = len(bars)
    start = max(2, n - config.scan_bars)
    if n - start < config.min_separation * 2 + 4:
        return None
    bottoms = [i for i in range(start, n - 1) if _is_bottom(bars, i)]
    if len(bottoms) < 2:
        return None
    first, second = bottoms[-2], bottoms[-1]
    if not (config.min_separation <= second - first <= config.max_separation):
        return None
    b1 = bars[first].low
    b2 = bars[second].low
    if b1 <= 0:
        return None
    tolerance = abs(b2 - b1) / b1
    if not (config.min_bottom_tolerance <= tolerance <= config.max_bottom_tolerance):
        return None
    if config.require_volume and bars[second].volume > 0 and bars[first].volume > 0:
        if bars[second].volume < bars[first].volume:
            return None
    neckline = max(bar.high for bar in bars[first : second + 1])
    if neckline <= 0:
        return None
    if bars[-1].close > neckline * (1.0 + config.min_neckline_break_pct):
        return neckline, first, second
    return None


def detect_m_curve(bars: list[Bar], config: PatternConfig) -> tuple[float, int, int] | None:
    """Mirror image: double top that just broke the neckline downward."""
    n = len(bars)
    start = max(2, n - config.scan_bars)
    if n - start < config.min_separation * 2 + 4:
        return None
    tops = [i for i in range(start, n - 1) if _is_top(bars, i)]
    if len(tops) < 2:
        return None
    first, second = tops[-2], tops[-1]
    if not (config.min_separation <= second - first <= config.max_separation):
        return None
    t1 = bars[first].high
    t2 = bars[second].high
    if t1 <= 0:
        return None
    tolerance = abs(t2 - t1) / t1
    if not (config.min_bottom_tolerance <= tolerance <= config.max_bottom_tolerance):
        return None
    neckline = min(bar.low for bar in bars[first : second + 1])
    if neckline <= 0:
        return None
    if bars[-1].close < neckline * (1.0 - config.min_neckline_break_pct):
        return neckline, first, second
    return None


def run_patterns(config: PatternConfig, daily: dict[str, list[Bar]]) -> dict[str, Any]:
    sessions = sorted({bar.timestamp.date() for bars in daily.values() for bar in bars})
    warmup = config.scan_bars + config.max_separation + 5
    if len(sessions) < warmup + 60:
        return {"status": "insufficient_data", "reason": "not enough sessions"}
    costs = CostModel(config.costs)
    index = {symbol: {bar.timestamp.date(): bar for bar in bars} for symbol, bars in daily.items()}
    history = {symbol: sorted(bars, key=lambda b: b.timestamp) for symbol, bars in daily.items()}

    cash = config.initial_capital
    positions: dict[str, dict[str, Any]] = {}
    curve: list[EquityPoint] = []
    trades: list[Trade] = []
    peak = config.initial_capital

    for session in range(warmup, len(sessions)):
        day = sessions[session]
        # ---- manage open positions: stop, target, time exit --------------------
        for symbol in list(positions):
            position = positions[symbol]
            bar = index[symbol].get(day)
            if bar is None:
                continue
            position["bars_held"] += 1
            entry = position["entry_price"]
            if position["side"] == Side.LONG:
                stop_hit = bar.low <= entry * (1.0 - config.stop_pct)
                target_hit = bar.high >= entry * (1.0 + config.target_pct)
            else:
                stop_hit = bar.high >= entry * (1.0 + config.stop_pct)
                target_hit = bar.low <= entry * (1.0 - config.target_pct)
            reason = None
            if stop_hit:
                reason = "stop_loss"
            elif target_hit:
                reason = "target"
            elif position["bars_held"] >= config.max_holding_bars:
                reason = "time_exit"
            if reason:
                cash += _close(positions.pop(symbol), bar, costs, reason, trades)

        # ---- scan for fresh patterns, filled at the next bar -------------------
        if not positions and len(positions) == 0:
            candidates: list[tuple[str, Side]] = []
            for symbol, bars in history.items():
                window = [b for b in bars if b.timestamp.date() <= sessions[session - 1]]
                if len(window) < warmup:
                    continue
                if config.allow_long:
                    if detect_w_curve(window, config) is not None:
                        candidates.append((symbol, Side.LONG))
                        continue
                if config.allow_short and detect_m_curve(window, config) is not None:
                    candidates.append((symbol, Side.SHORT))
            for symbol, side in candidates[: config.top_n]:
                bar = index[symbol].get(day)
                if bar is None or bar.open <= 0:
                    continue
                is_buy = side == Side.LONG
                price = costs.fill_price(bar.open, is_buy)
                weight = min(config.max_weight, 1.0 / max(1, min(len(candidates), config.top_n)))
                quantity = int((weight * max(peak, 1.0)) / max(price, 1e-9))
                if quantity < 1:
                    continue
                fee = costs.side_cost(is_buy, bar.open, quantity, 1.0).total
                cash -= fee
                positions[symbol] = {
                    "symbol": symbol, "side": side, "quantity": quantity, "entry_price": price,
                    "entry_time": bar.timestamp, "entry_fee": fee, "bars_held": 0,
                    "reason": "w_curve_breakout" if is_buy else "m_curve_breakdown",
                }

        equity = cash
        stamp = None
        price = 0.0
        for symbol, position in positions.items():
            bar = index[symbol].get(day)
            if bar is not None:
                direction = 1 if position["side"] == Side.LONG else -1
                equity += position["quantity"] * (bar.close - position["entry_price"]) * direction
                if stamp is None:
                    stamp = bar.timestamp
                    price = bar.close
        if stamp is None:
            if len(curve):
                curve.append(EquityPoint(curve[-1].timestamp, cash, 0.0, curve[-1].price, 0))
            continue
        peak = max(peak, equity)
        curve.append(EquityPoint(stamp, equity, equity / peak - 1.0 if peak else 0.0, price, sum(p["quantity"] for p in positions.values())))

    for symbol in list(positions):
        bar = index[symbol].get(sessions[-1])
        if bar is not None:
            cash += _close(positions.pop(symbol), bar, costs, "end_of_data", trades)
    if curve:
        curve[-1] = EquityPoint(curve[-1].timestamp, cash, 0.0, curve[-1].price, 0)

    metrics = calculate_metrics(curve, trades, config.initial_capital, 1440)
    metrics["excess_cagr_over_cash"] = float(metrics.get("cagr", 0.0)) - config.cash_hurdle_annual
    return {
        "status": "ok",
        "method": "w_curve_double_bottom_pattern",
        "sessions": len(sessions),
        "metrics": metrics,
        "exit_reason_counts": {
            reason: sum(1 for t in trades if t.exit_reason == reason)
            for reason in sorted({t.exit_reason for t in trades})
        },
    }



def _close(position: dict[str, Any], bar: Bar, costs: CostModel, reason: str, trades: list[Trade]) -> float:
    is_buy = position["side"] == Side.SHORT
    reference = bar.open if reason == "end_of_data" else bar.close
    exit_price = costs.fill_price(reference, is_buy)
    fee = costs.side_cost(is_buy, reference, position["quantity"], 1.0).total
    direction = 1 if position["side"] == Side.LONG else -1
    gross = (exit_price - position["entry_price"]) * direction * position["quantity"]
    trades.append(
        Trade(
            symbol=position["symbol"], side=position["side"], quantity=position["quantity"],
            entry_time=position["entry_time"], exit_time=bar.timestamp,
            entry_price=position["entry_price"], exit_price=exit_price,
            gross_pnl=gross, fees=position["entry_fee"] + fee,
            net_pnl=gross - position["entry_fee"] - fee,
            exit_reason=reason, entry_reason=position["reason"],
            bars_held=position["bars_held"], multiplier=1.0,
        )
    )
    return gross - position["entry_fee"] - fee