"""Intraday session strategy for NSE cash equity.

Two structural advantages over the daily work: positions are flat by the close
(overnight gap risk removed) and Indian intraday costs are materially lower than
delivery (no stamp duty, lower STT), so a signal only has to clear a fraction of
the daily hurdle to be worth trading.

Bars are hourly, 09:15 through 15:15, seven per session. Every entry carries a
hard session-exit so nothing is ever held overnight.
"""

from dataclasses import dataclass, field
from typing import Any

from .backtest import calculate_metrics
from .config import CostConfig
from .costs import CostModel
from .indicators import adx, atr, ema, rsi
from .models import Bar, EquityPoint, Regime, Side, Signal, Trade

SESSION_OPEN = "09:15"
SESSION_LAST_BAR = "15:15"
FLATTEN_AT = "15:10"          # exit before the final bar's close


@dataclass
class IntradayConfig:
    initial_capital: float = 5_000_000.0
    opening_range_bars: int = 2
    trend_ema: int = 8
    slow_ema: int = 21
    adx_period: int = 7
    atr_period: int = 7
    rsi_period: int = 7
    min_adx: float = 18.0
    max_atr_pct: float = 0.012
    min_atr_pct: float = 0.0015
    stop_atr: float = 1.2
    target_r: float = 1.6
    max_bars_in_trade: int = 4
    risk_per_trade_pct: float = 0.0075
    max_notional_pct: float = 1.0
    max_trades_per_day: int = 2
    allow_long: bool = True
    allow_short: bool = True
    cash_hurdle_annual: float = 0.06
    costs: CostConfig = field(
        default_factory=lambda: CostConfig(
            slippage_bps=3.0,
            brokerage_per_order=20.0,
            brokerage_pct=0.0,        # intraday delivery-brokerage is not charged
            exchange_txn_pct=0.0000297,
            stt_sell_pct=0.00025,     # intraday STT 0.025% on sell
            stamp_buy_pct=0.00002,    # intraday stamp 0.002% on buy
            sebi_pct=0.000001,
            gst_pct=0.18,
        )
    )


def _session(bars: list[Bar]) -> list[Bar]:
    return [bar for bar in bars if bar.timestamp.date() == bars[-1].timestamp.date()]


def run_intraday(config: IntradayConfig, bars: list[Bar]) -> dict[str, Any]:
    data = sorted(bars, key=lambda bar: bar.timestamp)
    warmup = max(config.slow_ema, config.adx_period * 2, config.atr_period * 2, config.opening_range_bars) + 6
    if len(data) < warmup + 40:
        return {"status": "insufficient_data", "reason": f"need {warmup + 40} bars, have {len(data)}"}

    cost_model = CostModel(config.costs)
    cash = config.initial_capital
    position: dict[str, Any] | None = None
    pending: dict[str, Any] | None = None
    equity_curve: list[EquityPoint] = []
    trades: list[Trade] = []
    peak = config.initial_capital
    trades_today = 0
    current_day = None

    for index in range(warmup, len(data)):
        bar = data[index]
        day = bar.timestamp.date()
        if day != current_day:
            current_day = day
            trades_today = 0
            # force flat across sessions
            if position is not None:
                cash = _close(position, bar, index, cash, trades, cost_model, "session_close", 1.0)
                position = None
            pending = None

        # 1. execute anything queued on the previous bar
        if pending is not None:
            if pending["kind"] == "enter" and position is None and trades_today < config.max_trades_per_day:
                position, cash = _open(pending["signal"], bar, index, cash, config, cost_model)
                if position is not None:
                    trades_today += 1
            elif pending["kind"] == "exit" and position is not None:
                cash = _close(position, bar, index, cash, trades, cost_model, pending["reason"], 1.0)
                position = None
            pending = None

        # 2. manage an open position
        if position is not None:
            exit_reason = None
            if position["side"] == Side.LONG:
                if bar.low <= position["stop"]:
                    exit_reason = "stop_loss"
                elif bar.high >= position["target"]:
                    exit_reason = "take_profit"
            else:
                if bar.high >= position["stop"]:
                    exit_reason = "stop_loss"
                elif bar.low <= position["target"]:
                    exit_reason = "take_profit"
            if exit_reason is None and position["bars_held"] + 1 >= config.max_bars_in_trade:
                exit_reason = "max_bars"
            if exit_reason is None and bar.timestamp.strftime("%H:%M") >= FLATTEN_AT:
                exit_reason = "session_flatten"
            if exit_reason is not None:
                cash = _close(position, bar, index, cash, trades, cost_model, exit_reason, 1.0)
                position = None
            else:
                position["bars_held"] += 1

        # 3. mark and record
        equity = cash + (position["unrealized"](bar.close) if position is not None else 0.0)
        peak = max(peak, equity)
        equity_curve.append(
            EquityPoint(bar.timestamp, equity, equity / peak - 1.0 if peak else 0.0, bar.close, position["quantity"] if position else 0)
        )

        # 4. queue a decision for the next bar
        if index == len(data) - 1:
            continue
        if position is not None or pending is not None:
            continue
        if bar.timestamp.strftime("%H:%M") >= FLATTEN_AT:
            continue
        if trades_today >= config.max_trades_per_day:
            continue
        signal = _signal(data[: index + 1], config)
        if signal is not None:
            pending = {"kind": "enter", "signal": signal}

    if position is not None:
        last = data[-1]
        cash = _close(position, last, len(data) - 1, cash, trades, cost_model, "end_of_data", 1.0)
        if equity_curve:
            equity_curve[-1] = EquityPoint(last.timestamp, cash, 0.0, last.close, 0)

    metrics = calculate_metrics(equity_curve, trades, config.initial_capital, 60)
    metrics["excess_cagr_over_cash"] = float(metrics.get("cagr", 0.0)) - config.cash_hurdle_annual
    return {
        "status": "ok",
        "method": "intraday_session_breakout_flat_by_close",
        "config": {
            "opening_range_bars": config.opening_range_bars,
            "trend_ema": config.trend_ema,
            "slow_ema": config.slow_ema,
            "min_adx": config.min_adx,
            "stop_atr": config.stop_atr,
            "target_r": config.target_r,
            "max_bars_in_trade": config.max_bars_in_trade,
            "risk_per_trade_pct": config.risk_per_trade_pct,
            "max_trades_per_day": config.max_trades_per_day,
            "initial_capital": config.initial_capital,
        },
        "bars": len(data),
        "first_bar": data[0].timestamp.isoformat(),
        "last_bar": data[-1].timestamp.isoformat(),
        "metrics": metrics,
        "exit_reason_counts": {
            reason: sum(1 for t in trades if t.exit_reason == reason)
            for reason in sorted({t.exit_reason for t in trades})
        },
        "trades_sample": [
            {"exit_reason": t.exit_reason, "net_pnl": round(t.net_pnl, 2), "exit_time": t.exit_time.isoformat()}
            for t in trades[:40]
        ],
    }


def _signal(bars: list[Bar], config: IntradayConfig) -> Signal | None:
    session = _session(bars)
    if len(session) < config.opening_range_bars + 2:
        return None
    current = session[-1]
    opening = session[: config.opening_range_bars]
    range_high = max(bar.high for bar in opening)
    range_low = min(bar.low for bar in opening)

    closes = [bar.close for bar in bars]
    atr_values = atr(bars, config.atr_period)
    adx_values = adx(bars, config.adx_period)
    fast = ema(closes, config.trend_ema)
    slow = ema(closes, config.slow_ema)
    rsi_values = rsi(closes, config.rsi_period)
    current_atr, current_adx, fast_ema, slow_ema, current_rsi = (
        atr_values[-1], adx_values[-1], fast[-1], slow[-1], rsi_values[-1]
    )
    if any(v is None for v in (current_atr, current_adx, fast_ema, slow_ema, current_rsi)) or current_atr <= 0:
        return None
    atr_pct = current_atr / current.close
    if atr_pct < config.min_atr_pct or atr_pct > config.max_atr_pct:
        return None
    if current_adx < config.min_adx:
        return None
    upward = fast_ema >= slow_ema
    if config.allow_long and current.close > range_high and upward and current_rsi < 78.0:
        confidence = 0.56 + 0.2 * min(1.0, (current.close - range_high) / max(current_atr, 1e-9))
        if confidence >= 0.56:
            return _make(current, Side.LONG, confidence, "intraday_breakout_long", current_atr, config, range_low)
    if config.allow_short and current.close < range_low and not upward and current_rsi > 22.0:
        confidence = 0.56 + 0.2 * min(1.0, (range_low - current.close) / max(current_atr, 1e-9))
        if confidence >= 0.56:
            return _make(current, Side.SHORT, confidence, "intraday_breakout_short", current_atr, config, range_high)
    return None


def _make(bar: Bar, side: Side, confidence: float, reason: str, current_atr: float, config: IntradayConfig, invalidation: float) -> Signal:
    stop_distance = max(current_atr * config.stop_atr, abs(bar.close - invalidation) * 0.6)
    return Signal(
        timestamp=bar.timestamp,
        symbol=bar.symbol,
        side=side,
        confidence=confidence,
        regime=Regime.TREND,
        reason=reason,
        atr=current_atr,
        stop_distance=stop_distance,
        target_r=config.target_r,
    )


def _open(signal: Signal, bar: Bar, index: int, cash: float, config: IntradayConfig, cost_model: CostModel) -> tuple[dict[str, Any] | None, float]:
    is_buy = signal.side == Side.LONG
    entry = cost_model.fill_price(bar.open, is_buy)
    stop_distance = max(signal.stop_distance, signal.atr * 0.5)
    direction = 1 if is_buy else -1
    stop = entry - direction * stop_distance
    target = entry + direction * stop_distance * signal.target_r
    risk_budget = cash * config.risk_per_trade_pct
    quantity = int(min(risk_budget / max(stop_distance, 1e-9), cash * config.max_notional_pct / max(entry, 1e-9)))
    if quantity < 1:
        return None, cash
    fee = cost_model.side_cost(is_buy, entry, quantity, 1.0).total
    # Cash is charged fees only. Notional is deliberately never booked into cash,
    # so equity = cash + signed unrealised holds for both directions. Booking the
    # short's proceeds here as well would double count it.
    cash -= fee
    multiplier = 1.0
    entry_outlay = quantity * entry + fee

    def unrealized(price: float, _d=direction, _q=quantity) -> float:
        return (price - entry) * _d * _q * multiplier

    position = {
        "symbol": signal.symbol,
        "side": signal.side,
        "quantity": quantity,
        "entry_price": entry,
        "entry_time": bar.timestamp,
        "entry_fee": fee,
        "stop": stop,
        "target": target,
        "direction": direction,
        "unrealized": unrealized,
        "entry_outlay": entry_outlay,
        "bars_held": 0,
        "reason": signal.reason,
    }
    return position, cash


def _close(position: dict[str, Any], bar: Bar, index: int, cash: float, trades: list[Trade], cost_model: CostModel, reason: str, multiplier: float) -> float:
    is_buy = position["side"] == Side.SHORT
    reference = bar.open if reason == "max_bars" else bar.close
    exit_price = cost_model.fill_price(reference, is_buy)
    fee = cost_model.side_cost(is_buy, exit_price, position["quantity"], multiplier).total
    cash += position["direction"] * position["quantity"] * (exit_price - position["entry_price"]) - fee
    gross = (exit_price - position["entry_price"]) * position["direction"] * position["quantity"] * multiplier
    trades.append(
        Trade(
            symbol=position["symbol"],
            side=position["side"],
            quantity=position["quantity"],
            entry_time=position["entry_time"],
            exit_time=bar.timestamp,
            entry_price=position["entry_price"],
            exit_price=exit_price,
            gross_pnl=gross,
            fees=position["entry_fee"] + fee,
            net_pnl=gross - position["entry_fee"] - fee,
            exit_reason=reason,
            entry_reason=position["reason"],
            bars_held=position["bars_held"],
            multiplier=multiplier,
        )
    )
    return cash