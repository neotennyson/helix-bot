"""Hybrid daily-signal / intraday-execution strategy.

The daily cross-sectional study found real selection alpha (90th percentile
against random universes) but capped it around 3% CAGR, and the intraday study
found that pure intraday breakout has a negative gross signal. The mechanism this
module tests: keep the daily signal that produces the alpha, but execute and exit
inside the session so cost drag falls, and use intraday bars for risk management
instead of overnight gaps.

Three distinct effects are separated on purpose:
  * daily signal, hold to next open  -> baseline delivery cost, overnight risk
  * daily signal, intraday exit      -> same signal, cheaper round trip
  * daily signal, intraday exit + ATR risk stop -> cheaper plus faster loss cuts

If the cost hypothesis is right, effect two beats effect one without the signal
changing at all. That is the claim under test.
"""

from dataclasses import dataclass, field
from typing import Any

from .backtest import calculate_metrics
from .config import CostConfig
from .costs import CostModel
from .indicators import atr, ema
from .models import Bar, EquityPoint, Side, Trade

DELIVERY_COSTS = CostConfig(
    slippage_bps=5.0, brokerage_per_order=20.0, brokerage_pct=0.001,
    exchange_txn_pct=0.0000297, stt_sell_pct=0.001, stamp_buy_pct=0.00015,
    sebi_pct=0.000001, gst_pct=0.18,
)
INTRADAY_COSTS = CostConfig(
    slippage_bps=3.0, brokerage_per_order=20.0, brokerage_pct=0.0,
    exchange_txn_pct=0.0000297, stt_sell_pct=0.00025, stamp_buy_pct=0.00002,
    sebi_pct=0.000001, gst_pct=0.18,
)


@dataclass
class HybridConfig:
    initial_capital: float = 5_000_000.0
    lookback_bars: int = 189
    trend_ema: int = 100
    rebalance_every: int = 21
    top_n: int = 8
    max_weight: float = 0.16
    exit_mode: str = "intraday"        # "overnight" | "intraday"
    intraday_exit_bar: int = 4         # which hourly bar of the next session to exit on
    atr_period: int = 14
    risk_stop_atr: float = 2.5         # 0 disables the intraday risk stop
    atr_exit_multiple: float = 1.0     # exit when day's move exceeds this x ATR
    cash_hurdle_annual: float = 0.06
    costs: CostConfig = field(default_factory=lambda: INTRADAY_COSTS)


def _daily_ranks(daily: dict[str, list[Bar]], stamp, config: HybridConfig) -> dict[str, float]:
    """Long-only momentum ranks computed strictly from bars at or before `stamp`."""
    scores: list[tuple[float, str]] = []
    for symbol, bars in daily.items():
        window = [b for b in bars if b.timestamp.date() <= stamp]
        if len(window) < config.lookback_bars + config.trend_ema + 10:
            continue
        closes = [b.close for b in window]
        trend = ema(closes, config.trend_ema)[-1]
        if trend is None or window[-1].close <= trend:
            continue
        momentum = window[-1].close / window[-1 - config.lookback_bars].close - 1.0
        if momentum > 0:
            scores.append((momentum, symbol))
    scores.sort(reverse=True)
    chosen = [symbol for _, symbol in scores[: config.top_n]]
    return {symbol: 1.0 / len(chosen) for symbol in chosen} if chosen else {}


def run_hybrid(
    config: HybridConfig,
    daily: dict[str, list[Bar]],
    hourly: dict[str, list[Bar]],
) -> dict[str, Any]:
    sessions = sorted({bar.timestamp.date() for bars in hourly.values() for bar in bars})
    if len(sessions) < 40:
        return {"status": "insufficient_data", "reason": "not enough sessions"}
    cost_model = CostModel(config.costs)
    cash = config.initial_capital
    positions: dict[str, dict[str, Any]] = {}
    pending_buys: list[tuple[str, float]] = []
    equity_curve: list[EquityPoint] = []
    trades: list[Trade] = []
    peak = config.initial_capital
    index_by_symbol = {symbol: {bar.timestamp.date(): bar for bar in bars} for symbol, bars in hourly.items()}

    for session_index, day in enumerate(sessions):
        # ---- signal and rebalance, decided on the prior close ----------------
        if session_index > 0 and (session_index - 1) % config.rebalance_every == 0:
            targets = _daily_ranks(daily, sessions[session_index - 1], config)
            for symbol in list(positions):
                if symbol not in targets:
                    bar = index_by_symbol[symbol].get(day)
                    if bar is None:
                        continue
                    cash += _exit(positions.pop(symbol), bar, cost_model, "signal_exit", 1.0, trades)
            for symbol, weight in targets.items():
                bar = index_by_symbol.get(symbol, {}).get(day)
                if bar is None:
                    continue
                price = cost_model.fill_price(bar.open, True)
                budget = min(weight * max(peak, 1.0), cash)
                quantity = int(budget / max(price, 1e-9))
                if quantity < 1:
                    continue
                fee = cost_model.side_cost(True, bar.open, quantity, 1.0).total
                cash -= fee
                positions[symbol] = {
                    "symbol": symbol, "quantity": quantity, "entry_price": price,
                    "entry_time": bar.timestamp, "entry_fee": fee, "reason": "daily_momentum",
                }
        # ---- intraday risk management ---------------------------------------
        for symbol in list(positions):
            position = positions[symbol]
            session_bars = [b for b in hourly[symbol] if b.timestamp.date() == day]
            if len(session_bars) < 2:
                continue
            # Bars elapsed since THIS position was opened, so exits are measured
            # from entry rather than from the session open.
            held = [b for b in session_bars if b.timestamp >= position["entry_time"]]
            reason = None
            if config.exit_mode == "overnight":
                pass
            else:
                daily_atr = None
                window = [b for b in daily[symbol] if b.timestamp.date() <= day - _one_day()]
                if len(window) > config.atr_period + 1:
                    atr_values = atr(window, config.atr_period)
                    daily_atr = atr_values[-1]
                if daily_atr and config.risk_stop_atr > 0:
                    low = min(b.low for b in session_bars)
                    if low <= position["entry_price"] - config.risk_stop_atr * daily_atr:
                        reason = "risk_stop"
                if reason is None and daily_atr and config.atr_exit_multiple > 0:
                    move = abs(session_bars[-1].close - position["entry_price"])
                    if move >= config.atr_exit_multiple * daily_atr:
                        reason = "range_exit"
                if reason is None and config.exit_mode == "intraday":
                    if len(held) - 1 >= config.intraday_exit_bar:
                        reason = "intraday_flat"
            if reason:
                cash += _exit(positions.pop(symbol), session_bars[min(2, len(session_bars) - 1)], cost_model, reason, 1.0, trades)
        # ---- mark -------------------------------------------------------------
        # Mark on UNREALISED P&L. Adding the full notional would double count,
        # because cash was only charged fees at entry, never the purchase outlay.
        equity = cash + sum(
            p["quantity"] * (
                (index_by_symbol[s][day].close if index_by_symbol.get(s, {}).get(day) else p["entry_price"]) - p["entry_price"]
            )
            for s, p in positions.items()
        )
        peak = max(peak, equity)
        reference = next(iter(index_by_symbol.values()))
        bar = reference.get(day)
        if bar is None:
            # reference symbol has no session on this date; skip rather than
            # emit an equity point with a null timestamp
            continue
        stamp = bar.timestamp
        price = bar.close
        equity_curve.append(EquityPoint(stamp, equity, equity / peak - 1.0 if peak else 0.0, price, sum(p["quantity"] for p in positions.values())))

    for symbol in list(positions):
        position = positions.pop(symbol)
        last = [b for b in hourly[symbol] if b.timestamp.date() <= sessions[-1]][-1]
        cash += _exit(position, last, cost_model, "end_of_data", 1.0, trades)
    if equity_curve:
        equity_curve[-1] = EquityPoint(equity_curve[-1].timestamp, cash, 0.0, equity_curve[-1].price, 0)

    metrics = calculate_metrics(equity_curve, trades, config.initial_capital, 1440)
    metrics["excess_cagr_over_cash"] = float(metrics.get("cagr", 0.0)) - config.cash_hurdle_annual
    return {
        "status": "ok",
        "method": f"hybrid_daily_signal_{config.exit_mode}_exit",
        "exit_mode": config.exit_mode,
        "cost_model": "intraday" if config.costs is INTRADAY_COSTS else "delivery",
        "sessions": len(sessions),
        "metrics": metrics,
        "exit_reason_counts": {
            reason: sum(1 for t in trades if t.exit_reason == reason)
            for reason in sorted({t.exit_reason for t in trades})
        },
    }


def _one_day():
    from datetime import timedelta

    return timedelta(days=1)


def _exit(position: dict[str, Any], bar: Bar, costs: CostModel, reason: str, mult: float, trades: list[Trade]) -> float:
    """Realise a long: book the P&L net of both fee legs and record the trade."""
    exit_price = costs.fill_price(bar.close, False)
    fee = costs.side_cost(False, bar.close, position["quantity"], mult).total
    gross = (exit_price - position["entry_price"]) * position["quantity"] * mult
    trades.append(
        Trade(
            symbol=position["symbol"], side=Side.LONG, quantity=position["quantity"],
            entry_time=position["entry_time"], exit_time=bar.timestamp,
            entry_price=position["entry_price"], exit_price=exit_price,
            gross_pnl=gross, fees=position["entry_fee"] + fee,
            net_pnl=gross - position["entry_fee"] - fee,
            exit_reason=reason, entry_reason=position["reason"],
            bars_held=0, multiplier=mult,
        )
    )
    return gross - position["entry_fee"] - fee
