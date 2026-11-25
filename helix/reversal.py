"""Cross-sectional short-term reversal for NSE equities.

Published NSE evidence (96-stock study, IC +0.020 large-cap / +0.025 midcap,
+40% over 5.4 years) attributes the edge to short-term cross-sectional reversal:
stocks that fall hardest over the prior 5-21 days tend to bounce within about a
week, driven by overreaction and panic selling rather than any technical pattern.

That is a different signal from anything else in this project. The trend and
volatility-breakout families are time-series momentum (own history, same
instrument). The mean-reversion family reverts an instrument to its own mean.
This ranks instruments against each other and buys the losers, which is what the
behavioural literature actually describes.

Holdings are equal weighted, exits are fixed-horizon, and every decision uses
only bars at or before the rebalance date.
"""

from dataclasses import dataclass, field
from typing import Any

from .backtest import calculate_metrics
from .config import CostConfig
from .costs import CostModel
from .models import Bar, EquityPoint, Side, Trade

DELIVERY_COSTS = CostConfig(
    slippage_bps=5.0, brokerage_per_order=20.0, brokerage_pct=0.001,
    exchange_txn_pct=0.0000297, stt_sell_pct=0.001, stamp_buy_pct=0.00015,
    sebi_pct=0.000001, gst_pct=0.18,
)


@dataclass
class ReversalConfig:
    initial_capital: float = 5_000_000.0
    lookback_bars: int = 10          # prior-window to rank on
    hold_bars: int = 5               # documented rebound horizon
    rebalance_every: int = 5
    top_n: int = 10                  # buy the `top_n` worst performers
    skip_most_extreme: bool = True   # avoid catching a falling knife
    extreme_cutoff: float = 0.12     # skip names down more than this in the window
    momentum_gate: int = 250         # require long-term uptrend context
    market_regime_bars: int = 200    # skip entries while the index is below its long MA
    stop_loss_pct: float = 0.08      # hard stop on a reversal entry
    max_weight: float = 0.12
    direction: str = "worst"        # "worst" buys losers, "best" buys leaders
    momentum_gate_window: int = 0    # optional cross-sectional momentum filter on the universe
    cash_hurdle_annual: float = 0.06
    costs: CostConfig = field(default_factory=lambda: DELIVERY_COSTS)


def run_reversal(
    config: ReversalConfig,
    daily: dict[str, list[Bar]],
    start_date: Any = None,
    end_date: Any = None,
) -> dict[str, Any]:
    sessions = sorted({bar.timestamp.date() for bars in daily.values() for bar in bars})
    warmup = config.lookback_bars + config.momentum_gate + 5
    # all_sessions drives the indicator warmup; sessions bounds what gets reported.
    # They must be the same index space or the warmup slice falls outside the loop.
    all_sessions = [day for day in sessions if (end_date is None or day <= end_date)]
    sessions = [day for day in all_sessions if (start_date is None or day >= start_date)]
    if len(sessions) < 60:
        return {"status": "insufficient_data", "reason": "evaluation window has fewer than 60 sessions"}
    warm = warmup
    loop_sessions = all_sessions
    cost_model = CostModel(config.costs)
    index = {symbol: {bar.timestamp.date(): bar for bar in bars} for symbol, bars in daily.items()}
    history = {symbol: sorted(bars, key=lambda b: b.timestamp) for symbol, bars in daily.items()}
    # equal-weight composite of the universe stands in for the index
    proxy_symbol = sorted(daily, key=lambda s: len(daily[s]))[-1] if daily else None

    cash = config.initial_capital
    positions: dict[str, dict[str, Any]] = {}
    equity_curve: list[EquityPoint] = []
    trades: list[Trade] = []
    peak = config.initial_capital

    for session in range(warm, len(loop_sessions)):
        day = loop_sessions[session]

        # ---- stops and horizon exits ------------------------------------------
        for symbol in list(positions):
            position = positions[symbol]
            bar_now = index[symbol].get(day)
            if day < (start_date or sessions[0]):
                continue
            if config.stop_loss_pct > 0 and bar_now is not None:
                if bar_now.low <= position["entry_price"] * (1.0 - config.stop_loss_pct):
                    cash += _close_with_list(positions.pop(symbol), bar_now, cost_model, "stop_loss", trades)
                    continue
            if position["exit_index"] <= session:
                bar = index[symbol].get(day)
                if bar is None:
                    continue
                cash += _close_with_list(positions.pop(symbol), bar, cost_model, "hold_expired", trades)

        # ---- rebalance on the worst performers -------------------------------
        if (session - warm) % config.rebalance_every == 0 and not positions:
            if day < (start_date or sessions[0]):
                continue
            if config.market_regime_bars > 0:
                proxy = history.get(proxy_symbol)
                if proxy is not None:
                    ref_window = [b for b in proxy if b.timestamp.date() <= loop_sessions[session - 1]]
                    if len(ref_window) > config.market_regime_bars:
                        ma = sum(b.close for b in ref_window[-config.market_regime_bars:]) / config.market_regime_bars
                        if ref_window[-1].close < ma:
                            continue
            ranked: list[tuple[float, str]] = []
            momentum_rank: dict[str, float] = {}
            if config.momentum_gate_window > 0:
                for symbol, bars in history.items():
                    window = [b for b in bars if b.timestamp.date() <= loop_sessions[session - 1]]
                    if len(window) <= config.momentum_gate_window:
                        continue
                    past = window[-1 - config.momentum_gate_window].close
                    if past > 0:
                        momentum_rank[symbol] = window[-1].close / past - 1.0
                if momentum_rank:
                    ordered = sorted(momentum_rank, key=lambda s: -momentum_rank[s])
                    keep = set(ordered[: max(config.top_n, len(ordered) // 2)])
                else:
                    keep = None
            else:
                keep = None
            for symbol, bars in history.items():
                if keep is not None and symbol not in keep:
                    continue
                window = [b for b in bars if b.timestamp.date() <= loop_sessions[session - 1]]
                if len(window) < warmup:
                    continue
                past = window[-1 - config.lookback_bars].close
                if past <= 0:
                    continue
                reversal_return = window[-1].close / past - 1.0
                context = window[-1].close / window[-1 - config.momentum_gate].close - 1.0 if len(window) > config.momentum_gate else 0.0
                if context < 0:
                    continue                      # only fade losers inside an uptrend
                if config.skip_most_extreme and reversal_return < -config.extreme_cutoff:
                    continue                      # avoid genuinely broken names
                ranked.append((reversal_return, symbol))
            if config.direction == "best":
                ranked.sort(reverse=True)         # strongest first
            else:
                ranked.sort()                      # most negative first
            chosen = ranked[: config.top_n]
            if not chosen:
                continue
            weight = min(config.max_weight, 1.0 / len(chosen))
            for _, symbol in chosen:
                bar = index[symbol].get(day)
                if bar is None or bar.open <= 0:
                    continue
                price = cost_model.fill_price(bar.open, True)
                quantity = int((weight * max(peak, 1.0)) / max(price, 1e-9))
                if quantity < 1:
                    continue
                fee = cost_model.side_cost(True, bar.open, quantity, 1.0).total
                cash -= fee
                positions[symbol] = {
                    "symbol": symbol, "quantity": quantity, "entry_price": price,
                    "entry_time": bar.timestamp, "entry_fee": fee,
                    "exit_index": session + config.hold_bars, "reason": "cross_sectional_reversal",
                }

        # ---- mark --------------------------------------------------------------
        if day < (start_date or sessions[0]):
            continue
        equity = cash
        reference_day = None
        for symbol, position in positions.items():
            bar = index[symbol].get(day)
            reference_day = reference_day or day
            if bar is not None:
                equity += position["quantity"] * (bar.close - position["entry_price"])
        if reference_day is None:
            continue
        any_bar = next((index[s][reference_day] for s in positions if index[s].get(reference_day)), None)
        price = any_bar.close if any_bar else 0.0
        stamp = any_bar.timestamp if any_bar else equity_curve[-1].timestamp
        peak = max(peak, equity)
        equity_curve.append(EquityPoint(stamp, equity, equity / peak - 1.0 if peak else 0.0, price, sum(p["quantity"] for p in positions.values())))

    for symbol in list(positions):
        bar = index[symbol].get(sessions[-1])
        if bar is not None:
            cash += _close_with_list(positions.pop(symbol), bar, cost_model, "end_of_data", trades)
    if equity_curve:
        equity_curve[-1] = EquityPoint(equity_curve[-1].timestamp, cash, 0.0, equity_curve[-1].price, 0)

    metrics = calculate_metrics(equity_curve, trades, config.initial_capital, 1440)
    metrics["excess_cagr_over_cash"] = float(metrics.get("cagr", 0.0)) - config.cash_hurdle_annual
    return {
        "status": "ok",
        "method": "cross_sectional_short_term_reversal",
        "config": {
            "lookback_bars": config.lookback_bars, "hold_bars": config.hold_bars,
            "rebalance_every": config.rebalance_every, "top_n": config.top_n,
            "momentum_gate": config.momentum_gate, "skip_most_extreme": config.skip_most_extreme,
        },
        "first_bar": sessions[0].isoformat(),
        "last_bar": sessions[-1].isoformat(),
        "metrics": metrics,
        "exit_reason_counts": {
            reason: sum(1 for t in trades if t.exit_reason == reason)
            for reason in sorted({t.exit_reason for t in trades})
        },
        "equity_curve": equity_curve,
        "trades_detail": trades,
    }


def _close_with_list(position, bar, costs, reason, trades):
    exit_price = costs.fill_price(bar.open, False)
    fee = costs.side_cost(False, bar.open, position["quantity"], 1.0).total
    gross = (exit_price - position["entry_price"]) * position["quantity"]
    trades.append(
        Trade(
            symbol=position["symbol"], side=Side.LONG, quantity=position["quantity"],
            entry_time=position["entry_time"], exit_time=bar.timestamp,
            entry_price=position["entry_price"], exit_price=exit_price,
            gross_pnl=gross, fees=position["entry_fee"] + fee,
            net_pnl=gross - position["entry_fee"] - fee,
            exit_reason=reason, entry_reason=position["reason"],
            bars_held=0, multiplier=1.0,
        )
    )
    return gross - position["entry_fee"] - fee