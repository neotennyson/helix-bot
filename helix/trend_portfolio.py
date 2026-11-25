"""Cross-sectional trend portfolio with volatility targeting.

This is the institutional design the per-symbol scans were missing:

  * holdings are chosen on a slow rebalance schedule, not bar by bar, so
    turnover (and therefore costs) stay low
  * weights are inverse-volatility, so a single volatile name cannot dominate
    risk and portfolio volatility is held near a target
  * names are long only, because shorting delivery equity in India is neither
    practical nor cheap
  * every decision uses data up to the rebalance date only

Cost model is shared with the rest of the project so results stay comparable.
"""

from dataclasses import dataclass, field
from typing import Any

from .backtest import calculate_metrics
from .config import CostConfig
from .costs import CostModel
from .indicators import ema
from .models import Bar, EquityPoint, Trade


@dataclass
class TrendPortfolioConfig:
    initial_capital: float = 5_000_000.0
    lookback_bars: int = 126           # ~6 months of momentum
    trend_ema: int = 100               # price must sit above this to be held
    top_n: int = 20
    rebalance_every: int = 21          # ~monthly
    atr_period: int = 20
    target_volatility: float = 0.16    # annualised
    max_weight: float = 0.12
    cash_hurdle_annual: float = 0.06
    costs: CostConfig = field(default_factory=CostConfig)


def _annualised_vol(closes: list[float], period: int = 20) -> float:
    if len(closes) < period + 1:
        return 0.0
    returns = [closes[i] / closes[i - 1] - 1.0 for i in range(len(closes) - period, len(closes)) if closes[i - 1] > 0]
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / len(returns)
    return variance**0.5 * (252.0**0.5)


def _momentum(bars: list[Bar], lookback: int) -> float:
    if len(bars) <= lookback:
        return 0.0
    past = bars[-lookback - 1].close
    if past <= 0:
        return 0.0
    return bars[-1].close / past - 1.0


def run_trend_portfolio(
    config: TrendPortfolioConfig,
    bars_by_symbol: dict[str, list[Bar]],
    start_index: int | None = None,
    end_index: int | None = None,
) -> dict[str, Any]:
    """Long-only cross-sectional momentum with inverse-vol weights.

    `start_index`/`end_index` slice the reference calendar so a walk-forward
    window can be evaluated without touching data outside it.
    """
    usable = {symbol: bars for symbol, bars in bars_by_symbol.items() if bars}
    if not usable:
        return {"status": "insufficient_data", "reason": "no bars"}
    reference = max(usable.values(), key=len)
    total = len(reference)
    first = start_index if start_index is not None else 0
    last = end_index if end_index is not None else total
    warmup = max(config.lookback_bars, config.trend_ema, config.atr_period) + 5
    first = max(first, warmup)
    if last - first < 60:
        return {"status": "insufficient_data", "reason": "window too short"}

    cost_model = CostModel(config.costs)
    cash = config.initial_capital
    positions: dict[str, dict[str, Any]] = {}
    equity_curve: list[EquityPoint] = []
    trades: list[Trade] = []
    peak = config.initial_capital
    weight_cap = 1.0 / max(1, config.top_n)

    for index in range(first, last):
        stamp = reference[index].timestamp
        # ---- rebalance ------------------------------------------------------
        if (index - first) % config.rebalance_every == 0:
            prices: dict[str, float] = {}
            scores: list[tuple[float, str]] = []
            vols: dict[str, float] = {}
            for symbol, bars in usable.items():
                window = [b for b in bars if b.timestamp <= stamp]
                if len(window) < warmup:
                    continue
                closes = [b.close for b in window]
                ema_values = ema(closes, config.trend_ema)
                if ema_values[-1] is None:
                    continue
                if window[-1].close <= ema_values[-1]:
                    continue
                momentum = _momentum(window, config.lookback_bars)
                if momentum <= 0:
                    continue
                volatility = _annualised_vol(closes, config.atr_period)
                if volatility <= 0.005:
                    continue
                scores.append((momentum, symbol))
                vols[symbol] = volatility
                prices[symbol] = window[-1].close
            scores.sort(reverse=True)
            selected = scores[: config.top_n]
            if not selected:
                continue
            current_equity = cash + sum(
                position["quantity"] * prices.get(symbol, position["last_price"]) for symbol, position in positions.items()
            )
            inverse_vol = {symbol: 1.0 / vols[symbol] for _, symbol in selected}
            total_inverse = sum(inverse_vol.values())
            target_weights = {
                symbol: min(weight_cap, (inverse_vol[symbol] / total_inverse) * config.top_n * weight_cap)
                for symbol in inverse_vol
            }
            # scale so the book targets the volatility budget
            book_vol = sum(target_weights[symbol] * vols[symbol] for symbol in target_weights)
            scale = 1.0
            if book_vol > 0:
                scale = min(1.0, config.target_volatility / book_vol)
            target_weights = {symbol: value * scale for symbol, value in target_weights.items()}
            target_symbols = set(target_weights)

            # exits
            for symbol in list(positions):
                if symbol not in target_symbols:
                    position = positions.pop(symbol)
                    price = prices.get(symbol, position["last_price"])
                    exit_price = cost_model.fill_price(price, False)
                    fee = cost_model.side_cost(False, price, position["quantity"], 1.0).total
                    cash += position["quantity"] * exit_price - fee
                    trades.append(_trade(position, exit_price, fee, stamp, "rebalance_exit"))
            # entries and resizes
            for symbol, weight in target_weights.items():
                price = prices[symbol]
                budget = weight * max(current_equity, 0.0)
                quantity = int(budget / max(price, 1e-9))
                if quantity <= 0:
                    continue
                existing = positions.get(symbol)
                if existing is not None and existing["quantity"] == quantity:
                    existing["weight"] = weight
                    continue
                if existing is not None:
                    delta = quantity - existing["quantity"]
                else:
                    delta = quantity
                if delta > 0:
                    fill = cost_model.fill_price(price, True)
                    fee = cost_model.side_cost(True, price, abs(delta), 1.0).total
                    cash -= abs(delta) * fill + fee
                elif delta < 0:
                    fill = cost_model.fill_price(price, False)
                    fee = cost_model.side_cost(False, price, abs(delta), 1.0).total
                    cash += abs(delta) * fill - fee
                else:
                    continue
                if existing is not None:
                    position = existing
                    position["quantity"] = quantity
                    position["weight"] = weight
                    position["cost_basis"] = (position["cost_basis"] * existing["quantity"] + delta * fill) / quantity if quantity else fill
                else:
                    positions[symbol] = {
                        "symbol": symbol,
                        "quantity": quantity,
                        "cost_basis": cost_model.fill_price(price, True),
                        "entry_time": stamp,
                        "weight": weight,
                        "last_price": price,
                    }
        # ---- mark ------------------------------------------------------------
        equity = cash
        for symbol, position in positions.items():
            bar = next((b for b in usable[symbol] if b.timestamp == stamp), None)
            price = bar.close if bar is not None else position["last_price"]
            position["last_price"] = price
            equity += position["quantity"] * price
        peak = max(peak, equity)
        drawdown = equity / peak - 1.0 if peak else 0.0
        total_quantity = sum(position["quantity"] for position in positions.values())
        equity_curve.append(EquityPoint(stamp, equity, drawdown, reference[index].close, total_quantity))
        if index == last - 1:
            for symbol in list(positions):
                position = positions.pop(symbol)
                price = position["last_price"]
                exit_price = cost_model.fill_price(price, False)
                fee = cost_model.side_cost(False, price, position["quantity"], 1.0).total
                cash += position["quantity"] * exit_price - fee
                trades.append(_trade(position, exit_price, fee, stamp, "end_of_window"))

    metrics = calculate_metrics(equity_curve, trades, config.initial_capital, 1440)
    metrics["average_holdings"] = (
        sum(1 for point in equity_curve if point.position > 0) / len(equity_curve) if equity_curve else 0.0
    )
    metrics["excess_cagr_over_cash"] = float(metrics.get("cagr", 0.0)) - config.cash_hurdle_annual
    return {
        "status": "ok",
        "method": "cross_sectional_momentum_inverse_vol_long_only",
        "config": {
            "lookback_bars": config.lookback_bars,
            "trend_ema": config.trend_ema,
            "top_n": config.top_n,
            "rebalance_every": config.rebalance_every,
            "target_volatility": config.target_volatility,
            "initial_capital": config.initial_capital,
        },
        "first_bar": reference[first].timestamp.isoformat(),
        "last_bar": reference[last - 1].timestamp.isoformat(),
        "metrics": metrics,
        "trade_sample": [
            {
                "exit_reason": t.exit_reason,
                "net_pnl": t.net_pnl,
                "exit_time": t.exit_time.isoformat(),
            }
            for t in trades[:30]
        ],
        "exit_reason_counts": {
            reason: sum(1 for t in trades if t.exit_reason == reason)
            for reason in sorted({t.exit_reason for t in trades})
        },
    }


def _trade(position: dict[str, Any], exit_price: float, fee: float, stamp: Any, reason: str) -> Trade:
    gross = (exit_price - position["cost_basis"]) * position["quantity"]
    net = gross - fee
    return Trade(
        symbol=position.get("symbol", ""),
        side=None,
        quantity=position["quantity"],
        entry_time=position["entry_time"],
        exit_time=stamp,
        entry_price=position["cost_basis"],
        exit_price=exit_price,
        gross_pnl=gross,
        fees=fee,
        net_pnl=net,
        exit_reason=reason,
        entry_reason="cross_sectional_momentum",
        bars_held=0,
        multiplier=1.0,
    )