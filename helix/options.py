"""Synthetic index-option research.

Neither broker adapter in this project can return historical option quotes
(Dhan returns DH-907 for index options, Groww returns an empty candle list), so
this module prices options with Black-Scholes on the *real* underlying futures
path and assumes an implied-volatility surface.

Consequences, which every report from this module carries:
  * premium levels, Greeks and settlement are modelled, not quoted
  * results are an upper bound on a real short-premium strategy, because real
    option prices embed bid/ask spreads, skew and jump risk that are absent here
  * `real_option_quotes` is always False, and no report may be treated as an
    approval to trade options
"""

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from .backtest import calculate_metrics
from .models import EquityPoint, Side, Trade

DAYS_PER_YEAR = 365.0
INTRADAY_VOL = 1.0
RISK_FREE_RATE = 0.065


@dataclass
class OptionLeg:
    side: Side
    right: str
    strike: float
    premium: float
    quantity: int
    entry_premium: float
    entry_time: Any = None
    multiplier: float = 65.0
    underlying: float = 0.0

    def market_value(self, underlying: float, iv: float, days_to_expiry: float) -> float:
        price = option_price(underlying, self.strike, days_to_expiry, iv, RISK_FREE_RATE, self.right)
        return (price - self.entry_premium) * self.side_sign * self.quantity * self.multiplier

    @property
    def side_sign(self) -> int:
        return 1 if self.side == Side.LONG else -1

    @property
    def intrinsic(self) -> float:
        return max(0.0, self.strike - self.underlying) if self.right == "PE" else max(0.0, self.underlying - self.strike)


@dataclass
class OptionConfig:
    underlying_symbol: str = "48704"
    lot_size: int = 65
    # For index options the lot size already carries the contract multiplier, so
    # this stays at 1.0 and is not multiplied on top of lot_size.
    multiplier: float = 1.0
    initial_capital: float = 5_000_000.0
    implied_volatility: float = 0.14
    delta_target: float = 0.25
    holding_days: int = 21
    premium_target_r: float = 0.5
    premium_stop_r: float = 2.0
    roll_on_exit: bool = True
    max_capital_at_risk_pct: float = 0.25
    cash_hurdle_annual: float = 0.06
    strategy: str = "covered_call"
    costs: dict[str, float] = field(
        default_factory=lambda: {
            "slippage_bps_per_premium": 0.01,
            "brokerage_per_order": 20.0,
            "brokerage_pct": 0.0,
            "exchange_txn_pct": 0.0000350,
            "stt_sell_pct": 0.001,
            "stamp_buy_pct": 0.00003,
            "sebi_pct": 0.0000001,
            "gst_pct": 0.18,
        }
    )

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "OptionConfig":
        value = value or {}
        base = cls()
        return cls(
            underlying_symbol=str(value.get("underlying_symbol", base.underlying_symbol)),
            lot_size=int(value.get("lot_size", base.lot_size)),
            multiplier=float(value.get("multiplier", base.multiplier)),
            initial_capital=float(value.get("initial_capital", base.initial_capital)),
            implied_volatility=float(value.get("implied_volatility", base.implied_volatility)),
            delta_target=float(value.get("delta_target", base.delta_target)),
            holding_days=int(value.get("holding_days", base.holding_days)),
            premium_target_r=float(value.get("premium_target_r", base.premium_target_r)),
            premium_stop_r=float(value.get("premium_stop_r", base.premium_stop_r)),
            roll_on_exit=bool(value.get("roll_on_exit", base.roll_on_exit)),
            max_capital_at_risk_pct=float(value.get("max_capital_at_risk_pct", base.max_capital_at_risk_pct)),
            cash_hurdle_annual=float(value.get("cash_hurdle_annual", base.cash_hurdle_annual)),
            strategy=str(value.get("strategy", base.strategy)),
            costs={**base.costs, **(value.get("costs") or {})},
        )


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def option_price(
    underlying: float,
    strike: float,
    days_to_expiry: float,
    volatility: float,
    rate: float,
    right: str,
) -> float:
    if days_to_expiry <= 0 or volatility <= 0 or underlying <= 0 or strike <= 0:
        intrinsic = max(0.0, underlying - strike) if right == "CE" else max(0.0, strike - underlying)
        return intrinsic
    tau = days_to_expiry / DAYS_PER_YEAR
    sqrt_tau = math.sqrt(tau)
    d1 = (math.log(underlying / strike) + (rate + 0.5 * volatility**2) * tau) / (volatility * sqrt_tau)
    d2 = d1 - volatility * sqrt_tau
    discounted_strike = strike * math.exp(-rate * tau)
    if right == "CE":
        return underlying * normal_cdf(d1) - discounted_strike * normal_cdf(d2)
    return discounted_strike * normal_cdf(-d2) - underlying * normal_cdf(-d1)


def option_delta(underlying: float, strike: float, days_to_expiry: float, volatility: float, rate: float, right: str) -> float:
    if days_to_expiry <= 0 or volatility <= 0:
        if right == "CE":
            return 1.0 if underlying > strike else 0.0
        return -1.0 if underlying < strike else 0.0
    tau = days_to_expiry / DAYS_PER_YEAR
    d1 = (math.log(underlying / strike) + (rate + 0.5 * volatility**2) * tau) / (volatility * math.sqrt(tau))
    delta = normal_cdf(d1) if right == "CE" else normal_cdf(d1) - 1.0
    return delta


def strike_for_delta(underlying: float, days_to_expiry: float, volatility: float, target_delta: float, right: str) -> float:
    """Invert the delta formula with a bisection search."""
    low, high = underlying * 0.5, underlying * 2.0
    for _ in range(60):
        mid = (low + high) / 2.0
        delta = option_delta(underlying, mid, days_to_expiry, volatility, RISK_FREE_RATE, right)
        if right == "CE":
            # call delta falls as strike rises
            if delta > target_delta:
                low = mid
            else:
                high = mid
        else:
            # put delta magnitude falls as strike falls, so it rises with strike
            if -delta > target_delta:
                high = mid
            else:
                low = mid
    return (low + high) / 2.0


def option_side_cost(is_buy: bool, premium: float, quantity: int, multiplier: float, costs: dict[str, float]) -> float:
    gross = abs(premium * quantity * multiplier)
    brokerage = min(costs["brokerage_per_order"], gross * costs["brokerage_pct"])
    exchange = gross * costs["exchange_txn_pct"]
    stt = 0.0 if is_buy else gross * costs["stt_sell_pct"]
    stamp = gross * costs["stamp_buy_pct"] if is_buy else 0.0
    sebi = gross * costs["sebi_pct"]
    slippage = gross * costs["slippage_bps_per_premium"]
    gst = (brokerage + exchange + sebi) * costs["gst_pct"]
    return brokerage + exchange + stt + stamp + sebi + gst + slippage


def run_option_research(config: OptionConfig, bars: list[Any]) -> dict[str, Any]:
    data = sorted(bars, key=lambda bar: bar.timestamp)
    if len(data) < 120:
        return {"status": "insufficient_data"}
    volatility = config.implied_volatility
    cash = config.initial_capital
    equity_curve: list[EquityPoint] = []
    trades: list[Trade] = []
    open_leg: OptionLeg | None = None
    open_entry_index = 0
    peak = config.initial_capital
    premium_basis = 0.0
    skipped_because_of_capital = 0
    for index, bar in enumerate(data):
        if index + config.holding_days + 1 >= len(data):
            break
        bars_in_leg = index - open_entry_index if open_leg is not None else 0
        days_left = max(0, config.holding_days - bars_in_leg)
        if open_leg is None:
            right = "CE" if config.strategy == "covered_call" else "PE"
            strike = strike_for_delta(bar.close, days_left, volatility, config.delta_target, right)
            premium = option_price(bar.close, strike, days_left, volatility, RISK_FREE_RATE, right)
            notional = strike * config.lot_size * config.multiplier
            if premium <= 0 or notional > cash * config.max_capital_at_risk_pct:
                skipped_because_of_capital += 1
                equity_curve.append(EquityPoint(bar.timestamp, cash, 0.0, bar.close, 0))
                continue
            cost = option_side_cost(False, premium, config.lot_size, config.multiplier, config.costs)
            cash += premium * config.lot_size * config.multiplier - cost
            open_leg = OptionLeg(
                side=Side.SHORT,
                right=right,
                strike=strike,
                premium=premium,
                quantity=config.lot_size,
                entry_premium=premium,
                entry_time=bar.timestamp,
                multiplier=config.multiplier,
            )
            open_entry_index = index
            premium_basis = premium * config.lot_size * config.multiplier
            equity_curve.append(EquityPoint(bar.timestamp, cash, 0.0, bar.close, config.lot_size))
            continue
        premium_now = option_price(bar.close, open_leg.strike, days_left, volatility, RISK_FREE_RATE, open_leg.right)
        unrealised = (premium - premium_now) * open_leg.quantity * open_leg.multiplier
        mtm_equity = cash + unrealised
        exit_reason: str | None = None
        realised = 0.0
        if bars_in_leg >= config.holding_days:
            exit_reason = "expiry_settlement"
            intrinsic = (
                max(0.0, open_leg.strike - bar.close) if open_leg.right == "PE" else max(0.0, bar.close - open_leg.strike)
            )
            realised = (open_leg.premium - intrinsic) * open_leg.quantity * open_leg.multiplier
        elif unrealised >= premium_basis * config.premium_target_r:
            exit_reason = "premium_target"
            realised = unrealised
        elif unrealised <= -premium_basis * config.premium_stop_r:
            exit_reason = "premium_stop"
            realised = unrealised
        if exit_reason is not None:
            exit_cost = option_side_cost(True, premium_now, open_leg.quantity, open_leg.multiplier, config.costs)
            cash += realised - exit_cost
            trades.append(
                Trade(
                    symbol=f"{config.underlying_symbol}-{open_leg.right}{int(open_leg.strike)}",
                    side=Side.SHORT,
                    quantity=open_leg.quantity,
                    entry_time=open_leg.entry_time,
                    exit_time=bar.timestamp,
                    entry_price=open_leg.entry_premium,
                    exit_price=premium_now,
                    gross_pnl=premium * open_leg.quantity * open_leg.multiplier - premium_now * open_leg.quantity * open_leg.multiplier,
                    fees=0.0,
                    net_pnl=realised - exit_cost,
                    exit_reason=exit_reason,
                    entry_reason="synthetic_short_premium",
                    bars_held=index - open_entry_index,
                    multiplier=open_leg.multiplier,
                )
            )
            open_leg = None
            equity_curve.append(EquityPoint(bar.timestamp, cash, 0.0, bar.close, 0))
            continue
        peak = max(peak, mtm_equity)
        drawdown = mtm_equity / peak - 1.0 if peak else 0.0
        equity_curve.append(EquityPoint(bar.timestamp, mtm_equity, drawdown, bar.close, open_leg.quantity))
    if open_leg is not None and data:
        bar = data[-1]
        premium_now = option_price(bar.close, open_leg.strike, 0, volatility, RISK_FREE_RATE, open_leg.right)
        settlement = max(0.0, (open_leg.strike - bar.close)) if open_leg.right == "PE" else max(0.0, (bar.close - open_leg.strike))
        cash += (open_leg.premium - settlement) * open_leg.quantity * open_leg.multiplier
        equity_curve.append(EquityPoint(bar.timestamp, cash, 0.0, bar.close, 0))
    metrics = calculate_metrics(equity_curve, trades, config.initial_capital, 1440)
    metrics["excess_cagr_over_cash"] = float(metrics.get("cagr", 0.0)) - config.cash_hurdle_annual
    return {
        "status": "ok",
        "research_only": True,
        "real_option_quotes": False,
        "pricing_model": "black_scholes_european",
        "iv_source": "assumed_constant",
        "disclaimer": "modelled premiums on a real underlying path; not a tradable option quote",
        "config": {
            "strategy": config.strategy,
            "underlying_symbol": config.underlying_symbol,
            "lot_size": config.lot_size,
            "implied_volatility": volatility,
            "delta_target": config.delta_target,
            "holding_days": config.holding_days,
            "premium_target_r": config.premium_target_r,
            "premium_stop_r": config.premium_stop_r,
            "initial_capital": config.initial_capital,
            "cash_hurdle_annual": config.cash_hurdle_annual,
            "max_capital_at_risk_pct": config.max_capital_at_risk_pct,
        },
        "bars": len(data),
        "first_bar": data[0].timestamp.isoformat(),
        "last_bar": data[-1].timestamp.isoformat(),
        "metrics": metrics,
        "promotion": {
            "passed": False,
            "status": "not_eligible_for_paper_review",
            "rules": {
                "real_option_quotes": False,
                "minimum_sharpe": 0.5,
                "positive_excess_cagr_over_cash": float(metrics.get("cagr", 0.0)) > config.cash_hurdle_annual,
                "minimum_trades": 20,
            },
            "note": "synthetic pricing can never clear the gate on its own",
        },
        "paper_review_candidate": False,
        "trades_sample": [
            {
                "exit_reason": trade.exit_reason,
                "net_pnl": trade.net_pnl,
                "exit_time": trade.exit_time.isoformat(),
                "bars_held": trade.bars_held,
            }
            for trade in trades[:40]
        ],
        "skipped_signals_for_capital_limit": skipped_because_of_capital,
        "exit_reason_counts": {
            reason: sum(1 for trade in trades if trade.exit_reason == reason)
            for reason in sorted({trade.exit_reason for trade in trades})
        },
    }
