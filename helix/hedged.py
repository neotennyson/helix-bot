"""Beta-hedged cross-sectional momentum.

The decomposition showed the momentum strategy carries only ~0.31 beta to the
equity universe, with 59% of its variance idiosyncratic. This module overlays a
short NIFTY futures position sized to neutralise the measured beta, so the
resulting return stream is dominated by the cross-sectional selection rather
than market direction.

If selection alpha is real, hedging raises Sharpe. If it is noise, hedging
removes return and buys nothing. That is the question this answers.
"""

from dataclasses import dataclass, field
from typing import Any

from .backtest import calculate_metrics
from .config import CostConfig
from .costs import CostModel
from .models import Bar, EquityPoint, Trade

FUTURES_COSTS = CostConfig(
    slippage_bps=2.0, brokerage_per_order=20.0, brokerage_pct=0.0003,
    exchange_txn_pct=0.0000173, stt_sell_pct=0.000125, stamp_buy_pct=0.0,
    sebi_pct=0.000001, gst_pct=0.18,
)


@dataclass
class HedgeConfig:
    beta_target: float = 0.0        # residual beta to aim for
    hedge_lag: int = 20             # sessions of trailing data used to estimate beta
    hedge_cap: float = 1.5          # max hedge as a multiple of equity
    futures_multiplier: float = 65.0
    initial_capital: float = 5_000_000.0
    cash_hurdle_annual: float = 0.06
    costs: CostConfig = field(default_factory=lambda: FUTURES_COSTS)


def estimate_beta(strategy_returns: list[float], market_returns: list[float]) -> float:
    if len(strategy_returns) < 20:
        return 0.0
    ms = sum(strategy_returns) / len(strategy_returns)
    mm = sum(market_returns) / len(market_returns)
    cov = sum((a - ms) * (b - mm) for a, b in zip(strategy_returns, market_returns)) / len(strategy_returns)
    var = sum((b - mm) ** 2 for b in market_returns) / len(market_returns)
    if var <= 0:
        return 0.0
    return cov / var


def run_hedged(
    hedge: HedgeConfig,
    strategy_curve: list[EquityPoint],
    futures: list[Bar],
) -> dict[str, Any]:
    """Overlay a rolling-beta short futures hedge on an equity strategy curve."""
    if len(strategy_curve) < 60:
        return {"status": "insufficient_data", "reason": "strategy curve too short"}
    fut_index = {bar.timestamp.date(): bar for bar in futures}
    strategy_returns = [
        strategy_curve[i].equity / strategy_curve[i - 1].equity - 1.0
        for i in range(1, len(strategy_curve))
    ]
    dates = [point.timestamp.date() for point in strategy_curve]

    costs = CostModel(hedge.costs)
    equity = hedge.initial_capital
    curve: list[EquityPoint] = []
    trades: list[Trade] = []
    hedge_units = 0.0
    hedge_open_price = 0.0
    peak = hedge.initial_capital
    betas: list[float] = []

    for index in range(1, len(dates)):
        day = dates[index]
        market_bar = fut_index.get(day)
        if market_bar is None:
            continue
        prev_market = fut_index.get(dates[index - 1])
        if prev_market is None or prev_market.close <= 0:
            continue
        market_return = market_bar.close / prev_market.close - 1.0
        strategy_return = strategy_returns[index - 1]

        # re-estimate beta from the trailing window and resize the hedge
        window = min(hedge.hedge_lag, index)
        if window >= 20:
            sr = strategy_returns[index - window : index]
            mr = []
            for k in range(index - window, index):
                pb = fut_index.get(dates[k])
                pp = fut_index.get(dates[k - 1]) if k - 1 >= 0 else None
                mr.append(pb.close / pp.close - 1.0 if pb and pp and pp.close > 0 else 0.0)
            if len(mr) == len(sr):
                beta = estimate_beta(sr, mr)
                betas.append(beta)
                target_units = -(beta - hedge.beta_target) * equity / (market_bar.close * hedge.futures_multiplier)
                target_units = max(-hedge.hedge_cap * equity / (market_bar.close * hedge.futures_multiplier),
                                   min(hedge.hedge_cap * equity / (market_bar.close * hedge.futures_multiplier),
                                       target_units))
                delta = target_units - hedge_units
                if abs(delta) >= 1.0:
                    traded = int(round(delta))
                    fill = costs.fill_price(market_bar.close, traded < 0)
                    fee = costs.side_cost(traded < 0, market_bar.close, abs(traded), hedge.futures_multiplier).total
                    equity -= traded * fill * hedge.futures_multiplier + fee
                    trades.append(
                        Trade(
                            symbol="NIFTYFUT", side=None, quantity=abs(traded),
                            entry_time=strategy_curve[index].timestamp, exit_time=strategy_curve[index].timestamp,
                            entry_price=market_bar.close, exit_price=fill,
                            gross_pnl=(fill - market_bar.close) * traded * hedge.futures_multiplier,
                            fees=fee, net_pnl=(fill - market_bar.close) * traded * hedge.futures_multiplier - fee,
                            exit_reason="hedge_resize", entry_reason="beta_neutralisation",
                            bars_held=0, multiplier=hedge.futures_multiplier,
                        )
                    )
                    hedge_units += traded
                    if hedge_open_price == 0.0:
                        hedge_open_price = fill

        strategy_pnl = equity * strategy_return
        futures_pnl = hedge_units * (market_bar.close - (fut_index[dates[index - 1]].close)) * hedge.futures_multiplier
        equity += strategy_pnl + futures_pnl
        peak = max(peak, equity)
        curve.append(
            EquityPoint(strategy_curve[index].timestamp, equity, equity / peak - 1.0 if peak else 0.0,
                        strategy_curve[index].price, int(hedge_units))
        )

    metrics = calculate_metrics(curve, trades, hedge.initial_capital, 1440)
    metrics["excess_cagr_over_cash"] = float(metrics.get("cagr", 0.0)) - hedge.cash_hurdle_annual
    metrics["average_beta"] = sum(betas) / len(betas) if betas else 0.0
    return {
        "status": "ok",
        "method": "beta_neutralised_momentum_over_nifty_futures",
        "hedge_target_beta": hedge.beta_target,
        "sessions": len(curve),
        "first_bar": str(curve[0].timestamp.date()) if curve else None,
        "last_bar": str(curve[-1].timestamp.date()) if curve else None,
        "metrics": metrics,
        "hedge_trades": len(trades),
        "total_hedge_fees": sum(t.fees for t in trades),
    }