import math
from datetime import datetime
from statistics import mean, pstdev
from typing import Any

from .config import BacktestConfig
from .costs import CostModel
from .models import BacktestResult, Bar, EquityPoint, Position, Side, Signal, Trade
from .risk import RiskManager
from .strategy import RegimeStrategy
from .trend_strategy import TrendBreakoutStrategy
from .mean_reversion_strategy import MeanReversionStrategy
from .opening_range_strategy import OpeningRangeBreakoutStrategy
from .volatility_breakout_strategy import VolatilityBreakoutStrategy


class BacktestEngine:
    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self.strategy = self._build_strategy()
        self.risk = RiskManager(config.risk)
        self.costs = CostModel(config.costs)

    def run(self, bars: list[Bar]) -> BacktestResult:
        data = sorted([bar for bar in bars if bar.symbol == self.config.symbol], key=lambda item: item.timestamp)
        required = max(self.config.strategy.slow_ema, self.config.strategy.atr_period * 2, self.config.strategy.breakout_period, self.config.strategy.long_term_ema if self.config.strategy.use_long_term_filter else 0) + 10
        if len(data) < required:
            raise ValueError("not enough bars for a valid backtest")
        cash = self.config.risk.initial_capital
        position: Position | None = None
        pending: dict[str, Any] | None = None
        trades: list[Trade] = []
        signals: list[Signal] = []
        equity_curve: list[EquityPoint] = []
        peak_equity = self.config.risk.initial_capital
        for index, bar in enumerate(data):
            current_equity = self._mark(cash, position, bar.open)
            self.risk.start_day(bar.timestamp, current_equity)
            if pending is not None:
                if pending["kind"] == "enter" and position is None:
                    position = self._enter(Signal.from_dict(pending["signal"]), bar, index, current_equity)
                elif pending["kind"] == "exit" and position is not None:
                    cash = self._exit(position, bar.open, bar.timestamp, index, pending["reason"], cash, trades)
                    position = None
                pending = None
            if position is not None:
                position, cash = self._check_bar_exit(position, bar, index, cash, trades)
                if position is not None:
                    position.bars_held += 1
                    if position.bars_held >= position.max_holding_bars:
                        cash = self._exit(position, bar.close, bar.timestamp, index, "max_holding_bars", cash, trades)
                        position = None
            equity = self._mark(cash, position, bar.close)
            if self.risk.mark(bar.timestamp, equity) and position is not None:
                cash = self._exit(position, bar.close, bar.timestamp, index, "daily_loss_limit", cash, trades)
                position = None
                equity = cash
            peak_equity = max(peak_equity, equity)
            drawdown = (equity - peak_equity) / peak_equity if peak_equity else 0.0
            equity_curve.append(
                EquityPoint(
                    timestamp=bar.timestamp,
                    equity=equity,
                    drawdown=drawdown,
                    price=bar.close,
                    position=position.quantity if position else 0,
                )
            )
            if index == len(data) - 1:
                if position is not None:
                    cash = self._exit(position, bar.close, bar.timestamp, index, "end_of_data", cash, trades)
                    position = None
                    equity = cash
                    peak_equity = max(peak_equity, equity)
                    drawdown = (equity - peak_equity) / peak_equity if peak_equity else 0.0
                    equity_curve[-1] = EquityPoint(bar.timestamp, equity, drawdown, bar.close, 0)
                continue
            if position is None and pending is None:
                signal = self._signal(data[: index + 1])
                if signal is not None:
                    signals.append(signal)
                    if self.risk.can_enter(bar.timestamp, equity, bar.close, signal.stop_distance, index):
                        pending = {"kind": "enter", "signal": signal.to_dict()}
            elif position is not None and pending is None:
                if self.strategy.should_flatten(bar):
                    pending = {"kind": "exit", "reason": "session_flatten"}
                else:
                    signal = self._signal(data[: index + 1])
                    if signal is not None and signal.side != position.side:
                        pending = {"kind": "exit", "reason": "opposing_signal"}
        final_equity = cash
        metrics = calculate_metrics(equity_curve, trades, self.config.risk.initial_capital, self.config.bar_minutes)
        return BacktestResult(
            symbol=self.config.symbol,
            initial_capital=self.config.risk.initial_capital,
            final_equity=final_equity,
            metrics=metrics,
            trades=trades,
            equity_curve=equity_curve,
            signals=signals,
            config=self.config.to_dict(),
        )

    def _build_strategy(self):
        if self.config.strategy_mode == "trend_breakout":
            return TrendBreakoutStrategy(self.config.strategy, self.config.timezone)
        if self.config.strategy_mode == "mean_reversion":
            return MeanReversionStrategy(self.config.strategy, self.config.timezone)
        if self.config.strategy_mode == "volatility_breakout":
            return VolatilityBreakoutStrategy(self.config.strategy, self.config.timezone)
        if self.config.strategy_mode == "opening_range":
            return OpeningRangeBreakoutStrategy(self.config.strategy, self.config.timezone)
        return RegimeStrategy(self.config.strategy, self.config.timezone)

    def _signal(self, bars: list[Bar]) -> Signal | None:
        return self.strategy.signal(bars)

    def _mark(self, cash: float, position: Position | None, price: float) -> float:
        if position is None:
            return cash
        return cash + position.unrealized(price) - position.entry_fee

    def _enter(self, signal: Signal, bar: Bar, index: int, equity: float) -> Position | None:
        if signal.side == Side.FLAT:
            return None
        is_buy = signal.side == Side.LONG
        entry_price = self.costs.fill_price(bar.open, is_buy)
        stop_distance = max(signal.stop_distance, signal.atr * self.config.risk.stop_buffer_atr)
        if not self.risk.can_enter(bar.timestamp, equity, entry_price, stop_distance, index):
            return None
        sizing = self.risk.size_position(equity, entry_price, stop_distance, signal.confidence)
        if sizing.quantity < 1:
            return None
        entry_fee = self.costs.side_cost(is_buy, entry_price, sizing.quantity, self.config.risk.contract_multiplier).total
        self.risk.register_entry(index)
        direction = 1 if signal.side == Side.LONG else -1
        stop_price = entry_price - direction * stop_distance
        target_price = entry_price + direction * stop_distance * signal.target_r
        return Position(
            symbol=signal.symbol,
            side=signal.side,
            quantity=sizing.quantity,
            entry_price=entry_price,
            entry_time=bar.timestamp,
            stop_price=stop_price,
            target_price=target_price,
            entry_fee=entry_fee,
            multiplier=self.config.risk.contract_multiplier,
            margin=sizing.margin,
            max_holding_bars=self.config.strategy.max_holding_bars,
            signal_reason=signal.reason,
        )

    def _check_bar_exit(
        self,
        position: Position,
        bar: Bar,
        index: int,
        cash: float,
        trades: list[Trade],
    ) -> tuple[Position | None, float]:
        if position.side == Side.LONG:
            if bar.low <= position.stop_price:
                return None, self._exit(position, position.stop_price, bar.timestamp, index, "stop_loss", cash, trades)
            if bar.high >= position.target_price:
                return None, self._exit(position, position.target_price, bar.timestamp, index, "take_profit", cash, trades)
        else:
            if bar.high >= position.stop_price:
                return None, self._exit(position, position.stop_price, bar.timestamp, index, "stop_loss", cash, trades)
            if bar.low <= position.target_price:
                return None, self._exit(position, position.target_price, bar.timestamp, index, "take_profit", cash, trades)
        return position, cash

    def _exit(
        self,
        position: Position,
        reference_price: float,
        exit_time: datetime,
        index: int,
        reason: str,
        cash: float,
        trades: list[Trade],
    ) -> float:
        is_buy = position.side == Side.SHORT
        exit_price = self.costs.fill_price(reference_price, is_buy)
        exit_fee = self.costs.side_cost(is_buy, exit_price, position.quantity, position.multiplier).total
        direction = 1 if position.side == Side.LONG else -1
        gross_pnl = (exit_price - position.entry_price) * direction * position.quantity * position.multiplier
        net_pnl = gross_pnl - position.entry_fee - exit_fee
        cash += gross_pnl - position.entry_fee - exit_fee
        trades.append(
            Trade(
                symbol=position.symbol,
                side=position.side,
                quantity=position.quantity,
                entry_time=position.entry_time,
                exit_time=exit_time,
                entry_price=position.entry_price,
                exit_price=exit_price,
                gross_pnl=gross_pnl,
                fees=position.entry_fee + exit_fee,
                net_pnl=net_pnl,
                exit_reason=reason,
                entry_reason=position.signal_reason,
                bars_held=position.bars_held,
                multiplier=position.multiplier,
            )
        )
        self.risk.register_exit(net_pnl, index)
        return cash


def calculate_metrics(
    equity_curve: list[EquityPoint],
    trades: list[Trade],
    initial_capital: float,
    bar_minutes: int = 15,
) -> dict[str, Any]:
    if not equity_curve:
        return {
            "total_return": 0.0,
            "final_equity": initial_capital,
            "trades": 0,
            "status": "insufficient_data",
        }
    values = [point.equity for point in equity_curve]
    returns = [values[index] / values[index - 1] - 1.0 for index in range(1, len(values)) if values[index - 1] > 0]
    final_equity = values[-1]
    total_return = final_equity / initial_capital - 1.0 if initial_capital else 0.0
    elapsed_days = max((equity_curve[-1].timestamp - equity_curve[0].timestamp).total_seconds() / 86400.0, 1.0 / 86400.0)
    years = elapsed_days / 365.25
    cagr = (final_equity / initial_capital) ** (1.0 / years) - 1.0 if initial_capital > 0 and final_equity > 0 else -1.0
    periods_per_year = 252.0 * max(1.0, 375.0 / max(bar_minutes, 1))
    volatility = pstdev(returns) * math.sqrt(periods_per_year) if len(returns) > 1 else 0.0
    # volatility above is already annualised, so annualising Sharpe needs the
    # full period count, not its square root.
    sharpe = mean(returns) / volatility * periods_per_year if volatility > 0 else 0.0
    downside = [min(value, 0.0) for value in returns]
    downside_deviation = math.sqrt(mean([value * value for value in downside])) if downside else 0.0
    sortino = mean(returns) / downside_deviation * math.sqrt(periods_per_year) if downside_deviation > 0 else 0.0
    peak = initial_capital
    max_drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        max_drawdown = min(max_drawdown, value / peak - 1.0)
    daily_values: dict[Any, float] = {}
    for point in equity_curve:
        daily_values[point.timestamp.date()] = point.equity
    daily_returns = []
    previous = initial_capital
    for value in daily_values.values():
        if previous > 0:
            daily_returns.append(value / previous - 1.0)
        previous = value
    winning = [trade.net_pnl for trade in trades if trade.net_pnl > 0]
    losing = [trade.net_pnl for trade in trades if trade.net_pnl < 0]
    gross_profit = sum(winning)
    gross_loss = abs(sum(losing))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
    monthly: dict[str, float] = {}
    month_end: dict[str, float] = {}
    for point in equity_curve:
        key = point.timestamp.strftime("%Y-%m")
        month_end[key] = point.equity
    previous_month_value = initial_capital
    for key in sorted(month_end):
        monthly[key] = month_end[key] / previous_month_value - 1.0 if previous_month_value else 0.0
        previous_month_value = month_end[key]
    total_fees = sum(trade.fees for trade in trades)
    turnover = sum(abs((trade.entry_price + trade.exit_price) * trade.quantity * trade.multiplier / 2.0) for trade in trades)
    exposure = sum(1 for point in equity_curve if point.position != 0) / len(equity_curve)
    benchmark_return = equity_curve[-1].price / equity_curve[0].price - 1.0 if equity_curve[0].price else 0.0
    return {
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": cagr / abs(max_drawdown) if max_drawdown < 0 else 0.0,
        "win_rate": len(winning) / len(trades) if trades else 0.0,
        "profit_factor": profit_factor,
        "expectancy": mean([trade.net_pnl for trade in trades]) if trades else 0.0,
        "trades": len(trades),
        "wins": len(winning),
        "losses": len(losing),
        "total_fees": total_fees,
        "turnover": turnover,
        "exposure_fraction": exposure,
        "average_holding_bars": mean([trade.bars_held for trade in trades]) if trades else 0.0,
        "benchmark_return": benchmark_return,
        "best_day": max(daily_returns) if daily_returns else 0.0,
        "worst_day": min(daily_returns) if daily_returns else 0.0,
        "monthly_returns": monthly,
        "start_equity": values[0],
        "final_equity": final_equity,
    }
