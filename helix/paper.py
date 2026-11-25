import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .backtest import calculate_metrics
from .config import BacktestConfig
from .costs import CostModel
from .data import parse_timestamp
from .models import Bar, EquityPoint, Position, Side, Signal, Trade
from .risk import RiskManager
from .strategy import RegimeStrategy
from .trend_strategy import TrendBreakoutStrategy
from .mean_reversion_strategy import MeanReversionStrategy
from .opening_range_strategy import OpeningRangeBreakoutStrategy
from .volatility_breakout_strategy import VolatilityBreakoutStrategy


class PaperTradingEngine:
    mode = "PAPER_ONLY"

    def __init__(self, config: BacktestConfig, state_path: str | Path) -> None:
        self.config = config
        self.state_path = Path(state_path)
        self.strategy = self._build_strategy()
        self.risk = RiskManager(config.risk)
        self.costs = CostModel(config.costs)
        self.state = self._load_state()
        self.risk.load_dict(self.state.get("risk"))

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

    def initialize_history(self, bars: list[Bar]) -> None:
        if self.state.get("last_timestamp") is not None or not bars:
            return
        history = bars[:-1]
        if history:
            self.state["bars"] = [self._bar_dict(bar) for bar in history[-self.config.history_bars:]]
            self.state["last_timestamp"] = history[-1].timestamp.isoformat()
        self._persist()

    def process_bar(self, bar: Bar) -> dict[str, Any]:
        if bar.symbol != self.config.symbol:
            raise ValueError(f"paper state is for {self.config.symbol}, received {bar.symbol}")
        last_timestamp = self.state.get("last_timestamp")
        if last_timestamp is not None and bar.timestamp <= parse_timestamp(last_timestamp):
            return self.state
        history = self._state_bars()
        if history and history[-1].timestamp >= bar.timestamp:
            return self.state
        history.append(bar)
        self.state["bars"] = [self._bar_dict(item) for item in history[-self.config.history_bars:]]
        self.risk.load_dict(self.state.get("risk"))
        position = self._position()
        pending = self.state.get("pending")
        cash = float(self.state.get("cash", self.config.risk.initial_capital))
        current_equity = self._mark(cash, position, bar.open)
        self.risk.start_day(bar.timestamp, current_equity)
        if pending is not None:
            if pending.get("kind") == "enter" and position is None:
                position = self._enter(Signal.from_dict(pending["signal"]), bar, len(history) - 1, current_equity)
            elif pending.get("kind") == "exit" and position is not None:
                trade = self._exit(position, bar.open, bar.timestamp, len(history) - 1, pending.get("reason", "signal"), cash)
                cash = trade[0]
                self._append_trade(trade[1])
                position = None
            self.state["pending"] = None
        if position is not None:
            position, cash, closed_trade = self._check_bar_exit(position, bar, len(history) - 1, cash)
            if closed_trade is not None:
                self._append_trade(closed_trade)
            if position is not None:
                position.bars_held += 1
                if position.bars_held >= position.max_holding_bars:
                    cash, closed_trade = self._exit(position, bar.close, bar.timestamp, len(history) - 1, "max_holding_bars", cash)
                    self._append_trade(closed_trade)
                    position = None
        equity = self._mark(cash, position, bar.close)
        if self.risk.mark(bar.timestamp, equity) and position is not None:
            cash, closed_trade = self._exit(position, bar.close, bar.timestamp, len(history) - 1, "daily_loss_limit", cash)
            self._append_trade(closed_trade)
            position = None
            equity = cash
        self.state["position"] = position.to_dict() if position else None
        self.state["cash"] = cash
        self.state["equity"] = equity
        unrealized = position.unrealized(bar.close) - (position.entry_fee if position else 0.0) if position else 0.0
        self.state["unrealized_pnl"] = unrealized
        self.state["realized_equity"] = cash
        self.state["equity_includes_unrealized"] = bool(position)
        self.state["last_timestamp"] = bar.timestamp.isoformat()
        self.state["risk"] = self.risk.to_dict()
        self.state["status"] = "halted" if self.risk.halted else ("position_open" if position else "flat")
        prior_equity = [float(item["equity"]) for item in self.state.get("equity_curve", [])]
        peak_equity = max([self.config.risk.initial_capital, *prior_equity])
        drawdown = equity / peak_equity - 1.0 if peak_equity else 0.0
        self._append_equity(EquityPoint(bar.timestamp, equity, drawdown, bar.close, position.quantity if position else 0))
        if position is None and self.state.get("pending") is None:
            signal = self.strategy.signal(history)
            if signal is not None and self.risk.can_enter(bar.timestamp, equity, bar.close, signal.stop_distance, len(history) - 1):
                self.state["pending"] = {"kind": "enter", "signal": signal.to_dict()}
        elif position is not None and self.state.get("pending") is None:
            if self.strategy.should_flatten(bar):
                self.state["pending"] = {"kind": "exit", "reason": "session_flatten"}
            else:
                signal = self.strategy.signal(history)
                if signal is not None and signal.side != position.side:
                    self.state["pending"] = {"kind": "exit", "reason": "opposing_signal"}
        self.state["metrics"] = self._state_metrics()
        self._persist()
        return self.state

    def reset(self) -> None:
        self.state = self._new_state()
        self.risk = RiskManager(self.config.risk)
        self._persist()

    def _new_state(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "broker": "none",
            "symbol": self.config.symbol,
            "initial_capital": self.config.risk.initial_capital,
            "cash": self.config.risk.initial_capital,
            "equity": self.config.risk.initial_capital,
            "position": None,
            "pending": None,
            "unrealized_pnl": 0.0,
            "realized_equity": self.config.risk.initial_capital,
            "equity_includes_unrealized": False,
            "trades": [],
            "equity_curve": [],
            "bars": [],
            "risk": self.risk.to_dict(),
            "last_timestamp": None,
            "status": "waiting_for_bar",
        }

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self._new_state()
        with self.state_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("symbol") != self.config.symbol:
            raise ValueError("paper state symbol does not match configuration")
        return state

    def _persist(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self.state, handle, indent=2, allow_nan=False)
        temporary.replace(self.state_path)

    def _state_bars(self) -> list[Bar]:
        return [self._bar_from_dict(value) for value in self.state.get("bars", [])]

    def _position(self) -> Position | None:
        value = self.state.get("position")
        return Position.from_dict(value) if value else None

    def _mark(self, cash: float, position: Position | None, price: float) -> float:
        return cash if position is None else cash + position.unrealized(price) - position.entry_fee

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
        return Position(
            symbol=signal.symbol,
            side=signal.side,
            quantity=sizing.quantity,
            entry_price=entry_price,
            entry_time=bar.timestamp,
            stop_price=entry_price - direction * stop_distance,
            target_price=entry_price + direction * stop_distance * signal.target_r,
            entry_fee=entry_fee,
            multiplier=self.config.risk.contract_multiplier,
            margin=sizing.margin,
            max_holding_bars=self.config.strategy.max_holding_bars,
            signal_reason=signal.reason,
        )

    def _check_bar_exit(self, position: Position, bar: Bar, index: int, cash: float) -> tuple[Position | None, float, Trade | None]:
        if position.side == Side.LONG:
            if bar.low <= position.stop_price:
                cash, trade = self._exit(position, position.stop_price, bar.timestamp, index, "stop_loss", cash)
                return None, cash, trade
            if bar.high >= position.target_price:
                cash, trade = self._exit(position, position.target_price, bar.timestamp, index, "take_profit", cash)
                return None, cash, trade
        else:
            if bar.high >= position.stop_price:
                cash, trade = self._exit(position, position.stop_price, bar.timestamp, index, "stop_loss", cash)
                return None, cash, trade
            if bar.low <= position.target_price:
                cash, trade = self._exit(position, position.target_price, bar.timestamp, index, "take_profit", cash)
                return None, cash, trade
        return position, cash, None

    def _exit(self, position: Position, reference_price: float, exit_time: datetime, index: int, reason: str, cash: float) -> tuple[float, Trade]:
        is_buy = position.side == Side.SHORT
        exit_price = self.costs.fill_price(reference_price, is_buy)
        exit_fee = self.costs.side_cost(is_buy, exit_price, position.quantity, position.multiplier).total
        direction = 1 if position.side == Side.LONG else -1
        gross_pnl = (exit_price - position.entry_price) * direction * position.quantity * position.multiplier
        cash += gross_pnl - position.entry_fee - exit_fee
        trade = Trade(
            symbol=position.symbol,
            side=position.side,
            quantity=position.quantity,
            entry_time=position.entry_time,
            exit_time=exit_time,
            entry_price=position.entry_price,
            exit_price=exit_price,
            gross_pnl=gross_pnl,
            fees=position.entry_fee + exit_fee,
            net_pnl=gross_pnl - position.entry_fee - exit_fee,
            exit_reason=reason,
            entry_reason=position.signal_reason,
            bars_held=position.bars_held,
            multiplier=position.multiplier,
        )
        self.risk.register_exit(trade.net_pnl, index)
        return cash, trade

    def _append_trade(self, trade: Trade) -> None:
        self.state.setdefault("trades", []).append(trade.to_dict())

    def _append_equity(self, point: EquityPoint) -> None:
        curve = self.state.setdefault("equity_curve", [])
        curve.append(point.to_dict())
        if len(curve) > 5000:
            del curve[:-5000]

    def _state_metrics(self) -> dict[str, Any]:
        curve = [EquityPoint(parse_timestamp(item["timestamp"]), float(item["equity"]), float(item["drawdown"]), float(item["price"]), int(item["position"])) for item in self.state.get("equity_curve", [])]
        trades = [Trade(
            symbol=item["symbol"],
            side=Side(item["side"]),
            quantity=int(item["quantity"]),
            entry_time=parse_timestamp(item["entry_time"]),
            exit_time=parse_timestamp(item["exit_time"]),
            entry_price=float(item["entry_price"]),
            exit_price=float(item["exit_price"]),
            gross_pnl=float(item["gross_pnl"]),
            fees=float(item["fees"]),
            net_pnl=float(item["net_pnl"]),
            exit_reason=item["exit_reason"],
            entry_reason=item["entry_reason"],
            bars_held=int(item["bars_held"]),
            multiplier=float(item["multiplier"]),
        ) for item in self.state.get("trades", [])]
        return calculate_metrics(curve, trades, self.config.risk.initial_capital, self.config.bar_minutes)

    @staticmethod
    def _bar_dict(bar: Bar) -> dict[str, Any]:
        return {
            "timestamp": bar.timestamp.isoformat(),
            "symbol": bar.symbol,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "expiry": bar.expiry,
        }

    @staticmethod
    def _bar_from_dict(value: dict[str, Any]) -> Bar:
        return Bar(
            timestamp=parse_timestamp(value["timestamp"]),
            symbol=value["symbol"],
            open=float(value["open"]),
            high=float(value["high"]),
            low=float(value["low"]),
            close=float(value["close"]),
            volume=int(value.get("volume", 0)),
            expiry=value.get("expiry"),
        )
