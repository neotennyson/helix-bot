import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class Regime(str, Enum):
    TREND = "TREND"
    RANGE = "RANGE"
    NONE = "NONE"


@dataclass(frozen=True)
class Bar:
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    expiry: str | None = None

    def validate(self) -> None:
        values = (self.open, self.high, self.low, self.close)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("bar prices must be finite")
        if any(value <= 0 for value in values):
            raise ValueError("bar prices must be positive")
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError("bar high/low is inconsistent")
        if self.low > self.high or self.volume < 0:
            raise ValueError("bar range or volume is invalid")


@dataclass
class Signal:
    timestamp: datetime
    symbol: str
    side: Side
    confidence: float
    regime: Regime
    reason: str
    atr: float
    stop_distance: float
    target_r: float = 2.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "side": self.side.value,
            "confidence": round(self.confidence, 6),
            "regime": self.regime.value,
            "reason": self.reason,
            "atr": round(self.atr, 6),
            "stop_distance": round(self.stop_distance, 6),
            "target_r": round(self.target_r, 6),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Signal":
        return cls(
            timestamp=datetime.fromisoformat(value["timestamp"]),
            symbol=value["symbol"],
            side=Side(value["side"]),
            confidence=float(value["confidence"]),
            regime=Regime(value["regime"]),
            reason=value["reason"],
            atr=float(value["atr"]),
            stop_distance=float(value["stop_distance"]),
            target_r=float(value.get("target_r", 2.0)),
            metadata=value.get("metadata", {}),
        )


@dataclass
class Position:
    symbol: str
    side: Side
    quantity: int
    entry_price: float
    entry_time: datetime
    stop_price: float
    target_price: float
    entry_fee: float
    multiplier: float
    margin: float
    max_holding_bars: int
    bars_held: int = 0
    signal_reason: str = ""

    def unrealized(self, price: float) -> float:
        direction = 1 if self.side == Side.LONG else -1
        return (price - self.entry_price) * direction * self.quantity * self.multiplier

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat(),
            "stop_price": self.stop_price,
            "target_price": self.target_price,
            "entry_fee": self.entry_fee,
            "multiplier": self.multiplier,
            "margin": self.margin,
            "max_holding_bars": self.max_holding_bars,
            "bars_held": self.bars_held,
            "signal_reason": self.signal_reason,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Position":
        return cls(
            symbol=value["symbol"],
            side=Side(value["side"]),
            quantity=int(value["quantity"]),
            entry_price=float(value["entry_price"]),
            entry_time=datetime.fromisoformat(value["entry_time"]),
            stop_price=float(value["stop_price"]),
            target_price=float(value["target_price"]),
            entry_fee=float(value["entry_fee"]),
            multiplier=float(value["multiplier"]),
            margin=float(value["margin"]),
            max_holding_bars=int(value["max_holding_bars"]),
            bars_held=int(value.get("bars_held", 0)),
            signal_reason=value.get("signal_reason", ""),
        )


@dataclass
class Trade:
    symbol: str
    side: Side
    quantity: int
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    gross_pnl: float
    fees: float
    net_pnl: float
    exit_reason: str
    entry_reason: str
    bars_held: int
    multiplier: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "gross_pnl": self.gross_pnl,
            "fees": self.fees,
            "net_pnl": self.net_pnl,
            "exit_reason": self.exit_reason,
            "entry_reason": self.entry_reason,
            "bars_held": self.bars_held,
            "multiplier": self.multiplier,
        }


@dataclass
class EquityPoint:
    timestamp: datetime
    equity: float
    drawdown: float
    price: float
    position: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "equity": self.equity,
            "drawdown": self.drawdown,
            "price": self.price,
            "position": self.position,
        }


@dataclass
class BacktestResult:
    symbol: str
    initial_capital: float
    final_equity: float
    metrics: dict[str, Any]
    trades: list[Trade]
    equity_curve: list[EquityPoint]
    signals: list[Signal]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "initial_capital": self.initial_capital,
            "final_equity": self.final_equity,
            "metrics": self.metrics,
            "trades": [trade.to_dict() for trade in self.trades],
            "equity_curve": [point.to_dict() for point in self.equity_curve],
            "signals": [signal.to_dict() for signal in self.signals],
            "config": self.config,
            "validation": {
                "status": "research_only",
                "message": "A positive result is not a guarantee of future profit.",
            },
        }
