import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from .config import RiskConfig
from .models import Bar, Trade


@dataclass
class SizingResult:
    quantity: int
    risk_budget: float
    notional: float
    margin: float


class RiskManager:
    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()
        self.day_key: date | None = None
        self.day_start_equity = self.config.initial_capital
        self.trades_today = 0
        self.consecutive_losses = 0
        self.halted = False
        self.last_exit_index: int | None = None
        self.last_entry_index: int | None = None

    def start_day(self, timestamp: datetime, equity: float) -> None:
        key = timestamp.date()
        if self.day_key != key:
            self.day_key = key
            self.day_start_equity = equity
            self.trades_today = 0
            self.consecutive_losses = 0
            self.halted = False

    def can_enter(self, timestamp: datetime, equity: float, price: float, stop_distance: float, bar_index: int) -> bool:
        self.start_day(timestamp, equity)
        if self.halted or self.trades_today >= self.config.max_trades_per_day:
            return False
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            return False
        if self.last_exit_index is not None and bar_index - self.last_exit_index <= self.config.cooldown_bars:
            return False
        return self.size_position(equity, price, stop_distance).quantity > 0

    def size_position(self, equity: float, price: float, stop_distance: float, confidence: float = 1.0) -> SizingResult:
        if price <= 0 or stop_distance <= 0 or equity <= 0:
            return SizingResult(0, 0.0, 0.0, 0.0)
        confidence_scale = min(1.0, max(0.5, confidence))
        risk_budget = equity * self.config.risk_per_trade_pct * confidence_scale
        multiplier = max(self.config.contract_multiplier, 0.000001)
        quantity_by_risk = int(math.floor(risk_budget / (stop_distance * multiplier)))
        max_notional = equity * self.config.max_notional_pct
        quantity_by_notional = int(math.floor(max_notional / (price * multiplier)))
        margin_pct = max(self.config.margin_pct, 0.000001)
        available_margin = max(0.0, equity * self.config.max_margin_pct)
        quantity_by_margin = int(math.floor(available_margin / (price * multiplier * margin_pct)))
        quantity = max(0, min(quantity_by_risk, quantity_by_notional, quantity_by_margin, self.config.max_quantity))
        notional = quantity * price * multiplier
        margin = notional * self.config.margin_pct
        return SizingResult(quantity, risk_budget, notional, margin)

    def register_entry(self, bar_index: int) -> None:
        self.trades_today += 1
        self.last_entry_index = bar_index

    def register_exit(self, pnl: float, bar_index: int) -> None:
        self.last_exit_index = bar_index
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def mark(self, timestamp: datetime, equity: float) -> bool:
        self.start_day(timestamp, equity)
        if self.day_start_equity > 0:
            loss = (self.day_start_equity - equity) / self.day_start_equity
            if loss >= self.config.max_daily_loss_pct:
                self.halted = True
        return self.halted

    def to_dict(self) -> dict[str, Any]:
        return {
            "day_key": self.day_key.isoformat() if self.day_key else None,
            "day_start_equity": self.day_start_equity,
            "trades_today": self.trades_today,
            "consecutive_losses": self.consecutive_losses,
            "halted": self.halted,
            "last_exit_index": self.last_exit_index,
            "last_entry_index": self.last_entry_index,
        }

    def load_dict(self, value: dict[str, Any] | None) -> None:
        value = value or {}
        day_key = value.get("day_key")
        self.day_key = date.fromisoformat(day_key) if day_key else None
        self.day_start_equity = float(value.get("day_start_equity", self.config.initial_capital))
        self.trades_today = int(value.get("trades_today", 0))
        self.consecutive_losses = int(value.get("consecutive_losses", 0))
        self.halted = bool(value.get("halted", False))
        self.last_exit_index = value.get("last_exit_index")
        self.last_entry_index = value.get("last_entry_index")
