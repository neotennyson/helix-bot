from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class StrategyConfig:
    fast_ema: int = 20
    slow_ema: int = 60
    atr_period: int = 14
    rsi_period: int = 14
    adx_period: int = 14
    breakout_period: int = 40
    trend_adx_min: float = 22.0
    range_adx_max: float = 20.0
    trend_threshold_atr: float = 0.45
    range_threshold_atr: float = 0.65
    rsi_trend_low: float = 50.0
    rsi_trend_high: float = 78.0
    rsi_range_long: float = 34.0
    rsi_range_short: float = 66.0
    min_atr_pct: float = 0.0007
    max_atr_pct: float = 0.015
    min_volume_ratio: float = 0.75
    confidence_threshold: float = 0.56
    target_r: float = 2.0
    stop_atr_multiple: float = 1.15
    max_holding_bars: int = 20
    allow_short: bool = True
    intraday_only: bool = True
    session_start: str = "09:30"
    flatten_at: str = "15:05"
    expiry_entry_cutoff: str = "13:30"
    expiry_flatten_at: str = "14:30"
    use_daily_bias: bool = True
    daily_fast_ema: int = 20
    daily_slow_ema: int = 50
    use_long_term_filter: bool = False
    long_term_ema: int = 200
    long_term_slope_bars: int = 20
    opening_range_bars: int = 6
    opening_range_max_atr: float = 2.5

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "StrategyConfig":
        if not value:
            return cls()
        fields = {field.name for field in self_fields(cls)}
        return cls(**{key: val for key, val in value.items() if key in fields})


@dataclass
class RiskConfig:
    initial_capital: float = 1_000_000.0
    cash_hurdle_annual: float = 0.06
    risk_per_trade_pct: float = 0.005
    max_notional_pct: float = 1.5
    max_margin_pct: float = 0.30
    margin_pct: float = 0.15
    max_daily_loss_pct: float = 0.02
    max_trades_per_day: int = 4
    max_consecutive_losses: int = 3
    cooldown_bars: int = 2
    max_quantity: int = 100
    contract_multiplier: float = 1.0
    stop_buffer_atr: float = 0.05

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "RiskConfig":
        if not value:
            return cls()
        fields = {field.name for field in self_fields(cls)}
        return cls(**{key: val for key, val in value.items() if key in fields})


@dataclass
class CostConfig:
    slippage_bps: float = 2.0
    brokerage_per_order: float = 20.0
    brokerage_pct: float = 0.0003
    exchange_txn_pct: float = 0.00003
    stt_sell_pct: float = 0.0002
    stamp_buy_pct: float = 0.00002
    sebi_pct: float = 0.0000001
    gst_pct: float = 0.18
    other_per_order: float = 0.0

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "CostConfig":
        if not value:
            return cls()
        fields = {field.name for field in self_fields(cls)}
        return cls(**{key: val for key, val in value.items() if key in fields})


@dataclass
class BacktestConfig:
    symbol: str = "NIFTY"
    instrument_scope: str = "INDEX_FUTURES_ONLY"
    strategy_mode: str = "regime"
    timezone: str = "Asia/Kolkata"
    bar_minutes: int = 15
    # Bars of history the paper engine must retain so its indicator state
    # matches a full backtest. Truncating this silently changes signals.
    history_bars: int = 6000
    strategy: StrategyConfig | None = None
    risk: RiskConfig | None = None
    costs: CostConfig | None = None

    def __post_init__(self) -> None:
        if self.strategy is None:
            self.strategy = StrategyConfig()
        if self.risk is None:
            self.risk = RiskConfig()
        if self.costs is None:
            self.costs = CostConfig()
        if self.strategy.fast_ema >= self.strategy.slow_ema:
            raise ValueError("fast_ema must be smaller than slow_ema")
        if min(self.strategy.atr_period, self.strategy.rsi_period, self.strategy.adx_period) <= 0:
            raise ValueError("indicator periods must be positive")
        if self.strategy.breakout_period <= 1:
            raise ValueError("breakout_period must exceed one")
        if self.strategy.use_long_term_filter and self.strategy.long_term_ema <= self.strategy.slow_ema:
            raise ValueError("long_term_ema must exceed slow_ema when enabled")
        if self.strategy.long_term_slope_bars <= 0:
            raise ValueError("long_term_slope_bars must be positive")
        if self.strategy.range_adx_max > self.strategy.trend_adx_min:
            raise ValueError("range_adx_max must not exceed trend_adx_min")
        if self.risk.initial_capital <= 0 or self.risk.contract_multiplier <= 0:
            raise ValueError("capital and contract multiplier must be positive")
        if self.history_bars < 100:
            raise ValueError("history_bars must retain at least 100 bars")
        if self.strategy.opening_range_bars < 2:
            raise ValueError("opening_range_bars must be at least two")
        if self.strategy.opening_range_max_atr <= 0:
            raise ValueError("opening_range_max_atr must be positive")
        if self.strategy_mode not in {"regime", "trend_breakout", "mean_reversion", "volatility_breakout", "opening_range"}:
            raise ValueError(f"unsupported strategy_mode: {self.strategy_mode}")

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "BacktestConfig":
        value = value or {}
        return cls(
            symbol=value.get("symbol", "NIFTY"),
            instrument_scope=value.get("instrument_scope", "INDEX_FUTURES_ONLY"),
            strategy_mode=value.get("strategy_mode", "regime"),
            timezone=value.get("timezone", "Asia/Kolkata"),
            bar_minutes=int(value.get("bar_minutes", 15)),
            history_bars=int(value.get("history_bars", 6000)),
            strategy=StrategyConfig.from_dict(value.get("strategy")),
            risk=RiskConfig.from_dict(value.get("risk")),
            costs=CostConfig.from_dict(value.get("costs")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "instrument_scope": self.instrument_scope,
            "strategy_mode": self.strategy_mode,
            "timezone": self.timezone,
            "bar_minutes": self.bar_minutes,
            "history_bars": self.history_bars,
            "strategy": asdict(self.strategy),
            "risk": asdict(self.risk),
            "costs": asdict(self.costs),
        }


def self_fields(cls: type) -> tuple[Any, ...]:
    from dataclasses import fields

    return fields(cls)


def load_config(path: str) -> BacktestConfig:
    import json

    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    return BacktestConfig.from_dict(value)
