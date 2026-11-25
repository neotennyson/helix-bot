from dataclasses import dataclass
from datetime import time
from zoneinfo import ZoneInfo

from .config import StrategyConfig
from .data import parse_time
from .models import Bar


@dataclass(frozen=True)
class SessionWindow:
    """Session/expiry gate shared by every strategy.

    Daily and monthly bars carry a midnight timestamp, so the clock gate is
    skipped for them; only expiry-day cutoffs still apply.
    """

    config: StrategyConfig
    timezone: ZoneInfo

    @classmethod
    def build(cls, config: StrategyConfig, timezone: str = "Asia/Kolkata") -> "SessionWindow":
        return cls(config=config, timezone=ZoneInfo(timezone))

    @property
    def session_start(self) -> time:
        return parse_time(self.config.session_start)

    @property
    def flatten_at(self) -> time:
        return parse_time(self.config.flatten_at)

    @property
    def expiry_entry_cutoff(self) -> time:
        return parse_time(self.config.expiry_entry_cutoff)

    @property
    def expiry_flatten_at(self) -> time:
        return parse_time(self.config.expiry_flatten_at)

    def local_time(self, bar: Bar) -> time:
        return bar.timestamp.astimezone(self.timezone).time().replace(tzinfo=None)

    def is_daily_bar(self, bar: Bar) -> bool:
        return self.local_time(bar) == time(0, 0)

    def allows_entry(self, bar: Bar) -> bool:
        local = self.local_time(bar)
        if not self.is_daily_bar(bar):
            if local < self.session_start:
                return False
            if self.config.intraday_only and local >= self.flatten_at:
                return False
        if bar.expiry and local >= self.expiry_entry_cutoff:
            return False
        return True

    def should_flatten(self, bar: Bar) -> bool:
        local = self.local_time(bar)
        if self.is_daily_bar(bar):
            return bool(bar.expiry and local >= self.expiry_flatten_at)
        if bar.expiry and local >= self.expiry_flatten_at:
            return True
        return self.config.intraday_only and local >= self.flatten_at
