from dataclasses import dataclass

from .config import CostConfig


@dataclass(frozen=True)
class CostBreakdown:
    brokerage: float
    exchange: float
    stt: float
    stamp: float
    sebi: float
    gst: float
    other: float

    @property
    def total(self) -> float:
        return self.brokerage + self.exchange + self.stt + self.stamp + self.sebi + self.gst + self.other


class CostModel:
    def __init__(self, config: CostConfig | None = None) -> None:
        self.config = config or CostConfig()

    def fill_price(self, reference_price: float, is_buy: bool) -> float:
        slippage = self.config.slippage_bps / 10_000.0
        return reference_price * (1.0 + slippage if is_buy else 1.0 - slippage)

    def side_cost(self, is_buy: bool, price: float, quantity: int, multiplier: float) -> CostBreakdown:
        gross = abs(price * quantity * multiplier)
        brokerage = min(
            self.config.brokerage_per_order,
            gross * self.config.brokerage_pct,
        )
        exchange = gross * self.config.exchange_txn_pct
        stt = 0.0 if is_buy else gross * self.config.stt_sell_pct
        stamp = gross * self.config.stamp_buy_pct if is_buy else 0.0
        sebi = gross * self.config.sebi_pct
        taxable = brokerage + exchange + sebi
        gst = taxable * self.config.gst_pct
        return CostBreakdown(
            brokerage=brokerage,
            exchange=exchange,
            stt=stt,
            stamp=stamp,
            sebi=sebi,
            gst=gst,
            other=self.config.other_per_order,
        )
