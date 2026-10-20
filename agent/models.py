from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from decimal import Decimal


def _plain(x: Decimal) -> str:
    """Decimal as plain text, no exponent and no trailing zeros: 0.02073700 -> '0.020737', 1E+2 -> '100'."""
    return format(x.normalize(), "f")


@dataclass
class Quote:
    symbol: str
    last: Decimal
    bid: Decimal
    ask: Decimal
    prev_close: Decimal
    ts: datetime | None          # time of the last trade, timezone-aware
    state: str = "active"
    has_traded: bool = True
    asset: str = "stock"         # stock | crypto. Crypto symbols are pairs like "BTC-USD", so they never clash with a stock ticker


@dataclass
class CryptoPair:
    """One Robinhood crypto pair and its order steps. Orders off these steps are rejected."""
    symbol: str                  # "BTC-USD": the agent's name for the coin everywhere
    code: str                    # "BTC"
    qty_step: Decimal            # min_order_quantity_increment
    price_step: Decimal          # min_order_price_increment
    tradable: bool = True


@dataclass
class Position:
    symbol: str
    quantity: Decimal
    avg_cost: Decimal


@dataclass
class Candidate:
    """One proposed order. Created ONLY by strategy.py, never by the LLM."""
    symbol: str
    side: str                          # buy | sell
    type: str                          # limit | market
    reason: str
    quantity: Decimal | None = None
    dollar_amount: Decimal | None = None
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None  # protective stop orders only
    kind: str = ""                     # entry | exit | protect  (filled in automatically)
    is_exit: bool = False              # exits are allowed in windows where new buys are not
    source: str = "rules"              # rules | llm
    asset: str = "stock"               # stock | crypto
    time_in_force: str = "gfd"         # gfd | gtc. The daily sessions keep protective stops alive overnight (gtc)

    def __post_init__(self) -> None:
        if not self.kind:
            self.kind = "entry" if self.side == "buy" else "exit"

    def notional(self, ref_price: Decimal) -> Decimal:
        if self.dollar_amount is not None:
            return self.dollar_amount
        price = self.limit_price if self.limit_price is not None else ref_price
        return (self.quantity or Decimal(0)) * price

    def to_order_args(self, account_number: str) -> dict:
        """Arguments for review_equity_order / place_equity_order. All values are strings."""
        args = {"account_number": account_number, "symbol": self.symbol, "side": self.side,
                "type": self.type, "time_in_force": self.time_in_force, "market_hours": "regular_hours"}
        if self.dollar_amount is not None:
            args["dollar_amount"] = f"{self.dollar_amount:.2f}"
        else:
            args["quantity"] = format(self.quantity.normalize(), "f")
        if self.limit_price is not None:
            args["limit_price"] = f"{self.limit_price:.2f}"
        if self.stop_price is not None:
            args["stop_price"] = f"{self.stop_price:.2f}"
        return args

    def to_crypto_args(self, rhs_account: str) -> dict:
        """Arguments for preview_crypto_order / place_crypto_order. Crypto calls a stop order 'stop_loss',
        and its market and limit orders only take 'gtc'."""
        stop = self.type == "stop_market"
        args = {"rhs_account_number": rhs_account, "symbol": self.symbol, "side": self.side,
                "type": "stop_loss" if stop else self.type, "time_in_force": self.time_in_force if stop else "gtc"}
        if self.dollar_amount is not None:
            args["dollar_amount"] = f"{self.dollar_amount:.2f}"
        else:
            args["quantity"] = _plain(self.quantity)
        if self.limit_price is not None:
            args["limit_price"] = _plain(self.limit_price)
        if self.stop_price is not None:
            args["stop_price"] = _plain(self.stop_price)
        return args

    def as_dict(self) -> dict:
        return {k: (str(v) if isinstance(v, Decimal) else v) for k, v in asdict(self).items()}


@dataclass
class Snapshot:
    now: datetime                      # timezone-aware, US/Eastern
    quotes: dict[str, Quote]
    positions: dict[str, Position]
    buying_power: Decimal | None
    equity: Decimal | None


@dataclass
class DayState:
    date: str
    start_equity: str | None = None
    buy_usd_total: str = "0"
    trades: list[dict] = field(default_factory=list)   # {symbol, side, kind, ref_id, status}
    last_llm_decision: str | None = None               # ISO time of the last decider call
    paper_pnl: str = "0"                                # realized P&L in paper mode
