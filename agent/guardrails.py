"""Hard limits. Plain code, no LLM. If this module says no, nothing else can say yes.

check() returns a list of violations. Empty list = the order may go on to the next step.
"""
from __future__ import annotations

from datetime import datetime, time
from decimal import Decimal
from zoneinfo import ZoneInfo

from .models import Candidate, DayState, Snapshot

ET = ZoneInfo("America/New_York")
OPEN, CLOSE = time(9, 30), time(16, 0)


def now_et() -> datetime:
    return datetime.now(ET)


def market_open(now: datetime) -> bool:
    """Weekday + regular hours. Holidays are caught by the stale-quote check."""
    return now.weekday() < 5 and OPEN <= now.time() < CLOSE


def minutes_since_open(now: datetime) -> float:
    return (now - now.replace(hour=9, minute=30, second=0, microsecond=0)).total_seconds() / 60


def minutes_to_close(now: datetime) -> float:
    return (now.replace(hour=16, minute=0, second=0, microsecond=0) - now).total_seconds() / 60


def check(c: Candidate, snap: Snapshot, day: DayState, cfg: dict, kill: bool = False) -> list[str]:
    """Same limits for stocks and crypto. Crypto differs only where the market does:
    it trades 24/7 (no session rules), takes fractional amounts, and has its own spread limit."""
    g = cfg["guardrails"]
    v: list[str] = []
    D = Decimal
    crypto = c.asset == "crypto"

    if kill:
        v.append("KILL file present")
    if c.symbol not in cfg["universe"]:
        v.append(f"{c.symbol} is not in universe")
    if c.side not in ("buy", "sell"):
        v.append(f"bad side {c.side!r}")
    allowed = set(g["allowed_order_types"])
    if (cfg.get("protect") or {}).get("broker_stop") and c.kind == "protect":
        allowed.add("stop_market")          # protective stops are allowed whenever protection is switched on
    if c.type not in allowed:
        v.append(f"order type {c.type!r} not allowed")
    if (c.quantity is None) == (c.dollar_amount is None):
        v.append("need exactly one of quantity / dollar_amount")
    if c.quantity is not None and c.quantity <= 0:
        v.append("quantity must be positive")
    if c.dollar_amount is not None and c.type != "market" and not crypto:     # crypto limit orders take dollars
        v.append("dollar_amount only works with market orders")
    if c.type == "limit" and c.limit_price is None:
        v.append("limit order without limit_price")
    if c.type == "stop_market":
        if c.kind != "protect" or c.side != "sell":
            v.append("stop orders are only allowed as protective sells")
        if c.stop_price is None or c.stop_price <= 0:
            v.append("stop order without stop_price")
        if c.quantity is None or (not crypto and c.quantity != c.quantity.to_integral_value()):
            v.append("stop orders need whole shares" if not crypto else "stop orders need a quantity")

    # --- market session (stocks only: crypto trades around the clock) --------
    # A protective stop is exempt: placed after hours or during a halt, Robinhood simply holds it until trading
    # resumes. Blocking it would only leave the holding with no protection at all.
    protect = c.kind == "protect"
    if crypto or protect:
        pass
    elif g.get("regular_hours_only", True) and not market_open(snap.now):
        v.append("market is closed")
    elif c.side == "buy":
        if minutes_since_open(snap.now) < g["no_new_buys_first_minutes"]:
            v.append("inside no-buy window after the open")
        if minutes_to_close(snap.now) < g["no_new_buys_last_minutes"]:
            v.append("inside no-buy window before the close")

    # --- quote quality ------------------------------------------------------
    q = snap.quotes.get(c.symbol)
    if q is None:
        return v + ["no quote"]
    if not protect and (q.state != "active" or not q.has_traded):
        v.append(f"instrument state={q.state} has_traded={q.has_traded}")
    if not protect and (q.ts is None or (snap.now - q.ts).total_seconds() > g["max_quote_age_seconds"]):
        v.append("quote is stale (holiday, halt, or feed problem)")
    if q.bid <= 0 or q.ask <= 0 or q.ask < q.bid:
        if not protect:                     # with no usable bid a stop is checked against nothing, and still placed
            v.append("bad bid/ask")
    else:
        mid = (q.bid + q.ask) / 2
        spread_pct = (q.ask - q.bid) / mid * 100
        # Buys only. A wide spread is a reason not to buy, never a reason to be stuck without an exit or a stop.
        if c.side == "buy" and spread_pct > D(str(g.get("crypto_max_spread_pct", 1.0) if crypto else g["max_spread_pct"])):
            v.append(f"spread {spread_pct:.2f}% too wide")
        slip = D(str(g["max_limit_slippage_pct"])) / 100
        if c.limit_price is not None:
            if c.side == "buy" and c.limit_price > q.ask * (1 + slip):
                v.append("buy limit too far above the ask")
            if c.side == "sell" and c.limit_price < q.bid * (1 - slip):
                v.append("sell limit too far below the bid")
        if c.stop_price is not None and c.stop_price >= q.bid:
            v.append("stop price is not below the bid (it would trigger at once)")

    # --- once-per-day and count limits ---------------------------------------
    live = [t for t in day.trades if t.get("status") in ("placed", "pending", "unknown")]
    kind_of = lambda t: t.get("kind") or ("entry" if t["side"] == "buy" else "exit")
    if any(t["symbol"] == c.symbol and kind_of(t) == c.kind for t in live):
        v.append(f"already did a {c.kind} order on {c.symbol} today")
    # The count cap applies to BUYS only. A counter must never be able to block an exit.
    if c.side == "buy" and len([t for t in live if t["side"] == "buy"]) >= g["max_trades_per_day"]:
        v.append("max_trades_per_day reached")

    pos = snap.positions.get(c.symbol)
    if c.side == "sell":
        if pos is None or pos.quantity <= 0:
            v.append("no position to sell (shorting is never allowed)")
        elif c.quantity is not None and c.quantity > pos.quantity:
            v.append("sell quantity is larger than the position")
        return v

    # --- buy-only money limits ------------------------------------------------
    notional = c.notional(q.ask)
    if notional > D(str(g["max_order_usd"])):
        v.append(f"order ${notional:.2f} is over max_order_usd")
    if D(day.buy_usd_total) + notional > D(str(g["max_daily_buy_usd"])):
        v.append("max_daily_buy_usd would be exceeded")
    held = (pos.quantity * q.last) if pos else D(0)
    if held + notional > D(str(g["max_position_usd"])):
        v.append("max_position_usd would be exceeded")
    if pos is None and len([p for p in snap.positions.values() if p.quantity > 0]) >= g["max_open_positions"]:
        v.append("max_open_positions reached")
    if snap.buying_power is None:
        v.append("buying power unknown")
    elif notional > snap.buying_power:
        v.append("not enough buying power")
    if day.start_equity and snap.equity is not None:
        start = D(day.start_equity)
        if start > 0 and (start - snap.equity) / start * 100 >= D(str(g["daily_loss_stop_pct"])):
            v.append("daily loss stop hit: no new buys today")
    elif snap.equity is None:
        v.append("account equity unknown")
    return v
