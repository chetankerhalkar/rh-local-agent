"""Deterministic rules. Same inputs -> same output, every time. No LLM in here.

This is a TEMPLATE, not advice. It mirrors the simple "buy the dip" example in
Robinhood's own agentic docs and adds day-trade style exits. Swap in your own
rules by writing another propose() with the same signature.
"""
from __future__ import annotations

from decimal import ROUND_DOWN, Decimal

from .guardrails import minutes_to_close
from .models import Candidate, DayState, Snapshot

CENT = Decimal("0.01")


def _exit_order(sym: str, qty: Decimal, bid: Decimal, reason: str, slack_pct: Decimal = Decimal(0)) -> Candidate:
    whole = qty == qty.to_integral_value()
    if whole:   # marketable limit: fills at the bid or better, never below (bid - slack)
        price = (bid * (1 - slack_pct / 100)).quantize(CENT)
        return Candidate(sym, "sell", "limit", reason, quantity=qty, limit_price=price, is_exit=True)
    # Robinhood only accepts fractional shares on market orders
    return Candidate(sym, "sell", "market", reason, quantity=qty, is_exit=True)


def in_flatten_window(snap: Snapshot, cfg: dict) -> bool:
    s = cfg["strategy"]
    return bool(s.get("flatten_before_close")) and \
        0 <= minutes_to_close(snap.now) <= s.get("flatten_minutes_before_close", 10)


def build_entry(sym: str, snap: Snapshot, cfg: dict, reason: str, source: str = "rules") -> Candidate | str:
    """Size and price an entry. The caller (rules or LLM) only picks the symbol. Returns a note on failure."""
    s, q = cfg["strategy"], snap.quotes[sym]
    usd = Decimal(str(s["order_usd"]))
    if s.get("order_style") == "market_dollars":
        return Candidate(sym, "buy", "market", reason, dollar_amount=usd, source=source)
    qty = (usd / q.ask).to_integral_value(rounding=ROUND_DOWN) if q.ask > 0 else Decimal(0)
    if qty < 1:
        return f"{sym}: one share (${q.ask}) costs more than order_usd, skipped"
    return Candidate(sym, "buy", "limit", reason, quantity=qty, limit_price=q.ask.quantize(CENT), source=source)


def build_exit(sym: str, snap: Snapshot, reason: str, source: str = "rules") -> Candidate:
    c = _exit_order(sym, snap.positions[sym].quantity, snap.quotes[sym].bid, reason)
    c.source = source
    return c


def propose_flatten(snap: Snapshot, cfg: dict, reason: str) -> list[Candidate]:
    """Sell every position in the universe. Used by the evening session.
    The limit sits a little under the bid so the order fills; you still get the bid or better."""
    slack = Decimal(str(cfg["guardrails"].get("max_limit_slippage_pct", 0.2))) / 2
    return [_exit_order(sym, pos.quantity, snap.quotes[sym].bid, reason, slack)
            for sym, pos in snap.positions.items()
            if sym in cfg["universe"] and sym in snap.quotes and pos.quantity > 0]


def propose_exits(snap: Snapshot, cfg: dict) -> list[Candidate]:
    """Hard exits. Pure code. These run in EVERY strategy mode and nothing can veto them."""
    s = cfg["strategy"]
    D = lambda x: Decimal(str(x))
    out: list[Candidate] = []
    flatten = in_flatten_window(snap, cfg)
    for sym, pos in snap.positions.items():
        q = snap.quotes.get(sym)
        if sym not in cfg["universe"] or q is None or pos.quantity <= 0 or pos.avg_cost <= 0:
            continue
        change = (q.last - pos.avg_cost) / pos.avg_cost * 100
        if flatten:
            out.append(_exit_order(sym, pos.quantity, q.bid, f"flatten before close ({change:+.2f}%)"))
        elif change <= -D(s["stop_loss_pct"]):
            out.append(_exit_order(sym, pos.quantity, q.bid, f"stop loss hit ({change:+.2f}%)"))
        elif change >= D(s["take_profit_pct"]):
            out.append(_exit_order(sym, pos.quantity, q.bid, f"take profit hit ({change:+.2f}%)"))
    return out


def propose(snap: Snapshot, day: DayState, cfg: dict) -> tuple[list[Candidate], list[str]]:
    s = cfg["strategy"]
    D = lambda x: Decimal(str(x))
    notes: list[str] = []
    out = propose_exits(snap, cfg)          # exits first: protecting money beats spending it
    if in_flatten_window(snap, cfg):
        return out, notes + ["flatten window: no new entries"]

    # ---- entries --------------------------------------------------------------
    for sym in cfg["universe"]:
        q = snap.quotes.get(sym)
        if q is None or q.prev_close <= 0:
            notes.append(f"{sym}: no usable quote")
            continue
        if sym in snap.positions and snap.positions[sym].quantity > 0:
            continue
        drop = (q.prev_close - q.last) / q.prev_close * 100
        if drop < D(s["entry_drop_pct"]):
            notes.append(f"{sym}: {-drop:+.2f}% vs prev close, no entry signal")
            continue
        reason = f"down {drop:.2f}% vs previous close (rule: >= {s['entry_drop_pct']}%)"
        entry = build_entry(sym, snap, cfg, reason)
        (out if isinstance(entry, Candidate) else notes).append(entry)
    return out, notes
