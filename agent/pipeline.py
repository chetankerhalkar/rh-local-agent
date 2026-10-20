"""One trading cycle.

    quotes -> decide (rules or LLM) -> guardrails (code) -> [LLM veto for rule-made entries]
           -> fresh quote + guardrails again -> Robinhood review -> [you] -> place
           -> after a buy fills: protective STOP order placed at Robinhood (if enabled)

Sessions:
    auto     normal cycle (used by `once` and `loop`)
    morning  same as auto: decide and buy, then protect
    evening  cancel the protective stops, sell everything, then check nothing is left

Every step is written to the journal, including the trades that did NOT happen.
"""
from __future__ import annotations

import asyncio
import json
import sys
import uuid
from datetime import timedelta, timezone
from decimal import Decimal

from . import decider, guardrails, strategy
from .config import ROOT, kill_switch_on
from .journal import load_day, log, save_day
from .llm import LocalLLM
from .models import Candidate, DayState, Position, Snapshot
from .rh import Robinhood, dec, extract_alerts, find_dicts_with

PAPER_BOOK = ROOT / "state" / "paper_book.json"
CENT = Decimal("0.01")


# ---------------------------------------------------------------- paper book ---
def _load_paper() -> tuple[dict[str, Position], dict[str, dict]]:
    if not PAPER_BOOK.exists():
        return {}, {}
    raw = json.loads(PAPER_BOOK.read_text(encoding="utf-8"))
    book = {s: Position(s, Decimal(p["quantity"]), Decimal(p["avg_cost"])) for s, p in raw.items()}
    stops = {s: p["stop"] for s, p in raw.items() if p.get("stop")}
    return book, stops


def _save_paper(book: dict[str, Position], stops: dict[str, dict]) -> None:
    PAPER_BOOK.parent.mkdir(exist_ok=True)
    PAPER_BOOK.write_text(json.dumps({s: {"quantity": str(p.quantity), "avg_cost": str(p.avg_cost),
                                          "stop": stops.get(s)} for s, p in book.items() if p.quantity > 0},
                                     indent=2), encoding="utf-8")


async def _paper_stop_hit(rh: Robinhood, sym: str, stop: dict) -> Decimal | None:
    """Did the price touch the paper stop since we bought? Looks at 5-minute lows. Keeps paper P&L honest."""
    try:
        data = await rh.call("get_equity_historicals", {"symbols": [sym], "start_time": stop["since"], "interval": "5minute"})
        bars = find_dicts_with(data, ("low_price",)) or find_dicts_with(data, ("low",))
        lows = [dec(b.get("low_price", b.get("low"))) for b in bars if not b.get("interpolated")]
        lows = [x for x in lows if x is not None]
        if lows and min(lows) <= Decimal(stop["price"]):
            return Decimal(stop["price"])
    except Exception as e:                                   # noqa: BLE001
        log("warning", symbol=sym, why=f"paper stop check unavailable ({type(e).__name__}); using the current price")
    return None


# ------------------------------------------------------------------ helpers ---
async def _snapshot(cfg: dict, rh: Robinhood, paper_book: dict[str, Position]) -> Snapshot:
    quotes = await rh.quotes(cfg["universe"])
    if cfg["mode"] == "paper":
        cash = Decimal(str((cfg.get("paper") or {}).get("buying_power", 500)))
        held = sum((p.quantity * quotes[s].last for s, p in paper_book.items() if s in quotes), Decimal(0))
        return Snapshot(guardrails.now_et(), quotes, dict(paper_book), cash, cash + held)
    positions = await rh.positions()
    bp, eq = await rh.money()
    return Snapshot(guardrails.now_et(), quotes, positions, bp, eq)


def _ask_human(c: Candidate, args: dict, alerts: list) -> bool:
    if not sys.stdin.isatty():
        log("approval_skipped", symbol=c.symbol, why="approve mode needs an open console window")
        return False
    print("\n" + "=" * 62 + f"\nPROPOSED ORDER  {c.side.upper()} {c.symbol}  ({c.kind})\nWhy:    {c.reason}\nOrder:  {args}")
    print(f"Robinhood review alerts: {alerts if alerts else 'none'}\n" + "=" * 62)
    return input("Place this order with real money? type 'yes' to confirm: ").strip().lower() == "yes"


async def _cancel_open(rh: Robinhood, sym: str, why: str) -> None:
    """Shares sitting under an open stop order cannot be sold. Cancel first, then sell."""
    try:
        for o in await rh.open_orders(sym):
            await rh.cancel_order(str(o["id"]))
            log("order_cancelled", symbol=sym, order_id=o["id"], why=why)
    except Exception as e:                                   # noqa: BLE001
        log("cancel_error", symbol=sym, error=f"{type(e).__name__}: {e}"[:300])


class Cycle:
    def __init__(self, cfg: dict, rh: Robinhood, llm: LocalLLM, day: DayState) -> None:
        self.cfg, self.rh, self.llm, self.day = cfg, rh, llm, day
        self.mode = cfg["mode"]
        self.book, self.stops = _load_paper() if self.mode == "paper" else ({}, {})
        self.snap: Snapshot | None = None

    async def refresh(self) -> Snapshot:
        self.snap = await _snapshot(self.cfg, self.rh, self.book)
        return self.snap

    async def execute(self, c: Candidate) -> bool:
        cfg, day = self.cfg, self.day
        bad = guardrails.check(c, self.snap, day, cfg, kill_switch_on())
        if bad:
            log("blocked_by_guardrails", order=c.as_dict(), violations=bad)
            return False

        if c.kind == "entry" and c.source == "rules":   # LLM may veto rule-made entries. Never an exit.
            q = self.snap.quotes[c.symbol]
            facts = {"last": str(q.last), "prev_close": str(q.prev_close), "bid": str(q.bid), "ask": str(q.ask)}
            news = await self.rh.headlines(c.symbol) if cfg["llm"].get("use_news") else []
            verdict = await self.llm.review(c.as_dict(), facts, news)
            log("llm_review", symbol=c.symbol, **verdict, headlines=len(news))
            if verdict["verdict"] != "approve":
                return False
        if c.kind == "entry" or c.source == "llm":      # a local model can take a minute: never trade on an old quote
            await self.refresh()
            bad = guardrails.check(c, self.snap, day, cfg, kill_switch_on())
            if bad:
                log("blocked_after_recheck", order=c.as_dict(), violations=bad)
                return False

        q = self.snap.quotes[c.symbol]
        ref_price = q.ask if c.side == "buy" else q.bid
        trade = {"symbol": c.symbol, "side": c.side, "kind": c.kind, "ref_id": str(uuid.uuid4()), "status": "pending"}

        if self.mode == "paper":
            await self._paper(c, ref_price)
            trade["status"] = "placed"
        else:
            if c.kind == "exit":
                await _cancel_open(self.rh, c.symbol, "free the shares before selling")
            args = c.to_order_args(self.rh.account)
            alerts = extract_alerts(await self.rh.review_order(args))
            log("robinhood_review", order=args, alerts=alerts)
            if self.mode == "auto" and alerts and cfg["guardrails"].get("block_on_any_review_alert", True):
                log("blocked_by_review_alert", symbol=c.symbol, kind=c.kind)
                return False
            if self.mode == "approve" and not _ask_human(c, args, alerts):
                log("not_approved", symbol=c.symbol)
                return False
            day.trades.append(trade)
            save_day(day)                   # write the ref_id BEFORE sending: a crash cannot double-order
            try:
                result = await self.rh.place_order(args, trade["ref_id"])
                trade["status"] = "placed"
                log("order_placed", order=args, ref_id=trade["ref_id"], result=result)
            except Exception as e:          # noqa: BLE001
                trade["status"] = "unknown"    # counts against today's limits until you check it
                log("order_error", order=args, ref_id=trade["ref_id"], error=str(e)[:500])

        if self.mode == "paper":
            day.trades.append(trade)
        if c.side == "buy":
            day.buy_usd_total = str(Decimal(day.buy_usd_total) + c.notional(ref_price))
        save_day(day)
        if c.kind == "entry" and trade["status"] == "placed" and (cfg.get("protect") or {}).get("broker_stop"):
            await self.protect(c)
        return trade["status"] == "placed"

    async def _paper(self, c: Candidate, ref_price: Decimal) -> None:
        if c.kind == "protect":
            self.stops[c.symbol] = {"price": str(c.stop_price), "since": guardrails.now_et().astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
            _save_paper(self.book, self.stops)
            log("paper_stop_set", symbol=c.symbol, stop_price=str(c.stop_price), note="simulation only")
            return
        price = c.limit_price or ref_price
        if c.limit_price is not None:        # a marketable limit fills at the quote or better, not at the limit
            price = min(c.limit_price, ref_price) if c.side == "buy" else max(c.limit_price, ref_price)
        note = "simulation only, no order sent"
        if c.side == "sell" and c.symbol in self.stops:
            hit = await _paper_stop_hit(self.rh, c.symbol, self.stops[c.symbol])
            if hit is not None:
                price, note = hit, "paper stop was touched earlier today: filled at the stop price (a real stop can slip lower)"
        pnl = Decimal(0)
        qty = c.quantity if c.quantity is not None else (c.dollar_amount / price)
        if c.side == "buy":
            old = self.book.get(c.symbol, Position(c.symbol, Decimal(0), Decimal(0)))
            total = old.quantity + qty
            self.book[c.symbol] = Position(c.symbol, total, (old.quantity * old.avg_cost + qty * price) / total)
        else:
            old = self.book.pop(c.symbol, None)
            self.stops.pop(c.symbol, None)
            pnl = (price - old.avg_cost) * old.quantity if old else pnl
        _save_paper(self.book, self.stops)
        self.day.paper_pnl = str(Decimal(self.day.paper_pnl) + pnl)
        log("paper_fill", order=c.as_dict(), assumed_price=str(price), realized_pnl=f"{pnl:.2f}",
            paper_pnl_today=f"{Decimal(self.day.paper_pnl):.2f}", note=note)

    async def protect(self, buy: Candidate) -> None:
        """After a buy: wait for the fill, then park a real stop order at Robinhood.
        That order protects you while the agent sleeps, and even if this PC is off."""
        p = self.cfg.get("protect") or {}
        wait, poll = float(p.get("fill_wait_seconds", 90)), float(p.get("poll_seconds", 5))
        pos = None
        for _ in range(max(1, int(wait / max(poll, 0.01)))):
            await self.refresh()
            pos = self.snap.positions.get(buy.symbol)
            if pos and buy.quantity is not None and pos.quantity >= buy.quantity:
                break
            if self.mode == "paper":
                break
            await asyncio.sleep(poll)
        if self.mode != "paper" and not (pos and buy.quantity is not None and pos.quantity >= buy.quantity):
            await _cancel_open(self.rh, buy.symbol, "buy did not fill in time")     # no late, unprotected fills
            await self.refresh()
            pos = self.snap.positions.get(buy.symbol)
        whole = pos.quantity.to_integral_value(rounding="ROUND_DOWN") if pos else Decimal(0)
        if whole < 1:
            if pos and pos.quantity > 0:
                log("UNPROTECTED_POSITION", symbol=buy.symbol, why="stop orders need whole shares (use order_style: limit_whole_shares)")
            else:
                log("buy_not_filled", symbol=buy.symbol)
            return
        basis = pos.avg_cost if pos.avg_cost > 0 else (buy.limit_price or self.snap.quotes[buy.symbol].last)
        stop = (basis * (1 - Decimal(str(self.cfg["strategy"]["stop_loss_pct"])) / 100)).quantize(CENT)
        c = Candidate(buy.symbol, "sell", "stop_market", f"protective stop {self.cfg['strategy']['stop_loss_pct']}% under cost {basis:.2f}",
                      quantity=whole, stop_price=stop, kind="protect", is_exit=True)
        if not await self.execute(c):
            log("UNPROTECTED_POSITION", symbol=buy.symbol, why="protective stop was not placed. Check the journal lines above.")


async def run_cycle(cfg: dict, rh: Robinhood, llm: LocalLLM, session: str = "auto") -> None:
    now = guardrails.now_et()
    day: DayState = load_day(f"{now:%Y-%m-%d}")
    if kill_switch_on():
        log("cycle_skipped", why="KILL file present")
        return
    if cfg["guardrails"].get("regular_hours_only", True) and not guardrails.market_open(now):
        log("cycle_skipped", why="market closed", now_et=f"{now:%a %H:%M}", session=session)
        return
    no_trade = [str(d) for d in (cfg.get("schedule") or {}).get("no_trade_dates") or []]
    skip_buys = f"{now:%Y-%m-%d}" in no_trade

    cy = Cycle(cfg, rh, llm, day)
    snap = await cy.refresh()
    if rh.missing:
        log("warning", tickers=rh.missing, why="Robinhood does not know these symbols. Fix the universe list in config.yaml.")
    if day.start_equity is None and snap.equity is not None:
        day.start_equity = str(snap.equity)
        save_day(day)
    for p in snap.positions.values():
        if p.avg_cost <= 0:
            log("warning", symbol=p.symbol, why="average cost not found: code-side exits for this position are disabled. Run discover.")

    if session == "evening":
        candidates, notes = strategy.propose_flatten(snap, cfg, "evening sell before the close"), ["evening session: sell everything, no new buys"]
    elif cfg["strategy"].get("name") == "llm_decider" and not skip_buys:
        candidates, notes = await decider.propose(snap, day, cfg, rh, llm)
        save_day(day)                                   # remembers when the LLM last decided
    else:
        candidates, notes = strategy.propose(snap, day, cfg)
    if skip_buys and session != "evening":
        candidates = [c for c in candidates if c.side == "sell"]
        notes.append("today is in schedule.no_trade_dates: no new buys")

    log("cycle", mode=cfg["mode"], session=session, quotes={s: str(q.last) for s, q in snap.quotes.items()},
        positions={s: str(p.quantity) for s, p in snap.positions.items()}, candidates=len(candidates), notes=notes)
    for c in candidates:
        await cy.execute(c)

    if session == "evening" and candidates:
        if cfg["mode"] != "paper":
            await asyncio.sleep(float((cfg.get("schedule") or {}).get("evening_verify_seconds", 20)))
        left = {s: str(p.quantity) for s, p in (await cy.refresh()).positions.items() if s in cfg["universe"] and p.quantity > 0}
        if left:
            log("STILL_HOLDING", positions=left, why="evening sell did not fully fill. These carry overnight with NO stop. Check the Robinhood app now.")
        else:
            log("flat", note="no positions left in the universe")
    log("cycle_done", session=session, llm_calls=llm.calls, llm_tokens=llm.tokens, trades_today=len(day.trades))
