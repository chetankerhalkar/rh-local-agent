"""Two runs a day, you in charge:   python -m agent morning      python -m agent evening

MORNING  code scans stocks and coins -> your local model says which look good -> you pick
         -> each pick: fresh price, guardrails, Robinhood's own review, then YOUR "yes"
         -> buy (never over max_order_usd) -> a stop order at Robinhood protects it, overnight too
EVENING  each holding the agent bought: up or flat -> sell. Down -> you choose: sell, or keep it with its stop
         -> the guardrails first, then cancel the stop, sell, check the fill. No fill -> its stop order goes back.

Nothing runs on a schedule and nothing is bought without you at the keyboard. Every step is journaled.
The agent only sells what it bought itself (state/agent_book.json) and only cancels orders it placed itself.

Robust by construction:
  * every order's ref_id is written to disk BEFORE it is sent. A retry re-sends the same ref_id and
    Robinhood never places one ref_id twice, so a dropped connection can never double an order
  * each step opens a fresh connection and nothing stays connected while you think
  * every run first reconciles with Robinhood, a week back: stops that sold, orders cut off mid-send,
    cancels never confirmed, buys that filled but were never recorded
  * mode: paper never sends and never cancels anything
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from decimal import ROUND_DOWN, Decimal

from . import analyst, config, guardrails, journal, scan
from .journal import load_day, log, save_day
from .laya import make_engine
from .llm import LocalLLM
from .models import Candidate, CryptoPair, DayState, Snapshot
from .rh import (DONE_STATES, OPEN_STATES, RHError, Robinhood, connect, dec, extract_alerts, order_fill, order_id_of,
                 parse_ts, step_down, step_up)
from .strategy import _exit_order, build_entry

CENT = Decimal("0.01")
LIVE_STATUSES = ("placed", "pending", "unknown")
LOOKBACK_DAYS = 7                    # how far back each run looks for orders whose ending was never recorded
NETWORK_ERRORS = {"ConnectError", "ConnectTimeout", "ReadTimeout", "ReadError", "WriteError", "WriteTimeout",
                  "PoolTimeout", "RemoteProtocolError", "NetworkError", "TransportError", "TimeoutException",
                  "HTTPStatusError", "ClosedResourceError", "BrokenResourceError", "EndOfStream", "McpError"}


class Busy(Exception):
    pass


# ------------------------------------------------------------------ console ---
def ask(prompt: str) -> str:
    """Every question to you goes through here (the tests script the answers)."""
    return input(prompt).strip()


def console_ok() -> bool:
    return sys.stdin.isatty()


def say(text: str = "") -> None:
    print(text, flush=True)


def is_crypto(symbol: str) -> bool:
    return symbol.endswith("-USD")          # a stock ticker never ends in -USD, so the two can never clash


def money(x) -> str:
    x = Decimal(x)
    return f"-${-x:,.2f}" if x < 0 else f"${x:,.2f}"


def price(x) -> str:
    """$2,410.55 for most prices, more digits for coins under a dollar ($0.123456)."""
    x = Decimal(x)
    if x >= 1:
        return f"${x:,.2f}"
    whole, _, frac = format(x.quantize(Decimal("0.00000001")).normalize(), "f").partition(".")
    return f"${whole}.{frac.ljust(2, '0')}"


def amount(symbol: str, qty) -> str:
    q = Decimal(qty)
    text = format(q.normalize(), "f")
    return f"{text} {symbol[:-4]}" if is_crypto(symbol) else f"{text} share{'' if q == 1 else 's'}"


def change(pnl: Decimal, cost: Decimal) -> str:
    return f"{money(pnl)} ({pnl / cost * 100:+.2f}%)" if cost else money(pnl)


def clock(t: datetime) -> str:
    return t.astimezone().strftime("%I:%M %p").lstrip("0")      # the PC's own time zone


def table(headers: list[str], rows: list[list[str]]) -> str:
    w = [max(len(str(r[i])) for r in [headers, *rows]) for i in range(len(headers))]
    line = lambda r: "  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)).rstrip()
    return "\n".join([line(headers), line(["-" * x for x in w]), *map(line, rows)])


# ---------------------------------------------------------- book and lock ---
def book_path():
    return config.ROOT / "state" / "agent_book.json"


def load_book() -> dict[str, dict]:
    """What the agent bought and still holds, keyed by symbol ("CHP", "ETH-USD")."""
    p = book_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_book(book: dict[str, dict]) -> None:
    p = book_path()
    p.parent.mkdir(exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(book, indent=2), encoding="utf-8")
    tmp.replace(p)                                   # atomic: a crash never leaves half a file


@contextmanager
def single_session():
    """Two sessions at once could buy twice. The operating system drops this lock if the process dies."""
    p = config.ROOT / "state" / "session.lock"
    p.parent.mkdir(exist_ok=True)
    f = open(p, "a+")
    try:
        f.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        raise Busy("another morning or evening session is already running. Finish it (or close its window) first.")
    try:
        yield
    finally:
        if os.name == "nt":
            import msvcrt
            try:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        f.close()


def _leaves(e: BaseException):
    if isinstance(e, BaseExceptionGroup):
        for sub in e.exceptions:
            yield from _leaves(sub)
    else:
        yield e


def _network(e: BaseException) -> bool:
    """Connection trouble is worth one retry. Robinhood saying no, a config problem or a bug is not."""
    errs = list(_leaves(e))
    if any(isinstance(x, (RHError, config.ConfigError)) for x in errs):
        return False
    return all(isinstance(x, (ConnectionError, TimeoutError, socket.gaierror)) or type(x).__name__ in NETWORK_ERRORS
               for x in errs)


@dataclass
class Idea:
    symbol: str
    confidence: float
    reason: str
    order: Candidate
    today: str | None                 # today's move in percent, for the table
    facts: dict = field(default_factory=dict)   # the data the engine looked at, shown with the idea
    second: dict | None = None                  # your local model's written view, when Laya made the call
    drivers: list = field(default_factory=list)  # which data moved Laya's score, and how far (measured)


# The numbers behind an idea, in words: (key in the facts, label)
FACT_LABELS = (("change_today_pct", "today"), ("change_since_midnight_et_pct", "since midnight ET"),
               ("change_24h_pct", "24 hours"), ("change_5d_pct", "5 days"), ("change_7d_pct", "7 days"),
               ("change_20d_pct", "20 days"), ("vs_20d_average_pct", "vs its 20-day average"),
               ("avg_daily_range_pct", "typical daily swing"), ("range_24h_pct", "24-hour range"),
               ("spread_pct", "buy/sell gap"))


def data_line(f: dict) -> str:
    return ", ".join(f"{label} {f[k]}%" for k, label in FACT_LABELS if f.get(k) not in (None, ""))


# ------------------------------------------------------------------ session ---
class Session:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.d = cfg.get("daily") or {}
        self.live = cfg["mode"] in ("approve", "auto")       # paper = dry run: nothing is sent, nothing is cancelled
        self.day = load_day(f"{guardrails.now_et():%Y-%m-%d}")
        self.days = [self.day] + [d for d in (self._on_disk(n) for n in range(1, LOOKBACK_DAYS + 1)) if d]
        self.book = load_book()
        want = bool(((cfg.get("scan") or {}).get("crypto") or {}).get("enabled", True))
        self.crypto_on = want or any(is_crypto(s) for s in self.book)
        self.crypto_account = str(cfg["robinhood"].get("crypto_account") or "")
        self.pairs: dict[str, CryptoPair] = {}
        self.allowed: set[str] = set(self.book)             # symbols this run may place orders on
        self.pending: dict[str, Candidate] = {}             # ref_id -> order, so a retry re-sends the very same order
        self.realized = Decimal(0)

    @staticmethod
    def _on_disk(n: int) -> DayState | None:
        date = f"{guardrails.now_et() - timedelta(days=n):%Y-%m-%d}"
        return load_day(date) if (config.ROOT / "state" / f"{date}.json").exists() else None

    # ---- the journal of orders ---------------------------------------------------
    def trades(self) -> list[tuple[DayState, dict]]:
        """(day, trade) for every order the daily sessions recorded in the last week, today first."""
        return [(d, t) for d in self.days for t in d.trades if t.get("asset")]

    def find_trade(self, ref_id: str | None) -> tuple[DayState, dict] | None:
        return next(((d, t) for d, t in self.trades() if ref_id and t.get("ref_id") == ref_id), None)

    def known_ids(self) -> set[str]:
        """Every order id and ref_id the agent itself created. Anything else at Robinhood is yours, and left alone."""
        ids = {str(v) for _, t in self.trades() for v in (t.get("order_id"), t.get("ref_id")) if v}
        ids |= {str(v) for e in self.book.values()
                for v in (e.get("stop_order_id"), e.get("stop_ref_id"), e.get("target_order_id"), e.get("target_ref_id"),
                          e.get("sell_ref_id"), e.get("buy_ref_id")) if v}
        return ids

    # ---- two ways to hold -------------------------------------------------------
    # day  (stocks, and crypto with daily.crypto_style: day): a STOP order waits at Robinhood; the evening sells.
    # hold (crypto with daily.crypto_style: hold): a TARGET sell waits at Robinhood and sells by itself when the price
    #      gets there. Robinhood keeps only one sell order per coin, so no stop can sit next to it: every session
    #      checks the stop level instead, and the evening offers to sell once the price is under it.
    @staticmethod
    def holds(e: dict) -> bool:
        return e.get("style") == "hold"

    def parked_id(self, e: dict) -> str | None:
        """The order this holding keeps waiting at Robinhood: its target sell (hold style) or its stop (day style)."""
        return e.get("target_order_id") if self.holds(e) else e.get("stop_order_id")

    def new_entry(self, sym: str, asset: str, opened: str, buy_ref: str, buy_oid: str | None,
                  qty: Decimal, cost: Decimal, avg: Decimal) -> dict:
        hold = asset == "crypto" and str(self.d.get("crypto_style", "day")).lower() == "hold"
        e = {"symbol": sym, "asset": asset, "style": "hold" if hold else "day", "opened": opened,
             "buy_ref_id": buy_ref, "buy_order_id": buy_oid, "quantity": str(qty), "cost": str(cost), "avg_price": str(avg),
             "stop_ref_id": str(uuid.uuid4()), "stop_order_id": None, "stop_price": None, "sell_ref_id": None}
        if hold:
            pct = Decimal(str(self.d.get("crypto_stop_loss_pct", 3.0)))
            pair = self.pairs.get(sym) or CryptoPair(sym, sym[:-4], Decimal("0.00000001"), Decimal("0.01"))
            e.update(target_ref_id=str(uuid.uuid4()), target_order_id=None, target_price=None,
                     stop_level=str(step_down(avg * (1 - pct / 100), pair.price_step)))
        return e

    async def park(self, rh: Robinhood, e: dict) -> bool:
        return await (self.place_target(rh, e) if self.holds(e) else self.protect(rh, e))

    def commit(self, c: Candidate, ref_id: str) -> tuple[DayState, dict]:
        """The ref_id goes to disk BEFORE the order is sent. Buys count against today's budget right away."""
        found = self.find_trade(ref_id)
        if found is None:
            t = {"symbol": c.symbol, "side": c.side, "kind": c.kind, "asset": c.asset, "ref_id": ref_id,
                 "status": "pending", "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            if c.side == "buy":
                t["usd"] = str(c.notional(c.limit_price or Decimal(0)))
                self.day.buy_usd_total = str(Decimal(self.day.buy_usd_total) + Decimal(t["usd"]))
            self.day.trades.append(t)
            save_day(self.day)
            found = (self.day, t)
        self.pending[ref_id] = c
        return found

    @staticmethod
    def rebudget(day: DayState, t: dict, spent: Decimal) -> None:
        """Replace a buy's planned dollars with what it really cost (0 when it never filled)."""
        if t.get("side") == "buy":
            day.buy_usd_total = str(Decimal(day.buy_usd_total) - Decimal(t.get("usd") or 0) + spent)
            t["usd"] = str(spent)
        save_day(day)

    # ---- plumbing ------------------------------------------------------------
    @asynccontextmanager
    async def broker(self):
        async with connect(self.cfg) as rh:
            rh.crypto_account = self.crypto_account
            yield rh

    async def step(self, what: str, fn) -> bool:
        """Run fn(rh) on a fresh connection, with one retry after a network problem. Every order inside
        re-sends a ref_id that is already on disk, so the retry can never place anything twice."""
        for attempt in (1, 2):
            try:
                async with self.broker() as rh:
                    await fn(rh)
                return True
            except Exception as e:                                # noqa: BLE001
                if not _network(e):
                    raise
                err = "; ".join(f"{type(x).__name__}: {x}" for x in _leaves(e))[:300]
                log("network_problem", step=what, attempt=attempt, error=err)
                if attempt == 2:
                    say(f"Lost the connection to Robinhood while {what}.\n"
                        "Run the same command again: it checks with Robinhood first and never sends an order twice.")
                    return False
                say(f"Connection problem while {what}. Trying once more...")
                await asyncio.sleep(2)
        return False

    def gcfg(self) -> dict:
        return {**self.cfg, "universe": sorted(self.allowed)}

    def checks(self, c: Candidate, snap: Snapshot, ref_id: str | None = None) -> list[str]:
        """The guardrails. On a retry the order's own earlier attempt (same ref_id) must not count against it."""
        day = replace(self.day, trades=[t for t in self.day.trades if not ref_id or t.get("ref_id") != ref_id])
        return guardrails.check(c, snap, day, self.gcfg(), config.kill_switch_on())

    async def setup(self, rh: Robinhood) -> bool:
        if not rh.account:
            say("No agentic account in config.yaml yet. Run  python -m agent discover  and copy the account number in.")
            return False
        if not await rh.agentic_account_ok():
            say("The account in config.yaml is not agentic_allowed at Robinhood. Nothing was done.")
            return False
        if self.crypto_on:
            try:
                self.crypto_account = rh.crypto_account = await rh.find_crypto_account()
                self.pairs = await rh.crypto_pairs() if self.crypto_account else {}
            except RHError as e:
                log("warning", why=f"crypto unavailable: {e}"[:300])
            if not self.crypto_account or not self.pairs:
                self.crypto_on = False
                why = "Robinhood's crypto list did not load" if self.crypto_account else \
                      "no crypto account number (rhs_account_number) was found for your agentic account" + await self._onboarding(rh)
                say(f"Crypto: skipped today, {why}.")
                if any(is_crypto(s) for s in self.book):
                    say("         The crypto the agent holds is left as it is, with its stop orders.")
        return True

    async def _onboarding(self, rh: Robinhood) -> str:
        try:
            data = await rh.call("get_crypto_account_onboarding_info", {})
            url = next((v for v in json.dumps(data).split('"') if v.startswith("https://")), "")
            return f". Open one here: {url}" if url else ""
        except Exception:                                         # noqa: BLE001  just a hint
            return ""

    async def snapshot(self, rh: Robinhood, symbols: list[str]) -> Snapshot:
        stocks = [s for s in symbols if not is_crypto(s)]
        coins = [s for s in symbols if is_crypto(s)] if self.crypto_on else []
        quotes = await rh.quotes(stocks) if stocks else {}
        if coins:
            quotes.update(await rh.crypto_quotes(coins))
            for s in coins:
                if s in quotes and not (s in self.pairs and self.pairs[s].tradable):
                    quotes[s].state = "halted"
        positions = await rh.positions()
        if self.crypto_on:
            positions.update(await rh.crypto_positions())
        bp, eq = await rh.money()
        return Snapshot(guardrails.now_et(), quotes, positions, bp, eq)

    # ---- orders ----------------------------------------------------------------
    async def order(self, rh: Robinhood, asset: str, order_id: str | None) -> dict | None:
        if not order_id:
            return None
        return await (rh.crypto_order(order_id) if asset == "crypto" else rh.equity_order(order_id))

    async def find(self, rh: Robinhood, t: dict) -> tuple[dict | None, bool]:
        """The order behind a journal trade: by its order id, else by its ref_id. Returns (order, sure).
        When nothing is found, sure says whether Robinhood's answer proves the order does not exist."""
        if t.get("order_id"):
            o = await self.order(rh, t["asset"], t["order_id"])
            if o:
                return o, True
        sent = parse_ts(t.get("ts")) or datetime.now(timezone.utc)
        since = (sent - timedelta(days=1)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            orders, complete = await rh.recent_orders(t["asset"], since)
        except RHError as e:
            log("warning", why=f"could not list {t['asset']} orders: {e}"[:300])
            return None, False
        o = next((o for o in orders if str(o.get("ref_id")) == t["ref_id"]), None)
        return o, o is not None or (complete and (not orders or any(x.get("ref_id") for x in orders)))

    async def review(self, rh: Robinhood, c: Candidate) -> tuple[dict, list]:
        if c.asset == "crypto":
            args = c.to_crypto_args(rh.crypto_account)
            return args, extract_alerts(await rh.review_crypto(args))
        args = c.to_order_args(rh.account)
        return args, extract_alerts(await rh.review_order(args))

    async def place(self, rh: Robinhood, c: Candidate, ref_id: str) -> str | None:
        """Send one order. Returns its order id, or None when Robinhood rejected it (already reported)."""
        day, t = self.commit(c, ref_id)
        try:
            if c.asset == "crypto":
                result = await rh.place_crypto(c.to_crypto_args(rh.crypto_account), ref_id)
            else:
                result = await rh.place_order(c.to_order_args(rh.account), ref_id)
            oid = order_id_of(result)
        except RHError as e:
            found, sure = await self.find(rh, t)       # a re-sent ref_id may be refused although the order exists
            if found is None and not sure:             # an error, and no way to tell whether the order exists
                t["status"] = "unknown"                # never written off: the next run asks Robinhood again
                save_day(day)
                log("order_unconfirmed", symbol=c.symbol, kind=c.kind, ref_id=ref_id, error=str(e)[:500])
                say(f"Robinhood answered the {c.kind} order for {c.symbol} with an error ({str(e)[:150]}), and I could not "
                    "confirm whether it went through. Check the Robinhood app. The next run checks it.")
                return None
            if found is None:
                t["status"] = "rejected"
                self.rebudget(day, t, Decimal(0))
                log("order_rejected", symbol=c.symbol, kind=c.kind, ref_id=ref_id, error=str(e)[:500])
                say(f"Robinhood rejected the {c.kind} order for {c.symbol}: {str(e)[:200]}")
                return None
            result, oid = found, str(found["id"])
            log("order_found_after_error", symbol=c.symbol, ref_id=ref_id, error=str(e)[:300])
        if not oid:
            found, _ = await self.find(rh, t)
            oid = str(found["id"]) if found else None
        t["status"], t["order_id"] = "placed", oid or t.get("order_id")
        save_day(day)
        log("order_placed", symbol=c.symbol, kind=c.kind, asset=c.asset, ref_id=ref_id, order=c.as_dict(), result=result)
        if not t["order_id"]:
            say(f"WARNING: Robinhood sent no order id for {c.symbol}. Check the order in the Robinhood app.")
        return t["order_id"]

    async def wait_fill(self, rh: Robinhood, asset: str, order_id: str) -> tuple[str, Decimal, Decimal | None]:
        p = self.cfg.get("protect") or {}
        end, poll = time.monotonic() + float(p.get("fill_wait_seconds", 90)), float(p.get("poll_seconds", 5))
        while True:
            state, qty, avg = order_fill(await self.order(rh, asset, order_id))
            if state in DONE_STATES or time.monotonic() >= end:
                return state, qty, avg
            await asyncio.sleep(poll)

    async def cancel(self, rh: Robinhood, asset: str, order_id: str, why: str) -> bool:
        """Cancel, then confirm with Robinhood. True once the order is no longer open (cancelled, or it filled
        first). False means it may still be live: the caller must not treat it as gone."""
        try:
            await (rh.cancel_crypto(order_id) if asset == "crypto" else rh.cancel_order(order_id))
            log("order_cancelled", asset=asset, order_id=order_id, why=why)
        except RHError as e:                                      # already filled or cancelled? the check decides
            log("cancel_error", asset=asset, order_id=order_id, error=str(e)[:300])
        poll = min(float((self.cfg.get("protect") or {}).get("poll_seconds", 5)), 2.0)
        for _ in range(8):
            state, qty, _ = order_fill(await self.order(rh, asset, order_id))
            if state in DONE_STATES:
                if state != "filled":
                    self.mark_cancelled(order_id, filled_some=bool(qty))
                return True
            await asyncio.sleep(poll)
        log("cancel_unconfirmed", asset=asset, order_id=order_id, why=why)
        return False

    def mark_cancelled(self, order_id: str, filled_some: bool = False) -> None:
        """The journal says so too, or a new order on the same holding today would look like a second one.
        A buy that filled part of the way stays counted; a sell or stop that did is simply finished."""
        for d, t in self.trades():
            if t.get("order_id") == order_id and t.get("status") in LIVE_STATUSES:
                if not filled_some:
                    t["status"] = "cancelled"
                elif t.get("side") != "buy":
                    t["status"] = "done"
                save_day(d)

    # ---- protection ------------------------------------------------------------
    async def protect(self, rh: Robinhood, e: dict) -> bool:
        """Park a stop order at Robinhood under what was paid. It stays until cancelled (gtc), so it protects the
        holding while this PC is off and overnight. If the price already fell past that level, the stop goes
        the same distance under today's price instead: a stop at or over the bid would sell at once."""
        if not (self.cfg.get("protect") or {}).get("broker_stop", True):
            return False
        sym, crypto = e["symbol"], e["asset"] == "crypto"
        pct = Decimal(str(self.d.get("crypto_stop_loss_pct" if crypto else "stock_stop_loss_pct", 3.0 if crypto else 1.0)))
        pair = self.pairs.get(sym) or CryptoPair(sym, sym[:-4], Decimal("0.00000001"), Decimal("0.01"))
        down = (lambda x: step_down(x, pair.price_step)) if crypto else (lambda x: x.quantize(CENT, rounding=ROUND_DOWN))
        usual = down(Decimal(e["avg_price"]) * (1 - pct / 100))
        snap = await self.snapshot(rh, [sym])
        c = self.pending.get(e["stop_ref_id"])                    # a retry sends the very same order
        if c is None:
            q, qty = snap.quotes.get(sym), Decimal(e["quantity"])
            stop = usual if not (q and q.bid > 0 and usual >= q.bid) else down(q.bid * (1 - pct / 100))
            qty = step_down(qty, pair.qty_step) if crypto else qty.to_integral_value(rounding=ROUND_DOWN)
            c = Candidate(sym, "sell", "stop_market", f"protective stop {pct}% under {'cost' if stop == usual else 'the price'}",
                          quantity=qty, stop_price=stop, kind="protect", is_exit=True, asset=e["asset"], time_in_force="gtc")
        bad = ["too small for a stop order"] if c.quantity <= 0 else self.checks(c, snap, e["stop_ref_id"])
        if bad:
            log("UNPROTECTED_POSITION", symbol=sym, why=bad)
            say(f"WARNING: no stop order for {sym} ({'; '.join(bad)}). Watch it in the Robinhood app.")
            return False
        oid = await self.place(rh, c, e["stop_ref_id"])
        if not oid:
            log("UNPROTECTED_POSITION", symbol=sym, why="stop order rejected")
            say(f"WARNING: {sym} has no stop order. Watch it in the Robinhood app.")
            return False
        e["stop_order_id"], e["stop_price"] = oid, str(c.stop_price)
        save_book(self.book)
        log("stop_placed", symbol=sym, stop_price=str(c.stop_price), quantity=str(c.quantity))
        where = f"{pct}% under what you paid" if c.stop_price == usual else \
                f"{pct}% under today's price: it had already fallen past {price(usual)}"
        say(f"Stop order placed: Robinhood sells {sym} if it drops to {price(c.stop_price)} ({where}).")
        return True

    async def place_target(self, rh: Robinhood, e: dict) -> bool:
        """Hold style: park a limit SELL at Robinhood at the target price. It waits there (good till cancelled; crypto
        limit orders last 90 days) and sells by itself once the price gets there."""
        sym = e["symbol"]
        pct = Decimal(str(self.d.get("crypto_take_profit_pct", 5.0)))
        pair = self.pairs.get(sym) or CryptoPair(sym, sym[:-4], Decimal("0.00000001"), Decimal("0.01"))
        snap = await self.snapshot(rh, [sym])
        c = self.pending.get(e["target_ref_id"])                  # a retry sends the very same order
        if c is None:
            limit = step_up(Decimal(e["avg_price"]) * (1 + pct / 100), pair.price_step)
            q = snap.quotes.get(sym)
            if q and q.bid > limit:                               # already past the target: sell at today's better price
                limit = step_down(q.bid * (1 - Decimal(str(self.d.get("sell_slack_pct", 0.10))) / 100), pair.price_step)
            c = Candidate(sym, "sell", "limit", f"target {pct}% over cost", quantity=step_down(Decimal(e["quantity"]), pair.qty_step),
                          limit_price=limit, kind="target", is_exit=True, asset="crypto")
        bad = ["too small for a sell order"] if c.quantity <= 0 else self.checks(c, snap, e["target_ref_id"])
        if bad:
            log("NO_TARGET_ORDER", symbol=sym, why=bad)
            say(f"WARNING: no target sell order for {sym} ({'; '.join(bad)}), so it will not sell by itself. "
                "Watch it in the Robinhood app.")
            return False
        oid = await self.place(rh, c, e["target_ref_id"])
        if not oid:
            log("NO_TARGET_ORDER", symbol=sym, why="target sell rejected")
            say(f"WARNING: {sym} has no target sell order, so it will not sell by itself. Watch it in the Robinhood app.")
            return False
        e["target_order_id"], e["target_price"] = oid, str(c.limit_price)
        save_book(self.book)
        log("target_placed", symbol=sym, limit_price=str(c.limit_price), quantity=str(c.quantity), stop_level=e.get("stop_level"))
        say(f"Target sell order placed: Robinhood sells {sym} by itself at {price(c.limit_price)} ({pct}% over what you paid).\n"
            f"         Robinhood allows one sell order per coin, so there is no stop order next to it. Each time you run\n"
            f"         the agent it checks the stop level, {price(e['stop_level'])}, and the evening offers to sell under it.")
        return True

    async def open_on(self, rh: Robinhood, e: dict) -> list[dict]:
        """The agent's own open SELL orders on this holding: its stop or target, or a sell left from a cut-off run.
        Only orders whose id or ref_id the agent recorded. Never an order you placed yourself."""
        sym, asset = e["symbol"], e["asset"]
        mine = self.known_ids()
        rows = await (rh.open_crypto_orders(sym) if asset == "crypto" else rh.open_orders(sym, any_day=True))
        rows = [o for o in rows if str(o.get("side", "sell")) == "sell" and (str(o["id"]) in mine or str(o.get("ref_id")) in mine)]
        for key in ("stop_order_id", "target_order_id"):
            if e.get(key) and all(str(o["id"]) != e[key] for o in rows):
                o = await self.order(rh, asset, e[key])
                if o and str(o.get("state")) in OPEN_STATES:
                    rows.append(o)
        return rows

    async def protect_missing(self, rh: Robinhood, only: set[str] | None = None) -> None:
        """Every holding keeps its order waiting at Robinhood: a stop (day style) or a target sell (hold style)."""
        for sym, e in list(self.book.items()):
            if self.parked_id(e) or (only is not None and sym not in only) or (e["asset"] == "crypto" and not self.crypto_on):
                continue
            if not self.holds(e) and not (self.cfg.get("protect") or {}).get("broker_stop", True):
                continue
            what = "target sell order" if self.holds(e) else "stop order"
            if not self.live:
                say(f"{sym} has no {what} (dry run: not placing one).")
            elif ask(f"{sym} has no {what} at Robinhood. Place one now? Type yes: ").lower() == "yes":
                e["target_ref_id" if self.holds(e) else "stop_ref_id"] = str(uuid.uuid4())   # a new order needs a new ref_id
                save_book(self.book)
                await self.park(rh, e)

    def warn_if_unprotected(self, e: dict) -> None:
        if e["symbol"] in self.book and not self.parked_id(e):
            what = "target sell order" if self.holds(e) else "stop order"
            log("UNPROTECTED_POSITION", symbol=e["symbol"], why=f"a step was cut short: no {what}")
            say(f"WARNING: {e['symbol']} may have no {what} right now. Run the session again soon, or check the Robinhood app.")

    # ---- reconcile: the book against Robinhood, before anything new ------------
    async def reconcile(self, rh: Robinhood) -> None:
        await self._settle_trades(rh)
        positions = (await self.snapshot(rh, [])).positions     # read AFTER settling: a buy may have filled just now
        parked = (("stop", "stop_order_id", "stop_price"), ("target", "target_order_id", "target_price"))
        for sym, e in list(self.book.items()):
            if e["asset"] == "crypto" and not self.crypto_on:
                continue                                          # cannot see crypto right now: leave it alone
            fills = {key: order_fill(await self.order(rh, e["asset"], e.get(key))) for _, key, _ in parked}
            sold = next(((how, key, at) for how, key, at in parked if fills[key][0] == "filled" and fills[key][1]), None)
            if sold:                                  # its stop or its target sold it. Shares still there are not the agent's
                how, key, at = sold
                self._close(e, fills[key][1], fills[key][2] or Decimal(e.get(at) or e["avg_price"]), how)
                continue
            held = positions[sym].quantity if sym in positions else Decimal(0)
            if held <= 0:
                await self._gone(rh, e)
                continue
            mine = Decimal(e["quantity"])
            if held < mine:
                e["cost"], e["quantity"] = str((Decimal(e["cost"]) * held / mine).quantize(CENT)), str(held)
                log("book_adjusted", symbol=sym, quantity=str(held))
                say(f"{sym}: Robinhood shows {amount(sym, held)}, less than the agent bought. Using that.")
            for how, key, _ in parked:
                if e.get(key) and fills[key][0] in DONE_STATES:  # cancelled in the app, or expired
                    log(f"{how}_gone", symbol=sym, state=fills[key][0])
                    self.mark_cancelled(e[key])
                    e[key] = None
            working = await self.open_on(rh, e)
            if not self.holds(e) and not e.get("stop_order_id"):  # a stop sent just before a crash: adopt it
                stop = next((o for o in working if dec(o.get("stop_price")) is not None), None)
                if stop:
                    e["stop_order_id"], e["stop_price"] = str(stop["id"]), str(dec(stop["stop_price"]))
            if self.holds(e) and not e.get("target_order_id"):   # the same for a target sell
                t = self.find_trade(e.get("target_ref_id"))
                ids = {str(e.get("target_ref_id")), str((t[1] if t else {}).get("order_id"))}
                target = next((o for o in working if str(o.get("ref_id")) in ids or str(o["id"]) in ids), None)
                if target:
                    lp = dec(target.get("limit_price"))
                    e["target_order_id"], e["target_price"] = str(target["id"]), (str(lp) if lp is not None else None)
            for o in working:                                     # a sell left open by an earlier run: it could still
                if str(o["id"]) in (e.get("stop_order_id"), e.get("target_order_id")) or dec(o.get("stop_price")) is not None:
                    continue                                      # sell what you keep
                if not self.live:
                    say(f"{sym}: a sell order from an earlier run is still open at Robinhood (dry run: left alone).")
                elif await self.cancel(rh, e["asset"], str(o["id"]), "a sell left open by an earlier run"):
                    say(f"{sym}: a sell order from an earlier run was still open. Cancelled it; tonight's plan decides again.")
                else:
                    say(f"WARNING: {sym} has a sell order from an earlier run that would not cancel. Check the Robinhood app.")
        save_book(self.book)

    async def _settle_trades(self, rh: Robinhood) -> None:
        """Orders from the last week whose ending was never recorded: sent while the connection dropped, a cancel
        never confirmed, a buy that filled but never reached the book. Robinhood is asked about each one."""
        for day, t in self.trades():
            sym, asset = t["symbol"], t["asset"]
            if asset == "crypto" and not self.crypto_on:
                continue
            open_buy = t.get("side") == "buy" and not t.get("booked") and t.get("status") in LIVE_STATUSES
            if not open_buy and t.get("status") not in ("pending", "unknown"):
                continue
            o, sure = await self.find(rh, t)
            if o is None:
                if sure:
                    t["status"] = "not_sent"
                    self.rebudget(day, t, Decimal(0))
                    log("order_not_sent", symbol=sym, ref_id=t["ref_id"], day=day.date)
                else:
                    say(f"Could not confirm a {t['kind']} order for {sym} from {day.date}. Check the Robinhood app.")
                continue
            t["order_id"] = str(o["id"])
            state, qty, avg = order_fill(o)
            log("order_found", symbol=sym, ref_id=t["ref_id"], state=state, day=day.date)
            if t.get("side") != "buy":
                t["status"] = "cancelled" if state in DONE_STATES and not qty else "placed"
                save_day(day)
                continue
            if state in OPEN_STATES:                              # an unfinished buy from a run that was cut short
                if not self.live:
                    say(f"A {sym} buy from an earlier run is still open at Robinhood. Dry run: left alone. "
                        "Cancel it in the app, or run in approve mode.")
                    continue
                if not await self.cancel(rh, asset, t["order_id"], "unfinished buy from a run that was cut short"):
                    t["status"] = "unknown"
                    save_day(day)
                    say(f"WARNING: a {sym} buy from an earlier run is still open and would not cancel. Check the Robinhood app.")
                    continue
                state, qty, avg = order_fill(await self.order(rh, asset, t["order_id"]))
            if qty and qty > 0:                                   # holdings read fresh: it may have filled just now
                self._adopt(day, t, (await self.snapshot(rh, [])).positions, qty, avg)
            else:
                t["status"] = "cancelled"
                self.rebudget(day, t, Decimal(0))
                log("buy_never_filled", symbol=sym, ref_id=t["ref_id"])

    def _adopt(self, day: DayState, t: dict, positions: dict, filled: Decimal, avg: Decimal | None) -> None:
        """A buy that filled but never reached the book. Only what THAT order bought is adopted."""
        sym = t["symbol"]
        avg = avg or (Decimal(t.get("usd") or 0) / filled)
        t["status"], t["booked"] = "placed", True
        self.rebudget(day, t, (filled * avg).quantize(CENT))
        held = positions[sym].quantity if sym in positions else Decimal(0)
        qty = min(filled, held)
        if qty <= 0 or sym in self.book:
            why = "it is no longer in the account" if qty <= 0 else "the agent already tracks that symbol"
            log("filled_not_adopted", symbol=sym, why=why)
            say(f"{sym}: a buy from an earlier run had filled ({amount(sym, filled)}), but {why}. Check the Robinhood app.")
            return
        cost = (qty * avg).quantize(CENT)
        self.book[sym] = self.new_entry(sym, t["asset"], day.date, t["ref_id"], t["order_id"], qty, cost, avg)
        self.allowed.add(sym)
        save_book(self.book)
        log("adopted", symbol=sym, quantity=str(qty), cost=str(cost))
        say(f"{sym}: a buy from a run that was cut short had filled ({amount(sym, qty)}). It is back on the agent's list.")

    async def _gone(self, rh: Robinhood, e: dict) -> None:
        """The holding is no longer in the account: an evening sell whose ending was never recorded, or sold in the app."""
        sym = e["symbol"]
        found = self.find_trade(e.get("sell_ref_id"))
        if found:
            state, qty, avg = order_fill((await self.find(rh, found[1]))[0])
            if qty and avg:
                self._close(e, qty, avg, "sell")
                return
        log("holding_gone", symbol=sym)
        say(f"{sym}: no longer in the account (sold in the app?). Removed from the agent's list.")
        del self.book[sym]
        save_book(self.book)

    def _close(self, e: dict, qty: Decimal, avg: Decimal, how: str) -> None:
        sym, held, cost = e["symbol"], Decimal(e["quantity"]), Decimal(e["cost"])
        part = cost * min(qty, held) / held if held else cost
        pnl = (min(qty, held) * avg - part).quantize(CENT)
        self.realized += pnl
        log(f"closed_by_{how}", symbol=sym, quantity=str(qty), price=str(avg), pnl=str(pnl))
        who = {"stop": "its stop order", "target": "its target sell order"}.get(how, "an earlier sell")
        say(f"{sym}: {who} sold it at {price(avg)}, {change(pnl, part)}.")
        del self.book[sym]
        save_book(self.book)

    # ---- morning ---------------------------------------------------------------
    async def morning(self) -> None:
        now = guardrails.now_et()
        say(f"\nMORNING SESSION  {now:%a %Y-%m-%d} {clock(now)} your time   mode: {self.cfg['mode']}"
            + ("" if self.live else "   (DRY RUN: nothing will be sent)"))
        if config.kill_switch_on():
            say("The KILL file is there, so no orders today. Delete it to trade.")
            return
        laya = str(self.cfg.get("engine", "llm")).lower() == "laya"
        if not laya and not self.cfg["llm"].get("enabled", True):
            say("The morning session needs your local model: set llm.enabled: true in config.yaml.")
            return
        ctx: dict = {}
        if not await self.step("looking at the market", lambda rh: self._look(rh, ctx)) or not ctx.get("facts"):
            return

        who = "Laya" if laya else "your local model"
        say(f"\nAsking {who} about {len(ctx['facts'])} stocks and coins" + ("..." if laya else " (a local model can take a few minutes)..."))
        llm = make_engine(self.cfg)
        raw = await analyst.analyze(llm, ctx["facts"], int(self.d.get("analysis_batch", 4)))
        if not raw:
            if llm.last_finish == "length" or llm.last_thinking:
                why = ("its answers were cut off, because it thinks before answering and runs out of room. "
                       "In config.yaml under llm: set  extra_body: {reasoning_effort: none}  (Ollama)")
            elif laya:
                why = "is laya-serve running? Try  python -m agent doctor"
            else:
                why = "is Ollama running? Try  python -m agent doctor"
            say(f"The local model gave no usable answer: {why}. Nothing bought.")
            return
        ideas, dropped = self.rank(raw, ctx["snap"])
        await self.explain(ideas, ctx["facts"], laya)
        log("ideas", shown=[i.symbol for i in ideas], dropped=dropped, llm_calls=llm.calls, llm_tokens=llm.tokens,
            why={i.symbol: {"engine": i.reason, "drivers": i.drivers, "second_opinion": i.second, "facts": i.facts}
                 for i in ideas})
        if not ideas:
            say("No buy ideas today." + (" The model liked some, but a safety limit said no:\n  " + "\n  ".join(dropped)
                                          if dropped else " The model did not like anything enough. That is a fine answer."))
            if ctx.get("crypto_note"):
                say(f"Crypto: no coins today. {ctx['crypto_note']}.")
            return
        self.show_ideas(ideas, ctx["snap"], ctx.get("crypto_note", ""))
        if ctx.get("news_note"):
            say(f"News: {ctx['news_note']}.")
        for idea in self.choose(ideas, ctx["room"]):
            await self.buy(idea)
        say(f"\nDone. Run  python -m agent evening  before {clock(now.replace(hour=15, minute=50))} your time "
            "to sell (crypto can be sold any time).")

    async def _look(self, rh: Robinhood, ctx: dict) -> None:
        ctx.clear()
        if not await self.setup(rh):
            return
        await self.reconcile(rh)
        snap = await self.snapshot(rh, list(self.book))
        if self.day.start_equity is None and snap.equity is not None:
            self.day.start_equity = str(snap.equity)
            save_day(self.day)
        self.show_holdings(snap)
        await self.protect_missing(rh)
        room, why = self.room(snap)
        if room <= 0:
            say(f"No room for new buys today ({why}).")
            if self.book:
                say("Your cash is in what the agent holds. To sell it and free the cash, run  python -m agent evening\n"
                    "(despite the name it works any time the stock market is open, and any time for crypto).")
            return
        stocks: list[str] = []
        ok, why_not = self.stock_buys_open(snap.now)
        if ok:
            stocks = [s for s in await scan.scan_stocks(rh, self.cfg) if s not in snap.positions]
        else:
            say(f"Stocks: skipped, {why_not}.")
        coins: list[str] = []
        crypto_note = ""
        if self.crypto_on:
            try:
                found, _, _, crypto_note = await scan.scan_crypto(rh, self.cfg, self.pairs)
                coins = [c for c in found if c not in snap.positions]
                if crypto_note:
                    say(f"Crypto: nothing to look at. {crypto_note}.")
            except RHError as e:
                log("scan_failed", asset="crypto", error=str(e)[:300])
                crypto_note = f"the crypto scan failed ({str(e)[:150]})"
                say(f"Crypto: {crypto_note}, skipped.")
        if not stocks and not coins:
            say("Nothing to look at right now.")
            return
        self.allowed |= set(stocks) | set(coins)
        snap = await self.snapshot(rh, stocks + coins)
        say(f"Scanned: {len(stocks)} stocks and {len(coins)} coins passed the rules. Reading their prices and news...")
        cs = (self.cfg.get("scan") or {}).get("crypto") or {}
        use_news = bool(self.cfg["llm"].get("use_news", True))
        facts = await analyst.stock_facts(rh, {s: snap.quotes[s] for s in stocks if s in snap.quotes}, snap.now, use_news)
        news_note = ""
        if facts and use_news and not any(f.get("headlines") for f in facts):     # say so: never hide missing news
            news_note = ("no news came back from Robinhood for any of these stocks"
                         + (f" ({rh.news_problem})" if rh.news_problem else "")
                         + ", so they were judged on prices alone. Run  python -m agent discover  to see why")
            log("news_missing", stocks=len(facts), problem=rh.news_problem)
            say(f"News: {news_note}.")
        coin_plan = "sell this evening"                  # the model judges each coin by how it will really be held
        if str(self.d.get("crypto_style", "day")).lower() == "hold":
            coin_plan = (f"hold until {self.d.get('crypto_take_profit_pct', 5.0)}% up, "
                         f"sell if {self.d.get('crypto_stop_loss_pct', 3.0)}% down first")
        facts += await analyst.crypto_facts({c: snap.quotes[c] for c in coins if c in snap.quotes},
                                            str(cs.get("history", "coinbase")), coin_plan)
        ctx.update(snap=snap, room=room, facts=facts, crypto_note=crypto_note, news_note=news_note)

    def room(self, snap: Snapshot) -> tuple[int, str]:
        """How many more buys today's limits allow. The guardrails still check every order on its own."""
        g = self.cfg["guardrails"]
        per = Decimal(str(self.cfg["strategy"]["order_usd"]))
        left = Decimal(str(g["max_daily_buy_usd"])) - Decimal(self.day.buy_usd_total)
        buys = len([t for t in self.day.trades if t.get("side") == "buy" and t.get("status") in LIVE_STATUSES])
        limits = {"open holdings left": int(g["max_open_positions"]) - len([p for p in snap.positions.values() if p.quantity > 0]),
                  "buys left today": int(g["max_trades_per_day"]) - buys,
                  "budget left for": int(left // per) if per > 0 else 0}
        if snap.buying_power is not None and per > 0:          # the cash in the account is a limit too
            limits["buying power left for"] = int(snap.buying_power // per)
        return max(0, min(limits.values())), ", ".join(f"{k} {max(0, v)}" for k, v in limits.items())

    def stock_buys_open(self, now: datetime) -> tuple[bool, str]:
        g = self.cfg["guardrails"]
        start = now.replace(hour=9, minute=30, second=0, microsecond=0) + timedelta(minutes=g["no_new_buys_first_minutes"])
        end = now.replace(hour=16, minute=0, second=0, microsecond=0) - timedelta(minutes=g["no_new_buys_last_minutes"])
        window = f"stock buys run {clock(start)} to {clock(end)} your time on weekdays"
        if f"{now:%Y-%m-%d}" in [str(x) for x in (self.cfg.get("schedule") or {}).get("no_trade_dates") or []]:
            return False, "today is a no-trade date in config.yaml (a half day)"
        if not guardrails.market_open(now):
            return False, f"the stock market is closed ({window})"
        if guardrails.minutes_since_open(now) < g["no_new_buys_first_minutes"] or \
                guardrails.minutes_to_close(now) < g["no_new_buys_last_minutes"]:
            return False, f"outside the buy window ({window})"
        return True, ""

    def entry(self, sym: str, snap: Snapshot, reason: str) -> Candidate | str:
        """Size and price one buy. Code decides both; the model only named the symbol."""
        if sym in self.book or (sym in snap.positions and snap.positions[sym].quantity > 0):
            return "already held"
        if not is_crypto(sym):
            return build_entry(sym, snap, self.cfg, reason, source="llm")
        pair, q = self.pairs.get(sym), snap.quotes[sym]
        if not pair:
            return "Robinhood does not list this pair"
        g, s = self.cfg["guardrails"], self.cfg["strategy"]
        usd = min(Decimal(str(s["order_usd"])), Decimal(str(g["max_order_usd"]))) - CENT   # a fee may round up one cent
        return Candidate(sym, "buy", "limit", reason, dollar_amount=usd, limit_price=step_up(q.ask, pair.price_step),
                         source="llm", asset="crypto")

    def rank(self, raw: dict[str, dict], snap: Snapshot) -> tuple[list[Idea], list[str]]:
        floor = float(self.d.get("min_confidence", 0.6))
        wanted = sorted(((s, x) for s, x in raw.items() if x["action"] == "buy" and x["confidence"] >= floor),
                        key=lambda t: -t[1]["confidence"])
        ideas: list[Idea] = []
        dropped: list[str] = []
        for sym, x in wanted:
            if sym not in snap.quotes:
                continue
            c = self.entry(sym, snap, f"model {x['confidence']:.2f}: {x['reason']}")
            bad = [c] if isinstance(c, str) else self.checks(c, snap)
            if bad:
                dropped.append(f"{sym}: {'; '.join(bad)}")
                continue
            q = snap.quotes[sym]
            ideas.append(Idea(sym, x["confidence"], x["reason"], c, analyst.pct(q.last, q.prev_close)))
            if len(ideas) >= int(self.d.get("max_suggestions", 5)):
                break
        return ideas, dropped

    async def explain(self, ideas: list[Idea], facts: list[dict], laya: bool) -> None:
        """Every idea carries the data behind it. Laya gives a score, not a reason, so your local model reads
        the same data and says buy or skip in words: a second opinion you see before you decide.
        It never adds, removes or reorders an idea."""
        by_sym = {f["symbol"]: f for f in facts}
        for i in ideas:
            i.facts = by_sym.get(i.symbol, {})
        if ideas and laya:                                        # measure what drove each Laya score
            say("Measuring what drove Laya's scores...")
            eng = make_engine(self.cfg)
            for i in ideas:
                try:
                    i.drivers = await eng.explain(i.facts, base=i.confidence)
                except Exception as e:                            # noqa: BLE001  only an explanation: never blocks
                    log("explain_failed", symbol=i.symbol, error=f"{type(e).__name__}: {e}"[:200])
        if not (ideas and laya and self.cfg["llm"].get("enabled", True)):
            return
        say("Asking your local model for a written second opinion on these...")
        try:
            views = await analyst.analyze(LocalLLM(self.cfg), [i.facts for i in ideas if i.facts],
                                          int(self.d.get("analysis_batch", 4)))
        except Exception as e:                                    # noqa: BLE001  only an opinion: never blocks
            log("second_opinion_failed", error=f"{type(e).__name__}: {e}"[:200])
            views = {}
        if not views:
            say("  (your local model did not answer, so no second opinion this time: is Ollama running?)")
        for i in ideas:
            i.second = views.get(i.symbol)

    def show_ideas(self, ideas: list[Idea], snap: Snapshot, crypto_note: str = "") -> None:
        rows = []
        for i, idea in enumerate(ideas, 1):
            c = idea.order
            size = (f"{money(c.dollar_amount)} of {idea.symbol[:-4]}" if c.dollar_amount is not None
                    else f"{amount(idea.symbol, c.quantity)} = {money(c.quantity * c.limit_price)}")
            rows.append([str(i), "crypto" if is_crypto(idea.symbol) else "stock", idea.symbol, price(snap.quotes[idea.symbol].last),
                         f"{idea.today}%" if idea.today else "-", size, f"{idea.confidence:.2f}"])
        say("\nThe model's ideas (it can be wrong, and a local model has no proven edge):")
        say(table(["#", "Type", "Symbol", "Price", "Today", "Order", "Model"], rows))
        say("Why:")
        for i, idea in enumerate(ideas, 1):
            score_only = idea.reason.startswith("Laya:")
            say(f"  {i}. {idea.symbol}: {idea.reason}" + ("  (Laya gives a score, not a written reason)" if score_only else ""))
            if score_only and idea.drivers:
                say("     What moved Laya's score (measured by hiding each piece of data; not Laya's own words):")
                moved = [d for d in idea.drivers if abs(d["points"]) >= 1][:3]
                for d in moved:
                    say(f"       {d['data']} ({d['value']}): {d['points']:+.0f} points "
                        + ("toward buy" if d["points"] > 0 else "toward skip"))
                if not moved:
                    say("       nothing moved it by even 1 point: the score barely depends on this stock's data")
            if data_line(idea.facts):
                say(f"     Data:    {data_line(idea.facts)}")
            for h in (idea.facts.get("headlines") or [])[:2]:
                say(f"     News:    {h}")
            if idea.second:
                s = idea.second
                verdict = "agrees" if s["action"] == "buy" else "DISAGREES"
                say(f"     Second opinion, your local model {verdict}: {s['action']} ({s['confidence']:.2f}), {s['reason']}")
        if crypto_note:
            say(f"Crypto: no coins today. {crypto_note}.")

    NONE_WORDS = {"", "none", "no", "n", "0", "skip", "nothing", "nope"}

    def choose(self, ideas: list[Idea], room: int) -> list[Idea]:
        one = len(ideas) == 1
        say(f"Today's limits leave room for {room} buy{'' if room == 1 else 's'}.")
        prompt = (f"Buy {ideas[0].symbol}? Type yes to buy it, or no to skip: " if one else
                  "Which should I buy? Type their numbers (like 1 or 1 3), all, or none: ")
        for _ in range(3):
            raw = ask(prompt).strip().lower()
            if raw in self.NONE_WORDS:
                say("Nothing bought.")
                return []
            if raw == "all" or (one and raw in ("yes", "y")):
                nums = list(range(1, len(ideas) + 1))
            else:
                try:
                    nums = list(dict.fromkeys(int(x) for x in raw.replace(",", " ").split()))
                except ValueError:
                    nums = [0]
            if not nums or any(n < 1 or n > len(ideas) for n in nums):
                say("Please type yes or no." if one else f"Please type numbers from 1 to {len(ideas)}, all, or none.")
                continue
            if len(nums) > room:
                say(f"Only {room} fit today's limits. Taking the first {room}.")
            return [ideas[n - 1] for n in nums[:room]]
        say("No clear answer, so nothing was bought.")
        return []

    async def buy(self, idea: Idea) -> None:
        sym = idea.symbol
        say(f"\n{'=' * 64}\n{sym}  ({'crypto' if is_crypto(sym) else 'stock'})")
        prep: dict = {}

        async def check(rh: Robinhood) -> None:          # fresh price, fresh size, limits, Robinhood's review
            prep.clear()
            snap = await self.snapshot(rh, [sym])
            c = self.entry(sym, snap, idea.order.reason) if sym in snap.quotes else "no fresh price"
            bad = [c] if isinstance(c, str) else self.checks(c, snap)
            if bad:
                log("blocked_by_guardrails", symbol=sym, violations=bad)
                say("Not bought: " + "; ".join(bad))
                return
            try:
                args, alerts = await self.review(rh, c)
            except RHError as e:
                log("review_refused", symbol=sym, error=str(e)[:500])
                say(f"Not bought. Robinhood's review refused it: {str(e)[:300]}")
                return
            log("robinhood_review", order=args, alerts=alerts)
            prep.update(c=c, alerts=alerts)

        if not await self.step(f"checking {sym}", check) or "c" not in prep:
            return
        c = prep["c"]
        self.show_order(c, idea, prep["alerts"])
        if not self.live:
            log("dry_run", order=c.as_dict())
            say("DRY RUN (mode: paper): nothing was sent.")
            return
        if ask("Place this order with real money? Type yes to confirm: ").lower() != "yes":
            log("not_approved", symbol=sym)
            say("Skipped.")
            return
        ref, sent = str(uuid.uuid4()), {"yes": False}

        async def send(rh: Robinhood) -> None:
            if not sent["yes"]:                            # one last look at the market before committing
                bad = self.checks(c, await self.snapshot(rh, [sym]))
                if bad:
                    log("blocked_after_recheck", order=c.as_dict(), violations=bad)
                    say("Not bought, the market moved: " + "; ".join(bad))
                    return
                sent["yes"] = True
            oid = await self.place(rh, c, ref)
            if oid:
                await self._after_buy(rh, c, ref, oid)

        try:
            ok = await self.step(f"buying {sym}", send)
        except BaseException:
            if sym in self.book:
                self.warn_if_unprotected(self.book[sym])
            raise
        if not ok and sym in self.book:
            self.warn_if_unprotected(self.book[sym])

    def show_order(self, c: Candidate, idea: Idea, alerts: list) -> None:
        crypto = c.asset == "crypto"
        if crypto:
            say(f"BUY      {money(c.dollar_amount)} of {c.symbol[:-4]}, paying at most {price(c.limit_price)} per coin")
        else:
            say(f"BUY      {amount(c.symbol, c.quantity)} of {c.symbol}, paying at most {price(c.limit_price)} each"
                f" = {money(c.quantity * c.limit_price)}")
        say(f"Why      {idea.reason} (model confidence {idea.confidence:.2f})")
        top = [d for d in idea.drivers if d["points"] >= 1][:2]
        if top:
            say("Driven   " + "; ".join(f"{d['data']} ({d['value']}) {d['points']:+.0f} points" for d in top))
        elif idea.drivers:
            say("Driven   by nothing in particular: no single piece of data moved Laya's score by 1 point")
        if data_line(idea.facts):
            say(f"Data     {data_line(idea.facts)}")
        if idea.second:
            say(f"Second   your local model says {idea.second['action']} ({idea.second['confidence']:.2f}): {idea.second['reason']}")
        if (self.cfg.get("protect") or {}).get("broker_stop", True):
            pct = self.d.get("crypto_stop_loss_pct" if crypto else "stock_stop_loss_pct", 3.0 if crypto else 1.0)
            say(f"Then     a stop order sells it if it falls {pct}% under what you pay")
        say(f"Review   {'Robinhood shows no warnings' if not alerts else 'Robinhood says: ' + json.dumps(alerts)[:400]}")

    async def _after_buy(self, rh: Robinhood, c: Candidate, ref: str, oid: str) -> None:
        day, t = self.find_trade(ref)
        state, qty, avg = await self.wait_fill(rh, c.asset, oid)
        if state not in DONE_STATES:
            if not await self.cancel(rh, c.asset, oid, "buy did not fill in time: no late, unprotected fills"):
                t["status"] = "unknown"                    # may still fill: the next run checks it, never writes it off
                save_day(day)
                log("UNCONFIRMED_CANCEL", symbol=c.symbol, order_id=oid)
                say(f"WARNING: the {c.symbol} buy did not fill in time and its cancel is not confirmed, so it may still fill. "
                    "Check the Robinhood app. The next run checks it and offers its stop or target order if it filled.")
                return
            state, qty, avg = order_fill(await self.order(rh, c.asset, oid))        # it may have filled meanwhile
        if not qty:
            t["status"] = "cancelled"
            self.rebudget(day, t, Decimal(0))
            log("buy_not_filled", symbol=c.symbol, state=state)
            say("It did not fill in time, so I cancelled it. Nothing was bought.")
            return
        avg = avg or c.limit_price
        cost = (qty * avg).quantize(CENT)
        t["booked"] = True
        self.rebudget(day, t, cost)
        e = self.book.get(c.symbol)
        if not e or e.get("buy_ref_id") != ref:                   # a retry keeps the entry, and so its ref_ids
            e = self.new_entry(c.symbol, c.asset, self.day.date, ref, oid, qty, cost, avg)
        e.update(quantity=str(qty), cost=str(cost), avg_price=str(avg), buy_order_id=oid)
        self.book[c.symbol] = e
        save_book(self.book)
        log("bought", symbol=c.symbol, asset=c.asset, quantity=str(qty), avg_price=str(avg), cost=str(cost), style=e["style"])
        say(f"Bought {amount(c.symbol, qty)} at {price(avg)}. Paid {money(cost)}.")
        await self.park(rh, e)

    def guard_text(self, e: dict) -> str:
        """What waits at Robinhood for this holding, in words."""
        if self.holds(e):
            target = f"target {price(e['target_price'])}" if e.get("target_order_id") and e.get("target_price") else "NO TARGET"
            return f"{target}, stop level {price(e['stop_level'])}"
        return f"stop {price(e['stop_price'])}" if e.get("stop_order_id") and e.get("stop_price") else "NO STOP"

    def show_holdings(self, snap: Snapshot) -> None:
        if not self.book:
            say("The agent holds nothing right now.")
            return
        rows, under = [], []
        for sym, e in self.book.items():
            q, cost = snap.quotes.get(sym), Decimal(e["cost"])
            worth = (Decimal(e["quantity"]) * q.bid).quantize(CENT) if q and q.bid > 0 else None
            rows.append([sym, amount(sym, e["quantity"]), money(cost), money(worth) if worth is not None else "-",
                         change(worth - cost, cost) if worth is not None else "-", self.guard_text(e)])
            if self.holds(e) and q and 0 < q.bid <= Decimal(e["stop_level"]):
                under.append(sym)
        say("What the agent holds:")
        say(table(["Symbol", "Amount", "Paid", "Worth now", "Change", "At Robinhood"], rows))
        for sym in under:
            say(f"WARNING: {sym} is under its stop level ({price(self.book[sym]['stop_level'])}). "
                "Run  python -m agent evening  to sell it.")

    # ---- evening ---------------------------------------------------------------
    async def evening(self) -> None:
        now = guardrails.now_et()
        say(f"\nEVENING SESSION  {now:%a %Y-%m-%d} {clock(now)} your time   mode: {self.cfg['mode']}"
            + ("" if self.live else "   (DRY RUN: nothing will be sent)"))
        if config.kill_switch_on():
            say("The KILL file is there, so no orders. Delete it to trade. Your stop orders stay at Robinhood.")
            return
        ctx: dict = {}

        async def look(rh: Robinhood) -> None:
            ctx.clear()
            if await self.setup(rh):
                await self.reconcile(rh)
                ctx["snap"] = await self.snapshot(rh, list(self.book))

        if not await self.step("checking your holdings", look) or "snap" not in ctx:
            return
        if not self.book:
            say("The agent holds nothing. Nothing to sell." + (f" Today: {money(self.realized)}." if self.realized else ""))
            return
        plan = self.plan(ctx["snap"])
        say(table(["#", "Symbol", "Amount", "Paid", "Worth now", "Change", "Plan"],
                  [[str(i), r["e"]["symbol"], amount(r["e"]["symbol"], r["e"]["quantity"]), money(r["e"]["cost"]),
                    money(r["worth"]) if r["worth"] is not None else "-",
                    change(r["pnl"], Decimal(r["e"]["cost"])) if r["pnl"] is not None else "-",
                    {"sell": "sell", "ask": "you decide", "keep": "keep", "wait": "wait", "hold": "hold"}[r["action"]]
                    + f" ({r['note']})"]
                   for i, r in enumerate(plan, 1)]))
        for r in plan:
            if r["action"] == "ask":
                r["action"] = self.down_choice(r)
        sells = [r for r in plan if r["action"] == "sell"]
        kept = {r["e"]["symbol"] for r in plan if r["action"] != "sell"}
        if sells and not self.live:
            log("dry_run", would_sell=[r["e"]["symbol"] for r in sells])
            say("DRY RUN (mode: paper): nothing was sold.")
            return
        if sells and self.d.get("confirm_sells", True):
            total = sum((r["worth"] for r in sells), Decimal(0))
            if ask(f"Sell {len(sells)} holding{'' if len(sells) == 1 else 's'} now for about {money(total)}? "
                   "Type yes to confirm: ").lower() != "yes":
                say("Nothing sold. Everything keeps its order at Robinhood.")
                kept |= {r["e"]["symbol"] for r in sells}
                sells = []
        for r in sells:
            await self.sell(r["e"])
        if any(not self.parked_id(self.book[s]) for s in kept if s in self.book):
            await self.step("protecting what you keep", lambda rh: self.protect_missing(rh, kept))
        say(f"\nDone. Sold in this session: {money(self.realized)}.")
        for s, e in self.book.items():
            waiting = self.guard_text(e)
            say(f"Still held: {s} ({amount(s, e['quantity'])}), at Robinhood: {waiting}"
                + (" (watch it in the Robinhood app)" if "NO " in waiting else ""))

    def plan(self, snap: Snapshot) -> list[dict]:
        """Up or flat -> sell. Down -> daily.when_down (ask by default). Stocks wait while their market is closed.
        Hold style: keep it (its target sell does the selling), unless the price is under its stop level."""
        when_down = str(self.d.get("when_down", "ask")).lower()
        open_ = snap.now.replace(hour=9, minute=30, second=0, microsecond=0)
        rows = []
        for sym, e in self.book.items():
            r = {"e": e, "worth": None, "pnl": None, "action": "wait", "note": ""}
            q = snap.quotes.get(sym)
            if e["asset"] == "crypto" and not self.crypto_on:
                r["note"] = "crypto unavailable right now"
            elif q is None or q.bid <= 0:
                r["note"] = "no price right now"
            else:
                r["worth"] = (Decimal(e["quantity"]) * q.bid).quantize(CENT)
                r["pnl"] = r["worth"] - Decimal(e["cost"])
                if self.holds(e):
                    level = Decimal(e.get("stop_level") or 0)
                    if level and q.bid <= level:
                        r["action"], r["note"] = "sell", f"under its stop level {price(level)}"
                    else:
                        r["action"], r["note"] = "hold", (f"its target sell waits at {price(e['target_price'])}"
                                                          if e.get("target_order_id") and e.get("target_price")
                                                          else "no target sell order")
                elif e["asset"] == "stock" and not guardrails.market_open(snap.now):
                    r["note"] = f"market closed, stocks sell {clock(open_)} to {clock(open_.replace(hour=16, minute=0))} your time"
                elif r["pnl"] >= 0:
                    r["action"], r["note"] = "sell", "up" if r["pnl"] > 0 else "flat"
                else:
                    r["action"], r["note"] = (when_down if when_down in ("sell", "keep") else "ask"), "down"
            rows.append(r)
        return rows

    def down_choice(self, r: dict) -> str:
        sym = r["e"]["symbol"]
        for _ in range(3):
            a = ask(f"{sym} is down {money(-r['pnl'])}. Sell it now at a loss (s), or keep it with its stop order (k)? ").lower()
            if a in ("s", "sell"):
                return "sell"
            if a in ("k", "keep"):
                return "keep"
        say(f"No clear answer, so {sym} is kept.")
        return "keep"

    async def sell(self, e: dict) -> None:
        e["sell_ref_id"] = str(uuid.uuid4())             # a new sell, on disk before anything is sent
        save_book(self.book)
        progress: dict = {}                               # what this sell has already done, so a retry skips it
        try:
            ok = await self.step(f"selling {e['symbol']}", lambda rh: self._sell(rh, e, progress))
        except BaseException:
            self.warn_if_unprotected(e)
            raise
        if not ok:
            self.warn_if_unprotected(e)

    async def _sell(self, rh: Robinhood, e: dict, p: dict) -> None:
        sym, asset = e["symbol"], e["asset"]
        if "order" not in p:
            if not p.get("told"):
                say(f"\n{sym}: selling {amount(sym, e['quantity'])}")
                p["told"] = True
            # 1. our own checks FIRST: if the sell cannot go ahead, nothing is cancelled and the stop stays
            snap = await self.snapshot(rh, [sym])
            c = self.exit_order(e, snap)
            bad = [c] if isinstance(c, str) else self.checks(c, snap)
            if not bad:
                # 2. free the holding (cancel its stop), then Robinhood's review of the sell
                if not await self._free(rh, e, p):
                    return
                try:
                    args, alerts = await self.review(rh, c)
                    log("robinhood_review", order=args, alerts=alerts)
                    if alerts and ask(f"Robinhood flagged this sell: {json.dumps(alerts)[:300]}\nSell anyway? Type yes: ").lower() != "yes":
                        bad = ["you said no after Robinhood's warning"]
                except RHError as err:
                    bad = [f"Robinhood's review refused it: {str(err)[:200]}"]
            if bad:
                log("sell_blocked", symbol=sym, why=bad)
                if not p.get("freed"):
                    say(f"{sym}: not sold ({'; '.join(bad)}). Its order at Robinhood stays in place.")
                else:
                    say(f"{sym}: not sold ({'; '.join(bad)}). Putting its order back at Robinhood.")
                    await self._restore(rh, e, p)
                return
            p["order"] = c
        c = p["order"]
        # 3. send (a retry re-sends the same ref_id), wait for the fill, settle
        if not p.get("settled"):
            oid = await self.place(rh, c, e["sell_ref_id"])
            if not oid:
                p["settled"] = True                              # rejected: nothing was sold
                await self._restore(rh, e, p)
                return
            state, qty, avg = await self.wait_fill(rh, asset, oid)
            if state not in DONE_STATES:
                if not await self.cancel(rh, asset, oid, "sell did not fill in time"):
                    found = self.find_trade(e["sell_ref_id"])
                    if found:
                        found[1]["status"] = "unknown"
                        save_day(found[0])
                    p["settled"] = True
                    log("UNCONFIRMED_CANCEL", symbol=sym, order_id=oid)
                    say(f"WARNING: the {sym} sell did not fill in time and its cancel is not confirmed, so it may still sell. "
                        "Check the Robinhood app. The next run checks it.")
                    return                                       # the coins or shares sit under that sell: no stop possible
                state, qty, avg = order_fill(await self.order(rh, asset, oid))
            self.settle(e, qty, avg or c.limit_price)
            p["settled"] = True
        # 4. whatever is left keeps its order at Robinhood
        if sym in self.book and not self.parked_id(e):
            if not p.get("warned"):
                log("STILL_HOLDING", symbol=sym, quantity=e["quantity"])
                say(f"WARNING: {sym} did not fully sell. {amount(sym, e['quantity'])} left. Putting its order back at Robinhood.")
                p["warned"] = True
            await self._restore(rh, e, p)

    async def _free(self, rh: Robinhood, e: dict, p: dict) -> bool:
        """Cancel the agent's own open sell orders on this holding (its stop or target): shares or coins under an
        open sell order cannot be sold again. Orders you placed yourself are never touched."""
        if p.get("freed"):
            return True
        sym, asset = e["symbol"], e["asset"]
        for o in await self.open_on(rh, e):
            oid = str(o["id"])
            if not await self.cancel(rh, asset, oid, "free the holding before selling"):
                say(f"{sym}: an open order on it would not cancel, so it was not sold. Check the Robinhood app.")
                if not self.parked_id(e):                        # its stop or target is already gone: put it back
                    await self._restore(rh, e, p)
                return False
            for key in ("stop_order_id", "target_order_id"):
                if oid == e.get(key):
                    e[key] = None
                    save_book(self.book)
        e["stop_order_id"] = None
        if self.holds(e):
            e["target_order_id"] = None
        save_book(self.book)
        p["freed"] = True
        return True

    async def _restore(self, rh: Robinhood, e: dict, p: dict) -> bool:
        """Put its order back at Robinhood (a stop, or a target sell for hold style). The new order gets ONE new
        ref_id per sell, so a retry re-sends the same one."""
        if e["symbol"] not in self.book or self.parked_id(e):
            return True
        if "park_ref" not in p:
            p["park_ref"] = e["target_ref_id" if self.holds(e) else "stop_ref_id"] = str(uuid.uuid4())
            save_book(self.book)
        return await self.park(rh, e)

    def exit_order(self, e: dict, snap: Snapshot) -> Candidate | str:
        sym, q = e["symbol"], snap.quotes.get(e["symbol"])
        if q is None or q.bid <= 0:
            return "no price right now"
        slack, qty = Decimal(str(self.d.get("sell_slack_pct", 0.10))), Decimal(e["quantity"])
        if e["asset"] != "crypto":
            return _exit_order(sym, qty, q.bid, "evening sell", slack)       # a limit a hair under the bid: it fills
        pair = self.pairs.get(sym)
        if not pair:
            return "Robinhood does not list this pair right now"
        return Candidate(sym, "sell", "limit", "evening sell", quantity=step_down(qty, pair.qty_step),
                         limit_price=step_down(q.bid * (1 - slack / 100), pair.price_step), is_exit=True, asset="crypto")

    def settle(self, e: dict, sold: Decimal, avg: Decimal) -> None:
        sym = e["symbol"]
        held, cost = Decimal(e["quantity"]), Decimal(e["cost"])
        if sold and sold > 0:
            part = cost * min(sold, held) / held
            pnl = (min(sold, held) * avg - part).quantize(CENT)
            self.realized += pnl
            log("sold", symbol=sym, quantity=str(sold), price=str(avg), pnl=str(pnl))
            say(f"Sold {amount(sym, sold)} at {price(avg)}: {change(pnl, part)}.")
            held, cost = held - min(sold, held), cost - part
        step = self.pairs[sym].qty_step if is_crypto(sym) and sym in self.pairs else Decimal(1)
        if held < step:                              # all gone, or crypto dust too small to sell
            del self.book[sym]
            log("closed", symbol=sym, dust=str(held) if held > 0 else None)
        else:
            e["quantity"], e["cost"] = str(held), str(cost.quantize(CENT))
        save_book(self.book)


@contextmanager
def _quiet_journal():
    """The journal file still gets every event; the screen gets plain words only."""
    before, journal.ECHO = journal.ECHO, False
    try:
        yield
    finally:
        journal.ECHO = before


async def morning(cfg: dict) -> None:
    if not console_ok():
        say("The morning session asks you before every buy, so it needs a console window. Nothing was done.")
        return
    with single_session(), _quiet_journal():
        await Session(cfg).morning()


async def evening(cfg: dict) -> None:
    if not console_ok():
        say("The evening session may ask you questions, so it needs a console window. Nothing was done.")
        return
    with single_session(), _quiet_journal():
        await Session(cfg).evening()
