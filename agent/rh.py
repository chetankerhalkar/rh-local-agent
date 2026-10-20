"""Thin wrapper around the Robinhood Trading MCP.

Two rules are enforced here in code:
  1. The generic call() refuses every tool that can move money.
  2. Orders go only through the review / place / cancel methods below, and only for the one
     agentic account in config.yaml (its account_number for stocks, its rhs_account_number for crypto).
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from decimal import ROUND_CEILING, ROUND_DOWN, Decimal, InvalidOperation
from pathlib import Path
from zoneinfo import ZoneInfo

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

from .models import CryptoPair, Position, Quote

# Read-only tools the LLM research chat is allowed to use. Everything else is denied.
READ_ONLY_TOOLS = {
    "get_equity_quotes", "get_equity_fundamentals", "get_equity_historicals", "get_equity_news",
    "get_equity_technical_indicators", "get_earnings_results", "get_earnings_calendar",
    "get_financials", "get_index_quotes", "get_indexes", "search",
    "get_crypto_quotes", "get_currency_pairs", "preview_scan",
}
ORDER_TOOLS = {"review_equity_order", "place_equity_order", "cancel_equity_order",
               "preview_crypto_order", "place_crypto_order", "cancel_crypto_order"}
OPEN_STATES = {"new", "queued", "confirmed", "unconfirmed", "partially_filled"}
DONE_STATES = {"filled", "cancelled", "canceled", "rejected", "failed", "voided"}   # crypto spells it "canceled"
ET = ZoneInfo("America/New_York")


# Robinhood answers 400 to the SDK's polite "end session" request on exit. Harmless, so keep it off the console.
logging.getLogger("mcp.client.streamable_http").setLevel(logging.ERROR)


class RHError(Exception):
    pass


def dec(x) -> Decimal | None:
    try:
        return Decimal(str(x)) if x not in (None, "") else None
    except (InvalidOperation, ValueError):
        return None


def find_first(obj, keys: tuple[str, ...]) -> Decimal | None:
    """Walk nested JSON and return the first numeric value stored under any of `keys`."""
    if isinstance(obj, dict):
        for k in keys:
            if k in obj:
                v = obj[k]
                v = v.get("amount") if isinstance(v, dict) else v     # {"amount": "12.3", "currency": ...}
                if dec(v) is not None:
                    return dec(v)
        for v in obj.values():
            r = find_first(v, keys)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_first(v, keys)
            if r is not None:
                return r
    return None


def find_dicts_with(obj, required: tuple[str, ...]) -> list[dict]:
    found = []
    if isinstance(obj, dict):
        if all(k in obj for k in required):
            found.append(obj)
        for v in obj.values():
            found += find_dicts_with(v, required)
    elif isinstance(obj, list):
        for v in obj:
            found += find_dicts_with(v, required)
    return found


def parse_ts(x) -> datetime | None:
    """ISO time in any of Robinhood's flavours: nanoseconds, 'Z', '+00:00', or no zone at all
    (crypto quote times come in US Eastern). Unreadable -> None, and the guardrails treat that as stale."""
    m = re.match(r"^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})(?:\.(\d+))?(Z|[+-]\d{2}:?\d{2})?$", str(x or "").strip())
    if not m:
        return None
    day, clock, frac, zone = m.groups()
    dt = datetime.fromisoformat(f"{day}T{clock}.{((frac or '') + '000000')[:6]}")
    if zone is None:
        return dt.replace(tzinfo=ET)
    if zone == "Z":
        return dt.replace(tzinfo=timezone.utc)
    sign = 1 if zone[0] == "+" else -1
    return dt.replace(tzinfo=timezone(sign * timedelta(hours=int(zone[1:3]), minutes=int(zone[-2:]))))


def pair_symbol(x) -> str | None:
    """'BTC-USD', 'BTCUSD' or 'btc-usd' -> 'BTC-USD'. Anything that is not a USD pair -> None."""
    s = str(x or "").upper().strip()
    if s.endswith("-USD"):
        return s if re.fullmatch(r"[A-Z0-9]{1,12}-USD", s) else None
    return pair_symbol(s[:-3] + "-USD") if s.endswith("USD") and len(s) > 3 else None


def asset_code(p: dict) -> str | None:
    """The coin code of a crypto position row, whichever key the server used for it."""
    for k in ("asset_code", "currency_code", "code", "asset", "currency", "asset_currency", "symbol"):
        v = p.get(k)
        if isinstance(v, dict):
            v = v.get("code") or v.get("symbol")
        if isinstance(v, str) and v.strip():
            s = v.upper().strip()
            if s.endswith("-USD"):
                return s[:-4]
            return s[:-3] if k == "symbol" and s.endswith("USD") and len(s) > 3 else s
    return None


def step_down(x: Decimal, step: Decimal) -> Decimal:
    """Round down onto an order step (a quantity or price increment). Sells and stops round down."""
    return (x / step).to_integral_value(rounding=ROUND_DOWN) * step if step > 0 else x


def step_up(x: Decimal, step: Decimal) -> Decimal:
    """Round up onto a price step. A buy limit rounded up still reaches the ask."""
    return (x / step).to_integral_value(rounding=ROUND_CEILING) * step if step > 0 else x


def order_id_of(result: dict) -> str | None:
    rows = find_dicts_with(result, ("id", "state")) or find_dicts_with(result, ("id",))
    return str(rows[0]["id"]) if rows else None


def order_fill(o: dict | None) -> tuple[str, Decimal, Decimal | None]:
    """(state, filled quantity, average fill price) of a stock or crypto order. No order -> ("unknown", 0, None)."""
    if not o:
        return "unknown", Decimal(0), None
    state = str(o.get("state") or "unknown").lower()
    execs = [e for e in (o.get("executions") or []) if isinstance(e, dict)]
    qty = next((q for q in (dec(o.get(k)) for k in ("cumulative_quantity", "filled_quantity", "filled_asset_quantity"))
                if q is not None), None)
    if qty is None:
        qty = sum((dec(e.get("quantity")) or Decimal(0) for e in execs), Decimal(0))
    avg = dec(o.get("average_price"))
    if avg is None and execs and qty:
        paid = sum(((dec(e.get("effective_price") or e.get("price")) or Decimal(0)) * (dec(e.get("quantity")) or Decimal(0))
                    for e in execs), Decimal(0))
        avg = paid / qty if paid else None
    if state == "filled" and not qty:
        qty = dec(o.get("quantity")) or Decimal(0)          # some shapes only repeat the order quantity
    return state, qty, avg


def extract_alerts(review: dict) -> list:
    """Anything that looks like a warning in a review_equity_order result."""
    alerts = []
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("alerts", "alert", "order_checks", "warnings", "errors") and v:
                    alerts.append({k: v})
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(review)
    return alerts


class Robinhood:
    def __init__(self, session: ClientSession, cfg: dict) -> None:
        self.s = session
        self.account = cfg["robinhood"].get("account_number") or ""
        self.crypto_account = str(cfg["robinhood"].get("crypto_account") or "")   # numeric rhs_account_number
        self.missing: list[str] = []          # tickers Robinhood did not recognise on the last quotes() call
        self.news_problem = ""                # why the last headlines() call came back empty, if it did

    async def _raw(self, name: str, args: dict) -> dict:
        res = await self.s.call_tool(name, args)
        text = "".join(getattr(c, "text", "") for c in (res.content or []))
        if getattr(res, "is_error", False):
            raise RHError(f"{name} failed: {text[:500]}")
        if getattr(res, "structured_content", None):
            return res.structured_content
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"text": text}

    async def call(self, name: str, args: dict) -> dict:
        """Generic entry point. Read tools only."""
        if name.startswith(("place_", "cancel_", "exercise_", "review_")) or name in ORDER_TOOLS:
            raise RHError(f"{name} is not allowed through call(). Orders use review_order/place_order.")
        return await self._raw(name, args)

    async def list_tools(self) -> list:
        return (await self.s.list_tools()).tools

    # ---- typed reads -----------------------------------------------------------
    async def _quote_rows(self, symbols: list[str]) -> list[dict]:
        """Robinhood rejects the WHOLE request when one symbol is unknown. Drop the unknown ones and retry once."""
        for _ in range(2):
            if not symbols:
                return []
            try:
                data = await self.call("get_equity_quotes", {"symbols": symbols})
                return (data.get("data") or {}).get("results", [])
            except RHError as e:
                m = re.search(r'"missing_instruments"\s*:\s*(\[[^\]]*\])', str(e))
                if not m:
                    raise
                bad = json.loads(m.group(1))
                self.missing += [b for b in bad if b not in self.missing]
                symbols = [x for x in symbols if x not in bad]
        return []

    async def quotes(self, symbols: list[str]) -> dict[str, Quote]:
        self.missing = []
        rows: list[dict] = []
        for i in range(0, len(symbols), 10):                     # small batches keep every request well inside limits
            rows += await self._quote_rows(list(symbols[i:i + 10]))
        out = {}
        for row in rows:
            q = row.get("quote") or {}
            last, ts = dec(q.get("last_trade_price")), q.get("venue_last_trade_time")
            ext, ext_ts = dec(q.get("last_non_reg_trade_price")), q.get("venue_last_non_reg_trade_time")
            if ext is not None and ext_ts and (not ts or ext_ts > ts):
                last, ts = ext, ext_ts                       # newer extended-hours print
            if not q.get("symbol") or last is None:
                continue
            when = None
            if ts:   # Robinhood sends nanoseconds; Python wants microseconds
                head, _, frac = ts.rstrip("Z").partition(".")
                when = datetime.fromisoformat(f"{head}.{(frac + '000000')[:6]}+00:00")
            out[q["symbol"]] = Quote(
                symbol=q["symbol"], last=last, bid=dec(q.get("bid_price")) or Decimal(0),
                ask=dec(q.get("ask_price")) or Decimal(0),
                prev_close=dec(q.get("adjusted_previous_close")) or dec(q.get("previous_close")) or Decimal(0),
                ts=when, state=q.get("state", "unknown"), has_traded=bool(q.get("has_traded", False)))
        return out

    async def positions(self) -> dict[str, Position]:
        data = await self.call("get_equity_positions", {"account_number": self.account})
        out = {}
        for p in find_dicts_with(data, ("symbol", "quantity")):
            qty = dec(p.get("quantity"))
            cost = find_first(p, ("average_buy_price", "average_cost", "avg_cost", "average_price"))
            if qty and qty > 0:
                out[p["symbol"]] = Position(p["symbol"], qty, cost or Decimal(0))
        return out

    async def money(self) -> tuple[Decimal | None, Decimal | None]:
        """(buying_power, total_equity). Returns None when a field cannot be found,
        and the guardrails then block all buys. Run `discover` to see the real keys."""
        data = await self.call("get_portfolio", {"account_number": self.account})
        bp = find_first(data, ("buying_power", "cash_buying_power", "unleveraged_buying_power"))
        eq = find_first(data, ("total_equity", "equity", "portfolio_equity", "total_market_value", "total_value"))
        return bp, eq

    async def headlines(self, symbol: str, limit: int = 5) -> list[str]:
        """Recent headlines. News is optional, so a failure returns [] - but it is remembered in
        self.news_problem, so the session can say that news is missing instead of hiding it."""
        try:
            data = await self.call("get_equity_news", {"symbol": symbol})
        except Exception as e:                                   # noqa: BLE001
            self.news_problem = f"{type(e).__name__}: {e}"[:200]
            return []
        titles = [str(d.get("title") or d.get("headline"))[:160]
                  for d in find_dicts_with(data, ("title",)) + find_dicts_with(data, ("headline",))
                  if d.get("title") or d.get("headline")]
        if not titles:
            self.news_problem = getattr(self, "news_problem", "") or "the answer had no headlines in it"
        return list(dict.fromkeys(titles))[:limit]

    async def agentic_account_ok(self) -> bool:
        data = await self.call("get_accounts", {})
        for a in find_dicts_with(data, ("account_number",)):
            if str(a.get("account_number")) == self.account:
                return a.get("agentic_allowed") is True
        return False

    # ---- orders: the only two doors ---------------------------------------------
    def _guard_account(self, args: dict) -> None:
        if not self.account or args.get("account_number") != self.account:
            raise RHError("order account does not match robinhood.account_number in config.yaml")

    async def review_order(self, args: dict) -> dict:
        self._guard_account(args)
        return await self._raw("review_equity_order", args)

    async def place_order(self, args: dict, ref_id: str) -> dict:
        self._guard_account(args)
        uuid.UUID(ref_id)                       # must be a real UUID (idempotency key)
        return await self._raw("place_equity_order", {**args, "ref_id": ref_id})


    async def open_orders(self, symbol: str, any_day: bool = False) -> list[dict]:
        """The agent's own still-open orders for one symbol. Placed today by default; any_day also
        finds good-till-cancelled stop orders left from earlier days."""
        args = {"account_number": self.account, "symbol": symbol}
        if any_day:
            args["placed_agent"] = "agentic"
        else:
            args["created_at_gte"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        data = await self.call("get_equity_orders", args)
        return [o for o in find_dicts_with(data, ("id", "state"))
                if str(o.get("state")) in OPEN_STATES and o.get("symbol", symbol) == symbol]

    async def cancel_order(self, order_id: str) -> dict:
        if not self.account:
            raise RHError("no account configured")
        return await self._raw("cancel_equity_order", {"account_number": self.account, "order_id": order_id})

    async def equity_order(self, order_id: str) -> dict | None:
        data = await self.call("get_equity_orders", {"account_number": self.account, "order_id": order_id})
        return next((o for o in find_dicts_with(data, ("id", "state")) if str(o["id"]) == order_id), None)

    async def recent_orders(self, asset: str, since: str) -> tuple[list[dict], bool]:
        """Orders created since a UTC time, to find one by its ref_id after an interrupted run.
        Returns (orders, complete). complete is False when the answer says more pages exist, so
        'not in the list' proves nothing."""
        if asset == "crypto":
            data = await self.call("get_crypto_orders", {"rhs_account_number": self.crypto_account, "created_at_gte": since})
        else:
            data = await self.call("get_equity_orders", {"account_number": self.account, "created_at_gte": since})
        tops = [data, *(v for v in data.values() if isinstance(v, dict))] if isinstance(data, dict) else []
        more = any(t.get(k) for t in tops for k in ("next", "next_cursor", "next_page", "cursor"))
        return find_dicts_with(data, ("id", "state")), not more

    # ---- crypto: same account, its numeric rhs_account_number ----------------------------
    async def find_crypto_account(self) -> str:
        """Crypto tools need the numeric rhs_account_number. Take it from the agentic account's get_accounts entry."""
        if not self.crypto_account and self.account:
            for a in find_dicts_with(await self.call("get_accounts", {}), ("account_number",)):
                if str(a.get("account_number")) == self.account and a.get("rhs_account_number"):
                    self.crypto_account = str(a["rhs_account_number"])
        return self.crypto_account

    async def crypto_pairs(self) -> dict[str, CryptoPair]:
        """Every USD pair Robinhood trades, with its order steps. Halted or display-only pairs come back not tradable."""
        out: dict[str, CryptoPair] = {}
        for p in find_dicts_with(await self.call("get_currency_pairs", {"limit": 700}), ("symbol",)):
            sym = pair_symbol(p.get("symbol"))
            if not sym or sym in out:
                continue
            halted = any(p.get(k) for k in ("halted", "trading_halted", "is_halted", "display_only"))
            out[sym] = CryptoPair(sym, sym[:-4],
                                  dec(p.get("min_order_quantity_increment")) or Decimal("0.00000001"),
                                  dec(p.get("min_order_price_increment")) or Decimal("0.01"),
                                  tradable=str(p.get("tradability") or "tradable").lower() == "tradable" and not halted)
        return out

    async def crypto_quotes(self, pairs: list[str]) -> dict[str, Quote]:
        """Live bid / ask / mark per pair. The server answers 'BTCUSD'; the agent calls it 'BTC-USD'.
        prev_close is the price at midnight US Eastern (the tool's open_price)."""
        out: dict[str, Quote] = {}
        routing = {"rhs_account_number": self.crypto_account} if self.crypto_account else {}
        for i in range(0, len(pairs), 20):
            data = await self.call("get_crypto_quotes", {"symbols": list(pairs[i:i + 20]), **routing})
            for row in find_dicts_with(data, ("symbol",)):
                sym = pair_symbol(row.get("symbol"))
                if sym not in pairs or sym in out:
                    continue
                bid = find_first(row, ("bid_price", "bid", "bid_inclusive_of_sell_spread"))
                ask = find_first(row, ("ask_price", "ask", "ask_inclusive_of_buy_spread"))
                if bid is None or ask is None:
                    continue
                out[sym] = Quote(sym, find_first(row, ("mark_price", "mark", "mid_price")) or (bid + ask) / 2, bid, ask,
                                 find_first(row, ("open_price", "previous_close", "prev_close")) or Decimal(0),
                                 parse_ts(row.get("updated_at") or row.get("ask_time") or row.get("bid_time")),
                                 asset="crypto")
        return out

    async def crypto_positions(self) -> dict[str, Position]:
        if not self.crypto_account:
            return {}
        data = await self.call("get_crypto_positions", {"rhs_account_number": self.crypto_account})
        out: dict[str, Position] = {}
        for p in find_dicts_with(data, ("quantity",)):
            code, qty = asset_code(p), dec(p.get("quantity"))
            sym = pair_symbol(f"{code}-USD") if code and code != "USD" else None
            if not sym or not qty or qty <= 0 or sym in out:
                continue
            avg = find_first(p, ("average_cost", "average_buy_price", "avg_cost", "average_price"))
            if avg is None:
                cost = sum((dec(d.get("direct_cost_basis")) or Decimal(0) for d in find_dicts_with(p, ("direct_cost_basis",))),
                           Decimal(0))
                avg = cost / qty if cost else None
            out[sym] = Position(sym, qty, avg or Decimal(0))
        return out

    async def crypto_order(self, order_id: str) -> dict | None:
        data = await self.call("get_crypto_orders", {"rhs_account_number": self.crypto_account, "order_id": order_id})
        return next((o for o in find_dicts_with(data, ("id", "state")) if str(o["id"]) == order_id), None)

    async def open_crypto_orders(self, pair: str) -> list[dict]:
        data = await self.call("get_crypto_orders", {"rhs_account_number": self.crypto_account, "symbol": pair,
                                                     "state_group": "open"})
        return [o for o in find_dicts_with(data, ("id", "state")) if str(o.get("state")) in OPEN_STATES]

    def _guard_crypto(self, args: dict) -> None:
        if not self.crypto_account or args.get("rhs_account_number") != self.crypto_account:
            raise RHError("crypto order account does not match the agentic account's rhs_account_number")

    async def review_crypto(self, args: dict) -> dict:
        self._guard_crypto(args)
        return await self._raw("preview_crypto_order", args)

    async def place_crypto(self, args: dict, ref_id: str) -> dict:
        self._guard_crypto(args)
        uuid.UUID(ref_id)                       # must be a real UUID (idempotency key)
        return await self._raw("place_crypto_order", {**args, "ref_id": ref_id})

    async def cancel_crypto(self, order_id: str) -> dict:
        if not self.crypto_account:
            raise RHError("no crypto account")
        return await self._raw("cancel_crypto_order", {"rhs_account_number": self.crypto_account, "order_id": order_id})


@asynccontextmanager
async def connect(cfg: dict):
    rh = cfg["robinhood"]
    if rh.get("transport") == "mock":
        mock = Path(__file__).resolve().parent.parent / "tests" / "mock_robinhood_server.py"
        params = StdioServerParameters(command=sys.executable, args=[str(mock)],
                                       env=dict(os.environ))     # the SDK hides env vars by default
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                yield Robinhood(s, cfg)
        return
    from .oauth import build_provider
    auth = build_provider(rh["mcp_url"], int(rh.get("oauth_callback_port", 8765)))
    async with create_mcp_http_client(auth=auth) as http:
        async with streamable_http_client(rh["mcp_url"], http_client=http) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                yield Robinhood(s, cfg)
