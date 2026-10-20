"""A fake Robinhood MCP server for offline testing (transport: mock).

Same tool names and argument names as the real server. The equity quote shape matches the real one.
The accounts / portfolio / positions / orders / crypto / screener shapes are GUESSES: run `discover`
against the real server to see the true ones. Nothing here touches real money.

It copies the real-world behaviour that matters for exits: shares or coins sitting under an open
sell order (a stop, say) are locked, and another sell is rejected until that order is cancelled.

Control the fake market with environment variables:
    MOCK_DROP_PCT=3        stock price is 3% under the previous close (triggers the dip rule)
    MOCK_SHIFT_PCT=2       stock prices move 2% from where they were (the evening price, say)
    MOCK_CRYPTO_SHIFT_PCT  the same for coins
    MOCK_ALERT=1           review_equity_order / preview_crypto_order return a warning
    MOCK_NO_FILL=1         buy orders stay open and never fill
    MOCK_NO_SELL_FILL=1    limit sells stay open and never fill
    MOCK_SCAN_FAIL=1       the screener errors out
    MOCK_HALT=CHP,MID      these stocks report state "halted"
    MOCK_CANCEL_FAIL=1     cancels error out and the order stays open
    MOCK_FILL_ON_CANCEL=1  an open stock buy fills just before its cancel lands (the cancel then errors)
    MOCK_PARTIAL_FILL=2    stock limit sells fill only this many shares and stay open
    MOCK_NO_ASSET_SPEC=1   the screener has no asset-type filter (funds must then be caught by name)
    MOCK_MISSING=FOO,BAR   symbols the fake Robinhood does not know (it then rejects the whole quote call)
    MOCK_STATE_FILE=path   keep positions and orders between runs and connections (morning, then evening)
"""
import json
import math
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

app = MCPServer("mock-robinhood")
ACCOUNT, RHS = "MOCK-AGENTIC-1", "5550001"
OPEN = ("new", "queued", "confirmed", "unconfirmed", "partially_filled")
PRICES = {"CHP": 10.0, "MID": 24.5, "BIG": 180.0, "ETFX": 30.0}          # everything else: 100
# code: (price, quantity step, price step, spread %)
COINS = {"BTC": (60000.0, "0.00000001", "0.01", 0.2), "ETH": (2400.0, "0.000001", "0.01", 0.2),
         "DOGE": (0.12, "0.01", "0.000001", 0.4), "USDC": (1.0, "0.01", "0.0001", 0.02), "WIDE": (5.0, "0.01", "0.001", 3.0)}


def _load() -> dict:
    f = os.environ.get("MOCK_STATE_FILE")
    st = json.loads(Path(f).read_text()) if f and Path(f).exists() else {}
    for k in ("positions", "orders", "crypto", "crypto_orders"):
        st.setdefault(k, {})
    return st


def _save(st: dict) -> None:
    if os.environ.get("MOCK_STATE_FILE"):
        Path(os.environ["MOCK_STATE_FILE"]).write_text(json.dumps(st))


MEM = _load()


def _state() -> dict:
    return _load() if os.environ.get("MOCK_STATE_FILE") else MEM


def _pct(name: str) -> float:
    return float(os.environ.get(name, "0")) / 100


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------ stocks ---
def _quote(sym: str) -> dict:
    prev = PRICES.get(sym, 100.0)
    last = round(prev * (1 - _pct("MOCK_DROP_PCT")) * (1 + _pct("MOCK_SHIFT_PCT")), 2)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "000Z"     # nanoseconds like the real feed
    return {"quote": {"symbol": sym, "last_trade_price": f"{last:.6f}", "venue_last_trade_time": now,
                      "last_non_reg_trade_price": None, "venue_last_non_reg_trade_time": None,
                      "adjusted_previous_close": f"{prev:.6f}", "previous_close": f"{prev:.6f}",
                      "bid_price": f"{last - 0.01:.6f}", "ask_price": f"{last + 0.01:.6f}",
                      "has_traded": True, "state": "halted" if sym in os.environ.get("MOCK_HALT", "").split(",") else "active"},
            "close": {"symbol": sym, "price": str(prev)}}


@app.tool()
def get_equity_quotes(symbols: list[str]) -> dict:
    unknown = [s for s in symbols if s in os.environ.get("MOCK_MISSING", "").split(",")]
    if unknown:                                                   # the real server rejects the whole request
        raise ToolError('API error 404: {"missing_instruments":' + json.dumps(unknown) + "}")
    return {"data": {"results": [_quote(s.upper()) for s in symbols]}, "guide": "mock"}


@app.tool()
def get_accounts() -> dict:
    return {"data": {"accounts": [{"account_number": "MOCK-MAIN-0", "rhs_account_number": "5550000", "agentic_allowed": False},
                                  {"account_number": ACCOUNT, "rhs_account_number": RHS, "agentic_allowed": True}]}}


@app.tool()
def get_portfolio(account_number: str) -> dict:
    bp = os.environ.get("MOCK_BUYING_POWER", "500.00")
    return {"data": {"buying_power": {"amount": bp}, "total_equity": "500.00"}}


@app.tool()
def get_equity_positions(account_number: str) -> dict:
    return {"data": {"positions": [{"symbol": s, **p} for s, p in _state()["positions"].items()]}}


@app.tool()
def get_equity_news(symbol: str) -> dict:
    return {"data": {"articles": [{"title": f"{symbol} moves with the broader market"}]}}


@app.tool()
def get_equity_historicals(symbols: list[str], start_time: str, interval: str = "5minute") -> dict:
    low = os.environ.get("MOCK_DAY_LOW")
    out = []
    for s in symbols:
        base = PRICES.get(s.upper(), 100.0)
        bars = [{"begins_at": f"t{i}", "close_price": f"{base * (1 - i * 0.001):.2f}",
                 "high_price": f"{base * (1.01 - i * 0.001):.2f}",
                 "low_price": low or f"{base * (0.999 - i * 0.001):.2f}", "interpolated": False} for i in range(30)]
        out.append({"symbol": s.upper(), "bars": bars})
    return {"data": {"results": out}}


@app.tool()
def get_scanner_filter_specs() -> dict:
    specs = [{"filter_type": "FILTER_TYPE_VOLUME", "display_name": "Volume", "supported_predicates": ["PREDICATE_GREATER_THAN"]}]
    if not os.environ.get("MOCK_NO_ASSET_SPEC"):
        specs.append({"filter_type": "FILTER_TYPE_ASSET_TYPE", "display_name": "Asset type",
                      "supported_predicates": ["PREDICATE_IN_LIST"], "options": [{"value": "STOCK"}, {"value": "ETF"}]})
    return {"data": {"filter_specs": specs}}


@app.tool()
def preview_scan(filters: list[dict], columns: list[dict] | None = None) -> dict:
    if os.environ.get("MOCK_SCAN_FAIL"):
        raise ToolError("market-data provider rejected the expression")
    stocks_only = any(f.get("filter_type") == "FILTER_TYPE_ASSET_TYPE" and f.get("values") == ["STOCK"] for f in filters)
    # like the real server: every row says EQUITY, stock or fund. Only the asset-type filter or the name tells them apart
    rows = [("BIG", "Big Industries Inc", False, "8000000"), ("CHP", "Cheap Holdings Corp", False, "5200000"),
            ("ETFX", "Example Select Sector SPDR ETF", True, "9000000"), ("MID", "Middle Co", False, "1900000"),
            ("LEV", "T-Rex 2X Long XYZ Daily Target ETF", True, "7000000"), ("bad!", "Bad Row", False, "1")]
    return {"data": {"title": "Preview", "total": len(rows), "rows": [
        {"ticker": t, "instrument_id": f"i-{t}", "type": "EQUITY",
         "cells": [{"display_name": "Name", "value": name}, {"display_name": "Volume", "value": v},
                   {"display_name": "Price", "value": str(PRICES.get(t, 100.0))}]}
        for t, name, fund, v in rows if not (fund and stocks_only)]}}


def _reserved(orders: dict, key: str, value: str) -> float:
    """Shares or coins already promised to open sell orders (what is still unfilled on them)."""
    return sum(float(o.get("quantity") or 0) - float(o.get("cumulative_quantity") or 0) for o in orders.values()
               if o.get(key) == value and o["side"] == "sell" and o["state"] in OPEN)


def _dedupe(orders: dict, ref_id: str) -> dict | None:
    return next((o for o in orders.values() if ref_id and o.get("ref_id") == ref_id), None)


@app.tool()
def get_equity_orders(account_number: str, symbol: str = "", created_at_gte: str = "", state: str = "",
                      order_id: str = "", placed_agent: str = "") -> dict:
    rows = [o for o in _state()["orders"].values() if (not symbol or o["symbol"] == symbol)
            and (not state or o["state"] == state) and (not order_id or o["id"] == order_id)]
    return {"data": {"orders": rows}}


@app.tool()
def cancel_equity_order(account_number: str, order_id: str) -> dict:
    st = _state()
    if os.environ.get("MOCK_CANCEL_FAIL"):
        raise ToolError("cancel could not be processed right now")
    o = st["orders"].get(order_id)
    if os.environ.get("MOCK_FILL_ON_CANCEL") and o and o["side"] == "buy" and o["state"] in OPEN:
        price = o["limit_price"] or _quote(o["symbol"])["quote"]["ask_price"]
        o.update(state="filled", cumulative_quantity=o["quantity"], average_price=f"{float(price):.4f}")
        st["positions"][o["symbol"]] = {"quantity": o["quantity"], "average_buy_price": f"{float(price):.4f}"}
        _save(st)
        raise ToolError("order already filled")
    if order_id not in st["orders"] or st["orders"][order_id]["state"] not in OPEN:
        raise ValueError("order is not open")
    st["orders"][order_id]["state"] = "cancelled"
    _save(st)
    return {"data": {"order": st["orders"][order_id]}}


@app.tool()
def review_equity_order(account_number: str, side: str, symbol: str, type: str, quantity: str = "",
                        dollar_amount: str = "", limit_price: str = "", stop_price: str = "",
                        time_in_force: str = "gfd", market_hours: str = "regular_hours") -> dict:
    alerts = [{"code": "mock_warning", "detail": "pretend buying power warning"}] if os.environ.get("MOCK_ALERT") else []
    return {"data": {"quote": _quote(symbol)["quote"], "alerts": alerts}}


@app.tool()
def place_equity_order(account_number: str, side: str, symbol: str, type: str, ref_id: str, quantity: str = "",
                       dollar_amount: str = "", limit_price: str = "", stop_price: str = "",
                       time_in_force: str = "gfd", market_hours: str = "regular_hours") -> dict:
    if account_number != ACCOUNT:
        raise ValueError("account is not agentic_allowed")
    st = _state()
    if (old := _dedupe(st["orders"], ref_id)) is not None:        # idempotent: same ref_id, same order
        return {"data": {"order": old}}
    order = {"id": str(uuid.uuid4()), "ref_id": ref_id, "symbol": symbol, "side": side, "type": type,
             "quantity": quantity, "limit_price": limit_price or None, "stop_price": stop_price or None,
             "time_in_force": time_in_force, "state": "confirmed", "cumulative_quantity": "0",
             "average_price": None, "created_at": _now()}
    q = float(quantity or 0)
    pos = st["positions"].get(symbol)
    if side == "sell":
        held = float(pos["quantity"]) if pos else 0.0
        if q > held - _reserved(st["orders"], "symbol", symbol) + 1e-9:
            raise ValueError("not enough shares: they are held by an open order")
    now = _quote(symbol)["quote"]
    marketable = not limit_price or (float(limit_price) >= float(now["ask_price"]) - 1e-9 if side == "buy"
                                     else float(limit_price) <= float(now["bid_price"]) + 1e-9)   # else it waits, like real
    fills = marketable and ((side == "buy" and not os.environ.get("MOCK_NO_FILL")) or
                            (side == "sell" and type in ("limit", "market") and not os.environ.get("MOCK_NO_SELL_FILL")))
    if fills:                                                     # stop orders simply stay open
        price = float(limit_price or now["ask_price" if side == "buy" else "bid_price"])
        part = float(os.environ.get("MOCK_PARTIAL_FILL") or 0) if side == "sell" and type == "limit" else 0
        done = min(part, q) if part else q
        if side == "buy":
            old_q = float(pos["quantity"]) if pos else 0.0
            old_cost = old_q * float(pos["average_buy_price"]) if pos else 0.0
            st["positions"][symbol] = {"quantity": f"{old_q + q:g}", "average_buy_price": f"{(old_cost + q * price) / (old_q + q):.4f}"}
        else:
            left = float(pos["quantity"]) - done
            if left <= 1e-9:
                st["positions"].pop(symbol)
            else:
                pos["quantity"] = f"{left:g}"
        order.update(state="filled" if done >= q else "partially_filled", cumulative_quantity=f"{done:g}",
                     average_price=f"{price:.4f}")
    st["orders"][order["id"]] = order
    _save(st)
    return {"data": {"order": order}}


# ------------------------------------------------------------------ crypto ---
def _coin(sym: str) -> str:
    s = sym.upper().replace("-", "")
    return s[:-3] if s.endswith("USD") and s[:-3] in COINS else s


def _cquote(code: str) -> dict:
    base, _, _, spread = COINS[code]
    mark = base * (1 + _pct("MOCK_CRYPTO_SHIFT_PCT"))
    half = mark * spread / 200
    now = datetime.now(ZoneInfo("America/New_York")).isoformat()          # the real tool renders US Eastern
    return {"symbol": f"{code}USD", "bid_price": f"{mark - half:.8f}", "ask_price": f"{mark + half:.8f}",
            "mark_price": f"{mark:.8f}", "open_price": f"{base:.8f}", "updated_at": now}


@app.tool()
def get_currency_pairs(cursor: str = "", limit: int = 25) -> dict:
    pairs = [{"id": f"p-{c}", "symbol": f"{c}-USD", "asset_currency": {"code": c}, "quote_currency": {"code": "USD"},
              "min_order_quantity_increment": qs, "min_order_price_increment": ps, "tradability": "tradable"}
             for c, (_, qs, ps, _) in COINS.items()]
    pairs.append({"id": "p-HALT", "symbol": "HALT-USD", "asset_currency": {"code": "HALT"}, "tradability": "untradable",
                  "min_order_quantity_increment": "1", "min_order_price_increment": "0.01"})
    return {"data": {"results": pairs[:limit]}}


@app.tool()
def get_crypto_quotes(symbols: list[str], rhs_account_number: str = "", timezone: str = "") -> dict:
    return {"data": {"results": [_cquote(_coin(s)) for s in symbols if _coin(s) in COINS]}}


@app.tool()
def get_crypto_positions(rhs_account_number: str, cursor: str = "") -> dict:
    rows = [{"asset_code": c, "quantity": p["quantity"], "quantity_available": p["quantity"],
             "cost_bases": [{"direct_cost_basis": p["cost"], "direct_quantity": p["quantity"]}]}
            for c, p in _state()["crypto"].items()]
    return {"data": {"results": rows}}


@app.tool()
def get_crypto_account_onboarding_info() -> dict:
    return {"data": {"url": "https://robinhood.com/us/en/crypto/"}}


@app.tool()
def preview_crypto_order(rhs_account_number: str, symbol: str, side: str, type: str, quantity: str = "",
                         dollar_amount: str = "", limit_price: str = "", stop_price: str = "", time_in_force: str = "",
                         tax_lots: list | None = None) -> dict:
    errors = [{"code": "mock_warning", "detail": "pretend crypto warning"}] if os.environ.get("MOCK_ALERT") else []
    return {"data": {"order": {"symbol": symbol, "side": side, "type": type, "estimated_cost": dollar_amount},
                     "errors": errors}}


@app.tool()
def place_crypto_order(rhs_account_number: str, symbol: str, side: str, type: str, ref_id: str = "", quantity: str = "",
                       dollar_amount: str = "", limit_price: str = "", stop_price: str = "", time_in_force: str = "",
                       tax_lots: list | None = None) -> dict:
    if rhs_account_number != RHS:
        raise ValueError("not an agentic-enabled crypto account")
    if time_in_force and type in ("market", "limit") and time_in_force != "gtc":
        raise ValueError("market and limit crypto orders only take gtc")
    st = _state()
    if (old := _dedupe(st["crypto_orders"], ref_id)) is not None:
        return {"data": {"order": old}}
    code = _coin(symbol)
    _, qs, _, _ = COINS[code]
    step = float(qs)
    q = float(quantity) if quantity else math.floor(float(dollar_amount) / float(limit_price) / step) * step
    order = {"id": str(uuid.uuid4()), "ref_id": ref_id, "asset": code, "side": side, "type": type,
             "quantity": f"{q:.8f}", "limit_price": limit_price or None, "stop_price": stop_price or None,
             "time_in_force": time_in_force or "gtc", "state": "confirmed", "cumulative_quantity": "0",
             "average_price": None, "created_at": _now()}
    pos = st["crypto"].get(code)
    if side == "sell" and q > (float(pos["quantity"]) if pos else 0.0) - _reserved(st["crypto_orders"], "asset", code) + 1e-12:
        raise ValueError("insufficient quantity: held by an open order")
    now = _cquote(code)
    marketable = not limit_price or (float(limit_price) >= float(now["ask_price"]) - 1e-9 if side == "buy"
                                     else float(limit_price) <= float(now["bid_price"]) + 1e-9)   # a target above waits
    fills = marketable and ((side == "buy" and not os.environ.get("MOCK_NO_FILL")) or
                            (side == "sell" and type == "limit" and not os.environ.get("MOCK_NO_SELL_FILL")))
    if fills:
        price = float(limit_price)
        if side == "buy":
            old_q, old_c = (float(pos["quantity"]), float(pos["cost"])) if pos else (0.0, 0.0)
            st["crypto"][code] = {"quantity": f"{old_q + q:.8f}", "cost": f"{old_c + q * price:.2f}"}
        else:
            left = float(pos["quantity"]) - q
            if left <= 1e-12:
                st["crypto"].pop(code)
            else:
                st["crypto"][code] = {"quantity": f"{left:.8f}", "cost": f"{float(pos['cost']) * left / float(pos['quantity']):.2f}"}
        order.update(state="filled", cumulative_quantity=f"{q:.8f}", average_price=f"{price:.8f}")
    st["crypto_orders"][order["id"]] = order
    _save(st)
    return {"data": {"order": order}}


@app.tool()
def cancel_crypto_order(rhs_account_number: str, order_id: str) -> dict:
    st = _state()
    if os.environ.get("MOCK_CANCEL_FAIL"):
        raise ToolError("cancel could not be processed right now")
    if order_id not in st["crypto_orders"] or st["crypto_orders"][order_id]["state"] not in OPEN:
        raise ValueError("order is not open")
    st["crypto_orders"][order_id]["state"] = "canceled"
    _save(st)
    return {"data": {"order": st["crypto_orders"][order_id]}}


@app.tool()
def get_crypto_orders(rhs_account_number: str, order_id: str = "", symbol: str = "", state: str = "",
                      state_group: str = "", side: str = "", created_at_gte: str = "", updated_at_gte: str = "",
                      cursor: str = "") -> dict:
    rows = [o for o in _state()["crypto_orders"].values() if (not order_id or o["id"] == order_id)
            and (not symbol or o["asset"] == _coin(symbol)) and (not state or o["state"] == state)
            and (state_group != "open" or o["state"] in OPEN)]
    return {"data": {"results": rows}}


if __name__ == "__main__":
    app.run("stdio")
