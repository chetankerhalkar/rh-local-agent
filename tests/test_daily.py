"""Daily sessions: morning (scan -> model ideas -> you pick -> buy -> stop) and evening (sell up/flat, ask when down).
Against the mock Robinhood server over the real MCP protocol, with a stub local model and scripted answers."""
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agent import daily, guardrails, journal, pipeline, strategy
from agent.models import DayState
from tests.test_end_to_end import events

IDEAS: dict[str, tuple[str, float]] = {}
SEEN: list[dict] = []                                     # every item the stub model was asked about


class Model(BaseHTTPRequestHandler):
    """A stub local model: answers each item from IDEAS, anything else is a skip."""
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        items = json.loads(body["messages"][-1]["content"])["items"]
        SEEN.extend(items)
        rows = [{"symbol": i["symbol"], "action": IDEAS.get(i["symbol"], ("skip", 0.2))[0],
                 "confidence": IDEAS.get(i["symbol"], ("skip", 0.2))[1], "reason": f"test idea for {i['symbol']}"} for i in items]
        out = json.dumps({"choices": [{"message": {"role": "assistant", "content": json.dumps({"decisions": rows})}}],
                          "usage": {"total_tokens": 50}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def make_cfg(port, mode="approve"):
    return {"mode": mode, "universe": [],
            "robinhood": {"transport": "mock", "account_number": "MOCK-AGENTIC-1"},
            "llm": {"enabled": True, "base_url": f"http://127.0.0.1:{port}/v1", "model": "stub", "timeout_seconds": 5,
                    "fail_closed": True, "use_news": True},
            "strategy": {"name": "llm_decider", "order_style": "limit_whole_shares", "order_usd": 50,
                         "stop_loss_pct": 1.0, "take_profit_pct": 1.5},
            "protect": {"broker_stop": True, "fill_wait_seconds": 0.2, "poll_seconds": 0.05},
            "guardrails": {"max_order_usd": 50, "max_daily_buy_usd": 150, "max_trades_per_day": 4, "max_position_usd": 50,
                           "max_open_positions": 3, "daily_loss_stop_pct": 2.0, "allowed_order_types": ["limit"],
                           "regular_hours_only": True, "no_new_buys_first_minutes": 15, "no_new_buys_last_minutes": 20,
                           "max_quote_age_seconds": 90, "max_spread_pct": 0.30, "crypto_max_spread_pct": 1.0,
                           "max_limit_slippage_pct": 0.20, "block_on_any_review_alert": True},
            "daily": {"max_suggestions": 5, "min_confidence": 0.6, "analysis_batch": 3, "stock_stop_loss_pct": 1.0,
                      "crypto_stop_loss_pct": 3.0, "when_down": "ask", "confirm_sells": True, "sell_slack_pct": 0.10},
            "scan": {"stocks": {"enabled": True, "max_candidates": 8},
                     "crypto": {"enabled": True, "max_candidates": 3, "exclude": ["USDC"], "history": "none"}}}


@pytest.fixture()
def world(tmp_path, monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), Model)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    for mod in (pipeline, journal, __import__("agent.config", fromlist=["x"])):
        monkeypatch.setattr(mod, "ROOT", tmp_path, raising=False)
    monkeypatch.setattr(guardrails, "market_open", lambda now: True)          # tests must pass at any hour
    monkeypatch.setattr(guardrails, "minutes_since_open", lambda now: 120)
    monkeypatch.setattr(guardrails, "minutes_to_close", lambda now: 200)
    monkeypatch.setattr(strategy, "minutes_to_close", lambda now: 200)
    monkeypatch.setenv("MOCK_STATE_FILE", str(tmp_path / "broker.json"))      # one broker across all connections
    monkeypatch.setattr(daily, "console_ok", lambda: True)
    queue: list[str] = []

    def answer(prompt: str) -> str:
        assert queue, f"unexpected question: {prompt}"
        return queue.pop(0)
    monkeypatch.setattr(daily, "ask", answer)
    SEEN.clear()
    IDEAS.clear()
    IDEAS.update({"CHP": ("buy", 0.9), "ETH-USD": ("buy", 0.8), "MID": ("buy", 0.5), "BIG": ("buy", 0.95),
                  "BTC-USD": ("skip", 0.3)})
    yield make_cfg(srv.server_port), tmp_path, queue
    srv.shutdown()


def morning(c):
    asyncio.run(daily.morning(c))


def evening(c):
    asyncio.run(daily.evening(c))


def broker(tmp):
    return json.loads((tmp / "broker.json").read_text()) if (tmp / "broker.json").exists() else {}


def book(tmp):
    p = tmp / "state" / "agent_book.json"
    return json.loads(p.read_text()) if p.exists() else {}


def named(tmp, name):
    return [e for e in events(tmp) if e["event"] == name]


def test_morning_buys_only_what_you_pick_and_protects_it(world):
    c, tmp, queue = world
    queue += ["1 2", "yes", "yes"]           # the two ideas shown: CHP (0.9), ETH-USD (0.8). BIG is over $50 a share
    morning(c)
    assert queue == []
    b = broker(tmp)
    assert b["positions"]["CHP"]["quantity"] == "4"                                     # floor(50 / 10.01)
    assert b["crypto"]["ETH"]["quantity"] == "0.02080800"                               # $49.99 at 2402.40
    stops = [o for o in b["orders"].values() if o["type"] == "stop_market"]
    cstops = [o for o in b["crypto_orders"].values() if o["type"] == "stop_loss"]
    assert [(o["state"], o["time_in_force"], o["stop_price"]) for o in stops] == [("confirmed", "gtc", "9.90")]
    assert [(o["state"], o["time_in_force"], o["stop_price"]) for o in cstops] == [("confirmed", "gtc", "2330.32")]
    k = book(tmp)
    assert set(k) == {"CHP", "ETH-USD"} and all(e["stop_order_id"] for e in k.values())
    assert k["CHP"]["cost"] == "40.04" and k["ETH-USD"]["cost"] == "49.99"
    for e in named(tmp, "order_placed"):                                                # nothing over $50, ever
        o = e["order"]
        assert o["side"] == "sell" or float(o.get("dollar_amount") or float(o["quantity"]) * float(o["limit_price"])) <= 50
    ideas = named(tmp, "ideas")[0]
    assert ideas["shown"] == ["CHP", "ETH-USD"] and any(d.startswith("BIG") for d in ideas["dropped"])
    scans = {e["asset"]: e["kept"] for e in named(tmp, "scan")}
    assert scans["stock"] == ["BIG", "CHP", "MID"]              # most traded first; the fund and the bad ticker are out
    assert scans["crypto"] == ["BTC-USD", "ETH-USD", "DOGE-USD"]  # stablecoin excluded, 3% spread and halted pair out


def test_enter_means_nothing_is_bought(world):
    c, tmp, queue = world
    queue += [""]
    morning(c)
    assert not broker(tmp).get("orders") and not broker(tmp).get("crypto_orders")


def test_no_at_the_confirmation_skips_the_order(world):
    c, tmp, queue = world
    queue += ["1", "no"]
    morning(c)
    assert not broker(tmp).get("orders") and named(tmp, "not_approved")


def test_model_offline_means_no_ideas_and_no_orders(world):
    c, tmp, queue = world
    c["llm"]["base_url"] = "http://127.0.0.1:9/v1"                                      # nothing listens there
    morning(c)
    assert named(tmp, "analysis_failed") and not broker(tmp).get("orders")


def test_paper_mode_is_a_dry_run(world):
    c, tmp, queue = world
    c["mode"] = "paper"
    queue += ["1"]
    morning(c)
    assert named(tmp, "dry_run") and not broker(tmp).get("orders") and book(tmp) == {}


def test_no_console_no_session(world, monkeypatch):
    c, tmp, _ = world
    monkeypatch.setattr(daily, "console_ok", lambda: False)
    morning(c)
    assert not named(tmp, "scan")


def test_evening_sells_what_is_up_and_asks_about_what_is_down(world, monkeypatch):
    c, tmp, queue = world
    queue += ["1 2", "yes", "yes"]
    morning(c)
    monkeypatch.setenv("MOCK_SHIFT_PCT", "2")                                           # CHP up 2%
    monkeypatch.setenv("MOCK_CRYPTO_SHIFT_PCT", "-2")                                   # ETH down 2%
    queue += ["k", "yes"]                                                               # keep ETH, confirm the CHP sell
    evening(c)
    assert queue == []
    b = broker(tmp)
    assert "CHP" not in b["positions"] and "ETH" in b["crypto"]
    assert [o["state"] for o in b["orders"].values() if o["type"] == "stop_market"] == ["cancelled"]
    assert [o["state"] for o in b["crypto_orders"].values() if o["type"] == "stop_loss"] == ["confirmed"]
    assert set(book(tmp)) == {"ETH-USD"}
    sold = named(tmp, "sold")
    assert [(s["symbol"], s["pnl"]) for s in sold] == [("CHP", "0.68")]                 # 4 x 10.18 - 40.04


def test_evening_down_and_you_choose_to_sell(world, monkeypatch):
    c, tmp, queue = world
    queue += ["2", "yes"]                                                               # ETH only
    morning(c)
    monkeypatch.setenv("MOCK_CRYPTO_SHIFT_PCT", "-2")
    queue += ["s", "yes"]
    evening(c)
    assert "ETH" not in broker(tmp)["crypto"] and book(tmp) == {}
    assert float(named(tmp, "sold")[0]["pnl"]) < 0


def test_a_sell_that_does_not_fill_gets_its_stop_back(world, monkeypatch):
    c, tmp, queue = world
    queue += ["1", "yes"]
    morning(c)
    monkeypatch.setenv("MOCK_SHIFT_PCT", "2")
    monkeypatch.setenv("MOCK_NO_SELL_FILL", "1")
    queue += ["yes"]
    evening(c)
    b = broker(tmp)
    assert b["positions"]["CHP"]["quantity"] == "4" and named(tmp, "STILL_HOLDING")
    states = sorted((o["type"], o["state"]) for o in b["orders"].values() if o["side"] == "sell")
    assert states == [("limit", "cancelled"), ("stop_market", "cancelled"), ("stop_market", "confirmed")]
    assert book(tmp)["CHP"]["stop_order_id"]


def test_stock_market_closed_in_the_evening_keeps_stocks_and_still_sells_crypto(world, monkeypatch):
    c, tmp, queue = world
    queue += ["1 2", "yes", "yes"]
    morning(c)
    monkeypatch.setattr(guardrails, "market_open", lambda now: False)
    monkeypatch.setenv("MOCK_CRYPTO_SHIFT_PCT", "2")
    queue += ["yes"]
    evening(c)
    b = broker(tmp)
    assert "CHP" in b["positions"] and "ETH" not in b["crypto"]
    assert [o["state"] for o in b["orders"].values() if o["type"] == "stop_market"] == ["confirmed"]
    assert set(book(tmp)) == {"CHP"}


def test_a_stop_that_already_sold_is_reported_not_sold_again(world):
    c, tmp, queue = world
    queue += ["1", "yes"]
    morning(c)
    b = broker(tmp)
    stop = next(o for o in b["orders"].values() if o["type"] == "stop_market")
    stop.update(state="filled", cumulative_quantity="4", average_price="9.90")         # the price fell through the stop
    b["positions"].pop("CHP")
    (tmp / "broker.json").write_text(json.dumps(b))
    evening(c)
    closed = named(tmp, "closed_by_stop")
    assert [(e["symbol"], e["pnl"]) for e in closed] == [("CHP", "-0.44")] and book(tmp) == {}
    assert not [o for o in broker(tmp)["orders"].values() if o["side"] == "sell" and o["type"] == "limit"]


def test_a_buy_cut_off_mid_send_is_found_and_adopted(world):
    c, tmp, queue = world
    ref = "11111111-2222-3333-4444-555555555555"
    (tmp / "broker.json").write_text(json.dumps({
        "positions": {"CHP": {"quantity": "4", "average_buy_price": "10.0100"}}, "crypto": {}, "crypto_orders": {},
        "orders": {"o1": {"id": "o1", "ref_id": ref, "symbol": "CHP", "side": "buy", "type": "limit", "quantity": "4",
                          "state": "filled", "cumulative_quantity": "4", "average_price": "10.0100"}}}))
    day = DayState(f"{guardrails.now_et():%Y-%m-%d}", buy_usd_total="40.04",
                   trades=[{"symbol": "CHP", "side": "buy", "kind": "entry", "asset": "stock", "ref_id": ref,
                            "status": "unknown", "usd": "40.04"}])
    journal.save_day(day)
    queue += ["k", "yes"]                          # CHP is a cent under cost: keep it, then yes to a stop order for it
    evening(c)
    assert named(tmp, "order_found") and named(tmp, "adopted")
    assert book(tmp)["CHP"]["stop_order_id"]
    assert [o["state"] for o in broker(tmp)["orders"].values() if o["type"] == "stop_market"] == ["confirmed"]


def test_crypto_off_means_no_new_coins_even_while_one_is_held(world):
    c, tmp, queue = world
    queue += ["2", "yes"]                                   # first morning: buy ETH-USD, the second idea
    morning(c)
    assert book(tmp)["ETH-USD"]["stop_order_id"]
    c["scan"]["crypto"]["enabled"] = False
    SEEN.clear()
    scans = len(named(tmp, "scan"))
    queue += [""]                                           # second morning: buy nothing
    morning(c)
    assert queue == []
    assert not [e for e in named(tmp, "scan")[scans:] if e["asset"] == "crypto"]
    assert SEEN and not [i for i in SEEN if daily.is_crypto(i["symbol"])]     # the model never saw a coin
    assert book(tmp)["ETH-USD"]["stop_order_id"]                              # the held coin is still managed


def test_scan_failure_falls_back_to_your_universe(world, monkeypatch):
    c, tmp, queue = world
    monkeypatch.setenv("MOCK_SCAN_FAIL", "1")
    c["universe"] = ["MID"]
    c["scan"]["crypto"]["enabled"] = False
    IDEAS["MID"] = ("buy", 0.9)
    queue += ["1", "yes"]
    morning(c)
    assert named(tmp, "scan_failed") and broker(tmp)["positions"]["MID"]["quantity"] == "2"   # floor(50 / 24.51)


def test_no_room_means_no_scan_and_no_model(world):
    c, tmp, queue = world
    journal.save_day(DayState(f"{guardrails.now_et():%Y-%m-%d}", buy_usd_total="150"))          # budget used up
    morning(c)
    assert not named(tmp, "scan") and not named(tmp, "analysis")


def test_a_dropped_connection_never_doubles_an_order(world, monkeypatch):
    from agent.rh import Robinhood
    c, tmp, queue = world
    real, sent = Robinhood.place_order, []

    async def flaky(self, args, ref_id):
        result = await real(self, args, ref_id)            # the order reaches Robinhood...
        sent.append(ref_id)
        if len(sent) == 1:
            raise ConnectionError("the line dropped before the answer came back")      # ...but the answer is lost
        return result
    monkeypatch.setattr(Robinhood, "place_order", flaky)
    queue += ["1", "yes"]
    morning(c)
    buys = [o for o in broker(tmp)["orders"].values() if o["side"] == "buy"]
    assert len(buys) == 1 and sent[0] == sent[1] == buys[0]["ref_id"]      # retried with the same ref_id: one order
    assert named(tmp, "network_problem") and book(tmp)["CHP"]["stop_order_id"]


# ---- the independent review's findings, each pinned by a test --------------------------------------------
def test_a_blocked_sell_leaves_the_stop_alone(world, monkeypatch):
    c, tmp, queue = world
    queue += ["1", "yes"]
    morning(c)
    monkeypatch.setenv("MOCK_SHIFT_PCT", "2")
    monkeypatch.setenv("MOCK_HALT", "CHP")                 # the guardrails say no to the sell...
    queue += ["yes"]
    evening(c)
    b = broker(tmp)
    assert [o["state"] for o in b["orders"].values() if o["type"] == "stop_market"] == ["confirmed"]   # ...so it keeps its stop
    assert not [o for o in b["orders"].values() if o["side"] == "sell" and o["type"] == "limit"]
    assert book(tmp)["CHP"]["stop_order_id"] and named(tmp, "sell_blocked")


def test_an_unconfirmed_cancel_is_never_written_off(world, monkeypatch):
    c, tmp, queue = world
    monkeypatch.setenv("MOCK_NO_FILL", "1")
    monkeypatch.setenv("MOCK_CANCEL_FAIL", "1")
    queue += ["1", "yes"]
    morning(c)
    day = journal.load_day(f"{guardrails.now_et():%Y-%m-%d}")
    assert named(tmp, "UNCONFIRMED_CANCEL") and book(tmp) == {} and day.buy_usd_total == "40.04"
    assert [t["status"] for t in day.trades if t["side"] == "buy"] == ["unknown"]
    b = broker(tmp)                                         # later, the order fills after all
    buy = next(o for o in b["orders"].values() if o["side"] == "buy")
    buy.update(state="filled", cumulative_quantity="4", average_price="10.0100")
    b["positions"]["CHP"] = {"quantity": "4", "average_buy_price": "10.0100"}
    (tmp / "broker.json").write_text(json.dumps(b))
    monkeypatch.delenv("MOCK_NO_FILL")
    monkeypatch.delenv("MOCK_CANCEL_FAIL")
    queue += ["k", "yes"]                                   # a cent under cost: keep it, and yes to a stop order
    evening(c)
    assert named(tmp, "adopted") and book(tmp)["CHP"]["stop_order_id"]


def test_a_dropped_connection_while_restoring_a_stop_does_not_crash(world, monkeypatch):
    from agent.rh import Robinhood
    c, tmp, queue = world
    queue += ["1", "yes"]
    morning(c)
    real, stops = Robinhood.place_order, []

    async def flaky(self, args, ref_id):
        result = await real(self, args, ref_id)
        if args["type"] == "stop_market":
            stops.append(ref_id)
            if len(stops) == 1:
                raise ConnectionError("the line dropped while the stop order went out")
        return result
    monkeypatch.setattr(Robinhood, "place_order", flaky)
    monkeypatch.setenv("MOCK_SHIFT_PCT", "2")
    monkeypatch.setenv("MOCK_NO_SELL_FILL", "1")            # the sell does not fill, so the stop has to go back
    queue += ["yes"]
    evening(c)
    b = broker(tmp)
    assert len(stops) == 2 and stops[0] == stops[1]         # the retry re-sent the same stop, not a second one
    assert [o["state"] for o in b["orders"].values() if o["type"] == "stop_market"] == ["cancelled", "confirmed"]
    assert book(tmp)["CHP"]["stop_order_id"] and named(tmp, "network_problem")


def test_orders_you_placed_yourself_are_never_cancelled(world, monkeypatch):
    c, tmp, queue = world
    queue += ["2", "yes"]                                   # the agent buys ETH
    morning(c)
    b = broker(tmp)
    b["crypto_orders"]["mine"] = {"id": "mine", "ref_id": "", "asset": "ETH", "side": "buy", "type": "limit",
                                  "quantity": "0.01", "limit_price": "2000", "state": "confirmed"}   # yours, from the app
    (tmp / "broker.json").write_text(json.dumps(b))
    monkeypatch.setenv("MOCK_CRYPTO_SHIFT_PCT", "2")
    queue += ["yes"]
    evening(c)
    b = broker(tmp)
    assert "ETH" not in b["crypto"] and b["crypto_orders"]["mine"]["state"] == "confirmed"


def test_a_booked_buy_is_not_adopted_again_when_you_buy_the_same_stock(world, monkeypatch):
    c, tmp, queue = world
    queue += ["1", "yes"]
    morning(c)
    monkeypatch.setenv("MOCK_SHIFT_PCT", "2")
    queue += ["yes"]
    evening(c)                                              # the agent sells its CHP
    b = broker(tmp)
    b["positions"]["CHP"] = {"quantity": "3", "average_buy_price": "10.2000"}     # then you buy CHP in the app
    (tmp / "broker.json").write_text(json.dumps(b))
    evening(c)
    assert book(tmp) == {} and not named(tmp, "adopted")


def test_yesterdays_cut_off_buy_is_checked_today(world):
    c, tmp, queue = world
    ref = "22222222-3333-4444-5555-666666666666"
    yesterday = guardrails.now_et() - __import__("datetime").timedelta(days=1)
    (tmp / "broker.json").write_text(json.dumps({
        "positions": {"CHP": {"quantity": "4", "average_buy_price": "10.0100"}}, "crypto": {}, "crypto_orders": {},
        "orders": {"o9": {"id": "o9", "ref_id": ref, "symbol": "CHP", "side": "buy", "type": "limit", "quantity": "4",
                          "state": "filled", "cumulative_quantity": "4", "average_price": "10.0100"}}}))
    journal.save_day(DayState(f"{yesterday:%Y-%m-%d}", buy_usd_total="40.04", trades=[
        {"symbol": "CHP", "side": "buy", "kind": "entry", "asset": "stock", "ref_id": ref, "status": "pending",
         "usd": "40.04", "ts": yesterday.isoformat()}]))
    queue += ["k", "yes"]
    evening(c)
    assert named(tmp, "adopted") and book(tmp)["CHP"]["opened"] == f"{yesterday:%Y-%m-%d}"


def test_paper_mode_never_cancels_a_real_order(world):
    c, tmp, queue = world
    c["mode"] = "paper"
    ref = "33333333-4444-5555-6666-777777777777"
    (tmp / "broker.json").write_text(json.dumps({
        "positions": {}, "crypto": {}, "crypto_orders": {},
        "orders": {"o7": {"id": "o7", "ref_id": ref, "symbol": "CHP", "side": "buy", "type": "limit", "quantity": "4",
                          "state": "confirmed", "cumulative_quantity": "0", "average_price": None}}}))
    journal.save_day(DayState(f"{guardrails.now_et():%Y-%m-%d}", buy_usd_total="40.04", trades=[
        {"symbol": "CHP", "side": "buy", "kind": "entry", "asset": "stock", "ref_id": ref, "status": "placed",
         "order_id": "o7", "usd": "40.04"}]))
    evening(c)
    assert broker(tmp)["orders"]["o7"]["state"] == "confirmed" and not named(tmp, "order_cancelled")


def test_a_refused_resend_is_looked_up_not_written_off(world, monkeypatch):
    from agent.rh import RHError, Robinhood
    c, tmp, queue = world
    real, calls = Robinhood.place_order, []

    async def picky(self, args, ref_id):
        calls.append(ref_id)
        if len(calls) == 1:
            await real(self, args, ref_id)                  # the order reaches Robinhood...
            raise ConnectionError("the answer was lost")
        if len(calls) == 2:
            raise RHError("place_equity_order failed: duplicate ref_id")    # ...and the re-send is refused
        return await real(self, args, ref_id)
    monkeypatch.setattr(Robinhood, "place_order", picky)
    queue += ["1", "yes"]
    morning(c)
    buys = [o for o in broker(tmp)["orders"].values() if o["side"] == "buy"]
    assert len(buys) == 1 and named(tmp, "order_found_after_error")
    assert book(tmp)["CHP"]["stop_order_id"] and not named(tmp, "order_rejected")


# ---- the second review pass --------------------------------------------------------------------------
def today():
    return f"{guardrails.now_et():%Y-%m-%d}"


def test_an_unclear_error_is_never_written_off_as_rejected(world, monkeypatch):
    from agent.rh import RHError, Robinhood
    c, tmp, queue = world

    async def unclear(self, args, ref_id):
        raise RHError("place_equity_order failed: upstream timeout")

    async def no_list(self, asset, since):
        raise RHError("get_equity_orders failed: unavailable")
    monkeypatch.setattr(Robinhood, "place_order", unclear)
    monkeypatch.setattr(Robinhood, "recent_orders", no_list)
    queue += ["1", "yes"]
    morning(c)
    day = journal.load_day(today())
    assert [t["status"] for t in day.trades if t["side"] == "buy"] == ["unknown"] and day.buy_usd_total == "40.04"
    assert named(tmp, "order_unconfirmed") and not named(tmp, "order_rejected")


def test_a_sell_left_open_by_an_earlier_run_is_cancelled_first(world, monkeypatch):
    c, tmp, queue = world
    queue += ["2", "yes"]                                   # the agent buys ETH, with its stop
    morning(c)
    b = broker(tmp)
    next(o for o in b["crypto_orders"].values() if o["type"] == "stop_loss")["state"] = "canceled"
    ref = "44444444-5555-6666-7777-888888888888"            # an earlier evening freed the coins, and its sell
    b["crypto_orders"]["s1"] = {"id": "s1", "ref_id": ref, "asset": "ETH", "side": "sell", "type": "limit",   # never
                                "quantity": "0.020808", "limit_price": "2500", "state": "confirmed",       # confirmed
                                "cumulative_quantity": "0", "average_price": None}
    (tmp / "broker.json").write_text(json.dumps(b))
    day = journal.load_day(today())
    day.trades.append({"symbol": "ETH-USD", "side": "sell", "kind": "exit", "asset": "crypto", "ref_id": ref,
                       "status": "unknown"})
    journal.save_day(day)
    monkeypatch.setenv("MOCK_CRYPTO_SHIFT_PCT", "-2")
    queue += ["k", "yes"]                                   # down tonight: keep it, and yes to a new stop order
    evening(c)
    b = broker(tmp)
    assert b["crypto_orders"]["s1"]["state"] == "canceled" and "ETH" in b["crypto"]
    assert [o["state"] for o in b["crypto_orders"].values() if o["type"] == "stop_loss"] == ["canceled", "confirmed"]


def test_a_cut_short_buy_that_fills_as_it_is_cancelled_is_adopted(world, monkeypatch):
    c, tmp, queue = world
    ref = "55555555-6666-7777-8888-999999999999"
    (tmp / "broker.json").write_text(json.dumps({
        "positions": {}, "crypto": {}, "crypto_orders": {},
        "orders": {"o5": {"id": "o5", "ref_id": ref, "symbol": "CHP", "side": "buy", "type": "limit", "quantity": "4",
                          "limit_price": "10.01", "state": "confirmed", "cumulative_quantity": "0", "average_price": None}}}))
    journal.save_day(DayState(today(), buy_usd_total="40.04", trades=[
        {"symbol": "CHP", "side": "buy", "kind": "entry", "asset": "stock", "ref_id": ref, "status": "placed",
         "order_id": "o5", "usd": "40.04"}]))
    monkeypatch.setenv("MOCK_FILL_ON_CANCEL", "1")
    queue += ["k", "yes"]
    evening(c)
    assert named(tmp, "adopted") and book(tmp)["CHP"]["quantity"] == "4" and book(tmp)["CHP"]["stop_order_id"]


def test_a_partly_filled_sell_does_not_block_selling_the_rest(world, monkeypatch):
    c, tmp, queue = world
    queue += ["1", "yes"]
    morning(c)                                              # 4 CHP
    monkeypatch.setenv("MOCK_SHIFT_PCT", "2")
    monkeypatch.setenv("MOCK_PARTIAL_FILL", "2")            # only 2 of the 4 sell
    queue += ["yes"]
    evening(c)
    assert book(tmp)["CHP"]["quantity"] == "2" and book(tmp)["CHP"]["stop_order_id"]
    monkeypatch.delenv("MOCK_PARTIAL_FILL")
    queue += ["yes"]
    evening(c)                                              # the same day, the other 2 sell
    assert book(tmp) == {} and "CHP" not in broker(tmp)["positions"]


def test_the_morning_says_why_crypto_found_nothing(world):
    c, tmp, queue = world
    c["guardrails"]["crypto_max_spread_pct"] = 0.01          # tighter than any real coin
    c["scan"]["stocks"]["enabled"] = False
    morning(c)
    ev = next(e for e in events(tmp) if e["event"] == "scan" and e["asset"] == "crypto")
    assert ev["kept"] == [] and ev["quoted"] >= 3 and "wider than" in ev["note"] and "BTC-USD" in ev["tightest"]


# ---- first real morning (2026-09-21): funds in the scan, and a thinking model --------------------------
def stock_scan(tmp):
    return next(e for e in events(tmp) if e["event"] == "scan" and e["asset"] == "stock")


def test_funds_are_left_out_by_the_screeners_own_asset_filter(world):
    c, tmp, queue = world
    queue += [""]
    morning(c)
    ev = stock_scan(tmp)
    assert ev["asset_filter"] == "STOCK" and ev["kept"] == ["BIG", "CHP", "MID"] and ev["funds_left_out"] == []


def test_funds_are_still_left_out_by_name_without_that_filter(world, monkeypatch):
    c, tmp, queue = world
    monkeypatch.setenv("MOCK_NO_ASSET_SPEC", "1")
    queue += [""]
    morning(c)
    ev = stock_scan(tmp)
    assert ev["asset_filter"] is None and ev["kept"] == ["BIG", "CHP", "MID"]
    assert sorted(ev["funds_left_out"]) == ["ETFX", "LEV"]                 # "...SPDR ETF" and "...2X ... ETF"


def test_funds_can_be_allowed_on_purpose(world):
    c, tmp, queue = world
    c["scan"]["stocks"]["funds"] = True
    queue += [""]
    morning(c)
    assert {"ETFX", "LEV"} <= set(stock_scan(tmp)["kept"])


# ---- the pick question (2026-09-21: "none" was not understood) ---------------------------------------
@pytest.mark.parametrize("word", ["None", "none", "no", "skip", ""])
def test_none_no_skip_or_enter_all_mean_buy_nothing(world, word):
    c, tmp, queue = world
    IDEAS["ETH-USD"] = ("skip", 0.3)                         # one idea only: CHP
    queue += [word]
    morning(c)
    assert queue == [] and not broker(tmp).get("orders")


def test_one_idea_is_a_plain_yes_or_no(world, capsys):
    c, tmp, queue = world
    IDEAS["ETH-USD"] = ("skip", 0.3)
    queue += ["yes", "yes"]                                  # yes to CHP, then yes to the order
    morning(c)
    assert broker(tmp)["positions"]["CHP"]["quantity"] == "4"
    out = capsys.readouterr().out
    assert "  1. CHP: test idea for CHP" in out              # the full reason, on its own line


def test_all_picks_every_idea(world):
    c, tmp, queue = world
    queue += ["all", "yes", "yes"]
    morning(c)
    b = broker(tmp)
    assert "CHP" in b["positions"] and "ETH" in b["crypto"]


def test_the_crypto_reason_is_repeated_under_the_ideas(world, capsys):
    c, tmp, queue = world
    c["guardrails"]["crypto_max_spread_pct"] = 0.01          # no coin passes
    queue += ["no"]
    morning(c)
    out = capsys.readouterr().out
    assert out.index("Crypto: no coins today.") > out.index("The model's ideas")


# ---- crypto hold style: a target sell waits at Robinhood, no evening sell (2026-09-22) ---------------------
def hold(c):
    c["daily"].update(crypto_style="hold", crypto_take_profit_pct=5.0)
    IDEAS["CHP"] = ("skip", 0.3)                            # ETH only
    return c


def test_crypto_hold_parks_a_target_sell_and_no_stop(world):
    c, tmp, queue = world
    queue += ["yes", "yes"]
    morning(hold(c))
    b = broker(tmp)
    sells = [(o["type"], o["state"], o["limit_price"], o["time_in_force"]) for o in b["crypto_orders"].values() if o["side"] == "sell"]
    assert sells == [("limit", "confirmed", "2522.52", "gtc")]             # 2402.40 x 1.05, waiting at Robinhood
    e = book(tmp)["ETH-USD"]
    assert e["style"] == "hold" and e["target_price"] == "2522.52" and e["stop_level"] == "2330.32"
    assert not e["stop_order_id"]


def test_the_evening_keeps_a_held_coin_between_its_levels(world, monkeypatch):
    c, tmp, queue = world
    queue += ["yes", "yes"]
    morning(hold(c))
    monkeypatch.setenv("MOCK_CRYPTO_SHIFT_PCT", "2")                        # up, but short of the target
    evening(c)                                                              # nothing to confirm: no question asked
    b = broker(tmp)
    assert "ETH" in b["crypto"] and [o["state"] for o in b["crypto_orders"].values() if o["side"] == "sell"] == ["confirmed"]
    assert set(book(tmp)) == {"ETH-USD"}


def test_the_target_sells_it_by_itself(world):
    c, tmp, queue = world
    queue += ["yes", "yes"]
    morning(hold(c))
    b = broker(tmp)                                                         # later the price reaches the target
    target = next(o for o in b["crypto_orders"].values() if o["side"] == "sell")
    target.update(state="filled", cumulative_quantity=target["quantity"], average_price="2522.52")
    b["crypto"].pop("ETH")
    (tmp / "broker.json").write_text(json.dumps(b))
    evening(c)
    closed = named(tmp, "closed_by_target")
    assert [(e["symbol"], e["pnl"]) for e in closed] == [("ETH-USD", "2.50")] and book(tmp) == {}


def test_under_its_stop_level_the_evening_offers_to_sell(world, monkeypatch):
    c, tmp, queue = world
    queue += ["yes", "yes"]
    morning(hold(c))
    monkeypatch.setenv("MOCK_CRYPTO_SHIFT_PCT", "-4")                       # under the 2330.32 stop level
    queue += ["yes"]
    evening(c)
    b = broker(tmp)
    assert "ETH" not in b["crypto"] and book(tmp) == {}
    assert [o["state"] for o in b["crypto_orders"].values() if o["type"] == "limit" and o["side"] == "sell"
            and o["limit_price"] == "2522.52"] == ["canceled"]              # the target was cancelled first


def test_the_model_judges_each_item_by_how_it_will_be_held(world):
    c, tmp, queue = world
    queue += [""]
    morning(hold(c))
    plans = {i["type"]: i["plan"] for i in SEEN}
    assert plans == {"stock": "sell this evening", "crypto": "hold until 5.0% up, sell if 3.0% down first"}


def test_buying_power_is_counted_in_the_room_for_buys(world, monkeypatch, capsys):
    c, tmp, queue = world
    monkeypatch.setenv("MOCK_BUYING_POWER", "60.00")        # enough for one $50 order, not two
    queue += ["1 2", "yes"]
    morning(c)
    out = capsys.readouterr().out
    assert "room for 1 buy." in out and "Taking the first 1" in out
    b = broker(tmp)
    assert "CHP" in b["positions"] and not b["crypto"]


def test_no_cash_left_points_to_selling_what_is_held(world, monkeypatch, capsys):
    c, tmp, queue = world
    queue += ["1", "yes"]
    morning(c)                                              # the cash is now in CHP
    monkeypatch.setenv("MOCK_BUYING_POWER", "10.00")
    morning(c)
    out = capsys.readouterr().out
    assert "buying power left for 0" in out and "python -m agent evening" in out


def test_laya_ideas_come_with_their_data_and_a_written_second_opinion(world, capsys):
    """Laya gives a score, not a reason (2026-09-28: 'why did I buy BN?'). Each idea now shows its data,
    and your local model's own buy/skip in words. The opinion never changes which ideas are shown."""
    from tests.test_laya import REPLY, LayaStub
    c, tmp, queue = world
    srv = HTTPServer(("127.0.0.1", 0), LayaStub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        REPLY.clear()
        REPLY["decision"] = "buy"                           # Laya likes everything, at 0.80
        c.update(engine="laya", laya={"base_url": f"http://127.0.0.1:{srv.server_port}", "timeout_seconds": 5})
        queue += ["none"]
        morning(c)
    finally:
        srv.shutdown()
    out = capsys.readouterr().out
    assert "(Laya gives a score, not a written reason)" in out and "Data:    today" in out
    assert "What moved Laya's score" in out                  # this stub scores everything 0.80, whatever the data:
    assert "nothing moved it by even 1 point" in out         # the screen says the score does not come from the data
    assert "your local model agrees: buy (0.90), test idea for CHP" in out
    assert "your local model DISAGREES: skip (0.30), test idea for BTC-USD" in out
    why = named(tmp, "ideas")[0]["why"]
    assert why["CHP"]["second_opinion"]["action"] == "buy" and why["CHP"]["facts"]["symbol"] == "CHP"


def test_missing_news_is_said_out_loud(world, monkeypatch, capsys):
    """2026-09-30: news had silently stopped arriving for days. Now the morning says so, with the cause."""
    from agent.rh import RHError, Robinhood
    real = Robinhood.call

    async def no_news(self, name, args):
        if name == "get_equity_news":
            raise RHError("get_equity_news failed: unknown tool")
        return await real(self, name, args)
    monkeypatch.setattr(Robinhood, "call", no_news)
    c, tmp, queue = world
    queue += ["none"]
    morning(c)
    out = capsys.readouterr().out
    assert out.count("News: no news came back from Robinhood") == 2 and "unknown tool" in out     # at the scan, and under the ideas
    assert named(tmp, "news_missing")


def test_one_session_at_a_time(world):
    with daily.single_session():
        with pytest.raises(daily.Busy):
            with daily.single_session():
                pass
    with daily.single_session():                   # released again afterwards
        pass
