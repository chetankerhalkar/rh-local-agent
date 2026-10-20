"""Morning buy -> protective stop at the broker -> evening cancel + sell. Against the mock Robinhood server."""
import asyncio
import json

import pytest

from agent import guardrails, pipeline, strategy
from agent.llm import LocalLLM
from agent.rh import connect
from tests.test_end_to_end import cfg as base_cfg, events


@pytest.fixture()
def world(tmp_path, monkeypatch):
    for mod in (pipeline, __import__("agent.journal", fromlist=["x"]), __import__("agent.config", fromlist=["x"])):
        monkeypatch.setattr(mod, "ROOT", tmp_path, raising=False)
    monkeypatch.setattr(pipeline, "PAPER_BOOK", tmp_path / "state" / "paper_book.json")
    monkeypatch.setattr(guardrails, "market_open", lambda now: True)
    monkeypatch.setattr(guardrails, "minutes_since_open", lambda now: 30)
    monkeypatch.setattr(guardrails, "minutes_to_close", lambda now: 300)
    monkeypatch.setattr(strategy, "minutes_to_close", lambda now: 300)
    monkeypatch.setenv("MOCK_STATE_FILE", str(tmp_path / "broker.json"))      # broker remembers between sessions
    monkeypatch.setenv("MOCK_DROP_PCT", "3")

    def make(mode):
        c = base_cfg(mode)
        c["protect"] = {"broker_stop": True, "fill_wait_seconds": 0.2, "poll_seconds": 0.05}
        c["schedule"] = {"evening_verify_seconds": 0}
        return c
    return make, tmp_path


def run(c, session):
    async def go():
        async with connect(c) as rh:
            await pipeline.run_cycle(c, rh, LocalLLM(c), session)
    asyncio.run(go())


def broker(tmp):
    return json.loads((tmp / "broker.json").read_text())


def test_morning_buy_places_real_stop_then_evening_cancels_and_sells(world, monkeypatch):
    make, tmp = world
    run(make("auto"), "morning")
    b = broker(tmp)
    stops = [o for o in b["orders"].values() if o["type"] == "stop_market"]
    assert "AAA" in b["positions"] and len(stops) == 1 and stops[0]["state"] == "confirmed"
    placed = [e["order"] for e in events(tmp) if e["event"] == "order_placed"]
    assert placed[1]["stop_price"] == "96.04" and placed[1]["quantity"] == "2"        # 1% under the 97.01 fill

    monkeypatch.setattr(guardrails, "minutes_to_close", lambda now: 20)               # it is now 15:40 ET
    run(make("auto"), "evening")
    b = broker(tmp)
    names = [e["event"] for e in events(tmp)]
    assert b["positions"] == {} and "order_cancelled" in names and "flat" in names
    assert [o["state"] for o in b["orders"].values() if o["type"] == "stop_market"] == ["cancelled"]


def test_evening_does_nothing_new_and_never_buys(world):
    make, tmp = world
    run(make("auto"), "evening")                                   # nothing held: no orders at all
    assert not [e for e in events(tmp) if e["event"] == "order_placed"]


def test_unfilled_buy_is_cancelled_and_not_protected(world, monkeypatch):
    make, tmp = world
    monkeypatch.setenv("MOCK_NO_FILL", "1")
    run(make("auto"), "morning")
    names = [e["event"] for e in events(tmp)]
    assert "order_cancelled" in names and "buy_not_filled" in names
    assert not [o for o in broker(tmp)["orders"].values() if o["type"] == "stop_market"]


def test_exit_is_never_blocked_by_the_trade_count(world, monkeypatch):
    make, tmp = world
    c = make("auto")
    c["guardrails"]["max_trades_per_day"] = 1                      # the one buy uses the whole budget
    run(c, "morning")
    monkeypatch.setattr(guardrails, "minutes_to_close", lambda now: 20)
    run(c, "evening")
    assert broker(tmp)["positions"] == {}                          # the sell still went through


def test_paper_stop_touched_during_the_day_fills_at_stop_price(world, monkeypatch):
    make, tmp = world
    run(make("paper"), "morning")
    assert [e for e in events(tmp) if e["event"] == "paper_stop_set"][0]["stop_price"] == "96.04"
    monkeypatch.setenv("MOCK_DAY_LOW", "95.00")                    # price dipped under the stop at midday
    run(make("paper"), "evening")
    sell = [e for e in events(tmp) if e["event"] == "paper_fill" and e["order"]["side"] == "sell"][0]
    assert sell["assumed_price"] == "96.04" and sell["realized_pnl"] == "-1.94"       # (96.04 - 97.01) x 2


def test_no_trade_date_blocks_buys(world):
    make, tmp = world
    c = make("auto")
    c["schedule"]["no_trade_dates"] = [f"{guardrails.now_et():%Y-%m-%d}"]
    run(c, "morning")
    assert not [e for e in events(tmp) if e["event"] == "order_placed"]
