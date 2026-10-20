"""Full cycle against the fake Robinhood MCP server: real MCP protocol, fake money."""
import asyncio
import json
import os

import pytest

from agent import guardrails, pipeline, strategy
from agent.llm import LocalLLM
from agent.rh import RHError, connect


def cfg(mode, llm_enabled=False):
    return {"mode": mode, "universe": ["AAA"], "paper": {"buying_power": 500},
            "robinhood": {"transport": "mock", "account_number": "MOCK-AGENTIC-1"},
            "llm": {"enabled": llm_enabled, "base_url": "http://127.0.0.1:9/v1", "model": "none",
                    "timeout_seconds": 2, "fail_closed": True, "use_news": True},
            "strategy": {"name": "dip_buy_with_exits", "order_style": "limit_whole_shares", "entry_drop_pct": 2.0,
                         "order_usd": 200, "take_profit_pct": 1.5, "stop_loss_pct": 1.0, "flatten_before_close": False},
            "guardrails": {"max_order_usd": 250, "max_daily_buy_usd": 300, "max_trades_per_day": 4,
                           "max_position_usd": 300, "max_open_positions": 3, "daily_loss_stop_pct": 2.0,
                           "allowed_order_types": ["limit"], "regular_hours_only": True,
                           "no_new_buys_first_minutes": 15, "no_new_buys_last_minutes": 20,
                           "max_quote_age_seconds": 90, "max_spread_pct": 0.30, "max_limit_slippage_pct": 0.20,
                           "block_on_any_review_alert": True}}


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    for mod in (pipeline, __import__("agent.journal", fromlist=["x"]), __import__("agent.config", fromlist=["x"])):
        monkeypatch.setattr(mod, "ROOT", tmp_path, raising=False)
    monkeypatch.setattr(pipeline, "PAPER_BOOK", tmp_path / "state" / "paper_book.json")
    monkeypatch.setattr(guardrails, "market_open", lambda now: True)       # tests must pass at any hour
    monkeypatch.setattr(guardrails, "minutes_since_open", lambda now: 120)
    monkeypatch.setattr(guardrails, "minutes_to_close", lambda now: 200)
    monkeypatch.setattr(strategy, "minutes_to_close", lambda now: 200)
    return tmp_path


def events(tmp_path):
    files = list((tmp_path / "journal").glob("*.jsonl"))
    return [json.loads(l) for f in files for l in f.read_text().splitlines()]


def run(c):
    async def go():
        async with connect(c) as rh:
            await pipeline.run_cycle(c, rh, LocalLLM(c))
    asyncio.run(go())


def test_no_signal_no_trade(sandbox, monkeypatch):
    monkeypatch.setenv("MOCK_DROP_PCT", "0.5")
    run(cfg("paper"))
    names = [e["event"] for e in events(sandbox)]
    assert "paper_fill" not in names and "cycle_done" in names


def test_paper_dip_buy_then_no_duplicate(sandbox, monkeypatch):
    monkeypatch.setenv("MOCK_DROP_PCT", "3")
    run(cfg("paper"))
    run(cfg("paper"))                                   # second cycle must not buy again
    fills = [e for e in events(sandbox) if e["event"] == "paper_fill"]
    assert len(fills) == 1 and fills[0]["order"]["quantity"] == "2"      # floor(200 / 97.01)


def test_llm_down_means_veto(sandbox, monkeypatch):
    monkeypatch.setenv("MOCK_DROP_PCT", "3")
    run(cfg("paper", llm_enabled=True))                 # nothing listens on that port
    ev = events(sandbox)
    assert [e for e in ev if e["event"] == "llm_review"][0]["verdict"] == "veto"
    assert not [e for e in ev if e["event"] == "paper_fill"]


def test_auto_places_once_with_ref_id(sandbox, monkeypatch):
    monkeypatch.setenv("MOCK_DROP_PCT", "3")
    run(cfg("auto"))
    placed = [e for e in events(sandbox) if e["event"] == "order_placed"]
    assert len(placed) == 1 and len(placed[0]["ref_id"]) == 36
    assert placed[0]["order"]["account_number"] == "MOCK-AGENTIC-1"


def test_auto_blocked_by_review_alert(sandbox, monkeypatch):
    monkeypatch.setenv("MOCK_DROP_PCT", "3")
    monkeypatch.setenv("MOCK_ALERT", "1")
    run(cfg("auto"))
    names = [e["event"] for e in events(sandbox)]
    assert "blocked_by_review_alert" in names and "order_placed" not in names


def test_generic_call_refuses_order_tools_and_wrong_account():
    async def go():
        c = cfg("auto")
        async with connect(c) as rh:
            with pytest.raises(RHError):
                await rh.call("place_equity_order", {})
            with pytest.raises(RHError):
                await rh.review_order({"account_number": "MOCK-MAIN-0"})
    asyncio.run(go())
