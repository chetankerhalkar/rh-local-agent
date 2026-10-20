"""The morning analysis: only sane model answers become ideas. Anything odd is dropped, never guessed."""
import asyncio
import json
from decimal import Decimal as D

import pytest

from agent import analyst, journal
from agent.models import Quote


class FakeLLM:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    async def chat(self, messages, tools=None, json_mode=False):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return {"content": reply}


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, "ROOT", tmp_path)


def test_only_sane_answers_become_ideas():
    facts = [{"symbol": s} for s in ("AAA", "BBB", "ETH-USD", "CCC", "DDD")]
    first = json.dumps({"decisions": [
        {"symbol": "AAA", "action": "buy", "confidence": 0.8, "reason": "ok"},
        {"symbol": "BBB", "action": "YOLO", "confidence": 0.9},                          # not buy or skip
        {"symbol": "ZZZ", "action": "buy", "confidence": 0.9},                           # never asked about
        {"symbol": "ETH", "action": "buy", "confidence": "0.7", "reason": "short name"}]})   # ETH means ETH-USD
    second = json.dumps({"decisions": [{"symbol": "CCC", "action": "buy", "confidence": 1.5},        # out of range
                                       {"symbol": "DDD", "action": "skip", "confidence": 0.9, "reason": "meh"}]})
    llm = FakeLLM([first, second])
    ideas = asyncio.run(analyst.analyze(llm, facts, batch=3))
    assert llm.calls == 2
    assert ideas == {"AAA": {"action": "buy", "confidence": 0.8, "reason": "ok"},
                     "ETH-USD": {"action": "buy", "confidence": 0.7, "reason": "short name"},
                     "DDD": {"action": "skip", "confidence": 0.9, "reason": "meh"}}


def test_a_cut_off_batch_is_asked_again_one_item_at_a_time():
    facts = [{"symbol": "AAA"}, {"symbol": "BBB"}]
    cut_off = '{"decisions": [{"symbol": "AAA", "action": "skip", "confidence": 0.1, "reason": "x"}'      # no closing
    one = lambda s: json.dumps({"decisions": [{"symbol": s, "action": "buy", "confidence": 0.8, "reason": "ok"}]})
    llm = FakeLLM([cut_off, one("AAA"), one("BBB")])
    ideas = asyncio.run(analyst.analyze(llm, facts, batch=2))
    assert llm.calls == 3 and set(ideas) == {"AAA", "BBB"}


def test_a_dead_model_is_not_asked_again_item_by_item():
    llm = FakeLLM([ConnectionError("Ollama is off")])
    assert asyncio.run(analyst.analyze(llm, [{"symbol": "AAA"}, {"symbol": "BBB"}], batch=2)) == {} and llm.calls == 1


def test_junk_or_an_offline_model_means_no_ideas():
    facts = [{"symbol": "AAA"}, {"symbol": "BBB"}]
    assert asyncio.run(analyst.analyze(FakeLLM(["I feel bullish today!"]), facts, batch=4)) == {}
    assert asyncio.run(analyst.analyze(FakeLLM([ConnectionError("Ollama is off")]), facts, batch=4)) == {}


def test_crypto_facts_from_hourly_candles(monkeypatch):
    rows = [[D(t), D(99), D(101), D(100), D(100 + t % 5), D(1)] for t in range(200)]       # oldest first

    async def candles(http, pair):
        return rows
    monkeypatch.setattr(analyst, "coinbase_hourly", candles)
    q = Quote("ETH-USD", D("110"), D("109.9"), D("110.1"), D("105"), None, asset="crypto")
    f = asyncio.run(analyst.crypto_facts({"ETH-USD": q}, "coinbase"))[0]
    assert f["change_since_midnight_et_pct"] == "+4.76"
    assert f["change_24h_pct"] == "+10.00"                       # 25 candles back closed at 100
    assert f["change_7d_pct"] == "+8.91"                         # 169 candles back closed at 101
    assert f["range_24h_pct"] == "1.82" and len(f["hourly_closes_oldest_first"]) == 12


def test_crypto_facts_without_history_still_describe_the_quote():
    q = Quote("BTC-USD", D("60000"), D("59940"), D("60060"), D("59000"), None, asset="crypto")
    f = asyncio.run(analyst.crypto_facts({"BTC-USD": q}, "none"))[0]
    assert f["spread_pct"] == "0.20" and "change_24h_pct" not in f
