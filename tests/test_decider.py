"""Decider mode: the LLM picks the action, code keeps the power. Uses a stub LLM + the mock Robinhood server."""
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agent import guardrails, pipeline, strategy
from agent.llm import LocalLLM
from agent.rh import connect
from tests.test_end_to_end import cfg as base_cfg, events  # noqa: F401

REPLY = {"content": ""}


class Stub(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        out = json.dumps({"choices": [{"message": {"role": "assistant", "content": REPLY["content"]}}],
                          "usage": {"total_tokens": 50}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture()
def world(tmp_path, monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    for mod in (pipeline, __import__("agent.journal", fromlist=["x"]), __import__("agent.config", fromlist=["x"])):
        monkeypatch.setattr(mod, "ROOT", tmp_path, raising=False)
    monkeypatch.setattr(pipeline, "PAPER_BOOK", tmp_path / "state" / "paper_book.json")
    monkeypatch.setattr(guardrails, "market_open", lambda now: True)
    monkeypatch.setattr(guardrails, "minutes_since_open", lambda now: 120)
    monkeypatch.setattr(guardrails, "minutes_to_close", lambda now: 200)
    monkeypatch.setattr(strategy, "minutes_to_close", lambda now: 200)
    c = base_cfg("paper", llm_enabled=True)
    c["universe"] = ["AAA", "BBB"]
    c["llm"]["base_url"] = f"http://127.0.0.1:{srv.server_port}/v1"
    c["strategy"]["name"] = "llm_decider"
    c["decider"] = {"min_confidence": 0.7, "decide_every_minutes": 15, "max_buys_per_decision": 1}
    yield c, tmp_path
    srv.shutdown()


def run(c):
    async def go():
        async with connect(c) as rh:
            await pipeline.run_cycle(c, rh, LocalLLM(c))
    asyncio.run(go())


def decide(*rows):
    REPLY["content"] = json.dumps({"decisions": list(rows)})


def test_llm_buy_is_sized_by_code_not_by_llm(world):
    c, tmp = world
    decide({"symbol": "AAA", "action": "buy", "confidence": 0.9, "reason": "test",
            "quantity": 5000, "dollar_amount": 999999, "limit_price": 1})          # LLM tries to set size: ignored
    run(c)
    fills = [e for e in events(tmp) if e["event"] == "paper_fill"]
    assert len(fills) == 1
    assert fills[0]["order"]["quantity"] == "1" and fills[0]["order"]["source"] == "llm"   # floor(200 / 100.01)


def test_bad_decisions_become_hold(world):
    c, tmp = world
    decide({"symbol": "TSLA", "action": "buy", "confidence": 0.99, "reason": "not in universe"},
           {"symbol": "AAA", "action": "buy", "confidence": 0.4, "reason": "low confidence"},
           {"symbol": "BBB", "action": "sell", "confidence": 0.95, "reason": "nothing held"},
           {"symbol": "BBB", "action": "yolo", "confidence": 0.95, "reason": "repeat + bad action"})
    run(c)
    ev = events(tmp)
    assert not [e for e in ev if e["event"] == "paper_fill"]
    results = [e["result"] for e in ev if e["event"] == "llm_decision"]
    assert len(results) == 4 and all(r != "accepted" for r in results)


def test_only_most_confident_buy_goes_ahead(world):
    c, tmp = world
    decide({"symbol": "AAA", "action": "buy", "confidence": 0.75, "reason": "ok"},
           {"symbol": "BBB", "action": "buy", "confidence": 0.95, "reason": "better"})
    run(c)
    fills = [e for e in events(tmp) if e["event"] == "paper_fill"]
    assert [f["order"]["symbol"] for f in fills] == ["BBB"]


def test_junk_reply_means_hold_and_llm_is_rate_limited(world):
    c, tmp = world
    REPLY["content"] = "I feel bullish today!"
    run(c)
    decide({"symbol": "AAA", "action": "buy", "confidence": 0.9, "reason": "second call"})
    run(c)                                             # within 15 minutes: the LLM must not be asked again
    ev = events(tmp)
    assert [e for e in ev if e["event"] == "llm_decision_failed"]
    assert not [e for e in ev if e["event"] == "paper_fill"]


def test_guardrails_still_apply_to_llm_decisions(world):
    c, tmp = world
    c["guardrails"]["max_order_usd"] = 50              # one share costs ~100, so the code-sized order must be blocked
    decide({"symbol": "AAA", "action": "buy", "confidence": 0.99, "reason": "very sure"})
    run(c)
    ev = events(tmp)
    assert [e for e in ev if e["event"] == "blocked_by_guardrails"] and not [e for e in ev if e["event"] == "paper_fill"]


def test_llm_buy_then_llm_sell_tracks_paper_pnl(world):
    c, tmp = world
    c["decider"]["decide_every_minutes"] = 0
    decide({"symbol": "AAA", "action": "buy", "confidence": 0.9, "reason": "in"})
    run(c)
    decide({"symbol": "AAA", "action": "sell", "confidence": 0.9, "reason": "out"})
    run(c)
    fills = [e for e in events(tmp) if e["event"] == "paper_fill"]
    assert [f["order"]["side"] for f in fills] == ["buy", "sell"]
    assert fills[1]["realized_pnl"] == "-0.02"         # bought at the ask 100.01, sold at the bid 99.99
