"""Laya engine: the fixed answer space is enforced, and anything unexpected fails closed."""
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from agent import analyst
from agent.laya import Laya, make_engine
from agent.llm import LocalLLM

SEEN: list = []
REPLY: dict = {}          # question name -> answer, or "bad" to send an off-list choice


class LayaStub(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append((self.path, body))
        answers = {}
        for name, q in body["questions"].items():
            opts = list(q["criteria"])
            pick = REPLY.get(name, opts[-1])
            if pick == "bad":
                answers[name] = {"choice": "yolo", "probabilities": {"yolo": 1.0}}
            else:
                answers[name] = {"choice": pick, "confidence": 0.8,
                                 "probabilities": {o: (0.8 if o == pick else 0.2 / (len(opts) - 1)) for o in opts}}
        out = json.dumps({"answers": answers, "usage": {"input_tokens": 50, "output_tokens": 0}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def _run(fn):
    srv = HTTPServer(("127.0.0.1", 0), LayaStub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    SEEN.clear()
    try:
        return fn(Laya({"laya": {"base_url": f"http://127.0.0.1:{srv.server_port}", "timeout_seconds": 5}}))
    finally:
        srv.shutdown()


def test_make_engine():
    assert isinstance(make_engine({"engine": "laya"}), Laya)
    assert isinstance(make_engine({"llm": {}}), LocalLLM)


def test_morning_analysis_uses_laya():
    REPLY.clear()
    REPLY["decision"] = "buy"
    facts = [{"symbol": "AAA", "type": "stock", "price": "10", "headlines": ["x" * 500] * 5,
              "today_5min_closes_oldest_first": [str(i) for i in range(12)]}]
    ideas = _run(lambda eng: asyncio.run(analyst.analyze(eng, facts)))
    assert SEEN[0][0] == "/v1/systemone"
    state = SEEN[0][1]["state"]
    assert len(state["headlines"]) == 3 and len(state["headlines"][0]) == 120
    assert len(state["today_5min_closes_oldest_first"]) == 8
    assert ideas["AAA"]["action"] == "buy" and ideas["AAA"]["confidence"] == 0.8
    assert ideas["AAA"]["reason"].startswith("Laya:")


def test_off_list_answer_is_dropped():
    REPLY.clear()
    REPLY["decision"] = "bad"
    ideas = _run(lambda eng: asyncio.run(eng.analyze([{"symbol": "AAA"}])))
    assert ideas == {}


def test_decider_only_offers_valid_actions():
    REPLY.clear()
    out = _run(lambda eng: asyncio.run(eng.decide([{"symbol": "AAA", "position": None},
                                                    {"symbol": "BBB", "position": {"quantity": "1"}}])))
    assert set(SEEN[0][1]["questions"]["action"]["criteria"]) == {"buy", "hold"}
    assert set(SEEN[1][1]["questions"]["action"]["criteria"]) == {"sell", "hold"}
    assert [d["action"] for d in out] == ["hold", "hold"]


class ScoringStub(BaseHTTPRequestHandler):
    """A Laya whose buy score depends on the data: the 20-day move adds 12 points, news 3, a big swing takes 2."""
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        s = body["state"]
        p = 0.50 + (0.12 if "change_20d_pct" in s else 0) + (0.03 if s.get("headlines") else 0) - \
            (0.02 if "avg_daily_range_pct" in s else 0)
        answers = {n: {"choice": "buy" if p >= 0.5 else "skip", "probabilities": {"buy": p, "skip": 1 - p}}
                   for n in body["questions"]}
        out = json.dumps({"answers": answers}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def test_explain_measures_what_moved_the_score():
    srv = HTTPServer(("127.0.0.1", 0), ScoringStub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        eng = Laya({"laya": {"base_url": f"http://127.0.0.1:{srv.server_port}", "timeout_seconds": 5}})
        fact = {"symbol": "BROS", "type": "stock", "change_today_pct": "+3.37", "change_20d_pct": "-19.39",
                "avg_daily_range_pct": "3.43", "headlines": ["Dutch Bros beats estimates"]}
        drivers = asyncio.run(eng.explain(fact))
    finally:
        srv.shutdown()
    assert [(d["data"], d["points"]) for d in drivers] == [
        ("the last 20 days / 7 days", 12.0), ("the news headlines", 3.0), ("how much it usually swings", -2.0),
        ("today's move", 0.0)]
    assert drivers[0]["value"] == "-19.39%" and drivers[1]["value"] == '"Dutch Bros beats estimates"'


def test_review_veto_and_fail_closed():
    REPLY.clear()
    REPLY["verdict"] = "approve"
    assert _run(lambda eng: asyncio.run(eng.review({"symbol": "X"}, {}, [])))["verdict"] == "approve"
    REPLY["verdict"] = "bad"
    out = _run(lambda eng: asyncio.run(eng.review({"symbol": "X"}, {}, [])))
    assert out["verdict"] == "veto" and out["llm_error"]
    dead = Laya({"laya": {"base_url": "http://127.0.0.1:9", "timeout_seconds": 2}})
    out = asyncio.run(dead.review({"symbol": "X"}, {}, []))
    assert out["verdict"] == "veto" and out["llm_error"]
