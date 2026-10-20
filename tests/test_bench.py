"""Bench against a stub OpenAI-compatible server: a 'good' model and a 'junk' model."""
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from agent.bench import CASES, run_bench

SHOCK = ("fraud", "bankruptcy", "halts", "fails phase", "going-concern", "delisted")


class Stub(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        user = body["messages"][-1]["content"].lower()
        if body["model"] == "junk":
            content = "Sure! I think you should probably buy it."          # not JSON -> must count as error
        else:
            verdict = "veto" if any(w in user for w in SHOCK) else "approve"
            content = "<think>hmm</think>" + json.dumps({"verdict": verdict, "reason": "stub"})
        out = json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}],
                          "usage": {"total_tokens": 10}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def test_bench_scores_good_and_junk_models():
    srv = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    cfg = {"llm": {"base_url": f"http://127.0.0.1:{srv.server_port}/v1", "model": "x", "timeout_seconds": 5}}
    rows = {r["model"]: r for r in asyncio.run(run_bench(cfg, ["good", "junk"]))}
    srv.shutdown()
    assert rows["good"]["correct"] == len(CASES) and rows["good"]["errors"] == 0
    assert rows["junk"]["errors"] == len(CASES) and rows["junk"]["correct"] == 0
