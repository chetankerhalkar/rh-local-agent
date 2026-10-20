"""llama.cpp specifics: extra_body is forwarded, and a reply that lands in reasoning_content is still used."""
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from agent.llm import LocalLLM

SEEN = {}


class LlamaCppStub(BaseHTTPRequestHandler):
    def do_POST(self):
        SEEN["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        msg = {"role": "assistant", "content": "", "reasoning_content": 'thinking... {"verdict": "veto", "reason": "halted"}'}
        out = json.dumps({"choices": [{"message": msg}], "usage": {"total_tokens": 7}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def test_extra_body_and_reasoning_content_rescue():
    srv = HTTPServer(("127.0.0.1", 0), LlamaCppStub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    cfg = {"llm": {"enabled": True, "base_url": f"http://127.0.0.1:{srv.server_port}/v1", "model": "local",
                   "timeout_seconds": 5, "fail_closed": True,
                   "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}}
    out = asyncio.run(LocalLLM(cfg).review({"symbol": "X"}, {}, ["X halted"]))
    srv.shutdown()
    assert SEEN["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert SEEN["body"]["response_format"] == {"type": "json_object"}
    assert out["verdict"] == "veto" and not out.get("llm_error")
