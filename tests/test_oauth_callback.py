"""Simulates the browser: the real redirect, plus the stray extra requests that crashed the old server on Windows."""
import asyncio
import urllib.request

from agent.oauth import build_provider


def test_callback_server_catches_code_and_survives_extra_requests():
    provider = build_provider("https://example.invalid/mcp", 8799)
    handler = provider.context.callback_handler

    async def browser():
        await asyncio.sleep(0.3)
        get = lambda path: urllib.request.urlopen(f"http://127.0.0.1:8799{path}", timeout=5).read()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, get, "/favicon.ico")                       # no code: must not finish the login
        body = await loop.run_in_executor(None, get, "/callback?code=abc123&state=xyz")
        assert b"Login complete" in body
        try:
            await loop.run_in_executor(None, get, "/favicon.ico")                   # late request while shutting down
        except Exception:                                                           # noqa: BLE001  refused is fine
            pass

    async def go():
        result, _ = await asyncio.gather(handler(), browser())
        return result

    r = asyncio.run(go())
    assert r.code == "abc123" and r.state == "xyz"
