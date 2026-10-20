"""OAuth login for the Robinhood MCP. You log in on Robinhood's own page in your
browser. This code never sees your password. It only stores the OAuth tokens that
Robinhood hands back, in .secrets/ on your own disk.

Treat .secrets/ like a password: never commit it, never show it on camera.
To cut the agent off: delete .secrets/ and disconnect the agent inside Robinhood.
"""
from __future__ import annotations

import asyncio
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.shared.auth import (AuthorizationCodeResult, OAuthClientInformationFull,
                             OAuthClientMetadata, OAuthToken)

from .config import ROOT


class FileTokenStorage(TokenStorage):
    def __init__(self) -> None:
        self.path = ROOT / ".secrets" / "robinhood_oauth.json"

    def _read(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def _write(self, key: str, value: dict) -> None:
        self.path.parent.mkdir(exist_ok=True)
        data = self._read()
        data[key] = value
        self.path.write_text(json.dumps(data), encoding="utf-8")

    async def get_tokens(self) -> OAuthToken | None:
        d = self._read().get("tokens")
        return OAuthToken.model_validate(d) if d else None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self._write("tokens", tokens.model_dump(mode="json"))

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        d = self._read().get("client")
        return OAuthClientInformationFull.model_validate(d) if d else None

    async def set_client_info(self, info: OAuthClientInformationFull) -> None:
        self._write("client", info.model_dump(mode="json"))


def build_provider(url: str, port: int) -> OAuthClientProvider:
    redirect = f"http://localhost:{port}/callback"

    async def open_browser(auth_url: str) -> None:
        print("\nOpening Robinhood login in your browser. If nothing opens, paste this URL:\n" + auth_url + "\n")
        webbrowser.open(auth_url)

    async def wait_for_callback() -> AuthorizationCodeResult:
        """Small web server on localhost that catches Robinhood's redirect.
        It runs in its own thread, so a late browser request (favicon, keep-alive) cannot upset
        the asyncio loop on Windows."""
        loop = asyncio.get_running_loop()
        got: asyncio.Future[dict] = loop.create_future()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:                                    # noqa: N802
                qs = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
                ok = "code" in qs
                body = (b"Login complete. You can close this tab and go back to PowerShell." if ok
                        else b"Waiting for the Robinhood login to finish...")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
                if ok:
                    loop.call_soon_threadsafe(lambda: got.done() or got.set_result(qs))

            def log_message(self, *args) -> None:                        # keep the console clean
                pass

        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            qs = await asyncio.wait_for(got, timeout=300)
        finally:
            await loop.run_in_executor(None, server.shutdown)
            server.server_close()
        return AuthorizationCodeResult(code=qs["code"], state=qs.get("state"), iss=qs.get("iss"))

    meta = OAuthClientMetadata(
        client_name="rh-local-agent",
        redirect_uris=[redirect],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="none",     # public client + PKCE, no client secret on disk
    )
    return OAuthClientProvider(server_url=url, client_metadata=meta, storage=FileTokenStorage(),
                               redirect_handler=open_browser, callback_handler=wait_for_callback)
