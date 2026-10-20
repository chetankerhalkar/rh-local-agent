"""Command line entry point.

  You run these two, the agent asks before every buy:
    python -m agent morning    scan stocks + crypto, the local model suggests, you pick, buy, stop orders protect
    python -m agent evening    sell what is up or flat; for what is down, you choose: sell or keep

  Checks and tools:
    python -m agent discover   log in, list tools, show your accounts, save real response shapes
    python -m agent scan       what the morning scan would hand the model right now (no model, no orders)
    python -m agent report     summary of today's journal
    python -m agent doctor     check Python, packages, config and the local LLM. Run this first when something fails
    python -m agent chat       research chat (local model + read-only tools)
    python -m agent tickers    check that the tickers in your universe fit the agent's rules (price, spread, state)
    python -m agent bench --models gemma4:12b qwen3:14b     compare local models on THIS agent's job

  The unattended style (your universe list only, no questions asked):
    python -m agent once / loop                  one cycle / cycles until the close
    python -m agent auto-morning / auto-evening  scheduled two-session style
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal

from . import __version__, daily, guardrails, scan
from .config import ROOT, ConfigError, load
from .journal import log
from .laya import Laya, make_engine
from .llm import LocalLLM
from .pipeline import run_cycle
from .rh import connect, find_dicts_with


async def discover(cfg: dict) -> None:
    out = ROOT / "discovery"
    out.mkdir(exist_ok=True)
    async with connect(cfg) as rh:
        tools = await rh.list_tools()
        (out / "tools.json").write_text(json.dumps([t.model_dump(mode="json") for t in tools], indent=2), encoding="utf-8")
        print(f"Connected. {len(tools)} tools saved to discovery/tools.json\n")
        accounts = await rh.call("get_accounts", {})
        (out / "get_accounts.json").write_text(json.dumps(accounts, indent=2), encoding="utf-8")
        print("Your accounts (the agent can only trade where agentic_allowed is True):")
        for a in find_dicts_with(accounts, ("account_number",)):
            print(f"  {a.get('account_number')}   type={a.get('type') or a.get('brokerage_account_type')}   "
                  f"agentic_allowed={a.get('agentic_allowed')}   rhs_account_number={a.get('rhs_account_number')}")
        if not rh.account:
            print("\nPut the agentic account number into config.yaml, then run discover again.")
            return
        print(f"\nConfigured account is agentic: {await rh.agentic_account_ok()}")
        for name in ("get_portfolio", "get_equity_positions"):
            data = await rh.call(name, {"account_number": rh.account})
            (out / f"{name}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        bp, eq = await rh.money()
        pos = await rh.positions()
        print(f"Parser check -> buying_power={bp}  equity={eq}  positions={ {s: str(p.quantity) for s, p in pos.items()} }")
        print("If buying_power or equity shows None, open discovery/get_portfolio.json, find the right\n"
              "key names, and add them to the tuples in agent/rh.py -> money(). Until then, buys stay blocked.")

        # what the morning and evening sessions use: orders, the screener, crypto
        crypto = await rh.find_crypto_account()
        print(f"\nCrypto account number (rhs_account_number) of the agentic account: {crypto or 'NOT FOUND'}")
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        shapes = [("get_equity_orders", {"account_number": rh.account, "created_at_gte": today}),
                  ("preview_scan", {"filters": scan.stock_filters((cfg.get("scan") or {}).get("stocks") or {},
                                                                  cfg["strategy"].get("order_usd", 50))}),
                  ("get_currency_pairs", {"limit": 5}),
                  ("get_equity_news", {"symbol": "AAPL"})]
        if crypto:
            shapes += [("get_crypto_quotes", {"symbols": ["BTC-USD"], "rhs_account_number": crypto}),
                       ("get_crypto_positions", {"rhs_account_number": crypto}),
                       ("get_crypto_orders", {"rhs_account_number": crypto, "state_group": "open"})]
        for name, args in shapes:
            try:
                (out / f"{name}.json").write_text(json.dumps(await rh.call(name, args), indent=2, default=str), encoding="utf-8")
                print(f"  saved discovery/{name}.json")
            except Exception as e:                               # noqa: BLE001  report it and keep going
                print(f"  {name}: FAILED  {type(e).__name__}: {str(e)[:200]}")
        stocks = await scan.scan_stocks(rh, cfg)
        print(f"Parser check -> screener picked: {stocks or 'NOTHING (see discovery/preview_scan.json)'}")
        news = await rh.headlines("AAPL", limit=3)
        print(f"Parser check -> news for AAPL: {len(news)} headlines" + (f", first: {news[0][:70]}" if news else
              f"   PROBLEM: {rh.news_problem}"))
        print(f"                Robinhood tools with 'news' in the name: {[t.name for t in tools if 'news' in t.name.lower()] or 'NONE'}")
        if crypto:
            pairs = await rh.crypto_pairs()
            q = (await rh.crypto_quotes(["BTC-USD"])).get("BTC-USD")
            print(f"Parser check -> {len(pairs)} crypto pairs   BTC-USD bid={q.bid if q else None} "
                  f"ask={q.ask if q else None} time={q.ts if q else None}")
            print("If the time shows None, crypto buys stay blocked (stale quote): check discovery/get_crypto_quotes.json.")


async def once(cfg: dict, session: str = "auto") -> None:
    async with connect(cfg) as rh:
        if cfg["mode"] != "paper" and not await rh.agentic_account_ok():
            raise ConfigError("configured account is not agentic_allowed. Refusing to run.")
        await run_cycle(cfg, rh, make_engine(cfg), session)


async def auto_morning(cfg: dict) -> None:
    await once(cfg, "morning")


async def auto_evening(cfg: dict) -> None:
    await once(cfg, "evening")


async def scan_now(cfg: dict) -> None:
    """What the morning scan would hand the model right now, and why anything was left out. No model, no orders."""
    coins: list[str] = []
    cq: dict = {}
    spreads: dict = {}
    note = ""
    async with connect(cfg) as rh:
        stocks = await scan.scan_stocks(rh, cfg)
        quotes = await rh.quotes(stocks) if stocks else {}
        if ((cfg.get("scan") or {}).get("crypto") or {}).get("enabled", True):
            if await rh.find_crypto_account():
                coins, cq, spreads, note = await scan.scan_crypto(rh, cfg, await rh.crypto_pairs())
            else:
                note = "no crypto account number (rhs_account_number) was found for your agentic account"
    usd = cfg["strategy"]["order_usd"]
    rows = []
    for s in stocks:
        q = quotes.get(s)
        spread = f"{(q.ask - q.bid) / ((q.ask + q.bid) / 2) * 100:.2f}%" if q and q.bid > 0 else "-"
        rows.append([s, daily.price(q.last) if q else "-", spread, f"{int(usd // q.ask)} sh" if q and q.ask > 0 else "-"])
    print(f"STOCKS  ({len(stocks)} would go to the model)")
    print(daily.table(["Symbol", "Price", "Spread", f"Per ${usd} order"], rows) if rows else "  nothing")
    if not guardrails.market_open(guardrails.now_et()):
        print("  The stock market is closed now, so the morning session skips stocks until it opens.")
    limit = cfg["guardrails"].get("crypto_max_spread_pct", 1.0)
    print(f"\nCRYPTO  ({len(coins)} would go to the model; a coin needs a spread of {limit}% or tighter)")
    crows = [[s, daily.price(cq[s].last) if s in cq else "-", f"{sp:.2f}%", "yes" if s in coins else "no"]
             for s, sp in sorted(spreads.items(), key=lambda kv: kv[1])[:12]]
    print(daily.table(["Coin", "Price", "Spread", "Goes to the model"], crows) if crows else "  no prices came back")
    if note:
        print(f"  {note}.")


async def loop(cfg: dict) -> None:
    wait = int((cfg.get("loop") or {}).get("interval_seconds", 300))
    log("loop_start", mode=cfg["mode"], interval=wait, universe=cfg["universe"])
    while True:
        now = guardrails.now_et()
        if now.weekday() >= 5 or now.time() >= guardrails.CLOSE:
            log("loop_end", why="market closed for today")
            return
        try:
            await once(cfg)                     # fresh connection each cycle: survives network drops
        except Exception as e:                  # noqa: BLE001
            leaves = list(_leaves(e))
            if any(isinstance(x, ConfigError) for x in leaves):
                raise
            log("cycle_error", error="; ".join(f"{type(x).__name__}: {x}" for x in leaves)[:500])
        await asyncio.sleep(wait)


async def tickers(cfg: dict) -> None:
    """Does each ticker YOU picked fit the agent's mechanical rules? This is a fit check, not a recommendation."""
    from decimal import Decimal
    usd = Decimal(str(cfg["strategy"]["order_usd"]))
    max_spread = Decimal(str(cfg["guardrails"]["max_spread_pct"]))
    whole = cfg["strategy"].get("order_style") != "market_dollars"
    async with connect(cfg) as rh:
        quotes = await rh.quotes(cfg["universe"])
        unknown = list(rh.missing)
    is_open = guardrails.market_open(guardrails.now_et())
    print(f"order_usd=${usd}  order_style={'whole shares' if whole else 'fractional dollars'}  max_spread_pct={max_spread}%\n")
    print(f"{'ticker':8s} {'last':>10s} {'spread %':>9s} {'shares/order':>13s}   result")
    good = 0
    for sym in cfg["universe"]:
        q = quotes.get(sym)
        if q is None:
            why = "Robinhood does not know this symbol. Check the spelling" if sym in unknown else "no quote came back"
            print(f"{sym:8s} {'-':>10s} {'-':>9s} {'-':>13s}   PROBLEM: {why}")
            continue
        spread = (q.ask - q.bid) / ((q.ask + q.bid) / 2) * 100 if q.bid > 0 and q.ask >= q.bid else None
        shares = int(usd // q.ask) if q.ask > 0 else 0
        problems = []
        if q.state != "active":
            problems.append(f"state is {q.state}")
        if whole and shares < 1:
            problems.append(f"one share (${q.ask:.2f}) costs more than order_usd. It will be skipped every day")
        if spread is None:
            problems.append("no usable bid/ask right now")
        elif spread > max_spread and is_open:
            problems.append(f"spread is over max_spread_pct. Buys will be blocked")
        good += not problems
        print(f"{sym:8s} {q.last:>10.2f} {(f'{spread:.2f}' if spread is not None else '-'):>9s} "
              f"{(str(shares) if whole else 'fractional'):>13s}   {'fits' if not problems else 'PROBLEM: ' + '; '.join(problems)}")
    print(f"\n{good} of {len(cfg['universe'])} tickers fit.")
    if not is_open:
        print("The market is closed, so spreads are not judged. Run this again during market hours for a true reading.")


async def chat(cfg: dict) -> None:
    from .chat import run_chat
    async with connect(cfg) as rh:
        await run_chat(rh, LocalLLM(cfg))


def doctor(config_path: str | None) -> None:
    """No Robinhood involved. Tells you which link in the chain is broken."""
    import sys
    import time

    import httpx
    ok = lambda good, text: print(("  OK    " if good else "  FAIL  ") + text) or good
    print(f"rh-local-agent {__version__}")
    print("1. Python")
    ok(sys.version_info >= (3, 10), f"Python {sys.version.split()[0]} (need 3.10 or newer)")
    ok("venv" in sys.prefix.lower() or sys.prefix != sys.base_prefix, f"running inside the project venv ({sys.prefix})")
    print("2. Config")
    try:
        cfg = load(config_path)
        ok(True, f"config.yaml loaded. mode={cfg['mode']}  strategy={cfg['strategy'].get('name')}  universe={cfg['universe'] or 'EMPTY'}")
    except ConfigError as e:
        ok(False, str(e))
        print("     Fix:  Copy-Item config.example.yaml config.yaml")
        return
    if str(cfg.get("engine", "llm")).lower() == "laya":
        base = str((cfg.get("laya") or {}).get("base_url", "http://localhost:8000")).rstrip("/")
        print(f"3. Laya decision server ({base})")
        eng, t0 = Laya(cfg), time.perf_counter()
        out = asyncio.run(eng.review({"symbol": "TEST", "side": "buy"}, {"last": "10"}, ["TEST moves with the market"]))
        if ok(not out.get("llm_error"), f"verdict={out['verdict']}  in {time.perf_counter() - t0:.2f}s  {out['reason'][:80]}"):
            print("\nAll good. Laya makes the decisions (engine: laya).")
        else:
            print("     Fix:  the Laya server is not running. Start it in its own window and leave it open:\n"
                  "           powershell -ExecutionPolicy Bypass -File .\\scripts\\run_laya.ps1")
        return
    print("3. Local LLM server")
    base, model = cfg["llm"]["base_url"].rstrip("/"), cfg["llm"]["model"]
    try:
        names = [m.get("id") for m in httpx.get(base + "/models", timeout=10).json().get("data", [])]
        ok(True, f"server answers at {base}. Models it reports: {names}")
        if names and model not in names:
            print(f"  NOTE  '{model}' is not in that list. Ollama: run  ollama pull {model}.  llama.cpp: fine, it ignores the name.")
    except Exception as e:                                   # noqa: BLE001
        ok(False, f"nothing answers at {base} ({type(e).__name__})")
        print("     Fix:  Ollama -> start the Ollama app, then  ollama ps\n"
              "           llama.cpp -> start llama-server and set base_url to http://localhost:8080/v1")
        return
    print("4. JSON answer test (the exact call the agent makes)")
    llm, t0 = LocalLLM(cfg), time.perf_counter()
    out = asyncio.run(llm.review({"symbol": "TEST", "side": "buy"}, {"last": "10"}, ["TEST moves with the market"]))
    secs = time.perf_counter() - t0
    if ok(not out.get("llm_error"), f"verdict={out['verdict']}  in {secs:.1f}s  tokens={llm.tokens}  reason={out['reason'][:80]}"):
        if llm.last_thinking:
            print(f"  NOTE  your model thought for {llm.last_thinking} characters before answering. On the morning's bigger\n"
                  "        questions that is slow and can cut the answer off. Switch it off in config.yaml under llm:\n"
                  "          Ollama:     extra_body: {reasoning_effort: none}\n"
                  "          llama.cpp:  extra_body: {chat_template_kwargs: {enable_thinking: false}}")
        elif secs > 60:
            print("  NOTE  slow. If this is llama.cpp + Gemma, thinking is probably on. See extra_body in config.example.yaml.")
        print("\nAll good. Next:  python -m agent bench")
    else:
        print("     The server is up but the reply was not usable JSON. llama.cpp: start it with --jinja and switch thinking off.")


def report() -> None:
    p = ROOT / "journal" / f"{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"
    if not p.exists():
        print("No journal for today yet.")
        return
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Journal {p.name}: {len(rows)} events")
    for event, n in Counter(r["event"] for r in rows).most_common():
        print(f"  {n:4d}  {event}")
    pnl = [r["paper_pnl_today"] for r in rows if r["event"] == "paper_fill" and "paper_pnl_today" in r]
    if pnl:
        print(f"  Paper P&L today (realized, simulated fills at the limit price): ${pnl[-1]}")
    sold = [Decimal(r["pnl"]) for r in rows if r["event"] in ("sold", "closed_by_stop", "closed_by_sell") and r.get("pnl")]
    if sold:
        print(f"  Realized today by the daily sessions: ${sum(sold):.2f}")
    for r in rows:
        if r["event"] in ("order_placed", "paper_fill", "blocked_by_guardrails", "blocked_after_recheck",
                          "llm_review", "llm_decision", "llm_decision_failed", "order_error", "order_cancelled",
                          "paper_stop_set", "UNPROTECTED_POSITION", "STILL_HOLDING", "flat", "buy_not_filled",
                          "ideas", "bought", "stop_placed", "sold", "closed_by_stop", "closed_by_sell", "holding_gone",
                          "not_approved", "dry_run", "order_rejected", "adopted", "network_problem", "scan_failed",
                          "analysis_failed", "sell_blocked"):
            print(f"  {r['ts'][11:19]}  {r['event']:24s} {json.dumps({k: v for k, v in r.items() if k not in ('ts', 'event')}, default=str)[:160]}")


def _leaves(e: BaseException):
    """The MCP library wraps errors in nested groups. Dig out the real ones."""
    if isinstance(e, BaseExceptionGroup):
        for sub in e.exceptions:
            yield from _leaves(sub)
    else:
        yield e


def _explain(e: BaseException) -> str:
    text = str(e)
    if isinstance(e, ConfigError):
        return f"CONFIG ERROR: {text}"
    if "missing_instruments" in text:
        return f"ROBINHOOD: it does not know one or more of your tickers. {text[-200:]}\n  Fix the universe list in config.yaml."
    if isinstance(e, TimeoutError):
        return "LOGIN TIMED OUT: the Robinhood login in the browser was not finished within 5 minutes. Run the command again."
    if isinstance(e, OSError) and "10048" in text or "address already in use" in text.lower():
        return "PORT BUSY: another copy of the agent is waiting for a login on port 8765. Close other PowerShell windows and retry."
    if type(e).__name__ == "RHError":
        return f"ROBINHOOD: {text[:400]}"
    if type(e).__name__ in ("ConnectError", "ConnectTimeout", "ReadTimeout"):
        return f"NETWORK: could not reach the server ({type(e).__name__}). Check your internet, then run:  python -m agent doctor"
    return f"{type(e).__name__}: {text[:400]}"


def main() -> None:
    ap = argparse.ArgumentParser(prog="agent", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["morning", "evening", "discover", "scan", "report", "doctor", "chat", "tickers",
                                        "bench", "once", "loop", "auto-morning", "auto-evening"])
    ap.add_argument("--version", action="version", version=f"rh-local-agent {__version__}")
    ap.add_argument("--config", default=None)
    ap.add_argument("--debug", action="store_true", help="show the full technical traceback on errors")
    ap.add_argument("--models", nargs="*", default=None, help="bench: Ollama model tags to compare")
    a = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):       # a model's fancy dash must never crash a session on a Windows code page
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    if a.command == "report":
        return report()
    if a.command == "doctor":
        return doctor(a.config)
    try:
        cfg = load(a.config)
        if a.command == "bench":
            from .bench import run_bench
            return asyncio.run(run_bench(cfg, a.models or [cfg["llm"]["model"]]))
        if a.command in ("once", "loop", "auto-morning", "auto-evening", "tickers") and not cfg["universe"]:
            raise ConfigError("universe is empty. This command trades only the stocks in that list, so open config.yaml\n"
                              "and put YOUR real stock symbols in it (the short codes shown in the Robinhood app).\n"
                              "Then run:  python -m agent tickers   to check that they fit the agent's rules.\n"
                              "(The morning / evening sessions do not need it: they scan the market.)")
        asyncio.run({"discover": discover, "once": once, "loop": loop, "chat": chat, "scan": scan_now,
                     "morning": daily.morning, "evening": daily.evening,
                     "auto-morning": auto_morning, "auto-evening": auto_evening, "tickers": tickers}[a.command](cfg))
    except KeyboardInterrupt:
        print("\nStopped by you.")
    except Exception as e:                       # noqa: BLE001
        if a.debug:
            raise
        lines = list(dict.fromkeys(_explain(x) for x in _leaves(e)))      # unique, in order
        raise SystemExit("\n".join(lines) + "\n(add --debug to see the full technical traceback)")


if __name__ == "__main__":
    main()
