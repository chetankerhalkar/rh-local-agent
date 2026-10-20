"""Laya decision engine: an open-source "System 1" model (Convai, Apache 2.0) that runs on your GPU.

Laya does not write text. For each question it returns one of the options WE list, with a
calibrated probability for every option. Malformed JSON, an invented symbol or an action outside
the list cannot happen: the answer space is fixed by this file. What stays true:
  * Laya can still be WRONG. It was trained on general decisions (tickets, email), not on trading.
  * it NEVER sets a size, a price, an account or which tickers exist. Code does, exactly as before
  * a server that is down, slow or answers in an unexpected shape -> skip / hold / veto (fail closed)

Run the server on the GPU machine:
    pip install "laya[serve]"
    $env:LAYA_DEVICE="cuda"; $env:LAYA_PRELOAD="1"; laya-serve       # http://localhost:8000
Then in config.yaml:  engine: laya
"""
from __future__ import annotations

import os
from decimal import Decimal, InvalidOperation

import httpx

from .journal import log
from .llm import LocalLLM

ANALYST_Q = {"type": "choice",
             "instructions": ("A small trading bot may buy this stock or coin now for about $50, then follow the item's plan. "
                              "Buy only with a clear reason for the price to be higher by the plan's exit. Skip is the default."),
             "criteria": {"buy": "clear reason the price should rise before the plan sells it",
                          "skip": "mixed, weak, falling or unclear picture; bad news; no edge"}}

REVIEW_Q = {"type": "choice",
            "instructions": ("A rule-based strategy proposed this order. Veto only for a clear company-specific shock in the "
                             "headlines. Having no headlines is not a reason to veto."),
            "criteria": {"approve": "nothing alarming is visible",
                         "veto": "fraud, bankruptcy, halted trading, failed drug trial, guidance collapse, delisting"}}

DECIDE_BUY = {"buy": "clear reason the price rises during today's session",
              "hold": "default: mixed, weak or unclear picture, or bad news"}
DECIDE_SELL = {"sell": "clear reason the price falls from here today",
               "hold": "default: no clear reason to exit now"}


# What Laya is shown, grouped the way a person would name it. Used to explain a score (see Laya.explain).
EXPLAIN_GROUPS = (
    ("the news headlines", ("headlines",)),
    ("today's move", ("change_today_pct", "change_since_midnight_et_pct")),
    ("the last 5 days / 24 hours", ("change_5d_pct", "change_24h_pct")),
    ("the last 20 days / 7 days", ("change_20d_pct", "change_7d_pct")),
    ("where it sits vs its 20-day average", ("vs_20d_average_pct",)),
    ("how much it usually swings", ("avg_daily_range_pct", "range_24h_pct")),
    ("today's price path", ("today_5min_closes_oldest_first", "hourly_closes_oldest_first")),
    ("the buy/sell gap", ("spread_pct",)),
)


def _shown(state: dict, keys: list[str]) -> str:
    """The value of one data group, short, for the explanation line."""
    k = keys[0]
    v = state[k]
    if k == "headlines":
        return f'"{str(v[0])[:70]}"' + (f" and {len(v) - 1} more" if len(v) > 1 else "")
    if isinstance(v, list):
        return f"{_num(v[0])} to {_num(v[-1])}"
    return f"{v}%"


def _num(x) -> str:
    """39.415000 -> 39.415"""
    try:
        return format(Decimal(str(x)).normalize(), "f")
    except InvalidOperation:
        return str(x)


def _compact(item: dict, closes: int = 8, headline_chars: int = 120) -> dict:
    """Laya reads at most 512 tokens (question text included). Keep the state short."""
    out = {}
    for k, v in item.items():
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v) and "closes" in k:
            v = v[-closes:]
        elif k == "headlines" and isinstance(v, list):
            v = [str(h)[:headline_chars] for h in v[:3]]
        out[k] = v
    return out


class Laya:
    def __init__(self, cfg: dict) -> None:
        self.c = cfg.get("laya") or {}
        self.calls = 0
        self.tokens = 0
        self.last_finish: str | None = None     # same attributes as LocalLLM, so callers need no changes
        self.last_thinking = 0

    async def ask(self, state: dict, questions: dict) -> dict:
        """{question_name: {"choice": str, "probabilities": {option: p}}}. Raises on anything unexpected."""
        headers = {}
        key = os.environ.get(self.c.get("api_key_env") or "LAYA_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        url = str(self.c.get("base_url", "http://localhost:8000")).rstrip("/") + "/v1/systemone"
        async with httpx.AsyncClient(timeout=float(self.c.get("timeout_seconds", 30))) as http:
            r = await http.post(url, json={"state": state, "questions": questions}, headers=headers)
            r.raise_for_status()
            data = r.json()
        self.calls += 1
        self.tokens += int((data.get("usage") or {}).get("input_tokens") or 0)
        answers = data.get("answers") or {}
        out = {}
        for name, q in questions.items():
            a = answers.get(name) or {}
            options = set(q["criteria"])
            probs = {k: float(v) for k, v in (a.get("probabilities") or {}).items() if k in options}
            choice = a.get("choice")
            if choice not in options or set(probs) != options or not all(0 <= p <= 1 for p in probs.values()):
                raise ValueError(f"unexpected answer for {name}: {str(a)[:150]}")
            out[name] = {"choice": choice, "probabilities": probs}
        return out

    async def analyze(self, facts: list[dict]) -> dict[str, dict]:
        """Morning ideas, same shape as analyst.analyze: {symbol: {"action", "confidence", "reason"}}.
        One item per call. A failed item is left out (fail closed: no idea shown)."""
        ideas: dict[str, dict] = {}
        for f in facts:
            sym = f["symbol"]
            try:
                a = (await self.ask(_compact(f), {"decision": ANALYST_Q}))["decision"]
            except Exception as e:                                     # noqa: BLE001
                log("analysis_failed", symbols=[sym], engine="laya", error=f"{type(e).__name__}: {e}"[:200])
                continue
            p = a["probabilities"]["buy"]
            ideas[sym] = {"action": a["choice"], "confidence": p,
                          "reason": f"Laya: {p:.0%} buy / {a['probabilities']['skip']:.0%} skip"}
            log("analysis", symbol=sym, engine="laya", **ideas[sym])
        return ideas

    async def explain(self, fact: dict, base: float | None = None) -> list[dict]:
        """WHY Laya scored an item the way it did. Laya writes no reasons, so this measures it instead: ask again
        with one kind of data hidden at a time and see how far the buy score moves. Biggest effect first:
        [{"data": label, "value": what Laya saw, "points": buy points it added (+) or took away (-)}]."""
        state = _compact(fact)
        if base is None:
            base = (await self.ask(state, {"decision": ANALYST_Q}))["decision"]["probabilities"]["buy"]
        out = []
        for label, keys in EXPLAIN_GROUPS:
            present = [k for k in keys if state.get(k) not in (None, "", [])]
            if not present:
                continue
            hidden = {k: v for k, v in state.items() if k not in present}
            p = (await self.ask(hidden, {"decision": ANALYST_Q}))["decision"]["probabilities"]["buy"]
            out.append({"data": label, "value": _shown(state, present), "points": round((base - p) * 100, 1)})
        return sorted(out, key=lambda d: -abs(d["points"]))

    async def decide(self, facts: list[dict]) -> list[dict]:
        """Decider mode, same shape as the LLM's "decisions" list. Only valid actions are offered:
        buy/hold without a position, sell/hold with one. Any failure raises -> the caller holds everything."""
        out = []
        for f in facts:
            criteria = DECIDE_SELL if f.get("position") else DECIDE_BUY
            q = {"type": "choice", "criteria": criteria,
                 "instructions": "Intraday trading bot. Choose the action for this symbol now. Hold is the default."}
            a = (await self.ask(_compact(f), {"action": q}))["action"]
            conf = a["probabilities"][a["choice"]]
            out.append({"symbol": f["symbol"], "action": a["choice"], "confidence": conf,
                        "reason": "Laya: " + " / ".join(f"{p:.0%} {k}" for k, p in a["probabilities"].items())})
        return out

    async def review(self, candidate: dict, facts: dict, headlines: list[str]) -> dict:
        """Returns {"verdict": "approve"|"veto", "reason": str}, like LocalLLM.review."""
        state = {"proposed_order": candidate, "facts": facts, "headlines": [str(h)[:120] for h in headlines[:3]]}
        try:
            a = (await self.ask(state, {"verdict": REVIEW_Q}))["verdict"]
            return {"verdict": a["choice"], "reason": f"Laya: {a['probabilities']['veto']:.0%} veto"}
        except Exception as e:                                         # noqa: BLE001  any failure -> fail closed
            return {"verdict": "veto", "llm_error": True,
                    "reason": f"Laya unavailable or unreadable ({type(e).__name__}): fail closed"}


def make_engine(cfg: dict) -> Laya | LocalLLM:
    """engine: laya -> Laya decides. Anything else -> the local LLM, as before."""
    return Laya(cfg) if str(cfg.get("engine", "llm")).lower() == "laya" else LocalLLM(cfg)
