"""Morning scan: which stocks and coins go to the model today. Plain code, no LLM.

The screen runs on fixed rules from config.yaml (the scan: section), so nothing the model
reads, headlines included, can put a symbol on the list. A fit check, not advice.
"""
from __future__ import annotations

import re
from decimal import Decimal

from .journal import log
from .models import CryptoPair, Quote
from .rh import Robinhood, dec, find_dicts_with, pair_symbol

TICKER = re.compile(r"[A-Z]{1,5}([.-][A-Z]{1,2})?")
FUND_TYPES = ("etf", "etp", "etn", "fund")            # leveraged and inverse funds live here
# Second net, on the fund's name: scan results call stocks AND funds "EQUITY", so the type alone never catches them
FUND_NAME = re.compile(r"\b(ETFS?|ETNS?|ETPS?|FUNDS?|[1-9]X|ULTRA|ULTRAPRO|INVERSE|LEVERAGED|DAILY TARGET|BULL|BEAR|"
                       r"ISHARES|PROSHARES|DIREXION|SPDR)\b", re.I)


def stock_filters(s: dict, order_usd: float = 50) -> list[dict]:
    """Robinhood screener filters. Boolean expressions are the form the tool documents
    (expression + PREDICATE_EQUAL + ["True"]). A list under scan.stocks.filters replaces these.
    The price cap defaults to one order's size: a pricier stock could never be bought in whole shares."""
    if s.get("filters"):
        return s["filters"]
    avg_volume = 'volumeAvg(candleCount=30, candlePeriod="1d", session="all")'
    rule = lambda expr, title: {"expression": expr, "predicate": "PREDICATE_EQUAL", "values": ["True"],
                                "display_title": title}
    return [rule(f"tradeAllDay.price >= {s.get('min_price', 5)}", "Price at least"),
            rule(f"tradeAllDay.price <= {s.get('max_price', order_usd)}", "Price at most"),
            rule(f"{avg_volume} >= {s.get('min_avg_volume', 1000000)}", "Average volume at least")]


def _volume(obj) -> Decimal | None:
    """The volume figure in one screener row, whatever shape the row has."""
    if isinstance(obj, dict):
        label = str(obj.get("display_name") or obj.get("column") or obj.get("name") or obj.get("title") or "")
        if "volume" in label.lower() and dec(obj.get("value")) is not None:
            return dec(obj.get("value"))
        for k, v in obj.items():
            if "volume" in str(k).lower() and not isinstance(v, (dict, list)) and dec(v) is not None:
                return dec(v)
        obj = list(obj.values())
    if isinstance(obj, list):
        for v in obj:
            if isinstance(v, (dict, list)) and (r := _volume(v)) is not None:
                return r
    return None


def _name(row: dict) -> str:
    """The Name cell of one screener row (a fund's name says ETF, 2X, Ultra, ...)."""
    for d in [row, *find_dicts_with(row, ("value",))]:
        label = str(d.get("display_name") or d.get("column") or d.get("title") or "").lower()
        if label == "name" and isinstance(d.get("value"), str):
            return d["value"]
    return str(row.get("name") or row.get("instrument_name") or row.get("simple_name") or "")


async def stock_only_filter(rh: Robinhood) -> dict | None:
    """The screener's own asset-type filter, set to plain stocks, in the exact words its spec lists.
    Robinhood's tool docs say to take those values verbatim from get_scanner_filter_specs, never from results."""
    try:
        data = await rh.call("get_scanner_filter_specs", {})
    except Exception as e:                                           # noqa: BLE001  the name net still works
        log("warning", why=f"screener filter specs unavailable: {type(e).__name__}")
        return None
    for spec in find_dicts_with(data, ("filter_type",)):
        ft = str(spec["filter_type"])
        label = " ".join(str(spec.get(k) or "") for k in ("display_name", "name", "title", "label")).lower()
        if "ASSET_TYPE" not in ft.upper() and "asset type" not in label:
            continue
        words = lambda xs: [x if isinstance(x, str) else str(x.get("value") or x.get("id") or x.get("name") or "") for x in xs or []]
        options = words(spec.get("options") or spec.get("values") or spec.get("allowed_values"))
        stock = next((o for o in options if o.upper() in ("STOCK", "STOCKS", "COMMON_STOCK")), None) or \
            next((o for o in options if "STOCK" in o.upper() and "ETF" not in o.upper()), None)
        preds = words(spec.get("supported_predicates") or spec.get("predicates"))
        pred = next((p for p in ("PREDICATE_IN_LIST", "PREDICATE_ANY_OF", "PREDICATE_EQUAL") if p in preds),
                    preds[0] if preds else "PREDICATE_IN_LIST")
        if stock:
            return {"filter_type": ft, "predicate": pred, "values": [stock]}
    log("warning", why="the screener has no asset-type filter with a stock option: funds are left out by name only")
    return None


async def scan_stocks(rh: Robinhood, cfg: dict) -> list[str]:
    """Your universe list first, then up to scan.stocks.max_candidates screener hits, most traded first.
    Funds (ETFs, leveraged and inverse ones included) are left out unless scan.stocks.funds is true.
    If the screener fails, the morning carries on with your universe list alone."""
    s = (cfg.get("scan") or {}).get("stocks") or {}
    mine = list(cfg.get("universe") or [])
    if not s.get("enabled", True):
        return mine
    no_funds = not s.get("funds", False)
    rules = stock_filters(s, cfg["strategy"].get("order_usd", 50))
    only = await stock_only_filter(rh) if no_funds and not s.get("filters") else None
    found: list[tuple[Decimal, int, str]] = []
    dropped: list[str] = []
    try:
        try:
            data = await rh.call("preview_scan", {"filters": rules + ([only] if only else [])})
        except Exception as e:                                       # noqa: BLE001
            if not only:
                raise
            log("warning", why=f"screener refused the asset-type filter ({type(e).__name__}): funds left out by name only")
            data = await rh.call("preview_scan", {"filters": rules})
        for i, row in enumerate(find_dicts_with(data, ("ticker",))):
            t, kind = str(row.get("ticker") or "").upper().strip(), str(row.get("type") or "").lower()
            if not TICKER.fullmatch(t):
                continue
            if no_funds and (any(f in kind for f in FUND_TYPES) or FUND_NAME.search(_name(row))):
                dropped.append(t)
                continue
            found.append((_volume(row) or Decimal(0), -i, t))         # screener order breaks volume ties
    except Exception as e:                                           # noqa: BLE001  the screen is optional
        log("scan_failed", asset="stock", error=f"{type(e).__name__}: {e}"[:300], fallback="universe only")
    ranked = [t for _, _, t in sorted(found, reverse=True) if t not in mine]
    picked = mine + list(dict.fromkeys(ranked))[: int(s.get("max_candidates", 8))]
    log("scan", asset="stock", screener_hits=len(found), kept=picked, funds_left_out=dropped,
        asset_filter=only["values"][0] if only else None)
    return picked


async def scan_crypto(rh: Robinhood, cfg: dict, pairs: dict[str, CryptoPair]
                      ) -> tuple[list[str], dict[str, Quote], dict[str, Decimal], str]:
    """scan.crypto.include first, then the coins with the tightest bid/ask spread (the most liquid ones).
    Stablecoins and anything in scan.crypto.exclude are skipped.
    Returns (pairs picked, every quote seen, every spread seen, and a plain-words reason when nothing was picked)."""
    c = (cfg.get("scan") or {}).get("crypto") or {}
    skip = {str(x).upper() for x in c.get("exclude") or []}
    usable = [p.symbol for p in pairs.values() if p.tradable and p.code not in skip]
    quotes = await rh.crypto_quotes(usable) if usable else {}
    limit = Decimal(str(cfg["guardrails"].get("crypto_max_spread_pct", 1.0)))
    spread = {s: (q.ask - q.bid) / ((q.ask + q.bid) / 2) * 100 for s, q in quotes.items() if q.bid > 0 and q.ask >= q.bid}
    mine = [p for p in (pair_symbol(x if "-" in str(x) else f"{x}-USD") for x in c.get("include") or []) if p in spread]
    ranked = sorted((s for s, sp in spread.items() if sp <= limit and s not in mine), key=lambda s: spread[s])
    picked = mine + ranked[: int(c.get("max_candidates", 5))]
    tightest = min(spread, key=lambda s: spread[s]) if spread else None
    note = ""
    if not picked:                                    # say why, instead of a silent empty list
        if not pairs:
            note = "Robinhood's coin list came back empty"
        elif not usable:
            note = f"all {len(pairs)} pairs are untradable right now or are in scan.crypto.exclude"
        elif not quotes:
            note = (f"no prices came back for the {len(usable)} coins. Run  python -m agent discover  and look at "
                    "discovery/get_crypto_quotes.json to see what the fields are called")
        elif not spread:
            note = f"the {len(quotes)} prices that came back had no usable bid and ask"
        else:
            note = (f"every coin's bid/ask spread is wider than guardrails.crypto_max_spread_pct ({limit}%) right now. "
                    f"The tightest is {tightest} at {spread[tightest]:.2f}%")
    log("scan", asset="crypto", pairs=len(pairs), usable=len(usable), quoted=len(quotes), kept=picked,
        tightest=f"{tightest} {spread[tightest]:.2f}%" if tightest else None, note=note)
    return picked, quotes, spread, note
