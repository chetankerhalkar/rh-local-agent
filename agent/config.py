from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
VALID_MODES = ("paper", "approve", "auto")


class ConfigError(Exception):
    pass


def load(path: str | None = None) -> dict:
    p = Path(path) if path else ROOT / "config.yaml"
    if not p.exists():
        raise ConfigError(f"{p} not found. Copy config.example.yaml to config.yaml and edit it.")
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    cfg["universe"] = [str(s).upper().strip() for s in (cfg.get("universe") or [])]
    validate(cfg)
    return cfg


PLACEHOLDER = re.compile(r"^(TICKER|SYMBOL|STOCK|YOUR)[_-]?\w*$|^(AAA|BBB|XXX|ZZZ)$")


def validate(cfg: dict) -> None:
    fake = [t for t in cfg["universe"] if PLACEHOLDER.match(t)]
    if fake and (cfg.get("robinhood") or {}).get("transport") != "mock":
        raise ConfigError(f"universe still has placeholder names: {', '.join(fake)}\n"
                          "Replace them with real stock symbols, the short codes you see in the Robinhood app.\n"
                          "Which stocks is your decision. Then run:  python -m agent tickers")
    bad = [t for t in cfg["universe"] if not re.fullmatch(r"[A-Z]{1,5}([.-][A-Z]{1,2})?", t)]
    if bad:
        raise ConfigError(f"these do not look like stock symbols: {', '.join(bad)}  (expected 1 to 5 letters, like the codes in the Robinhood app)")
    mode = cfg.get("mode")
    if mode not in VALID_MODES:
        raise ConfigError(f"mode must be one of {VALID_MODES}, got {mode!r}")
    for section in ("robinhood", "llm", "strategy", "guardrails"):
        if not isinstance(cfg.get(section), dict):
            raise ConfigError(f"missing config section: {section}")
    if mode != "paper" and not cfg["robinhood"].get("account_number"):
        raise ConfigError("robinhood.account_number is required for approve/auto mode.")
    if mode == "auto":
        if not cfg.get("i_understand_auto_places_real_orders"):
            raise ConfigError("auto mode needs i_understand_auto_places_real_orders: true")
        if os.environ.get("RH_AGENT_ALLOW_AUTO") != "1":
            raise ConfigError("auto mode also needs environment variable RH_AGENT_ALLOW_AUTO=1")
    g, s = cfg["guardrails"], cfg["strategy"]
    if float(s.get("order_usd", 0)) > float(g.get("max_order_usd", 0)):
        raise ConfigError("strategy.order_usd is larger than guardrails.max_order_usd")
    stocks, d = (cfg.get("scan") or {}).get("stocks") or {}, cfg.get("daily") or {}
    if stocks.get("enabled", True) and "max_price" in stocks and float(stocks["max_price"]) > float(s.get("order_usd", 0)):
        raise ConfigError("scan.stocks.max_price is over strategy.order_usd: the scan would pick stocks\n"
                          "where one share costs more than an order may spend, so they could never be bought.")
    if float(d.get("sell_slack_pct", 0.10)) > float(g.get("max_limit_slippage_pct", 0.20)):
        raise ConfigError("daily.sell_slack_pct is over guardrails.max_limit_slippage_pct: every evening sell would be blocked.")
    if str(d.get("when_down", "ask")).lower() not in ("ask", "sell", "keep"):
        raise ConfigError("daily.when_down must be ask, sell or keep")


def kill_switch_on() -> bool:
    """Create an empty file named KILL in the project folder to stop all orders."""
    return (ROOT / "KILL").exists()
