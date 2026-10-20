"""The exact failure a user hit: placeholder / unknown tickers. Must be caught early and explained plainly."""
import asyncio
import subprocess
import sys

import pytest

from agent import config
from agent.__main__ import _explain, _leaves, tickers
from agent.rh import RHError, connect
from tests.test_end_to_end import cfg as base_cfg


def test_placeholders_are_rejected_before_any_network_call():
    c = base_cfg("paper")
    c["robinhood"]["transport"] = "http"
    c["universe"] = ["TICKER1", "TICKER2", "F"]
    with pytest.raises(config.ConfigError, match="placeholder names: TICKER1, TICKER2"):
        config.validate(c)
    c["universe"] = ["F", "BRK.B", "SOFI"]
    config.validate(c)                                   # real-looking symbols pass
    c["universe"] = ["APPLE INC"]
    with pytest.raises(config.ConfigError, match="do not look like stock symbols"):
        config.validate(c)


def test_unknown_ticker_does_not_break_the_good_ones(monkeypatch, capsys):
    monkeypatch.setenv("MOCK_MISSING", "NOPE")
    c = base_cfg("paper")
    c["universe"] = ["AAA", "NOPE", "BBB"]

    async def go():
        async with connect(c) as rh:
            q = await rh.quotes(c["universe"])
            return sorted(q), rh.missing
    got, missing = asyncio.run(go())
    assert got == ["AAA", "BBB"] and missing == ["NOPE"]
    c["strategy"]["order_usd"] = 250
    asyncio.run(tickers(c))
    out = capsys.readouterr().out
    assert "Robinhood does not know this symbol" in out and "2 of 3 tickers fit" in out


def test_nested_error_groups_become_one_plain_line():
    inner = RHError('get_equity_quotes failed: API error 404: {"missing_instruments":["TICKER1"]}')
    wrapped = ExceptionGroup("unhandled errors in a TaskGroup", [ExceptionGroup("again", [inner])])
    lines = [_explain(x) for x in _leaves(wrapped)]
    assert len(lines) == 1 and lines[0].startswith("ROBINHOOD: it does not know") and "Traceback" not in lines[0]
