import asyncio

from agent.__main__ import tickers
from tests.test_end_to_end import cfg as base_cfg


def test_tickers_flags_price_over_order_size(capsys, monkeypatch):
    monkeypatch.setenv("MOCK_DROP_PCT", "0")
    c = base_cfg("paper")
    c["universe"] = ["AAA", "BBB"]
    c["strategy"]["order_usd"] = 50                 # mock price is ~100, so one share does not fit
    asyncio.run(tickers(c))
    out = capsys.readouterr().out
    assert "costs more than order_usd" in out and "0 of 2 tickers fit" in out
    c["strategy"]["order_usd"] = 250
    asyncio.run(tickers(c))
    assert "2 of 2 tickers fit" in capsys.readouterr().out
