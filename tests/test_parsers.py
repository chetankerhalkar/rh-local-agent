"""Robinhood response parsing and order arguments. These run offline; `discover` checks them against the real server."""
from datetime import datetime, timezone
from decimal import Decimal as D

from agent import scan
from agent.models import Candidate
from agent.rh import asset_code, order_fill, pair_symbol, parse_ts, step_down, step_up


def test_parse_ts_handles_every_flavour():
    utc = datetime(2026, 9, 21, 15, 0, 0, 123456, tzinfo=timezone.utc)
    assert parse_ts("2026-09-21T15:00:00.123456789Z") == utc                  # nanoseconds, UTC
    assert parse_ts("2026-09-21T11:00:00.123456-04:00") == utc
    assert parse_ts("2026-09-21T11:00:00.123456-0400") == utc
    assert parse_ts("2026-09-21 11:00:00.123456") == utc                      # no zone: US Eastern, like crypto quotes
    assert parse_ts("yesterday") is None and parse_ts(None) is None


def test_pair_symbols_and_coin_codes():
    assert pair_symbol("BTCUSD") == pair_symbol("btc-usd") == "BTC-USD"
    assert pair_symbol("USDC-USD") == "USDC-USD" and pair_symbol("BTC") is None and pair_symbol("") is None
    assert asset_code({"asset_code": "ETH"}) == "ETH"
    assert asset_code({"currency": {"code": "DOGE"}}) == "DOGE"
    assert asset_code({"symbol": "BTCUSD"}) == "BTC"
    assert asset_code({"asset": "SOL-USD"}) == "SOL"


def test_order_steps_round_the_safe_way():
    assert step_down(D("0.0208084"), D("0.000001")) == D("0.020808")        # sell / stop amounts: never more than held
    assert step_down(D("9.9099"), D("0.01")) == D("9.90")
    assert step_up(D("2402.401"), D("0.01")) == D("2402.41")                # a buy limit rounded up still reaches the ask


def test_order_fill_reads_every_shape():
    assert order_fill(None) == ("unknown", D(0), None)
    assert order_fill({"state": "filled", "cumulative_quantity": "4", "average_price": "10.01"}) == ("filled", D(4), D("10.01"))
    two_fills = {"state": "partially_filled", "executions": [{"quantity": "1", "price": "10"}, {"quantity": "1", "price": "12"}]}
    assert order_fill(two_fills) == ("partially_filled", D(2), D(11))
    assert order_fill({"state": "filled", "quantity": "3"})[1] == D(3)


def test_crypto_order_arguments():
    stop = Candidate("ETH-USD", "sell", "stop_market", "t", quantity=D("0.020808"), stop_price=D("2330.32"),
                     kind="protect", asset="crypto", time_in_force="gtc")
    assert stop.to_crypto_args("5550001") == {"rhs_account_number": "5550001", "symbol": "ETH-USD", "side": "sell",
                                              "type": "stop_loss", "time_in_force": "gtc", "quantity": "0.020808",
                                              "stop_price": "2330.32"}
    buy = Candidate("ETH-USD", "buy", "limit", "t", dollar_amount=D("49.99"), limit_price=D("2402.40"), asset="crypto")
    args = buy.to_crypto_args("5550001")
    assert args["time_in_force"] == "gtc" and args["dollar_amount"] == "49.99" and args["limit_price"] == "2402.4"
    assert "quantity" not in args


def test_stock_stop_keeps_its_time_in_force():
    stop = Candidate("AAA", "sell", "stop_market", "t", quantity=D(4), stop_price=D("9.90"), kind="protect", time_in_force="gtc")
    assert stop.to_order_args("ACC")["time_in_force"] == "gtc"
    assert Candidate("AAA", "buy", "limit", "t", quantity=D(1), limit_price=D(10)).to_order_args("ACC")["time_in_force"] == "gfd"


def test_crypto_quotes_accept_robinhoods_other_field_names():
    """Robinhood has used bid_price/ask_price and the spread-inclusive names, plain or as {"amount": ...}."""
    import asyncio

    from agent.rh import Robinhood
    payload = {"data": {"results": [{"symbol": "BTCUSD", "bid_inclusive_of_sell_spread": {"amount": "59900.00"},
                                     "ask_inclusive_of_buy_spread": {"amount": "60100.00"},
                                     "open_price": "59000", "updated_at": "2026-09-19 17:00:00"}]}}
    rh = Robinhood.__new__(Robinhood)
    rh.crypto_account = ""

    async def call(name, args):
        return payload
    rh.call = call
    q = asyncio.run(rh.crypto_quotes(["BTC-USD"]))["BTC-USD"]
    assert (q.bid, q.ask, q.last, q.prev_close) == (D("59900.00"), D("60100.00"), D(60000), D(59000))
    assert q.ts is not None and q.asset == "crypto"


def test_screener_volume_found_in_any_row_shape():
    assert scan._volume({"ticker": "A", "cells": [{"display_name": "Volume", "value": "123"}]}) == D(123)
    assert scan._volume({"ticker": "A", "values": {"Volume": "456"}}) == D(456)
    assert scan._volume({"ticker": "A", "volume": "789"}) == D(789)
    assert scan._volume({"ticker": "A", "cells": [{"column": "Price", "value": "5"}]}) is None


def test_screener_filters_cap_the_price_at_one_order():
    rules = [f["expression"] for f in scan.stock_filters({"min_price": 5}, order_usd=40)]
    assert "tradeAllDay.price <= 40" in rules and "tradeAllDay.price >= 5" in rules
    assert scan.stock_filters({"filters": [{"expression": "x"}]}) == [{"expression": "x"}]      # yours replace ours
