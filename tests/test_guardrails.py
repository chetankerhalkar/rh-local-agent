"""Every guardrail gets a test. If you add a rule, add a test. Run:  python -m pytest -q"""
from datetime import datetime, timedelta
from decimal import Decimal as D

import pytest

from agent import guardrails as g
from agent.models import Candidate, DayState, Position, Quote, Snapshot

NOW = datetime(2026, 9, 16, 11, 0, tzinfo=g.ET)       # a Wednesday, 11:00 ET

CFG = {"universe": ["AAA", "BBB"],
       "guardrails": {"max_order_usd": 60, "max_daily_buy_usd": 150, "max_trades_per_day": 4,
                      "max_position_usd": 120, "max_open_positions": 2, "daily_loss_stop_pct": 2.0,
                      "allowed_order_types": ["limit"], "regular_hours_only": True,
                      "no_new_buys_first_minutes": 15, "no_new_buys_last_minutes": 20,
                      "max_quote_age_seconds": 90, "max_spread_pct": 0.30, "max_limit_slippage_pct": 0.20}}


def quote(sym="AAA", last="50", bid="49.99", ask="50.01", age=5, **kw):
    return Quote(sym, D(last), D(bid), D(ask), D("51"), NOW - timedelta(seconds=age), **kw)


def snap(now=NOW, quotes=None, positions=None, bp="500", eq="500"):
    return Snapshot(now, quotes or {"AAA": quote()}, positions or {}, D(bp) if bp else None, D(eq) if eq else None)


def buy(sym="AAA", qty="1", limit="50.01"):
    return Candidate(sym, "buy", "limit", "test", quantity=D(qty), limit_price=D(limit))


def day(**kw):
    return DayState(date="2026-09-16", start_equity="500", **kw)


def test_clean_buy_passes():
    assert g.check(buy(), snap(), day(), CFG) == []


@pytest.mark.parametrize("label, cand, s, d, kill, expect", [
    ("kill switch",        buy(), snap(), day(), True, "KILL"),
    ("not in universe",    buy("ZZZ"), snap(quotes={"ZZZ": quote("ZZZ")}), day(), False, "universe"),
    ("market order type",  Candidate("AAA", "buy", "market", "t", dollar_amount=D("50")), snap(), day(), False, "not allowed"),
    ("weekend",            buy(), snap(now=datetime(2026, 9, 19, 11, 0, tzinfo=g.ET)), day(), False, "closed"),
    ("after hours",        buy(), snap(now=NOW.replace(hour=17)), day(), False, "closed"),
    ("just after open",    buy(), snap(now=NOW.replace(hour=9, minute=35)), day(), False, "after the open"),
    ("just before close",  buy(), snap(now=NOW.replace(hour=15, minute=50)), day(), False, "before the close"),
    ("stale quote",        buy(), snap(quotes={"AAA": quote(age=600)}), day(), False, "stale"),
    ("halted",             buy(), snap(quotes={"AAA": quote(state="halted")}), day(), False, "state="),
    ("wide spread",        buy(limit="50.00"), snap(quotes={"AAA": quote(bid="49.00", ask="51.00")}), day(), False, "spread"),
    ("limit far over ask", buy(limit="52.00"), snap(), day(), False, "above the ask"),
    ("order too big",      buy(qty="2", limit="50.01"), snap(), day(), False, "max_order_usd"),
    ("daily buy cap",      buy(), snap(), day(buy_usd_total="120"), False, "max_daily_buy_usd"),
    ("trade count cap",    buy(), snap(), day(trades=[{"symbol": f"B{i}", "side": "buy", "status": "placed"} for i in range(4)]), False, "max_trades_per_day"),
    ("same trade twice",   buy(), snap(), day(trades=[{"symbol": "AAA", "side": "buy", "status": "placed"}]), False, "already did"),
    ("unknown counts too", buy(), snap(), day(trades=[{"symbol": "AAA", "side": "buy", "status": "unknown"}]), False, "already did"),
    ("position cap",       buy(), snap(positions={"AAA": Position("AAA", D(2), D(50))}), day(), False, "max_position_usd"),
    ("too many positions", buy(), snap(positions={"X": Position("X", D(1), D(1)), "Y": Position("Y", D(1), D(1))}), day(), False, "max_open_positions"),
    ("no buying power",    buy(), snap(bp="10"), day(), False, "buying power"),
    ("bp unknown",         buy(), snap(bp=None), day(), False, "buying power unknown"),
    ("daily loss stop",    buy(), snap(eq="480"), day(), False, "daily loss stop"),
])
def test_blocks(label, cand, s, d, kill, expect):
    violations = g.check(cand, s, d, CFG, kill)
    assert any(expect in v for v in violations), f"{label}: expected {expect!r} in {violations}"


def test_stop_order_rules():
    cfg = {**CFG, "protect": {"broker_stop": True}}
    s = snap(positions={"AAA": Position("AAA", D(2), D(50))})
    good = Candidate("AAA", "sell", "stop_market", "t", quantity=D(2), stop_price=D("49.50"), kind="protect", is_exit=True)
    assert g.check(good, s, day(), cfg) == []
    assert any("not allowed" in v for v in g.check(good, s, day(), CFG))                       # protection switched off
    above = Candidate("AAA", "sell", "stop_market", "t", quantity=D(2), stop_price=D("50.50"), kind="protect", is_exit=True)
    assert any("trigger at once" in v for v in g.check(above, s, day(), cfg))
    frac = Candidate("AAA", "sell", "stop_market", "t", quantity=D("1.5"), stop_price=D("49.50"), kind="protect", is_exit=True)
    assert any("whole shares" in v for v in g.check(frac, s, day(), cfg))
    buy_stop = Candidate("AAA", "buy", "stop_market", "t", quantity=D(1), stop_price=D("49.50"))
    assert any("only allowed as protective sells" in v for v in g.check(buy_stop, s, day(), cfg))


def test_no_shorting():
    sell = Candidate("AAA", "sell", "limit", "t", quantity=D(1), limit_price=D("49.99"), is_exit=True)
    assert any("shorting" in v for v in g.check(sell, snap(), day(), CFG))


def test_cannot_sell_more_than_held():
    sell = Candidate("AAA", "sell", "limit", "t", quantity=D(5), limit_price=D("49.99"), is_exit=True)
    s = snap(positions={"AAA": Position("AAA", D(2), D(50))})
    assert any("larger than the position" in v for v in g.check(sell, s, day(), CFG))


CRYPTO_CFG = {**CFG, "universe": ["AAA", "ETH-USD"], "protect": {"broker_stop": True},
              "guardrails": {**CFG["guardrails"], "crypto_max_spread_pct": 1.0}}


def cquote(now=NOW, bid="2398", ask="2402"):
    return Quote("ETH-USD", D("2400"), D(bid), D(ask), D("2380"), now - timedelta(seconds=5), asset="crypto")


def cbuy(usd="49.99", limit="2402"):
    return Candidate("ETH-USD", "buy", "limit", "t", dollar_amount=D(usd), limit_price=D(limit), asset="crypto")


def test_crypto_buy_by_dollars_passes_any_day_any_hour():
    saturday_night = datetime(2026, 9, 19, 23, 0, tzinfo=g.ET)
    for now in (NOW, saturday_night, NOW.replace(hour=9, minute=31)):
        assert g.check(cbuy(), snap(now=now, quotes={"ETH-USD": cquote(now)}), day(), CRYPTO_CFG) == []


def test_crypto_keeps_every_money_limit_and_has_its_own_spread_limit():
    s = snap(quotes={"ETH-USD": cquote()})
    assert any("max_order_usd" in v for v in g.check(cbuy(usd="60.01"), s, day(), CRYPTO_CFG))
    assert any("max_daily_buy_usd" in v for v in g.check(cbuy(), s, day(buy_usd_total="120"), CRYPTO_CFG))
    wider = snap(quotes={"ETH-USD": cquote(bid="2390", ask="2410")})          # 0.83%: too wide for a stock, fine for a coin
    assert g.check(cbuy(limit="2410"), wider, day(), CRYPTO_CFG) == []
    widest = snap(quotes={"ETH-USD": cquote(bid="2370", ask="2430")})         # 2.5%: over the crypto limit
    assert any("spread" in v for v in g.check(cbuy(limit="2430"), widest, day(), CRYPTO_CFG))


def test_crypto_protective_stop_takes_fractional_amounts():
    s = snap(quotes={"ETH-USD": cquote()}, positions={"ETH-USD": Position("ETH-USD", D("0.020808"), D("2402.40"))})
    stop = Candidate("ETH-USD", "sell", "stop_market", "t", quantity=D("0.020808"), stop_price=D("2330.32"),
                     kind="protect", is_exit=True, asset="crypto", time_in_force="gtc")
    assert g.check(stop, s, day(), CRYPTO_CFG) == []


def test_protective_stop_goes_in_after_hours_or_during_a_halt():
    """Robinhood holds a stop placed after hours until the open. Blocking it would leave the holding bare."""
    cfg = {**CFG, "protect": {"broker_stop": True}}
    s = snap(now=NOW.replace(hour=17), quotes={"AAA": quote(age=3600, state="halted")},
             positions={"AAA": Position("AAA", D(2), D(50))})
    stop = Candidate("AAA", "sell", "stop_market", "t", quantity=D(2), stop_price=D("49.50"), kind="protect", is_exit=True)
    assert g.check(stop, s, day(), cfg) == []
    too_high = Candidate("AAA", "sell", "stop_market", "t", quantity=D(2), stop_price=D("50.50"), kind="protect", is_exit=True)
    assert any("trigger at once" in v for v in g.check(too_high, s, day(), cfg))
    sell = Candidate("AAA", "sell", "limit", "t", quantity=D(2), limit_price=D("49.99"), is_exit=True)
    assert any("closed" in v for v in g.check(sell, s, day(), cfg))       # an ordinary sell still waits for the market


def test_exit_allowed_near_close_and_after_loss_stop():
    """Exits must always get through: late in the day and when the account is down."""
    sell = Candidate("AAA", "sell", "limit", "t", quantity=D(2), limit_price=D("49.99"), is_exit=True)
    s = snap(now=NOW.replace(hour=15, minute=55), eq="400", positions={"AAA": Position("AAA", D(2), D(50))})
    s.quotes["AAA"].ts = s.now - timedelta(seconds=5)
    assert g.check(sell, s, day(), CFG) == []
