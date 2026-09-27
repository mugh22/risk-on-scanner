from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from src.portfolio import Holding
from src.coinbase_portfolio import holdings_from_accounts, merge_holdings
from src.portfolio_risk import weekly_stop, build_risk_plan, price_text
from src.reporting import _risk_plan_html, _risk_plan_text

NOW = datetime(2026, 9, 27, tzinfo=timezone.utc)


def frame():
    time = pd.date_range('2026-01-05', periods=259, tz='UTC')
    return pd.DataFrame(dict(time=time, open=100., high=110., low=90., close=100., volume=1000.))


def test_stop_uses_completed_weeks_and_does_not_widen():
    f = frame()
    original = weekly_stop(f, live=120, now=NOW)
    assert original['stop'] == pytest.approx(80)
    f.loc[f.time >= '2026-09-21', 'low'] = 1
    assert weekly_stop(f, live=120, now=NOW)['stop'] == original['stop']
    old = dict(original, stop=85, week='older week')
    newer = weekly_stop(f, old, 120, NOW)
    assert newer['stop'] == 85
    assert newer['status'] == 'KEEP PRIOR — LOWER LEVEL REJECTED'


def test_breach_persists_and_no_raise_on_same_week():
    f = frame()
    first = weekly_stop(f, live=120, now=NOW)
    old = dict(first, stop=70)
    assert weekly_stop(f, old, 120, NOW)['stop'] == 70
    breached = weekly_stop(f, first, 75, NOW)
    assert breached['breached']
    assert weekly_stop(f, breached, 120, NOW)['breached']
    old['week'] = 'older'
    assert weekly_stop(f, old, 120, NOW)['stop'] == 80


def test_cash_excludes_holds_and_manual_cash():
    holdings = holdings_from_accounts([{'currency':'USDC','available_balance':{'value':'1000'},'hold':{'value':'500'}}])
    merged = merge_holdings(holdings, [Holding('USDC',5000)])
    assert merged[0].quantity == 1500
    assert merged[0].available_quantity == 1000


def setup():
    coins = [SimpleNamespace(symbol=s, live_price=100., score=90-i, signal='BUY',
              deployment_status='BREAKOUT RETEST — ADD',entry_low=99.,entry_high=101.,target2=160.)
             for i,s in enumerate(['SOL','LINK','AAVE','ONDO'])]
    rows = [{'symbol':'USDC','value':10000.,'price':1.,'quantity':10000.,'signal':'CASH'}]
    holdings = [Holding('USDC',10000,available_quantity=8000)]
    return holdings, rows, {c.symbol:frame() for c in coins}, coins


def test_allocations_capped_by_cash_and_portfolio_stop_risk():
    holdings,rows,frames,coins = setup()
    plan,_ = build_risk_plan(holdings,rows,frames,coins,{},'Live quantities from Coinbase',2,now=NOW)
    allocated = list(plan['allocations'].values())
    assert allocated
    assert sum(a['usd'] for a in allocated) <= 8000*.3
    assert sum(a['risk_usd'] for a in allocated) <= 10000*.01 + 1e-8
    assert all(a['cash_pct'] <= 10 for a in allocated)
    assert all(a['risk_usd'] <= 50+1e-8 for a in allocated)


@pytest.mark.parametrize('case',['fallback','defensive','extended','outside_entry','poor_rr','unpriced','concentrated','no_cash'])
def test_unsafe_deployment_is_zero(case):
    holdings, rows, frames, coins = setup()
    source, score = 'Live quantities from Coinbase', 2
    if case == 'fallback': source = 'Coinbase unavailable; issue fallback'
    if case == 'defensive': score = 60
    if case == 'extended':
        for c in coins: c.deployment_status = 'WAIT — EXTENDED'
    if case == 'outside_entry':
        for c in coins: c.live_price = 110
    if case == 'poor_rr':
        for c in coins: c.target2 = 105
    if case == 'unpriced': holdings.append(Holding('UNKNOWN',10))
    if case == 'no_cash': holdings = [Holding('USDC',10000)]
    if case == 'concentrated':
        coins = coins[:1]
        rows.append(dict(symbol='SOL',value=10000,price=100,quantity=100,signal='BUY'))
    plan,_ = build_risk_plan(holdings,rows,frames,coins,{},source,score,now=NOW)
    assert not plan['allocations']


def test_tiny_prices_are_readable_in_both_report_formats():
    assert price_text(.00009512) == '0.00009512'
    row = dict(symbol='SPELL',cycle_stop=.00009512,stop_distance=15.,stop_loss_usd=1500,
               stop_status='NEW — NOT PLACED',signal='BUY')
    plan = dict(cash=1000,allocations={},note='DEPLOY 0%')
    assert '0.00009512' in _risk_plan_html([row],plan)
    assert '0.00009512' in _risk_plan_text([row],plan)
