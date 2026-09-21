import numpy as np
import pandas as pd
import pytest

from src.indicators import ema, period_return, rsi
from src.exit_risk import assess_exit_risk
from src.exit_risk import ExitRiskResult
from src.reporting import render, render_weekly
from src.portfolio import parse_portfolio_table
from src.coinbase_portfolio import holdings_from_accounts, merge_holdings
from src.portfolio import Holding
from src.scanner import build_portfolio_rows, previous_closed_score
from src.profit_protection import assess_profit_protection
from src.positioning import assess_positioning
from src.timeframes import completed_daily, completed_weekly_closes
from src.weekly import holding_action, snapshot as weekly_snapshot, weekly_bars, weekly_market
from src.scoring import CoinResult, regime, score_coin
from src.signals import classify, is_overextended, levels


def coin(**overrides):
    values = dict(symbol="SOL", price=120, usd_1d=2, usd_7d=8, usd_30d=25, rel_7d=5, rel_30d=15, ema20=110, ema50=100, ema200=80, rsi=62, macd_hist=1.2, volume_ratio=1.5, breakout=True, atr=5, weekly_rel=4, weekly_constructive=True, daily_higher_closes=3, daily_rel_confirmations=3)
    values.update(overrides)
    return CoinResult(**values)


def test_ema_and_returns():
    values = pd.Series(range(1, 61), dtype=float)
    assert ema(values, 20).iloc[-1] > ema(values, 50).iloc[-1]
    assert period_return(pd.Series([100.0, 110.0]), 1) == pytest.approx(10.0)


def test_rsi_bounds_and_direction():
    rising = rsi(pd.Series(np.arange(1, 40), dtype=float)).iloc[-1]
    falling = rsi(pd.Series(np.arange(40, 1, -1), dtype=float)).iloc[-1]
    assert rising == 100 and falling == 0


def test_regimes():
    assert [regime(x) for x in [0, 35, 55, 70, 85]] == ["RISK-OFF", "NEUTRAL", "EARLY RISK-ON", "RISK-ON", "STRONG RISK-ON"]


def test_score_and_buy_logic():
    c = coin()
    weights = {"relative_30d":30,"relative_7d":15,"trend":25,"momentum":15,"volume":8,"breakout":7}
    c.score, _ = score_coin(c, weights)
    cfg = {"min_buy_score":72,"min_watch_score":58,"min_risk_on_score":55,"min_volume_ratio":.9,"max_rsi_buy":74,"max_distance_ema20_pct":12,"max_daily_move_pct":15}
    assert c.score >= 72
    assert classify(c, 75, cfg) == "BUY"


def test_overextension_blocks_buy():
    c = coin(rsi=82); c.score=90
    cfg = {"min_buy_score":72,"min_watch_score":58,"min_risk_on_score":55,"min_volume_ratio":.9,"max_rsi_buy":74,"max_distance_ema20_pct":20,"max_daily_move_pct":15}
    assert is_overextended(c, cfg)[0]
    assert classify(c, 80, cfg) == "WATCH"


def test_volatility_levels():
    c=coin(); levels(c,130,105)
    assert c.invalidation < c.price < c.target1 < c.target2
    assert c.reward_risk > 1


def test_execution_levels_can_use_live_price_without_mutating_signal_close():
    c = coin(price=100, live_price=112); levels(c, 108, 90, c.live_price)
    assert c.price == 100
    assert c.entry_low < 112 < c.entry_high
    assert c.target1 > 112


def _frame(start=100, step=1.0, periods=120):
    close = pd.Series([start + step * i for i in range(periods)], dtype=float)
    time = pd.date_range(end=pd.Timestamp.now(tz="UTC").normalize()-pd.Timedelta(days=1), periods=periods, freq="D")
    return pd.DataFrame({"time": time, "close": close, "high": close + 2, "low": close - 2, "open": close - 1, "volume": 1000.0})


def test_exit_risk_constructive_market_is_not_exit():
    btc = _frame(100, 1.0)
    frames = {f"C{i}": _frame(50 + i, 1.2) for i in range(10)}
    coins = [coin(symbol=name, rel_7d=5, rel_30d=15, rsi=62) for name in frames]
    result = assess_exit_risk(btc, _frame(0.05, .0001), frames, coins)
    assert result.score < 45
    assert "EXIT" not in result.call


def test_exit_risk_detects_broad_deterioration():
    btc = _frame(220, -1.0)
    frames = {f"C{i}": _frame(180 + i, -1.25) for i in range(10)}
    coins = [coin(symbol=name, price=35, ema20=50, ema50=70, rel_7d=-12, rel_30d=-28, rsi=28) for name in frames]
    result = assess_exit_risk(btc, _frame(0.08, -.00025), frames, coins)
    assert result.score >= 60
    assert result.call in {"REDUCE ALT RISK", "EXIT MOST ALT RISK"}


def test_dominance_rotation_increases_concentration_component():
    btc = _frame(100, .5)
    frames = {f"C{i}": _frame(50 + i, .2) for i in range(10)}
    coins = [coin(symbol=name) for name in frames]
    current = {"concentration": 82.0, "stable_btc_ratio": .19}
    previous = {"concentration": 80.5, "stable_btc_ratio": .18}
    result = assess_exit_risk(btc, None, frames, coins, current, previous)
    assert result.components["Capital concentration"] >= 60


def test_buy_is_always_in_opportunity_table_and_bars_are_email_safe():
    coins = [coin(symbol=f"C{i}", price=100+i) for i in range(11)]
    for i, item in enumerate(coins):
        item.score = 99-i
        item.signal = "WATCH"
        levels(item, item.price+10, item.price-10)
    coins[-1].signal = "BUY"
    risk = ExitRiskResult(22, "LOW", "HOLD", "Healthy", {"Relative strength": 20}, {"Relative strength": "Healthy · stable"})
    context = {"btc_constructive": True, "eth_btc_positive": True, "breadth_20": 80, "breadth_rel30": 70, "weekly_breadth": 70, "daily_confirmation_breadth": 60, "month_regime": 80, "weekly_close": 75, "last_3_closes": 65}
    html, _ = render(80, 79, coins, [], context, risk, [20, 22])
    table = html.split("<h2>Market opportunities</h2>", 1)[1].split("</table>", 1)[0]
    assert "C10" in table and "BUY" in table
    assert "bar-track" in html and "background:#e2e8f0" in html
    assert html.index("Your portfolio — actions first") < html.index("Market evidence")
    assert "Live price" in html and "Signal close" in html


def test_report_prefers_live_price_but_keeps_signal_close():
    item = coin(price=100, live_price=107); item.score = 80; item.signal = "WATCH"; levels(item, 110, 90)
    risk = ExitRiskResult(10, "LOW", "HOLD", "Healthy", {"Relative strength": 0}, {"Relative strength": "Healthy · stable"})
    context = {"btc_constructive": True, "eth_btc_positive": True, "breadth_20": 80, "breadth_rel30": 70, "weekly_breadth": 70, "daily_confirmation_breadth": 60, "month_regime": 80, "weekly_close": 75, "last_3_closes": 65}
    html, _ = render(70, 70, [item], [], context, risk, [10, 10])
    assert "$107.0000" in html and "$100.0000" in html
    assert "Previous daily close: 70.0 → Current: 70.0 (+0.0)" in html


def test_zero_component_still_has_visible_bar_track():
    coins = [coin()]; coins[0].score = 80; coins[0].signal = "WATCH"; levels(coins[0], 130, 105)
    risk = ExitRiskResult(0, "LOW", "HOLD", "Healthy", {"Relative strength": 0}, {"Relative strength": "Healthy · stable"})
    context = {"btc_constructive": True, "eth_btc_positive": True, "breadth_20": 80, "breadth_rel30": 70, "weekly_breadth": 70, "daily_confirmation_breadth": 60, "month_regime": 80, "weekly_close": 75, "last_3_closes": 65}
    html, _ = render(80, 79, coins, [], context, risk, [0, 0])
    assert "width:100%;background:#e2e8f0" in html
    assert "width:0%;background:#16803c" not in html
    assert "Healthy · stable" in html


def test_exit_risk_direction_warns_before_confirmed_score_changes():
    btc = _frame(100, 1.0)
    frames = {f"C{i}": _frame(50 + i, 1.2) for i in range(10)}
    coins = [coin(symbol=name, rel_7d=5, rel_30d=15, rsi=62) for name in frames]
    baseline = assess_exit_risk(btc, _frame(0.05, .0001), frames, coins)
    weaker = [coin(symbol=name, rel_7d=2, rel_30d=11, rsi=62) for name in frames]
    result = assess_exit_risk(btc, _frame(0.05, .00008), frames, weaker, previous_metrics=baseline.metrics)
    assert result.components["Relative strength"] == 0
    assert result.directions["Relative strength"] == "Healthy · worsening"


def test_portfolio_issue_table_parser():
    body = """| Symbol | Quantity | Average Cost | Target % | Enabled |
|---|---:|---:|---:|---|
| BTC | 0.5 | $40,000 | 50% | Yes |
| ARB | 1000 | 0.50 | 10 | No |
| SOL | 4 | 100 | 20 | Yes |"""
    holdings = parse_portfolio_table(body)
    assert [h.symbol for h in holdings] == ["BTC", "SOL"]
    assert holdings[0].average_cost == 40000


def test_coinbase_balances_are_combined_and_issue_metadata_is_preserved():
    accounts = [
        {"currency": "ARB", "available_balance": {"value": "100"}, "hold": {"value": "5"}},
        {"currency": "ARB", "available_balance": {"value": "2"}},
        {"currency": "USD", "available_balance": {"value": "0"}},
    ]
    live = holdings_from_accounts(accounts)
    merged = merge_holdings(live, [Holding("ARB", 999, .55, 20), Holding("AERO", 10, 1.1)])
    assert merged[0] == Holding("ARB", 107, .55, 20)
    assert merged[1] == Holding("AERO", 10, 1.1)


def test_portfolio_dust_is_filtered_and_rows_are_sorted_by_allocation():
    raw = [
        {"holding": Holding("SMALL", 5), "price": 1, "signal": "CASH", "heat": 0, "action": "HOLD"},
        {"holding": Holding("MID", 2), "price": 10, "signal": "WATCH", "heat": 10, "action": "HOLD"},
        {"holding": Holding("BIG", 10), "price": 10, "signal": "WATCH", "heat": 20, "action": "HOLD"},
    ]
    rows, excluded = build_portfolio_rows(raw)
    assert excluded == 1
    assert [row["symbol"] for row in rows] == ["BIG", "MID"]
    assert round(sum(row["allocation"] for row in rows), 8) == 100


def test_profit_protection_separates_hot_rally_from_exit_risk():
    result = assess_profit_protection(coin(rsi=82, usd_30d=90, price=145, ema20=110, breakout=False), 5)
    assert result.score >= 60
    assert "TRIM" in result.action


def test_positioning_separates_btc_bull_from_weak_ethbtc():
    coins = [coin(symbol=f"C{i}", rel_7d=5, rel_30d=10,
                  weekly_constructive=i < 6, daily_rel_confirmations=3) for i in range(10)]
    btc = coin(symbol="BTC", live_price=121)
    context = {"btc_constructive": True, "eth_btc_positive": False,
               "breadth_20": 90, "breadth_rel30": 80,
               "weekly_breadth": 60, "daily_confirmation_breadth": 80}
    risk = ExitRiskResult(10, "LOW", "HOLD", "Healthy", {})
    result = assess_positioning(78, context, btc, coins, risk,
                                {"score": 30}, {}, {})
    assert result.market_regime == "CONFIRMED RISK-ON"
    assert result.rotation_phase == "BROAD ALT EXPANSION"
    assert "ETH/BTC is weak" in result.evidence[3]


def test_positioning_detects_hidden_relative_narrowing():
    coins = [coin(symbol=f"C{i}", rel_7d=-5, rel_30d=-2,
                  weekly_constructive=False, daily_rel_confirmations=1) for i in range(10)]
    btc = coin(symbol="BTC", live_price=121)
    context = {"btc_constructive": True, "eth_btc_positive": False,
               "breadth_20": 80, "breadth_rel30": 20,
               "weekly_breadth": 10, "daily_confirmation_breadth": 20}
    risk = ExitRiskResult(20, "LOW", "HOLD", "Healthy", {})
    result = assess_positioning(65, context, btc, coins, risk,
                                {"score": 30}, {}, {})
    assert result.rotation_phase in {"ALT RISK-OFF", "LATE / NARROWING ROTATION"}
    assert "weak alts" in result.existing_action or "BTC" in result.existing_action


def test_open_daily_and_current_week_are_excluded():
    now = pd.Timestamp("2026-09-19T12:00:00Z")
    time = pd.date_range("2026-09-01", "2026-09-19", freq="D", tz="UTC")
    frame = pd.DataFrame({"time": time, "close": range(len(time))})
    daily = completed_daily(frame, now.to_pydatetime())
    weekly = completed_weekly_closes(daily, now.to_pydatetime())
    assert daily.time.max() == pd.Timestamp("2026-09-18T00:00:00Z")
    assert weekly.index.max() == pd.Timestamp("2026-09-07T00:00:00Z")


def test_weekly_mode_uses_completed_weeks_and_multiweek_relative_strength():
    btc = _frame(100, 1.0, 240)
    alt = _frame(50, 1.2, 240)
    bars = weekly_bars(alt)
    item = weekly_snapshot("ALT", alt, btc)
    market = weekly_market({"BTC": weekly_snapshot("BTC", btc, btc), "ALT": item})
    assert len(bars) >= 30
    assert item.relative_4w > 0 and item.relative_12w > 0
    assert item.trend in {"ADVANCING", "CONSTRUCTIVE"}
    assert market["horizon"]
    assert "HOLD" in holding_action(item, 10, 10)
    risk = ExitRiskResult(20, "LOW", "HOLD", "Healthy", {"Relative strength": 0})
    html, text = render_weekly(market, {"BTC": weekly_snapshot("BTC", btc, btc), "ALT": item},
                               [{"symbol": "ALT", "allocation": 10, "weekly": item,
                                 "weekly_action": holding_action(item, 10, 10)}], None, risk, {}, None, None)
    assert "2–6 week scenario map" in html
    assert "WEEKLY CRYPTO OUTLOOK" in text
