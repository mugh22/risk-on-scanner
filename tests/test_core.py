import numpy as np
import pandas as pd
import pytest

from src.indicators import ema, period_return, rsi
from src.exit_risk import assess_exit_risk
from src.exit_risk import ExitRiskResult
from src.reporting import render
from src.portfolio import parse_portfolio_table
from src.profit_protection import assess_profit_protection
from src.timeframes import completed_daily, completed_weekly_closes
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


def test_profit_protection_separates_hot_rally_from_exit_risk():
    result = assess_profit_protection(coin(rsi=82, usd_30d=90, price=145, ema20=110, breakout=False), 5)
    assert result.score >= 60
    assert "TRIM" in result.action


def test_open_daily_and_current_week_are_excluded():
    now = pd.Timestamp("2026-09-19T12:00:00Z")
    time = pd.date_range("2026-09-01", "2026-09-19", freq="D", tz="UTC")
    frame = pd.DataFrame({"time": time, "close": range(len(time))})
    daily = completed_daily(frame, now.to_pydatetime())
    weekly = completed_weekly_closes(daily, now.to_pydatetime())
    assert daily.time.max() == pd.Timestamp("2026-09-18T00:00:00Z")
    assert weekly.index.max() == pd.Timestamp("2026-09-07T00:00:00Z")
