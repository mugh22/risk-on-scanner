import numpy as np
import pandas as pd
import pytest

from src.indicators import ema, period_return, rsi
from src.exit_risk import assess_exit_risk
from src.scoring import CoinResult, regime, score_coin
from src.signals import classify, is_overextended, levels


def coin(**overrides):
    values = dict(symbol="SOL", price=120, usd_1d=2, usd_7d=8, usd_30d=25, rel_7d=5, rel_30d=15, ema20=110, ema50=100, ema200=80, rsi=62, macd_hist=1.2, volume_ratio=1.5, breakout=True, atr=5)
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


def _frame(start=100, step=1.0, periods=120):
    close = pd.Series([start + step * i for i in range(periods)], dtype=float)
    return pd.DataFrame({"close": close, "high": close + 2, "low": close - 2, "open": close - 1, "volume": 1000.0})


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
