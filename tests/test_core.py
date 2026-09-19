import numpy as np
import pandas as pd
import pytest

from src.indicators import ema, period_return, rsi
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
