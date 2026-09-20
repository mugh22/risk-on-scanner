from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from .indicators import ema, period_return, rsi


@dataclass(frozen=True)
class WeeklySnapshot:
    symbol: str
    close: float
    return_4w: float
    return_12w: float
    relative_4w: float
    relative_12w: float
    ema4: float
    ema10: float
    rsi: float
    positive_weeks: int
    drawdown_12w: float
    trend: str


def weekly_bars(frame: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Aggregate daily bars into completed Monday-anchored UTC weeks."""
    if frame.empty:
        return frame.copy()
    now = now or datetime.now(timezone.utc)
    stamp = pd.Timestamp(now)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    current_week = stamp.normalize() - pd.Timedelta(days=stamp.weekday())
    weekly = frame.set_index("time").resample("W-MON", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"),
    )
    return weekly.loc[weekly.index < current_week].dropna().reset_index()


def snapshot(symbol: str, frame: pd.DataFrame, btc_frame: pd.DataFrame) -> WeeklySnapshot:
    weekly, btc = weekly_bars(frame), weekly_bars(btc_frame)
    if len(weekly) < 13 or len(btc) < 13:
        raise ValueError(f"Insufficient completed weekly history for {symbol}")
    close, btc_close = weekly.close, btc.close
    e4, e10 = ema(close, 4), ema(close, 10)
    ret4, ret12 = period_return(close, 4), period_return(close, 12)
    rel4 = ret4 - period_return(btc_close, 4)
    rel12 = ret12 - period_return(btc_close, 12)
    current = float(close.iloc[-1])
    weekly_rsi = float(rsi(close, 14).iloc[-1])
    positive = int((close.pct_change().tail(3) > 0).sum())
    drawdown = (current / float(weekly.high.tail(12).max()) - 1) * 100
    if current > e4.iloc[-1] > e10.iloc[-1] and rel4 > 0:
        trend = "ADVANCING"
    elif current > e10.iloc[-1] and rel12 > 0:
        trend = "CONSTRUCTIVE"
    elif current < e4.iloc[-1] and rel4 < 0:
        trend = "WEAKENING"
    else:
        trend = "TRANSITION"
    return WeeklySnapshot(symbol, current, ret4, ret12, rel4, rel12, float(e4.iloc[-1]),
                          float(e10.iloc[-1]), weekly_rsi, positive, drawdown, trend)


def weekly_market(snapshots: dict[str, WeeklySnapshot]) -> dict:
    btc = snapshots["BTC"]
    alts = [item for symbol, item in snapshots.items() if symbol != "BTC"]
    count = max(len(alts), 1)
    above4 = 100 * sum(item.close > item.ema4 for item in alts) / count
    above10 = 100 * sum(item.close > item.ema10 for item in alts) / count
    rel4 = 100 * sum(item.relative_4w > 0 for item in alts) / count
    rel12 = 100 * sum(item.relative_12w > 0 for item in alts) / count
    distribution = 100 * sum(item.trend == "WEAKENING" for item in alts) / count
    score = round((above4 + above10 + rel4 + rel12 + (100 if btc.trend in {"ADVANCING", "CONSTRUCTIVE"} else 25)) / 5, 1)
    if score >= 75:
        posture, horizon = "ADVANCING", "Constructive 2–6 week window"
    elif score >= 60:
        posture, horizon = "SELECTIVE RISK-ON", "Positive but selective 2–4 week window"
    elif score >= 45:
        posture, horizon = "TRANSITION", "Mixed 1–4 week window; require confirmation"
    else:
        posture, horizon = "DEFENSIVE", "Capital-protection window until weekly repair"
    return {"score": score, "posture": posture, "horizon": horizon, "above4": above4,
            "above10": above10, "rel4": rel4, "rel12": rel12, "distribution": distribution}


def holding_action(item: WeeklySnapshot, allocation: float, exit_risk: float) -> str:
    if item.trend == "WEAKENING" and (exit_risk >= 45 or allocation >= 20):
        return "REDUCE / REASSESS"
    if item.trend == "WEAKENING":
        return "PROTECT; NO NEW CAPITAL"
    if item.rsi >= 72 and item.return_4w >= 35:
        return "TRIM INTO STRENGTH"
    if allocation >= 25:
        return "REBALANCE CONCENTRATION"
    if item.trend == "ADVANCING":
        return "HOLD; TRAIL WEEKLY SUPPORT"
    return "HOLD / MONITOR WEEKLY CLOSE"
