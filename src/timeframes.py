from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from .indicators import ema, period_return


def completed_daily(frame: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Remove Binance's still-forming UTC daily candle."""
    if frame.empty:
        return frame
    now = now or datetime.now(timezone.utc)
    today = pd.Timestamp(now).tz_convert("UTC").normalize()
    return frame.loc[frame.time < today].reset_index(drop=True)


def completed_weekly_closes(frame: pd.DataFrame, now: datetime | None = None) -> pd.Series:
    """Monday 00:00 UTC anchored weeks, excluding the current incomplete week."""
    if frame.empty:
        return pd.Series(dtype=float)
    now = now or datetime.now(timezone.utc)
    current_week = pd.Timestamp(now).tz_convert("UTC").normalize() - pd.Timedelta(days=pd.Timestamp(now).weekday())
    indexed = frame.set_index("time").close
    weekly = indexed.resample("W-MON", label="left", closed="left").last()
    return weekly.loc[weekly.index < current_week].dropna()


def weekly_higher_low_confirmed(frame: pd.DataFrame, now: datetime | None = None) -> bool:
    """Require two confirmed weekly swing lows, with the newest above the prior one."""
    if frame.empty or "low" not in frame:
        return False
    now = now or datetime.now(timezone.utc)
    current_week = pd.Timestamp(now).tz_convert("UTC").normalize() - pd.Timedelta(days=pd.Timestamp(now).weekday())
    weekly = frame.set_index("time").resample("W-MON", label="left", closed="left").agg(
        low=("low", "min"), high=("high", "max"), close=("close", "last")
    ).loc[lambda x: x.index < current_week].dropna()
    if len(weekly) < 7:
        return False
    pivots = [i for i in range(1, len(weekly)-1)
              if weekly.low.iloc[i] < weekly.low.iloc[i-1] and weekly.low.iloc[i] <= weekly.low.iloc[i+1]]
    if len(pivots) < 2:
        return False
    prior, latest = pivots[-2], pivots[-1]
    return bool(weekly.low.iloc[latest] > weekly.low.iloc[prior]
                and weekly.close.iloc[-1] > weekly.high.iloc[latest])


def breakout_retest_confirmed(frame: pd.DataFrame) -> bool:
    """Detect a completed 20-day breakout followed by a hold near the old ceiling."""
    if len(frame) < 30:
        return False
    for index in range(max(20, len(frame)-6), len(frame)-1):
        level = float(frame.close.iloc[index-20:index].max())
        if frame.close.iloc[index] <= level:
            continue
        after = frame.iloc[index+1:]
        if not after.empty and after.low.min() <= level * 1.03 and after.close.iloc[-1] >= level:
            return True
    return False


def relative_daily_confirmations(frame: pd.DataFrame, btc: pd.DataFrame, days: int = 3) -> int:
    alt_returns = frame.close.pct_change().tail(days)
    btc_returns = btc.close.pct_change().tail(days)
    size = min(len(alt_returns), len(btc_returns))
    if size == 0:
        return 0
    return int((alt_returns.iloc[-size:].to_numpy() > btc_returns.iloc[-size:].to_numpy()).sum())


def timeframe_evidence(frame: pd.DataFrame, btc: pd.DataFrame, now: datetime | None = None) -> dict[str, float | bool | int]:
    alt_weekly = completed_weekly_closes(frame, now)
    btc_weekly = completed_weekly_closes(btc, now)
    weekly_rel = 0.0
    weekly_constructive = False
    if len(alt_weekly) >= 5 and len(btc_weekly) >= 2:
        weekly_rel = period_return(alt_weekly, 1) - period_return(btc_weekly, 1)
        weekly_constructive = bool(alt_weekly.iloc[-1] > ema(alt_weekly, 4).iloc[-1] and weekly_rel > 0)
    higher_closes = int((frame.close.diff().tail(3) > 0).sum())
    rel_confirmations = relative_daily_confirmations(frame, btc, 3)
    return {
        "weekly_rel": round(weekly_rel, 2),
        "weekly_constructive": weekly_constructive,
        "daily_higher_closes": higher_closes,
        "daily_rel_confirmations": rel_confirmations,
        "weekly_higher_low_confirmed": weekly_higher_low_confirmed(frame, now),
        "breakout_retest": breakout_retest_confirmed(frame),
    }
