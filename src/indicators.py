from __future__ import annotations

import numpy as np
import pandas as pd


def ema(values: pd.Series, span: int) -> pd.Series:
    return values.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(values: pd.Series, period: int = 14) -> pd.Series:
    delta = values.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    return result.where(loss.ne(0), 100.0).where(gain.ne(0), 0.0)


def macd(values: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(values, 12) - ema(values, 26)
    signal = line.ewm(span=9, adjust=False, min_periods=9).mean()
    return line, signal, line - signal


def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    prev = frame["close"].shift(1)
    tr = pd.concat([(frame["high"] - frame["low"]), (frame["high"] - prev).abs(), (frame["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def period_return(values: pd.Series, periods: int) -> float:
    if len(values) <= periods or values.iloc[-periods - 1] == 0:
        return float("nan")
    return float((values.iloc[-1] / values.iloc[-periods - 1] - 1) * 100)

