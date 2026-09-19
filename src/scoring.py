from __future__ import annotations

from dataclasses import dataclass, field
from math import isnan


def clamp(value: float, low: float = 0, high: float = 100) -> float:
    return max(low, min(high, value))


def regime(score: float) -> str:
    if score < 35: return "RISK-OFF"
    if score < 55: return "NEUTRAL"
    if score < 70: return "EARLY RISK-ON"
    if score < 85: return "RISK-ON"
    return "STRONG RISK-ON"


@dataclass
class CoinResult:
    symbol: str
    price: float
    usd_1d: float
    usd_7d: float
    usd_30d: float
    rel_7d: float
    rel_30d: float
    ema20: float
    ema50: float
    ema200: float | None
    rsi: float
    macd_hist: float
    volume_ratio: float
    breakout: bool
    atr: float
    score: float = 0
    signal: str = "NO SIGNAL"
    reasons: list[str] = field(default_factory=list)
    entry_low: float | None = None
    entry_high: float | None = None
    target1: float | None = None
    target2: float | None = None
    invalidation: float | None = None
    reward_risk: float | None = None


def score_coin(c: CoinResult, weights: dict[str, float]) -> tuple[float, list[str]]:
    reasons: list[str] = []
    rel30 = clamp(50 + c.rel_30d * 2.5)
    rel7 = clamp(50 + c.rel_7d * 4)
    trend = (40 if c.price > c.ema20 else 0) + (35 if c.ema20 > c.ema50 else 0) + (25 if c.price > c.ema50 else 0)
    momentum = clamp((c.rsi - 35) * 2 + (20 if c.macd_hist > 0 else 0))
    volume = clamp(c.volume_ratio * 60)
    breakout = 100 if c.breakout else (55 if c.price > c.ema20 else 20)
    parts = {"relative_30d": rel30, "relative_7d": rel7, "trend": trend, "momentum": momentum, "volume": volume, "breakout": breakout}
    score = sum(parts[k] * weights[k] for k in weights) / sum(weights.values())
    reasons.append(("+" if c.rel_30d > 0 else "-") + f" 30D vs BTC {c.rel_30d:+.1f}%")
    reasons.append(("+" if c.price > c.ema20 and c.ema20 > c.ema50 else "-") + " constructive EMA structure")
    reasons.append(("+" if c.macd_hist > 0 else "-") + " positive MACD" if c.macd_hist > 0 else "- negative MACD")
    reasons.append(("+" if c.volume_ratio >= 1 else "-") + f" volume {c.volume_ratio:.1f}x average")
    return round(score, 1), reasons

