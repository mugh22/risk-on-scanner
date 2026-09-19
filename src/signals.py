from __future__ import annotations

from .scoring import CoinResult


def is_overextended(c: CoinResult, cfg: dict) -> tuple[bool, list[str]]:
    flags = []
    distance = (c.price / c.ema20 - 1) * 100
    if c.rsi > cfg["max_rsi_buy"]: flags.append(f"RSI extended ({c.rsi:.1f})")
    if distance > cfg["max_distance_ema20_pct"]: flags.append(f"{distance:.1f}% above EMA20")
    if c.usd_1d > cfg["max_daily_move_pct"]: flags.append(f"one-day move {c.usd_1d:.1f}%")
    return bool(flags), flags


def classify(c: CoinResult, risk_score: float, cfg: dict) -> str:
    extended, flags = is_overextended(c, cfg)
    c.reasons.extend(f"- {flag}" for flag in flags)
    confirmations = sum([c.rel_30d > 0, c.rel_7d > 0, c.price > c.ema20, c.ema20 > c.ema50, c.macd_hist > 0, c.volume_ratio >= cfg["min_volume_ratio"], c.breakout])
    timeframe_confirmed = c.rel_30d > 0 and c.weekly_constructive and c.daily_rel_confirmations >= 2
    c.reasons.append(("+" if c.weekly_constructive else "-") + f" completed week vs BTC {c.weekly_rel:+.1f}%")
    c.reasons.append(("+" if c.daily_rel_confirmations >= 2 else "-") + f" last 3 closed days: {c.daily_rel_confirmations}/3 beat BTC")
    if risk_score >= cfg["min_risk_on_score"] and c.score >= cfg["min_buy_score"] and confirmations >= 6 and timeframe_confirmed and not extended:
        return "BUY"
    if c.score >= cfg["min_watch_score"] and confirmations >= 4 and c.rel_30d > 0:
        return "WATCH"
    return "NO SIGNAL"


def levels(c: CoinResult, recent_high: float, recent_low: float, base_price: float | None = None) -> None:
    price = c.price if base_price is None else base_price
    c.entry_low, c.entry_high = price - 0.35 * c.atr, price + 0.15 * c.atr
    c.invalidation = max(price - 1.5 * c.atr, recent_low - 0.25 * c.atr)
    risk = price - c.invalidation
    c.target1 = max(recent_high, price + 2 * c.atr)
    c.target2 = price + 3.5 * c.atr
    c.reward_risk = (c.target1 - price) / risk if risk > 0 else None
