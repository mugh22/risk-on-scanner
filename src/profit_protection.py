from __future__ import annotations

from dataclasses import dataclass

from .scoring import CoinResult, clamp


@dataclass(frozen=True)
class ProfitProtection:
    score: float
    level: str
    action: str
    reasons: tuple[str, ...]


def heat_call(score: float) -> tuple[str, str]:
    if score < 40: return "LOW", "HOLD"
    if score < 60: return "WARM", "HOLD / DO NOT ADD"
    if score < 75: return "HOT", "CONSIDER TRIM 15–25%"
    return "EXTREME", "CONSIDER TRIM 25–50%"


def assess_profit_protection(coin: CoinResult, exit_risk_score: float) -> ProfitProtection:
    """Measure rally heat separately from broad exit risk."""
    score = 0.0
    reasons: list[str] = []
    current_price = coin.live_price or coin.price
    distance = (current_price / coin.ema20 - 1) * 100 if coin.ema20 else 0.0

    if coin.rsi >= 80:
        score += 30; reasons.append(f"RSI is extremely stretched at {coin.rsi:.0f}")
    elif coin.rsi >= 72:
        score += 20; reasons.append(f"RSI is stretched at {coin.rsi:.0f}")
    elif coin.rsi >= 65:
        score += 10

    if distance >= 20:
        score += 25; reasons.append(f"Price is {distance:.0f}% above EMA20")
    elif distance >= 12:
        score += 15; reasons.append(f"Price is {distance:.0f}% above EMA20")
    elif distance >= 8:
        score += 8

    if coin.usd_30d >= 80:
        score += 20; reasons.append(f"30-day gain is {coin.usd_30d:.0f}%")
    elif coin.usd_30d >= 40:
        score += 12; reasons.append(f"30-day gain is {coin.usd_30d:.0f}%")
    elif coin.usd_30d >= 20:
        score += 6

    if coin.target1 and current_price >= coin.target1:
        score += 20; reasons.append("Price reached the first modeled target")
    elif coin.target1 and current_price >= coin.target1 * .95:
        score += 12; reasons.append("Price is within 5% of the first target")

    if coin.rsi >= 70 and not coin.breakout:
        score += 12; reasons.append("High RSI has not confirmed a fresh breakout")
    score += min(15, exit_risk_score * .15)
    score = round(clamp(score), 1)

    level, action = heat_call(score)
    return ProfitProtection(score, level, action, tuple(reasons[:3]))
