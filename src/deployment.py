from __future__ import annotations

from dataclasses import dataclass

from .scoring import CoinResult


@dataclass(frozen=True)
class AssetDeployment:
    status: str
    reason: str


def assess_asset_deployment(
    coin: CoinResult,
    market_status: str,
    rotation_phase: str,
    exit_risk: float,
) -> AssetDeployment:
    """Classify an entry using USD structure, BTC-relative structure, and market permission."""
    live = coin.live_price or coin.price
    distance20 = 100 * (live / coin.ema20 - 1) if coin.ema20 else 0.0
    near_support = bool(coin.atr and abs(live - coin.ema20) <= 1.25 * coin.atr)
    relative_weak = coin.symbol != "BTC" and (
        (coin.rel_7d < 0 and coin.rel_30d < 0)
        or (coin.weekly_rel < 0 and coin.daily_rel_confirmations < 2)
    )

    if "FAILED" in market_status:
        return AssetDeployment("FAILED DIP — DO NOT ADD", "Broad market protection overrides the individual setup.")
    if coin.symbol != "BTC" and exit_risk >= 60:
        if coin.rel_7d > 0 and coin.rel_30d > 0 and coin.breakout and distance20 <= 12:
            return AssetDeployment("SELECTIVE LEADER — PROBE ONLY", "Broad alt risk is high, but this asset is independently breaking out versus BTC.")
        return AssetDeployment("FAILED DIP — DO NOT ADD", "Broad alt-market protection overrides this individual setup.")
    if relative_weak:
        return AssetDeployment(
            "WAIT — BTC-RELATIVE WEAKNESS",
            f"The asset is not earning its risk versus BTC ({coin.rel_7d:+.1f}% 7D; {coin.rel_30d:+.1f}% 30D).",
        )
    if coin.rsi > 74 or distance20 > 12:
        return AssetDeployment("WAIT — EXTENDED", f"Price is {distance20:+.1f}% from EMA20 with RSI {coin.rsi:.0f}.")
    if "WAIT" in market_status or "BREAKOUT WATCH" in market_status:
        return AssetDeployment("WAIT — MARKET LOCATION", "The asset is acceptable, but BTC has not offered a favorable entry window.")
    if coin.breakout_retest and coin.rel_30d > 0:
        return AssetDeployment("BREAKOUT RETEST — ADD", "A recent breakout level held while relative strength remained positive.")
    if (coin.weekly_higher_low_confirmed and coin.rel_30d > 0
            and coin.daily_higher_closes >= 2 and coin.daily_rel_confirmations >= 2
            and coin.macd_hist > 0):
        return AssetDeployment("HIGHER LOW CONFIRMED — ADD", "Completed weekly structure confirmed a higher low versus a prior swing low.")
    if near_support and coin.price >= coin.ema50 and coin.rel_30d > 0:
        return AssetDeployment("SUPPORT HOLDING — PARTIAL ENTRY", "Price is near EMA20, above EMA50, and outperforming BTC over 30D.")
    if coin.daily_higher_closes >= 2 and coin.daily_rel_confirmations >= 2 and coin.rel_30d > 0:
        return AssetDeployment("REVERSAL CONFIRMING — PARTIAL ADD", "Two or more recent closes improved and beat BTC.")
    if "ALT RISK-OFF" in rotation_phase and coin.symbol != "BTC":
        return AssetDeployment("WAIT — ALT MARKET WEAK", "BTC may be constructive, but broad alt participation is not.")
    return AssetDeployment("WATCH — NO ENTRY TRIGGER", "Structure is not broken, but no support, retest, or confirmed reversal entry is present.")
