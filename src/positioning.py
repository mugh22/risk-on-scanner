from __future__ import annotations

from dataclasses import dataclass

from .exit_risk import ExitRiskResult
from .scoring import CoinResult


@dataclass(frozen=True)
class PositioningResult:
    market_regime: str
    rotation_phase: str
    deployment_status: str
    deployment_action: str
    existing_action: str
    cash_band: str
    evidence: tuple[str, ...]
    metrics: dict[str, float]


def assess_positioning(
    score: float,
    context: dict,
    btc: CoinResult,
    coins: list[CoinResult],
    exit_risk: ExitRiskResult,
    market_heat: dict,
    dominance: dict | None = None,
    previous_dominance: dict | None = None,
) -> PositioningResult:
    """Translate existing closed-candle evidence into portfolio decisions.

    Market direction and alt rotation are deliberately separate. ETH/BTC is a
    supporting rotation input only; it can never veto a BTC-led bull regime.
    """
    dominance = dominance or {}
    previous_dominance = previous_dominance or {}
    n = max(len(coins), 1)
    rel7 = 100 * sum(c.rel_7d > 0 for c in coins) / n
    rel30 = float(context.get("breadth_rel30", 0))
    weekly = float(context.get("weekly_breadth", 0))
    daily = float(context.get("daily_confirmation_breadth", 0))
    usd_breadth = float(context.get("breadth_20", 0))
    btc_live = btc.live_price or btc.price
    btc_distance20 = 100 * (btc_live / btc.ema20 - 1) if btc.ema20 else 0.0
    btc_trend = bool(context.get("btc_constructive"))
    eth_support = bool(context.get("eth_btc_positive"))

    if exit_risk.score >= 60 or (score < 35 and not btc_trend):
        market_regime = "BEAR / RISK-OFF"
    elif score < 55:
        market_regime = "BASE / MIXED"
    elif btc_trend and score < 70:
        market_regime = "EARLY RECOVERY"
    elif btc_trend:
        market_regime = "CONFIRMED RISK-ON"
    else:
        market_regime = "RISK-ON UNDER STRESS"

    participation = (rel7 + rel30 + weekly + daily) / 4
    if exit_risk.score >= 60 or (rel7 < 30 and rel30 < 35):
        rotation = "ALT RISK-OFF"
    elif usd_breadth >= 60 and rel7 < 45 and rel30 < 45:
        rotation = "LATE / NARROWING ROTATION"
    elif weekly >= 50 and rel7 >= 60 and rel30 >= 60:
        rotation = "BROAD ALT EXPANSION"
    elif rel7 >= 50 or rel30 >= 50 or daily >= 55:
        rotation = "EARLY SELECTIVE ROTATION"
    else:
        rotation = "BTC LEADERSHIP"

    # Dominance changes refine the label but never override broad price evidence.
    btc_d_change = 0.0
    stable_d_change = 0.0
    if dominance and previous_dominance:
        btc_d_change = float(dominance.get("btc_d", 0)) - float(previous_dominance.get("btc_d", 0))
        stable_d_change = float(dominance.get("stable_d", 0)) - float(previous_dominance.get("stable_d", 0))
    narrowing = rotation == "LATE / NARROWING ROTATION" or (btc_d_change > .45 and rel7 < rel30)

    recent_high = btc.recent_high or btc_live
    resistance_gap = 100 * (recent_high / btc_live - 1) if btc_live else 0.0
    pullback = 100 * (btc_live / recent_high - 1) if recent_high else 0.0

    if exit_risk.score >= 60 or market_regime == "BEAR / RISK-OFF":
        deploy_status = "FAILED / DEFENSIVE"
        deploy_action = "Do not add; require weekly repair before deploying new capital."
    elif -1.5 <= resistance_gap <= 2.5 and not btc.breakout_retest:
        deploy_status = "WAIT — BTC NEAR RESISTANCE"
        deploy_action = "No dip is present; wait for a pullback or completed breakout retest."
    elif btc.breakout and not btc.breakout_retest:
        deploy_status = "BREAKOUT WATCH — WAIT FOR RETEST"
        deploy_action = "Do not chase the breakout candle; require the old ceiling to hold as support."
    elif btc.breakout_retest:
        deploy_status = "RETEST ENTRY — STAGED ADDS"
        deploy_action = "The breakout retest held; deploy gradually while invalidation remains intact."
    elif btc.weekly_higher_low_confirmed and btc_trend and exit_risk.score < 45:
        deploy_status = "HIGHER LOW CONFIRMED — STAGED ADDS"
        deploy_action = "Weekly structure is confirmed; deploy in tranches rather than all at once."
    elif pullback <= -5 and btc.price >= btc.ema50:
        deploy_status = "EARLY DIP — PROBE ONLY"
        deploy_action = "Support is being tested but reversal is unconfirmed; use at most 10–20%."
    elif market_heat.get("score", 0) >= 60 or btc_distance20 >= 8:
        deploy_status = "WAIT — DO NOT CHASE"
        deploy_action = "Hold reserve; add only after support or a confirmed breakout retest."
    elif btc_trend and score >= 70 and weekly >= 40 and exit_risk.score < 45:
        deploy_status = "REVERSAL CONFIRMING — PARTIAL ADDS"
        deploy_action = "Conditions are constructive, but retain reserve until weekly confirmation."
    elif btc_trend and score >= 55:
        deploy_status = "BUILD — PARTIAL SIZE"
        deploy_action = "Use 20–40% of planned capital; add only after confirmation."
    else:
        deploy_status = "EARLY — PROBE ONLY"
        deploy_action = "Use at most 10–20%; the higher-low structure is not confirmed."

    if exit_risk.score >= 60:
        existing = "Reduce broad alt risk; protect cash and strongest relative leaders only."
        cash_band = "35–50%"
    elif narrowing:
        existing = "Stop adding weak alts; trim extended laggards and rotate selectively."
        cash_band = "25–35%"
    elif market_heat.get("score", 0) >= 60:
        existing = "Hold leaders, but harvest 15–25% from extended positions."
        cash_band = "20–30%"
    elif rotation in {"BROAD ALT EXPANSION", "EARLY SELECTIVE ROTATION"}:
        existing = "Hold leaders; remove positions that lose weekly BTC-relative structure."
        cash_band = "15–25%"
    else:
        existing = "Favor BTC and exceptional leaders; avoid broad alt exposure."
        cash_band = "20–30%"

    evidence = [
        f"BTC trend is {'constructive' if btc_trend else 'mixed/weak'}; price is {btc_distance20:+.1f}% from EMA20",
        f"Alt/BTC breadth: {rel7:.0f}% over 7D, {rel30:.0f}% over 30D, {weekly:.0f}% constructive weekly",
        f"USD breadth is {usd_breadth:.0f}% while recent BTC-relative confirmation is {daily:.0f}%",
        f"ETH/BTC is {'supportive' if eth_support else 'weak'} and is used only as supporting rotation evidence",
    ]
    if dominance and previous_dominance:
        evidence.append(f"Since prior scan: BTC.D {btc_d_change:+.2f} points; stablecoin dominance {stable_d_change:+.2f} points")

    return PositioningResult(
        market_regime, rotation, deploy_status, deploy_action, existing, cash_band,
        tuple(evidence),
        {"rel7": rel7, "rel30": rel30, "weekly": weekly, "daily": daily,
         "usd_breadth": usd_breadth, "participation": participation,
         "btc_distance20": btc_distance20, "btc_d_change": btc_d_change,
         "stable_d_change": stable_d_change, "resistance_gap": resistance_gap,
         "pullback_from_high": pullback},
    )
