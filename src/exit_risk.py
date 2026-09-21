from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from .indicators import ema, period_return, rsi
from .scoring import CoinResult, clamp
from .timeframes import completed_weekly_closes


@dataclass
class ExitRiskResult:
    score: float
    level: str
    call: str
    summary: str
    components: dict[str, float]
    directions: dict[str, str] = field(default_factory=dict)
    red_flags: list[str] = field(default_factory=list)
    supports: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


def _risk_level(score: float) -> tuple[str, str]:
    if score < 25:
        return "LOW", "HOLD / SELECTIVE ADDS"
    if score < 45:
        return "WATCH", "HOLD — DO NOT CHASE"
    if score < 60:
        return "CAUTION", "STOP ADDING / REVIEW WEAK ALTS"
    if score < 75:
        return "HIGH", "REDUCE ALT RISK"
    return "CRITICAL", "EXIT MOST ALT RISK"


def _breadth_at(frames: dict[str, pd.DataFrame], offset: int, window: int) -> float:
    observations = []
    for frame in frames.values():
        close = frame.close.iloc[: len(frame) - offset if offset else None]
        if len(close) >= window:
            observations.append(close.iloc[-1] > ema(close, window).iloc[-1])
    return 100 * sum(observations) / max(len(observations), 1)


def assess_exit_risk(
    btc_frame: pd.DataFrame,
    ethbtc: pd.DataFrame | None,
    frames: dict[str, pd.DataFrame],
    coins: list[CoinResult],
    dominance: dict | None = None,
    previous_dominance: dict | None = None,
    previous_metrics: dict[str, float] | None = None,
    now: datetime | None = None,
) -> ExitRiskResult:
    """Explainable, multi-confirmation warning model; higher means more alt downside risk."""
    dominance = dominance or {}
    previous_dominance = previous_dominance or {}
    previous_metrics = previous_metrics or {}
    flags: list[str] = []
    supports: list[str] = []

    # 1) Relative strength: the median alt and ETH/BTC should confirm a healthy rotation.
    median_rel7 = float(pd.Series([c.rel_7d for c in coins]).median())
    median_rel30 = float(pd.Series([c.rel_30d for c in coins]).median())
    rel_breadth7 = 100 * sum(c.rel_7d > 0 for c in coins) / max(len(coins), 1)
    rel_breadth30 = 100 * sum(c.rel_30d > 0 for c in coins) / max(len(coins), 1)
    relative = 0.0
    relative += 25 if median_rel7 < -3 else 12 if median_rel7 < 0 else 0
    relative += 25 if median_rel30 < -8 else 12 if median_rel30 < 0 else 0
    relative += 20 if rel_breadth7 < 35 else 10 if rel_breadth7 < 50 else 0
    relative += 15 if rel_breadth30 < 35 else 7 if rel_breadth30 < 50 else 0
    eth_rel7 = 0.0
    if ethbtc is not None:
        eth_rel7 = period_return(ethbtc.close, 7)
        eth_e20 = ema(ethbtc.close, 20)
        relative += 15 if eth_rel7 < 0 and ethbtc.close.iloc[-1] < eth_e20.iloc[-1] else 0
    relative = clamp(relative)
    if relative >= 55:
        flags.append(f"Alt/BTC participation is weak: median 7D {median_rel7:+.1f}%, only {rel_breadth7:.0f}% outperform BTC")
    elif median_rel7 > 0 and rel_breadth7 >= 55:
        supports.append(f"Alt/BTC participation remains broad ({rel_breadth7:.0f}% outperforming over 7D)")

    # 2) Breadth: weakening participation often appears before the headline index rolls over.
    breadth20 = _breadth_at(frames, 0, 20)
    breadth50 = _breadth_at(frames, 0, 50)
    prior_breadth20 = _breadth_at(frames, 7, 20)
    breadth_change = breadth20 - prior_breadth20
    breadth = 0.0
    breadth += 35 if breadth20 < 30 else 20 if breadth20 < 50 else 0
    breadth += 25 if breadth50 < 35 else 12 if breadth50 < 50 else 0
    breadth += 30 if breadth_change <= -25 else 18 if breadth_change <= -15 else 8 if breadth_change <= -8 else 0
    breadth += 10 if breadth20 < breadth50 else 0
    breadth = clamp(breadth)
    if breadth_change <= -15:
        flags.append(f"Breadth rolled over quickly: EMA20 participation fell {abs(breadth_change):.0f} points in 7D")
    elif breadth20 >= 60:
        supports.append(f"Breadth is healthy: {breadth20:.0f}% of tracked alts remain above EMA20")

    # 3) Exhaustion/distribution: high RSI alone is not bearish; failed highs and weak closes matter.
    extended = sum(c.rsi >= 70 or c.price > c.ema20 + 2.5 * c.atr for c in coins)
    failed_highs = 0
    weekly_red = 0
    for frame in frames.values():
        close = frame.close
        current_high = float(close.tail(21).max())
        prior_high = float(close.iloc[-90:-21].max()) if len(close) >= 90 else current_high
        if rsi(close).iloc[-1] >= 68 and current_high < prior_high * 0.99:
            failed_highs += 1
        weekly = completed_weekly_closes(frame, now)
        if len(weekly) >= 3 and weekly.iloc[-1] < weekly.iloc[-2] < weekly.iloc[-3]:
            weekly_red += 1
    n = max(len(coins), 1)
    extended_pct = 100 * extended / n
    failed_pct = 100 * failed_highs / n
    weekly_red_pct = 100 * weekly_red / n
    exhaustion = clamp(
        (25 if extended_pct >= 50 else 12 if extended_pct >= 30 else 0)
        + (40 if failed_pct >= 30 else 22 if failed_pct >= 15 else 0)
        + (35 if weekly_red_pct >= 40 else 18 if weekly_red_pct >= 20 else 0)
    )
    if failed_pct >= 15:
        flags.append(f"Failed-high pattern: {failed_pct:.0f}% are near overbought without clearing prior highs")
    if extended_pct >= 40 and exhaustion < 55:
        flags.append(f"Rally is stretched: {extended_pct:.0f}% of tracked alts are extended")

    # 4) BTC stress: alt losses commonly accelerate when BTC loses trend and volatility expands.
    btc_close = btc_frame.close
    btc_e20, btc_e50 = ema(btc_close, 20), ema(btc_close, 50)
    btc7 = period_return(btc_close, 7)
    btc_drawdown = (btc_close.iloc[-1] / btc_close.tail(30).max() - 1) * 100
    btc_stress = 0.0
    btc_stress += 25 if btc_close.iloc[-1] < btc_e20.iloc[-1] else 0
    btc_stress += 25 if btc_close.iloc[-1] < btc_e50.iloc[-1] else 0
    btc_stress += 25 if btc7 < -8 else 12 if btc7 < -3 else 0
    btc_stress += 25 if btc_drawdown < -15 else 12 if btc_drawdown < -8 else 0
    btc_stress = clamp(btc_stress)
    if btc_stress >= 50:
        flags.append(f"BTC trend stress is elevated (7D {btc7:+.1f}%, 30D drawdown {btc_drawdown:.1f}%)")
    elif btc_close.iloc[-1] > btc_e20.iloc[-1] > btc_e50.iloc[-1]:
        supports.append("BTC remains above rising short- and medium-term trend levels")

    # 5) Capital concentration: current snapshot is context; changes become useful after state accumulates.
    concentration = 35.0  # neutral until at least two comparable snapshots exist
    concentration_change = None
    stable_btc_change = None
    if dominance and previous_dominance:
        concentration_change = dominance.get("concentration", 0) - previous_dominance.get("concentration", dominance.get("concentration", 0))
        stable_btc_change = dominance.get("stable_btc_ratio", 0) - previous_dominance.get("stable_btc_ratio", dominance.get("stable_btc_ratio", 0))
        concentration = 15
        if concentration_change > 1.0:
            concentration += 45
        elif concentration_change > 0.35:
            concentration += 25
        if stable_btc_change > 0.002:
            concentration += 30
        elif stable_btc_change > 0:
            concentration += 10
        concentration = clamp(concentration)
        if concentration >= 60:
            flags.append("Capital is rotating toward BTC/ETH/stablecoins instead of broadening into alts")
        elif concentration_change < -0.35:
            supports.append("Market-cap concentration is falling, consistent with broader alt participation")

    components = {
        "Relative strength": round(relative, 1),
        "Breadth deterioration": round(breadth, 1),
        "Exhaustion / distribution": round(exhaustion, 1),
        "BTC trend stress": round(btc_stress, 1),
        "Capital concentration": round(concentration, 1),
    }
    weights = {"Relative strength": .30, "Breadth deterioration": .25, "Exhaustion / distribution": .20, "BTC trend stress": .15, "Capital concentration": .10}
    score = round(sum(components[k] * weights[k] for k in components), 1)
    level, call = _risk_level(score)
    summary = (
        "Multiple independent red flags agree; protect capital rather than waiting for a perfect top."
        if score >= 60 else
        "Deterioration is visible, but not yet broad enough for a full exit."
        if score >= 45 else
        "The alt regime is mixed; hold valid trends but avoid chasing extended moves."
        if score >= 25 else
        "Relative strength and participation remain constructive; no broad exit signal."
    )
    metrics = {
        "median_rel7": round(median_rel7, 2), "median_rel30": round(median_rel30, 2),
        "rel_breadth7": round(rel_breadth7, 1), "rel_breadth30": round(rel_breadth30, 1),
        "breadth20": round(breadth20, 1), "breadth50": round(breadth50, 1),
        "breadth_change7": round(breadth_change, 1), "ethbtc_7d": round(eth_rel7, 2),
        "btc_7d": round(btc7, 2), "btc_drawdown30": round(btc_drawdown, 2),
        "extended_pct": round(extended_pct, 1), "failed_high_pct": round(failed_pct, 1),
        "weekly_red_pct": round(weekly_red_pct, 1),
        "concentration": round(float(dominance.get("concentration", 0)), 2),
        "stable_btc_ratio": round(float(dominance.get("stable_btc_ratio", 0)), 4),
    }

    # Direction is deliberately separate from the confirmed component score. It can
    # warn that healthy evidence is weakening before a strict risk threshold is crossed.
    def movement(name: str, pairs: list[tuple[str, int, float]]) -> str:
        if not previous_metrics:
            return "Baseline"
        changes = []
        for key, risk_sign, tolerance in pairs:
            if key not in previous_metrics:
                continue
            delta = (metrics[key] - float(previous_metrics[key])) * risk_sign
            if abs(delta) >= tolerance:
                changes.append(delta)
        if not changes:
            return "Stable"
        net = sum(changes)
        return "Worsening" if net > 0 else "Improving" if net < 0 else "Mixed"

    movements = {
        "Relative strength": movement("Relative strength", [("median_rel7", -1, 1.5), ("median_rel30", -1, 2.0), ("rel_breadth7", -1, 5), ("rel_breadth30", -1, 5), ("ethbtc_7d", -1, 1)]),
        "Breadth deterioration": movement("Breadth deterioration", [("breadth20", -1, 5), ("breadth50", -1, 5), ("breadth_change7", -1, 5)]),
        "Exhaustion / distribution": movement("Exhaustion / distribution", [("extended_pct", 1, 5), ("failed_high_pct", 1, 5), ("weekly_red_pct", 1, 5)]),
        "BTC trend stress": movement("BTC trend stress", [("btc_7d", -1, 1.5), ("btc_drawdown30", -1, 2)]),
        "Capital concentration": movement("Capital concentration", [("concentration", 1, .25), ("stable_btc_ratio", 1, .001)]),
    }
    directions = {}
    for name, component_score in components.items():
        move = movements[name]
        condition = "Healthy" if component_score < 25 else "Watch" if component_score < 50 else "Elevated"
        directions[name] = f"{condition} · {move.lower()}" if move in {"Baseline", "Stable", "Worsening"} else move
    return ExitRiskResult(score, level, call, summary, components, directions, flags[:4], supports[:3], metrics)
