"""Persistent weekly protection and bounded, advisory cash deployment."""
from __future__ import annotations

import math
from datetime import datetime, timezone

from .indicators import atr
from .weekly import weekly_bars, entry_action

CASH = {"USD", "USDC", "USDT"}


def price_text(value):
    if value is None or not math.isfinite(value):
        return "N/A"
    return f"{value:,.4f}" if abs(value) >= 1 else f"{value:.8f}"


def weekly_stop(frame, previous=None, live=None, now=None):
    """Ratchet only on new completed weeks; never lower or clear a breached stop."""
    now = now or datetime.now(timezone.utc)
    old = dict(previous or {})
    weeks = weekly_bars(frame, now)
    candidate = None
    week = str(weeks.time.iloc[-1]) if len(weeks) else None
    if len(weeks) >= 15:
        volatility = float(atr(weeks).iloc[-1])
        lows = weeks.low.to_list()
        pivots = [i for i in range(max(2, len(lows)-26), len(lows)-2)
                  if lows[i] < min(lows[i-2:i]) and lows[i] <= min(lows[i+1:i+3])]
        support = float(lows[pivots[-1]]) if pivots else float(weeks.low.tail(12).min())
        candidate = support - .5 * volatility
        if not math.isfinite(candidate) or candidate <= 0:
            candidate = None
    # Migration: the original 12-week-low rule produced unusably distant stops.
    # Recalculate those once, then retain the new risk-capped level.
    if old.get("method", "").startswith("Confirmed weekly swing low"):
        old = {}
    stop = old.get("stop")
    status = "UNCHANGED" if stop else "INSUFFICIENT WEEKLY DATA"
    # Check the old level before raising it. Completed daily bars after the last
    # evaluation reveal breaches that recovered before today's quote.
    breached = bool(old.get("breached"))
    if stop and old.get("checked_at"):
        import pandas as pd
        subsequent = frame[frame.time >= pd.Timestamp(old["checked_at"])]
        breached = breached or bool(len(subsequent) and subsequent.low.min() <= stop)
    if stop and live is not None and live <= stop:
        breached = True
    if live and candidate is not None:
        # Structural support may be far away. Cap the suggested downside at
        # 15% of today's price, while keeping a tighter structural level.
        candidate = max(candidate, live * .85)
    if not breached and candidate is not None and week != old.get("week"):
        if stop is None:
            stop, status = candidate, "NEW — NOT PLACED"
        elif candidate >= stop * 1.05 and live is not None and candidate < live:
            stop, status = candidate, "RAISE SUGGESTED — NOT SYNCED"
        elif candidate < stop:
            status = "KEEP PRIOR — LOWER LEVEL REJECTED"
    if stop and live is not None and live <= stop:
        breached = True
    if breached:
        status = "BREACHED — REVIEW / NO ADD"
    return {"stop": stop, "candidate": candidate, "previous_stop": old.get("stop"),
            "status": status, "breached": breached, "week": week,
            "checked_at": now.isoformat(),
            "method": "Weekly structure with a 15% maximum price-risk cap; advisory"}


def build_risk_plan(holdings, rows, frames, coins, previous, source_note, exit_score,
                    config=None, weekly_items=None, weekly_context=None, now=None):
    config = config or {}
    states = dict(previous or {})
    current = {h.symbol for h in holdings if h.symbol not in CASH}
    states = {k: v for k, v in states.items() if k in current}
    stops = {}
    for coin in coins:
        if coin.symbol in frames:
            stops[coin.symbol] = weekly_stop(frames[coin.symbol], states.get(coin.symbol), coin.live_price, now)
            if coin.symbol in current:
                states[coin.symbol] = stops[coin.symbol]
    total = sum(r["value"] for r in rows)
    # Cap modeled loss across the whole priced portfolio at 6% and any one
    # holding at 2% of that portfolio. A tight cap is marked for review.
    proposed = {}
    for row in rows:
        stop = stops.get(row["symbol"], {})
        if row["symbol"] not in CASH and stop.get("stop") and row["price"] > stop["stop"] and total > 0:
            distance = min(1-stop["stop"]/row["price"], .02*total/row["value"])
            proposed[row["symbol"]] = distance * row["value"]
    scale = min(1., .06*total/sum(proposed.values())) if proposed else 1.
    for row in rows:
        stop = stops.get(row["symbol"], {})
        if row["symbol"] in proposed and not stop.get("breached"):
            distance = proposed[row["symbol"]] / row["value"] * scale
            adjusted = row["price"] * (1-distance)
            if adjusted > stop["stop"]:
                stop["stop"] = adjusted
                stop["status"] = "RISK CAP — REVIEW LEVEL"
                states[row["symbol"]] = stop
        row["cycle_stop"] = stop.get("stop")
        row["stop_status"] = "CASH — NOT APPLICABLE" if row["symbol"] in CASH else stop.get("status", "NO WEEKLY DATA")
        row["stop_distance"] = 100*(1-row["cycle_stop"]/row["price"]) if row["cycle_stop"] else None
        row["stop_loss_usd"] = max(0, row["price"]-row["cycle_stop"])*row["quantity"] if row["cycle_stop"] else None
        if stop.get("breached"):
            row["action"] = "STOP BREACHED — REVIEW POSITION"
    cash = sum(h.available_quantity for h in holdings if h.symbol in CASH and h.available_quantity is not None)
    values = {r["symbol"]: r["value"] for r in rows}
    live_source = bool(source_note and source_note.startswith("Live quantities from Coinbase"))
    covered = {r["symbol"] for r in rows} | {c.symbol for c in coins} | CASH
    missing = sorted(h.symbol for h in holdings if h.symbol not in covered)
    plan = {"cash": cash, "total": total, "allocations": {}, "note": "", "stops": stops}
    if not live_source:
        plan["note"] = "DEPLOY 0% — live Coinbase available cash could not be verified."
        return plan, states
    if missing:
        plan["unpriced"] = missing
    if exit_score >= 60 or cash <= 5 or total <= 0:
        plan["note"] = "DEPLOY 0% — defensive market or insufficient available cash."
        return plan, states
    reserve = total * config.get("minimum_reserve_pct", 15) / 100
    budget = min(cash * config.get("max_cash_per_report_pct", 30)/100, max(0, cash-reserve))
    risk_budget = total * config.get("max_portfolio_risk_per_report_pct", 1)/100
    actionable = {"BREAKOUT RETEST — ADD", "HIGHER LOW CONFIRMED — ADD", "SUPPORT HOLDING — PARTIAL ENTRY", "REVERSAL CONFIRMING — PARTIAL ADD"}
    for coin in sorted(coins, key=lambda c: (-c.score, c.symbol)):
        if coin.signal != "BUY" or coin.deployment_status not in actionable:
            continue
        if weekly_items is not None:
            item = weekly_items.get(coin.symbol)
            if not item or not entry_action(item, weekly_context, exit_score).startswith(("ADD", "PROBE")):
                continue
        stop = stops.get(coin.symbol, {})
        price = coin.live_price
        level = stop.get("stop")
        if not level or stop.get("breached") or not price or not (0 < level < price):
            continue
        if not (coin.entry_low <= price <= coin.entry_high):
            continue
        risk = (price-level)/price
        rr = (coin.target2-price)/(price-level)
        if rr < config.get("min_target2_reward_risk", 2):
            continue
        headroom = max(0, total*config.get("max_asset_allocation_pct",20)/100-values.get(coin.symbol,0))
        amount = min(cash*config.get("cash_per_asset_pct",10)/100, budget, headroom,
                     total*config.get("max_portfolio_risk_per_asset_pct",.5)/100/risk, risk_budget/risk)
        amount = math.floor(max(0, amount)*100)/100
        if amount <= 5:
            continue
        plan["allocations"][coin.symbol] = {"usd": amount, "cash_pct": amount/cash*100,
            "quantity": amount/price, "stop": level, "risk_usd": amount*risk, "reward_risk": rr}
        budget -= amount
        risk_budget -= amount*risk
    deployed = sum(a["usd"] for a in plan["allocations"].values())
    plan["deployed"] = deployed
    plan["note"] = (f"Suggested ${deployed:,.2f}; retain ${cash-deployed:,.2f} available cash. " if deployed else
                    "DEPLOY 0% — no BUY meets entry, weekly-stop reward/risk, reserve and concentration limits. ")
    if missing:
        plan["note"] += " Other Coinbase balances excluded from valuation: " + ", ".join(missing) + "."
    return plan, states
