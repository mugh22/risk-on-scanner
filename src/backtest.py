from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path

import pandas as pd
import yaml

from .deployment import assess_asset_deployment
from .exit_risk import assess_exit_risk
from .market_data import BinanceClient
from .positioning import assess_positioning
from .profit_protection import assess_profit_protection, heat_call
from .scanner import analyze, market_score
from .signals import classify, levels


def _cut(frame: pd.DataFrame, as_of: datetime) -> pd.DataFrame:
    return frame.loc[frame.time < pd.Timestamp(as_of)].reset_index(drop=True)


def _future_return(frame: pd.DataFrame, as_of: datetime, entry: float, days: int) -> float | None:
    future = frame.loc[(frame.time >= pd.Timestamp(as_of)) & (frame.time < pd.Timestamp(as_of + timedelta(days=days)))]
    if future.empty:
        return None
    return round((float(future.close.iloc[-1]) / entry - 1) * 100, 1)


def _max_drawdown(frame: pd.DataFrame, as_of: datetime, entry: float, days: int = 30) -> float | None:
    future = frame.loc[(frame.time >= pd.Timestamp(as_of)) & (frame.time < pd.Timestamp(as_of + timedelta(days=days)))]
    if future.empty:
        return None
    return round((float(future.low.min()) / entry - 1) * 100, 1)


def _grade(status: str, r30: float | None, r90: float | None, drawdown: float | None) -> str:
    if None in {r30, r90, drawdown}:
        return "NOT GRADED"
    add = status.startswith(("BREAKOUT RETEST", "HIGHER LOW CONFIRMED",
                             "SUPPORT HOLDING", "REVERSAL CONFIRMING"))
    defensive = status.startswith(("WAIT", "FAILED", "WATCH"))
    if add:
        return "GOOD" if (r30 > 0 and drawdown > -15) or r90 >= 20 else "BAD"
    if defensive:
        return "GOOD" if r30 <= 0 or drawdown <= -10 else "MISSED UPSIDE" if r90 >= 20 else "NEUTRAL"
    return "NEUTRAL"


def _exit_grade(action: str, r30: float | None, drawdown: float | None) -> str:
    """Grade an actual reduce/exit decision separately from entry avoidance."""
    if r30 is None or drawdown is None:
        return "NOT GRADED"
    defensive = action.startswith(("CONSIDER TRIM", "REDUCE", "EXIT"))
    if defensive:
        return "GOOD EXIT" if r30 <= -5 or drawdown <= -15 else "EARLY EXIT"
    if r30 <= -10 or drawdown <= -20:
        return "MISSED EXIT"
    return "NO EXIT NEEDED"


def _effective_exit_action(symbol: str, profit_action: str, broad_exit_call: str) -> str:
    """Combine asset heat with a confirmed broad alt-risk reduction call."""
    if symbol != "BTC" and broad_exit_call.startswith(("REDUCE", "EXIT")):
        return broad_exit_call
    return profit_action


def _expand_cases(spec: dict) -> list[dict]:
    cases = list(spec.get("cases", []))
    for sweep in spec.get("daily_sweeps", []):
        for signal_day in pd.date_range(sweep["start"], sweep["end"], freq="D"):
            # A signal labelled 2024-04-01 is evaluated immediately after that
            # UTC daily candle closes, so the no-lookahead cutoff is next midnight.
            cases.append({
                "date": (signal_day + pd.Timedelta(days=1)).date().isoformat(),
                "signal_date": signal_day.date().isoformat(),
                "name": sweep["name"],
                "focus": sweep["focus"],
                "rationale": sweep["rationale"],
            })
    return cases


def run_backtest(config_path: str, cases_path: str, output_dir: str) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text())
    cases = _expand_cases(yaml.safe_load(Path(cases_path).read_text()))
    # Stable, long-lived Binance pairs keep the historical breadth universe
    # comparable and avoid introducing assets that did not yet exist.
    symbols = ["ETH", "SOL", "XRP", "BNB", "ADA", "DOGE", "AVAX", "LINK",
               "NEAR", "ARB", "OP", "SUI", "APT", "INJ", "AAVE", "UNI", "FET"]
    latest = max(datetime.fromisoformat(str(case["date"])).replace(tzinfo=timezone.utc) for case in cases)
    end = latest + timedelta(days=100)
    client = BinanceClient(10, 1, trust_env=True)
    frames: dict[str, pd.DataFrame] = {"BTC": client.daily("BTCUSDT", 1000, end)}
    for symbol in symbols:
        try:
            frames[symbol] = client.daily(f"{symbol}USDT", 1000, end)
        except RuntimeError:
            pass
    try:
        ethbtc_all = client.daily("ETHBTC", 1000, end)
    except RuntimeError:
        ethbtc_all = None

    results = []
    for case in cases:
        as_of = datetime.fromisoformat(str(case["date"])).replace(tzinfo=timezone.utc)
        btc_frame = _cut(frames["BTC"], as_of)
        if len(btc_frame) < 62:
            continue
        btc = analyze("BTC", btc_frame, btc_frame, cfg["weights"]["alt_strength"], as_of)
        btc.live_price = btc.price
        levels(btc, float(btc_frame.high.tail(60).iloc[:-1].max()), float(btc_frame.low.tail(20).min()), btc.price)
        available, used_frames = [], {}
        for symbol, full in frames.items():
            if symbol == "BTC":
                continue
            sliced = _cut(full, as_of)
            if len(sliced) < 62:
                continue
            used_frames[symbol] = sliced
            item = analyze(symbol, sliced, btc_frame, cfg["weights"]["alt_strength"], as_of)
            item.live_price = item.price
            levels(item, float(sliced.high.tail(60).iloc[:-1].max()), float(sliced.low.tail(20).min()), item.price)
            available.append(item)
        ethbtc = _cut(ethbtc_all, as_of) if ethbtc_all is not None else None
        score, context = market_score(btc, ethbtc, available, cfg["weights"]["risk_on"])
        for item in available:
            item.signal = classify(item, score, cfg["signals"])
        risk = assess_exit_risk(btc_frame, ethbtc, used_frames, available, now=as_of)
        protections = {item.symbol: assess_profit_protection(item, risk.score) for item in [btc, *available]}
        heats = [protection.score for protection in protections.values()]
        heat_score = round(float(pd.Series(heats).quantile(.75)), 1)
        heat_level, heat_action = heat_call(heat_score)
        positioning = assess_positioning(score, context, btc, available, risk,
                                          {"score": heat_score, "level": heat_level, "action": heat_action})
        by_symbol = {"BTC": btc, **{item.symbol: item for item in available}}
        assets = []
        btc_entry = btc.price
        btc30 = _future_return(frames["BTC"], as_of, btc_entry, 30)
        for symbol in case["focus"]:
            item = by_symbol.get(symbol)
            full = frames.get(symbol)
            if item is None or full is None:
                continue
            readiness = assess_asset_deployment(item, positioning.deployment_status,
                                                positioning.rotation_phase, risk.score)
            r7 = _future_return(full, as_of, item.price, 7)
            r30 = _future_return(full, as_of, item.price, 30)
            r90 = _future_return(full, as_of, item.price, 90)
            dd30 = _max_drawdown(full, as_of, item.price)
            protection = protections[symbol]
            exit_action = _effective_exit_action(symbol, protection.action, risk.call)
            assets.append({"symbol": symbol, "price": round(item.price, 6), "signal": item.signal,
                           "readiness": readiness.status, "reason": readiness.reason,
                           "profit_protection_score": protection.score,
                           "profit_protection_action": protection.action,
                           "effective_exit_action": exit_action,
                           "rel_30d_at_call": round(item.rel_30d, 1), "weekly_rel_at_call": round(item.weekly_rel, 1),
                           "forward_7d": r7, "forward_30d": r30, "forward_90d": r90,
                           "forward_30d_vs_btc": None if r30 is None or btc30 is None else round(r30-btc30, 1),
                           "max_drawdown_30d": dd30, "entry_grade": _grade(readiness.status, r30, r90, dd30),
                           "exit_grade": _exit_grade(exit_action, r30, dd30)})
        results.append({"date": case.get("signal_date", str(case["date"])), "name": case["name"], "rationale": case["rationale"],
                        "risk_on": score, "exit_risk": risk.score, "broad_exit_call": risk.call,
                        "market_regime": positioning.market_regime,
                        "rotation": positioning.rotation_phase, "market_readiness": positioning.deployment_status,
                        "market_action": positioning.deployment_action, "assets": assets})

    decisive = [asset for case in results for asset in case["assets"] if asset["entry_grade"] in {"GOOD", "BAD", "MISSED UPSIDE"}]
    good = sum(asset["entry_grade"] == "GOOD" for asset in decisive)
    exit_decisions = [asset for case in results for asset in case["assets"] if asset["exit_grade"] in {"GOOD EXIT", "EARLY EXIT", "MISSED EXIT"}]
    summary = {"cases": len(results), "decisive_asset_calls": len(decisive), "good": good,
               "bad": sum(asset["entry_grade"] == "BAD" for asset in decisive),
               "missed_upside": sum(asset["entry_grade"] == "MISSED UPSIDE" for asset in decisive),
               "case_study_hit_rate": round(100 * good / max(len(decisive), 1), 1),
               "good_exits": sum(asset["exit_grade"] == "GOOD EXIT" for asset in exit_decisions),
               "early_exits": sum(asset["exit_grade"] == "EARLY EXIT" for asset in exit_decisions),
               "missed_exits": sum(asset["exit_grade"] == "MISSED EXIT" for asset in exit_decisions)}
    payload = {"methodology": "Signals use only candles before each as-of timestamp; later candles are used only for grading.",
               "limitations": "Selected-event case study, not a statistically representative strategy backtest. Historical dominance and live intraday quotes are excluded.",
               "summary": summary, "results": results}
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(payload, indent=2))
    rows = []
    for case in results:
        for asset in case["assets"]:
            rows.append(f"<tr><td>{escape(case['date'])}<br><small>{escape(case['name'])}</small></td><td><b>{escape(asset['symbol'])}</b></td>"
                        f"<td>{escape(case['market_readiness'])}</td><td>{escape(asset['readiness'])}</td><td>{escape(asset['signal'])}</td>"
                        f"<td>{asset['forward_7d']:+.1f}%</td><td>{asset['forward_30d']:+.1f}%</td><td>{asset['forward_90d']:+.1f}%</td>"
                        f"<td>{escape(asset['effective_exit_action'])}</td><td>{asset['profit_protection_score']:.0f}</td>"
                        f"<td>{asset['max_drawdown_30d']:+.1f}%</td><td><b>{escape(asset['entry_grade'])}</b></td>"
                        f"<td><b>{escape(asset['exit_grade'])}</b></td></tr>")
    html = f"""<!doctype html><html><head><meta name='viewport' content='width=device-width'><style>
body{{font-family:Arial,sans-serif;max-width:1100px;margin:auto;padding:16px;color:#17202a}}.hero{{background:#0f172a;color:white;padding:18px;border-radius:10px}}.grid{{display:flex;gap:8px;flex-wrap:wrap}}.metric{{background:#f1f5f9;padding:12px;border-radius:8px}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#f1f5f9}}small,.muted{{color:#64748b}}.scroll{{overflow-x:auto}}
</style></head><body><div class='hero'><h1>Historical Decision Replay</h1><p>No-lookahead selected-event validation</p></div>
<h2>Summary</h2><div class='grid'><div class='metric'><b>{summary['cases']}</b><br>critical dates</div><div class='metric'><b>{summary['decisive_asset_calls']}</b><br>decisive calls</div><div class='metric'><b>{summary['case_study_hit_rate']:.1f}%</b><br>case-study hit rate</div><div class='metric'><b>{summary['missed_upside']}</b><br>missed upside</div></div>
<p class='muted'>{escape(payload['limitations'])}</p><div class='scroll'><table><tr><th>Date</th><th>Asset</th><th>Market readiness</th><th>Asset readiness</th><th>Signal</th><th>+7D</th><th>+30D</th><th>+90D</th><th>Profit action</th><th>Heat</th><th>30D max DD</th><th>Entry grade</th><th>Exit grade</th></tr>{''.join(rows)}</table></div></body></html>"""
    (out / "report.html").write_text(html)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--cases", default="backtests/cases.yaml")
    parser.add_argument("--output", default="backtests/output")
    args = parser.parse_args()
    payload = run_backtest(args.config, args.cases, args.output)
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
