from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml

from .backtest import _cut, _future_return, _max_drawdown
from .decision_policy import persistent_exit_call
from .exit_risk import assess_exit_risk
from .market_data import BinanceClient
from .scanner import analyze, market_score
from .weekly import entry_action, holding_action, snapshot, weekly_market


SYMBOLS = ["ETH", "SOL", "XRP", "BNB", "ADA", "DOGE", "AVAX", "LINK",
           "NEAR", "ARB", "OP", "SUI", "APT", "INJ", "AAVE", "UNI", "FET",
           "TAO", "POL", "ATOM"]


def _entry_grade(action: str, r4: float, r12: float, drawdown: float) -> str:
    add = action.startswith(("ADD", "PROBE"))
    if add:
        return "GOOD" if (r4 > 0 and drawdown > -15) or r12 >= 20 else "BAD"
    return "MISSED UPSIDE" if r12 >= 20 and drawdown > -20 else "GOOD AVOID" if drawdown <= -15 else "NEUTRAL"


def _holder_grade(action: str, r4: float, drawdown: float) -> str:
    defensive = action.startswith(("REDUCE", "TRIM", "PROTECT"))
    if defensive:
        return "GOOD EXIT" if r4 <= -5 or drawdown <= -15 else "EARLY EXIT"
    return "MISSED EXIT" if r4 <= -10 or drawdown <= -20 else "NO EXIT NEEDED"


def run(config_path: str, cases_path: str, output_dir: str, candidate: bool) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text())
    spec = yaml.safe_load(Path(cases_path).read_text())
    dates = pd.date_range(spec["start"], spec["end"], freq="W-MON", tz="UTC")
    end = dates[-1].to_pydatetime() + timedelta(days=100)
    client = BinanceClient(10, 1, trust_env=True)
    frames = {"BTC": client.daily("BTCUSDT", 1000, end)}
    for symbol in SYMBOLS:
        try:
            frames[symbol] = client.daily(f"{symbol}USDT", 1000, end)
        except RuntimeError:
            pass
    try:
        ethbtc_all = client.daily("ETHBTC", 1000, end)
    except RuntimeError:
        ethbtc_all = None

    results, prior_scores = [], []
    for stamp in dates:
        as_of = stamp.to_pydatetime()
        btc_frame = _cut(frames["BTC"], as_of)
        if len(btc_frame) < 100:
            continue
        btc = analyze("BTC", btc_frame, btc_frame, cfg["weights"]["alt_strength"], as_of)
        coins, used = [], {}
        for symbol, frame in frames.items():
            if symbol == "BTC":
                continue
            sliced = _cut(frame, as_of)
            if len(sliced) < 100:
                continue
            used[symbol] = sliced
            coins.append(analyze(symbol, sliced, btc_frame, cfg["weights"]["alt_strength"], as_of))
        ethbtc = _cut(ethbtc_all, as_of) if ethbtc_all is not None else None
        _, _ = market_score(btc, ethbtc, coins, cfg["weights"]["risk_on"])
        risk = assess_exit_risk(btc_frame, ethbtc, used, coins, now=as_of)
        effective_call, persisted = persistent_exit_call(risk.score, prior_scores, risk.call) if candidate else (risk.call, False)
        prior_scores.append(risk.score)

        snaps = {"BTC": snapshot("BTC", btc_frame, btc_frame, as_of)}
        for symbol, frame in used.items():
            try:
                snaps[symbol] = snapshot(symbol, frame, btc_frame, as_of)
            except ValueError:
                pass
        market = weekly_market(snaps)
        assets = []
        for symbol in spec["focus"]:
            item, full = snaps.get(symbol), frames.get(symbol)
            if item is None or full is None:
                continue
            baseline_holder = holding_action(item, 10, risk.score)
            holder = (effective_call if candidate and symbol != "BTC" and effective_call.startswith(("REDUCE", "EXIT"))
                      else baseline_holder)
            new_capital = entry_action(item, market, risk.score) if candidate else "WATCH — NO EXPLICIT WEEKLY ENTRY"
            r4 = _future_return(full, as_of, item.close, 28)
            r12 = _future_return(full, as_of, item.close, 84)
            dd = _max_drawdown(full, as_of, item.close, 28)
            if None in {r4, r12, dd}:
                continue
            assets.append({"symbol": symbol, "trend": item.trend, "holder_action": holder,
                           "entry_action": new_capital, "forward_4w": r4, "forward_12w": r12,
                           "drawdown_4w": dd, "entry_grade": _entry_grade(new_capital, r4, r12, dd),
                           "holder_grade": _holder_grade(holder, r4, dd)})
        results.append({"date": stamp.date().isoformat(), "key_date": stamp.date().isoformat() in {str(x) for x in spec.get("key_dates", [])},
                        "market": market, "exit_risk": risk.score, "exit_call": effective_call,
                        "persisted": persisted, "assets": assets})

    flat = [a for row in results for a in row["assets"]]
    summary = {
        "weeks": len(results),
        "good_entries": sum(a["entry_grade"] == "GOOD" for a in flat),
        "bad_entries": sum(a["entry_grade"] == "BAD" for a in flat),
        "missed_upside": sum(a["entry_grade"] == "MISSED UPSIDE" for a in flat),
        "good_exits": sum(a["holder_grade"] == "GOOD EXIT" for a in flat),
        "early_exits": sum(a["holder_grade"] == "EARLY EXIT" for a in flat),
        "missed_exits": sum(a["holder_grade"] == "MISSED EXIT" for a in flat),
    }
    payload = {"model": "candidate" if candidate else "baseline", "summary": summary, "results": results}
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(payload, indent=2))
    return payload


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--cases", default="backtests/weekly_cases.yaml")
    p.add_argument("--output", default="backtests/output/weekly")
    p.add_argument("--candidate", action="store_true")
    args = p.parse_args()
    print(json.dumps(run(args.config, args.cases, args.output, args.candidate)["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
