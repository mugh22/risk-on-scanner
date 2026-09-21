from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from .emailer import send
from .dominance import safe_snapshot
from .deployment import assess_asset_deployment
from .exit_risk import assess_exit_risk
from .indicators import atr, ema, macd, period_return, rsi
from .market_data import BinanceClient
from .portfolio import fallback_spot_prices, load_portfolio
from .profit_protection import assess_profit_protection, heat_call
from .positioning import assess_positioning
from .reporting import render, render_weekly
from .scoring import CoinResult, regime, score_coin
from .signals import classify, levels
from .state import append_run, changes, load, save
from .timeframes import timeframe_evidence
from .weekly import holding_action, snapshot as weekly_snapshot, weekly_market

LOG = logging.getLogger(__name__)
MODEL_VERSION = "closed-candle-v1"
PORTFOLIO_MIN_VALUE_USD = 5.0


def analyze(symbol: str, frame: pd.DataFrame, btc: pd.DataFrame, weights: dict, now: datetime | None = None) -> CoinResult:
    close = frame.close
    e20, e50, e200 = ema(close, 20), ema(close, 50), ema(close, 200)
    _, _, histogram = macd(close)
    btc7, btc30 = period_return(btc.close, 7), period_return(btc.close, 30)
    result = CoinResult(symbol, float(close.iloc[-1]), period_return(close, 1), period_return(close, 7), period_return(close, 30), period_return(close, 7)-btc7, period_return(close, 30)-btc30, float(e20.iloc[-1]), float(e50.iloc[-1]), float(e200.iloc[-1]) if pd.notna(e200.iloc[-1]) else None, float(rsi(close).iloc[-1]), float(histogram.iloc[-1]), float(frame.volume.iloc[-1]/frame.volume.tail(20).mean()), bool(close.iloc[-1] > close.iloc[-21:-1].max()), float(atr(frame).iloc[-1]))
    evidence=timeframe_evidence(frame,btc,now)
    result.weekly_rel=float(evidence["weekly_rel"]); result.weekly_constructive=bool(evidence["weekly_constructive"])
    result.daily_higher_closes=int(evidence["daily_higher_closes"]); result.daily_rel_confirmations=int(evidence["daily_rel_confirmations"])
    result.weekly_higher_low_confirmed=bool(evidence["weekly_higher_low_confirmed"])
    result.breakout_retest=bool(evidence["breakout_retest"])
    result.score, result.reasons = score_coin(result, weights)
    levels(result, float(frame.high.tail(60).iloc[:-1].max()), float(frame.low.tail(20).min()))
    result.recent_high = float(frame.high.tail(60).iloc[:-1].max())
    return result


def market_score(btc: CoinResult, ethbtc: pd.DataFrame | None, coins: list[CoinResult], weights: dict) -> tuple[float, dict]:
    btc_component = sum([btc.price > btc.ema20, btc.price > btc.ema50, btc.ema20 > btc.ema50, btc.usd_30d > 0]) * 25
    if ethbtc is not None:
        ec = ethbtc.close; ee20, ee50 = ema(ec,20), ema(ec,50)
        eth_component = sum([period_return(ec,7)>0, period_return(ec,30)>0, ec.iloc[-1]>ee20.iloc[-1], ee20.iloc[-1]>ee50.iloc[-1]])*25
    else: eth_component = 50
    breadth20 = 100 * sum(c.price > c.ema20 for c in coins) / max(len(coins), 1)
    breadth50 = 100 * sum(c.price > c.ema50 for c in coins) / max(len(coins), 1)
    breadth7 = 100 * sum(c.rel_7d > 0 for c in coins) / max(len(coins), 1)
    breadth30 = 100 * sum(c.rel_30d > 0 for c in coins) / max(len(coins), 1)
    month_regime=(btc_component+eth_component+breadth20+breadth50+breadth30)/5
    weekly_breadth=100*sum(c.weekly_constructive for c in coins)/max(len(coins),1)
    weekly_close=(weekly_breadth+eth_component)/2
    daily_breadth=100*sum(c.daily_rel_confirmations>=2 for c in coins)/max(len(coins),1)
    last_3_closes=(daily_breadth+100*sum(c.daily_higher_closes>=2 for c in coins)/max(len(coins),1))/2
    parts = {"month_regime":month_regime,"weekly_close":weekly_close,"last_3_closes":last_3_closes}
    score = sum(parts[k]*weights[k] for k in weights)/sum(weights.values())
    return round(score,1), {"btc_constructive":btc_component>=75,"eth_btc_positive":eth_component>=75,"breadth_20":breadth20,"breadth_rel30":breadth30,"weekly_breadth":weekly_breadth,"daily_confirmation_breadth":daily_breadth,"month_regime":month_regime,"weekly_close":weekly_close,"last_3_closes":last_3_closes}


def build_portfolio_rows(raw_rows: list[dict], minimum_value: float = PORTFOLIO_MIN_VALUE_USD) -> tuple[list[dict], int]:
    """Exclude dust, calculate weights over visible holdings, and rank by allocation."""
    valued = [(row, row["holding"].quantity * row["price"]) for row in raw_rows]
    included = [(row, value) for row, value in valued if value > minimum_value]
    excluded_count = len(valued) - len(included)
    total_value = sum(value for _, value in included)
    portfolio_rows = []
    for row, value in included:
        holding = row["holding"]
        price = row["price"]
        pnl = "N/A" if holding.average_cost is None else f"{(price/holding.average_cost-1)*100:+.1f}%"
        portfolio_rows.append({
            **{key: item for key, item in row.items() if key != "holding"},
            "symbol": holding.symbol,
            "quantity": holding.quantity,
            "value": value,
            "allocation": 100 * value / max(total_value, .000001),
            "pnl": pnl,
        })
    portfolio_rows.sort(key=lambda row: row["allocation"], reverse=True)
    return portfolio_rows, excluded_count


def previous_closed_score(btc_frame: pd.DataFrame, ethbtc: pd.DataFrame | None,
                          frames: dict[str, pd.DataFrame], alt_weights: dict,
                          risk_weights: dict) -> float | None:
    """Recalculate the regime at the prior completed daily close."""
    # EMA200 is optional in scoring; 62 rows cover the longest required
    # lookback used by returns, recent levels, and the prior-close slice.
    if len(btc_frame) < 62 or any(len(frame) < 62 for frame in frames.values()):
        return None
    prior_btc_frame = btc_frame.iloc[:-1].copy()
    prior_btc = analyze("BTC", prior_btc_frame, prior_btc_frame, alt_weights)
    prior_ethbtc = ethbtc.iloc[:-1].copy() if ethbtc is not None and len(ethbtc) > 1 else None
    prior_coins = [
        analyze(symbol, frame.iloc[:-1].copy(), prior_btc_frame, alt_weights)
        for symbol, frame in frames.items()
    ]
    return market_score(prior_btc, prior_ethbtc, prior_coins, risk_weights)[0]


def run(config_path: str, state_path: str, report_dir: str, no_email: bool = False, history_path: str | None = None, report_mode: str = "daily") -> int:
    cfg = yaml.safe_load(Path(config_path).read_text()); data_cfg=cfg["data"]
    holdings, portfolio_note = load_portfolio()
    client = BinanceClient(data_cfg["timeout_seconds"], data_cfg["retries"]); quote=data_cfg["quote"]
    LOG.info("Scan started using Binance public daily candles")
    btc_frame=client.daily(f"BTC{quote}", data_cfg["days"])
    btc=analyze("BTC",btc_frame,btc_frame,cfg["weights"]["alt_strength"])
    try: ethbtc=client.daily("ETHBTC",data_cfg["days"])
    except RuntimeError: ethbtc=None
    coins=[]; frames={}
    portfolio_symbols = [h.symbol for h in holdings if h.symbol not in {"BTC", "USDT", "USDC", "USD"}]
    symbols = list(dict.fromkeys([*cfg["symbols"], *portfolio_symbols]))
    for symbol in symbols:
        try:
            frame=client.daily(f"{symbol}{quote}",data_cfg["days"]); frames[symbol]=frame
            coins.append(analyze(symbol,frame,btc_frame,cfg["weights"]["alt_strength"]))
        except Exception as exc: LOG.error("Skipping %s: %s",symbol,exc)
    if not coins: raise RuntimeError("No altcoin data was successfully analyzed")
    spot = client.spot_prices([f"BTC{quote}", *[f"{coin.symbol}{quote}" for coin in coins]])
    quote_time, quote_source = spot.fetched_at, spot.source
    btc.live_price = spot.prices[f"BTC{quote}"]
    levels(btc, float(btc_frame.high.tail(60).iloc[:-1].max()), float(btc_frame.low.tail(20).min()), btc.live_price)
    for coin in coins:
        coin.live_price = spot.prices[f"{coin.symbol}{quote}"]
        frame = frames[coin.symbol]
        levels(coin, float(frame.high.tail(60).iloc[:-1].max()), float(frame.low.tail(20).min()), coin.live_price)
    score, context=market_score(btc,ethbtc,coins,cfg["weights"]["risk_on"])
    prior_score = previous_closed_score(
        btc_frame, ethbtc, frames, cfg["weights"]["alt_strength"], cfg["weights"]["risk_on"]
    )
    for coin in coins: coin.signal=classify(coin,score,cfg["signals"])
    previous=load(state_path); comparable=previous if previous.get("model_version")==MODEL_VERSION else {}
    signals={c.symbol:c.signal for c in coins}
    comparison = dict(comparable)
    if prior_score is not None:
        comparison["risk_score"] = prior_score
    notes=changes(comparison,score,signals)
    dominance=safe_snapshot(data_cfg.get("timeout_seconds",15))
    exit_risk=assess_exit_risk(btc_frame,ethbtc,frames,coins,dominance,previous.get("dominance"),previous.get("exit_risk_metrics"))
    history=(previous.get("exit_risk_history") or [])[-19:]+[exit_risk.score]
    by_symbol = {"BTC": btc, **{coin.symbol: coin for coin in coins}}
    protections = {symbol: assess_profit_protection(coin, exit_risk.score) for symbol, coin in by_symbol.items()}
    heat_values = [item.score for item in protections.values()]
    market_heat_score = round(float(pd.Series(heat_values).quantile(.75)), 1) if heat_values else 0.0
    market_heat_level, market_heat_action = heat_call(market_heat_score)
    market_heat = {"score": market_heat_score, "level": market_heat_level, "action": market_heat_action}
    positioning = assess_positioning(
        score, context, btc, coins, exit_risk, market_heat, dominance,
        previous.get("dominance"),
    )
    for coin in [btc, *coins]:
        readiness = assess_asset_deployment(coin, positioning.deployment_status, positioning.rotation_phase, exit_risk.score)
        coin.deployment_status, coin.deployment_reason = readiness.status, readiness.reason
    raw_rows = []
    fallback_prices = fallback_spot_prices([holding.symbol for holding in holdings if holding.symbol not in by_symbol])
    for holding in holdings:
        coin = by_symbol.get(holding.symbol)
        if holding.symbol in {"USDT", "USDC", "USD"}:
            raw_rows.append({"holding": holding, "price": 1.0, "signal": "CASH", "heat": 0.0, "action": "HOLD AS RESERVE"})
        elif coin:
            protection = protections[holding.symbol]
            raw_rows.append({"holding": holding, "price": coin.live_price, "signal": coin.signal, "heat": protection.score, "action": protection.action,
                             "deployment": coin.deployment_status, "deployment_reason": coin.deployment_reason})
        elif holding.symbol in fallback_prices:
            raw_rows.append({"holding": holding, "price": fallback_prices[holding.symbol], "signal": "PRICE ONLY", "heat": 0.0, "action": "NOT SCORED"})
    portfolio_rows, dust_count = build_portfolio_rows(raw_rows)
    if dust_count:
        dust_note = f"{dust_count} balance{'s' if dust_count != 1 else ''} valued at ${PORTFOLIO_MIN_VALUE_USD:.0f} or less excluded."
        portfolio_note = f"{portfolio_note} {dust_note}" if portfolio_note else dust_note
    if (datetime.now(timezone.utc) - quote_time).total_seconds() > 180:
        raise RuntimeError("Live quotes became stale before report rendering")
    if report_mode == "weekly":
        weekly_frames = {"BTC": btc_frame, **frames}
        weekly_items = {}
        for symbol, frame in weekly_frames.items():
            try: weekly_items[symbol] = weekly_snapshot(symbol, frame, btc_frame)
            except ValueError as exc: LOG.warning("Weekly analysis skipped %s: %s", symbol, exc)
        weekly_context = weekly_market(weekly_items)
        for row in portfolio_rows:
            item = weekly_items.get(row["symbol"])
            row["weekly"] = item
            if item: row["weekly_action"] = holding_action(item, row["allocation"], exit_risk.score)
        html,text=render_weekly(weekly_context,weekly_items,portfolio_rows,portfolio_note,exit_risk,dominance,quote_time,quote_source)
    else:
        html,text=render(score,prior_score,coins,notes,context,exit_risk,history,dominance,portfolio_rows,portfolio_note,market_heat,positioning,quote_time,quote_source)
    out=Path(report_dir); out.mkdir(parents=True,exist_ok=True); (out/"report.html").write_text(html); (out/"report.txt").write_text(text)
    new_buys=[n for n in notes if n.startswith("NEW BUY")]
    if report_mode == "weekly": subject=f"Weekly Crypto Outlook: {weekly_context['posture']} | 2–6 Week Window"
    elif exit_risk.score >= 60: subject=f"🚨 {exit_risk.call} | Exit Risk {exit_risk.score:.0f}"
    elif new_buys: subject=f"🚨 {len(new_buys)} NEW BUY SIGNAL{'S' if len(new_buys)!=1 else ''} | Risk-On {score:.0f}"
    else: subject=f"Crypto Market Decision: {exit_risk.call} | Risk-On {score:.0f}"
    if cfg["email"]["enabled"] and not no_email: send(subject,html,text)
    state={"model_version":MODEL_VERSION,"risk_score":score,"exit_risk":exit_risk.score,"exit_risk_history":history,"exit_risk_components":exit_risk.components,"exit_risk_metrics":exit_risk.metrics,"dominance":dominance or previous.get("dominance",{}),"signals":signals,"last_email_window":previous.get("last_email_window")}
    save(state_path,state)
    append_run(history_path,{**state,"regime":regime(score),"buy_count":sum(c.signal=="BUY" for c in coins),"watch_count":sum(c.signal=="WATCH" for c in coins)})
    LOG.info("Analyzed %d assets; %s; exit risk %.0f; BUY=%d WATCH=%d",len(coins),regime(score),exit_risk.score,sum(c.signal=="BUY" for c in coins),sum(c.signal=="WATCH" for c in coins)); return 0


def main() -> int:
    load_dotenv(); p=argparse.ArgumentParser(); p.add_argument("--config",default="config.yaml"); p.add_argument("--state",default="state.json"); p.add_argument("--history"); p.add_argument("--report-dir",default="reports"); p.add_argument("--no-email",action="store_true"); p.add_argument("--report-mode",choices=("daily","weekly"),default="daily"); a=p.parse_args(); logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(name)s: %(message)s"); return run(a.config,a.state,a.report_dir,a.no_email,a.history,a.report_mode)


if __name__ == "__main__": raise SystemExit(main())
