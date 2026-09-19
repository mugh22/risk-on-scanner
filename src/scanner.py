from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from .emailer import send
from .dominance import safe_snapshot
from .exit_risk import assess_exit_risk
from .indicators import atr, ema, macd, period_return, rsi
from .market_data import BinanceClient
from .reporting import render
from .scoring import CoinResult, regime, score_coin
from .signals import classify, levels
from .state import append_run, changes, load, save

LOG = logging.getLogger(__name__)


def analyze(symbol: str, frame: pd.DataFrame, btc: pd.DataFrame, weights: dict) -> CoinResult:
    close = frame.close
    e20, e50, e200 = ema(close, 20), ema(close, 50), ema(close, 200)
    _, _, histogram = macd(close)
    btc7, btc30 = period_return(btc.close, 7), period_return(btc.close, 30)
    result = CoinResult(symbol, float(close.iloc[-1]), period_return(close, 1), period_return(close, 7), period_return(close, 30), period_return(close, 7)-btc7, period_return(close, 30)-btc30, float(e20.iloc[-1]), float(e50.iloc[-1]), float(e200.iloc[-1]) if pd.notna(e200.iloc[-1]) else None, float(rsi(close).iloc[-1]), float(histogram.iloc[-1]), float(frame.volume.iloc[-1]/frame.volume.tail(20).mean()), bool(close.iloc[-1] > close.iloc[-21:-1].max()), float(atr(frame).iloc[-1]))
    result.score, result.reasons = score_coin(result, weights)
    levels(result, float(frame.high.tail(60).iloc[:-1].max()), float(frame.low.tail(20).min()))
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
    breadth = (breadth20+breadth50+breadth7+breadth30)/4
    momentum = max(0, min(100, 50 + sum(c.rel_30d for c in coins)/max(len(coins),1)*4))
    parts = {"btc_trend":btc_component,"eth_btc":eth_component,"breadth":breadth,"alt_btc_momentum":momentum}
    score = sum(parts[k]*weights[k] for k in weights)/sum(weights.values())
    return round(score,1), {"btc_constructive":btc_component>=75,"eth_btc_positive":eth_component>=75,"breadth_20":breadth20,"breadth_rel30":breadth30}


def run(config_path: str, state_path: str, report_dir: str, no_email: bool = False, history_path: str | None = None) -> int:
    cfg = yaml.safe_load(Path(config_path).read_text()); data_cfg=cfg["data"]
    client = BinanceClient(data_cfg["timeout_seconds"], data_cfg["retries"]); quote=data_cfg["quote"]
    LOG.info("Scan started using Binance public daily candles")
    btc_frame=client.daily(f"BTC{quote}", data_cfg["days"])
    btc=analyze("BTC",btc_frame,btc_frame,cfg["weights"]["alt_strength"])
    try: ethbtc=client.daily("ETHBTC",data_cfg["days"])
    except RuntimeError: ethbtc=None
    coins=[]; frames={}
    for symbol in cfg["symbols"]:
        try:
            frame=client.daily(f"{symbol}{quote}",data_cfg["days"]); frames[symbol]=frame
            coins.append(analyze(symbol,frame,btc_frame,cfg["weights"]["alt_strength"]))
        except Exception as exc: LOG.error("Skipping %s: %s",symbol,exc)
    if not coins: raise RuntimeError("No altcoin data was successfully analyzed")
    score, context=market_score(btc,ethbtc,coins,cfg["weights"]["risk_on"])
    for coin in coins: coin.signal=classify(coin,score,cfg["signals"])
    previous=load(state_path); signals={c.symbol:c.signal for c in coins}; notes=changes(previous,score,signals)
    dominance=safe_snapshot(data_cfg.get("timeout_seconds",15))
    exit_risk=assess_exit_risk(btc_frame,ethbtc,frames,coins,dominance,previous.get("dominance"))
    history=(previous.get("exit_risk_history") or [])[-19:]+[exit_risk.score]
    html,text=render(score,previous.get("risk_score"),coins,notes,context,exit_risk,history,dominance)
    out=Path(report_dir); out.mkdir(parents=True,exist_ok=True); (out/"report.html").write_text(html); (out/"report.txt").write_text(text)
    new_buys=[n for n in notes if n.startswith("NEW BUY")]
    if exit_risk.score >= 60: subject=f"🚨 {exit_risk.call} | Exit Risk {exit_risk.score:.0f}"
    elif new_buys: subject=f"🚨 {len(new_buys)} NEW BUY SIGNAL{'S' if len(new_buys)!=1 else ''} | Risk-On {score:.0f}"
    else: subject=f"Crypto Market Decision: {exit_risk.call} | Risk-On {score:.0f}"
    if cfg["email"]["enabled"] and not no_email: send(subject,html,text)
    state={"risk_score":score,"exit_risk":exit_risk.score,"exit_risk_history":history,"dominance":dominance or previous.get("dominance",{}),"signals":signals}
    save(state_path,state)
    append_run(history_path,{**state,"regime":regime(score),"buy_count":sum(c.signal=="BUY" for c in coins),"watch_count":sum(c.signal=="WATCH" for c in coins)})
    LOG.info("Analyzed %d assets; %s; exit risk %.0f; BUY=%d WATCH=%d",len(coins),regime(score),exit_risk.score,sum(c.signal=="BUY" for c in coins),sum(c.signal=="WATCH" for c in coins)); return 0


def main() -> int:
    load_dotenv(); p=argparse.ArgumentParser(); p.add_argument("--config",default="config.yaml"); p.add_argument("--state",default="state.json"); p.add_argument("--history"); p.add_argument("--report-dir",default="reports"); p.add_argument("--no-email",action="store_true"); a=p.parse_args(); logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(name)s: %(message)s"); return run(a.config,a.state,a.report_dir,a.no_email,a.history)


if __name__ == "__main__": raise SystemExit(main())
