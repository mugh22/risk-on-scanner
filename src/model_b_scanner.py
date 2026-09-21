from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml
import pandas as pd
import httpx
from dotenv import load_dotenv

from .emailer import send
from .market_data import BASES, BinanceClient
from .model_b import current_market_features, predict, train_models, training_dataset
from .model_b_reporting import render_model_b
from .portfolio import load_portfolio
from .scanner import PORTFOLIO_MIN_VALUE_USD
from .timeframes import completed_daily

LOG = logging.getLogger(__name__)


def historical_daily(client: BinanceClient, pair: str, days: int):
    """Paginate Binance's 1,000-candle limit without changing Model A's data client."""
    pages = []
    end_time = None
    remaining = days
    while remaining > 0:
        limit = min(1000, remaining)
        error = None
        page = None
        for base in BASES:
            try:
                params = {"symbol": pair, "interval": "1d", "limit": limit}
                if end_time is not None:
                    params["endTime"] = int(end_time.timestamp() * 1000)
                response = client.client.get(f"{base}/api/v3/klines", params=params)
                response.raise_for_status()
                rows = response.json()
                if not rows:
                    raise ValueError(f"No candles for {pair}")
                raw_count = len(rows)
                page = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume",
                                    "close_time", "quote_volume", "trades", "buy_base", "buy_quote", "ignore"])
                for column in ("open", "high", "low", "close", "volume"):
                    page[column] = pd.to_numeric(page[column])
                page["time"] = pd.to_datetime(page["time"], unit="ms", utc=True)
                page = completed_daily(page[["time", "open", "high", "low", "close", "volume"]])
                break
            except (httpx.HTTPError, ValueError) as exc:
                error = exc
                time.sleep(.5)
        if page is None:
            raise RuntimeError(f"Unable to load historical {pair}: {error}")
        if page.empty:
            break
        pages.append(page)
        remaining -= len(page)
        if raw_count < limit:
            break
        earliest = page["time"].min().to_pydatetime()
        end_time = earliest - timedelta(milliseconds=1)
    if not pages:
        raise RuntimeError(f"No historical candles for {pair}")
    return pd.concat(reversed(pages), ignore_index=True).drop_duplicates("time").sort_values("time").tail(days).reset_index(drop=True)


def _portfolio_rows(holdings, predictions) -> tuple[list[dict], int]:
    prices = {item.symbol: item.price for item in predictions}
    valued = [(holding, holding.quantity * prices[holding.symbol]) for holding in holdings if holding.symbol in prices]
    included = [(holding, value) for holding, value in valued if value > PORTFOLIO_MIN_VALUE_USD]
    total = sum(value for _, value in included)
    rows = [{"symbol": holding.symbol, "value": value, "allocation": value / max(total, 1e-9) * 100}
            for holding, value in included]
    rows.sort(key=lambda row: row["allocation"], reverse=True)
    return rows, len(valued) - len(included)


def run(config_path: str, report_dir: str, no_email: bool = False) -> int:
    cfg = yaml.safe_load(Path(config_path).read_text())
    data_cfg, train_cfg = cfg["data"], cfg["training"]
    client = BinanceClient(data_cfg["timeout_seconds"], data_cfg["retries"])
    quote, days = data_cfg["quote"], int(data_cfg["history_days"])
    btc = historical_daily(client, f"BTC{quote}", days)
    frames = {}
    for symbol in cfg["symbols"]:
        try:
            frames[symbol] = historical_daily(client, f"{symbol}{quote}", days)
        except Exception as exc:
            LOG.warning("Model B skipped %s: %s", symbol, type(exc).__name__)
    if not frames:
        raise RuntimeError("Model B could not load any asset history")
    dataset = training_dataset(frames, btc, train_cfg)
    models, quality = train_models(dataset, float(train_cfg["validation_fraction"]))
    market_features = current_market_features(frames, btc)
    predictions = [predict(symbol, frame, btc, models, cfg["decision"], market_features, quality,
                           float(train_cfg["minimum_validated_balanced_accuracy"]))
                   for symbol, frame in frames.items()]
    holdings, portfolio_note = load_portfolio()
    portfolio_rows, dust = _portfolio_rows(holdings, predictions)
    if dust:
        suffix = f"{dust} supported balance{'s' if dust != 1 else ''} valued at ${PORTFOLIO_MIN_VALUE_USD:.0f} or less excluded."
        portfolio_note = f"{portfolio_note} {suffix}" if portfolio_note else suffix
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html, text = render_model_b(predictions, quality, portfolio_rows, portfolio_note,
                                generated_at, cfg["model_version"])
    output = Path(report_dir); output.mkdir(parents=True, exist_ok=True)
    (output / "report.html").write_text(html)
    (output / "report.txt").write_text(text)
    entries = sum(item.action == "CONSIDER STAGED ENTRY" for item in predictions)
    defensive = sum(item.action == "PROTECT / DO NOT ADD" for item in predictions)
    subject = f"{cfg['email']['subject_prefix']}: {entries} Entry · {defensive} Protect"
    if cfg["email"]["enabled"] and not no_email:
        send(subject, html, text)
    LOG.info("Model B trained on %d observations; %d entry and %d protect calls",
             quality.samples, entries, defensive)
    return 0


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/model_b.yaml")
    parser.add_argument("--report-dir", default="reports/model-b")
    parser.add_argument("--no-email", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return run(args.config, args.report_dir, args.no_email)


if __name__ == "__main__":
    raise SystemExit(main())
