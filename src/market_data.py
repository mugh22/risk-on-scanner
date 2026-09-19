from __future__ import annotations

import logging
import time

import httpx
import pandas as pd

from .timeframes import completed_daily

LOG = logging.getLogger(__name__)
BASES = (
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api.binance.us",
)


class BinanceClient:
    def __init__(self, timeout: float = 20, retries: int = 3) -> None:
        self.client = httpx.Client(timeout=timeout, headers={"User-Agent": "risk-on-scanner/1.0"}, trust_env=False)
        self.retries = retries

    def daily(self, pair: str, limit: int = 240) -> pd.DataFrame:
        error: Exception | None = None
        for base in BASES:
            for attempt in range(self.retries):
                try:
                    response = self.client.get(f"{base}/api/v3/klines", params={"symbol": pair, "interval": "1d", "limit": limit})
                    if response.status_code in (418, 429):
                        time.sleep(2 ** attempt)
                        continue
                    response.raise_for_status()
                    rows = response.json()
                    if not rows:
                        raise ValueError(f"No candles for {pair}")
                    frame = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "buy_base", "buy_quote", "ignore"])
                    for col in ("open", "high", "low", "close", "volume"):
                        frame[col] = pd.to_numeric(frame[col])
                    frame["time"] = pd.to_datetime(frame["time"], unit="ms", utc=True)
                    LOG.info("Loaded %s from %s", pair, base)
                    frame = completed_daily(frame[["time", "open", "high", "low", "close", "volume"]])
                    if frame.empty:
                        raise ValueError(f"No completed candles for {pair}")
                    return frame
                except (httpx.HTTPError, ValueError) as exc:
                    error = exc
                    LOG.warning("Data attempt %s/%s failed for %s via %s: %s", attempt + 1, self.retries, pair, base, exc)
                    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (400, 404, 451):
                        break
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"Unable to load {pair}: {error}")
