from __future__ import annotations

import logging

import httpx

LOG = logging.getLogger(__name__)


class CoinGeckoClient:
    """Optional market-cap context. Scanner operation never depends on this source."""

    def __init__(self, timeout: float = 15) -> None:
        self.client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": "risk-on-scanner/2.0"},
            trust_env=False,
        )

    def snapshot(self) -> dict:
        global_response = self.client.get("https://api.coingecko.com/api/v3/global")
        global_response.raise_for_status()
        data = global_response.json()["data"]
        total = float(data["total_market_cap"]["usd"])
        percentages = data["market_cap_percentage"]
        btc_d = float(percentages.get("btc", 0))
        eth_d = float(percentages.get("eth", 0))

        stable_response = self.client.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={"vs_currency": "usd", "category": "stablecoins", "order": "market_cap_desc", "per_page": 250, "page": 1},
        )
        stable_response.raise_for_status()
        stables = stable_response.json()
        stable_cap = sum(float(x.get("market_cap") or 0) for x in stables)
        by_symbol = {str(x.get("symbol", "")).lower(): float(x.get("market_cap") or 0) for x in stables}
        usdt_d = 100 * by_symbol.get("usdt", 0) / total
        usdc_d = 100 * by_symbol.get("usdc", 0) / total
        stable_d = 100 * stable_cap / total
        btc_cap = total * btc_d / 100
        eth_cap = total * eth_d / 100
        alt_ex_stables = max(0.0, total - btc_cap - eth_cap - stable_cap)
        return {
            "btc_d": round(btc_d, 3), "eth_d": round(eth_d, 3),
            "usdt_d": round(usdt_d, 3), "usdc_d": round(usdc_d, 3),
            "stable_d": round(stable_d, 3),
            "concentration": round(btc_d + eth_d + usdt_d + usdc_d, 3),
            "stable_btc_ratio": round(stable_cap / btc_cap, 5) if btc_cap else 0,
            "alt_ex_stable_btc": round(alt_ex_stables / btc_cap, 4) if btc_cap else 0,
        }


def safe_snapshot(timeout: float = 15) -> dict:
    try:
        return CoinGeckoClient(timeout).snapshot()
    except Exception as exc:
        LOG.warning("Optional dominance snapshot unavailable: %s", exc)
        return {}
