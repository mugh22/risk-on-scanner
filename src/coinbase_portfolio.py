from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Callable

from .portfolio import Holding

LOG = logging.getLogger(__name__)


def _amount(value: object) -> float:
    if isinstance(value, dict):
        value = value.get("value", 0)
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def holdings_from_accounts(accounts: list[dict]) -> list[Holding]:
    """Combine duplicate Coinbase wallets and include funds currently on hold."""
    quantities: dict[str, float] = {}
    for account in accounts:
        symbol = str(account.get("currency", "")).upper().strip()
        if not symbol:
            continue
        quantity = _amount(account.get("available_balance")) + _amount(account.get("hold"))
        if quantity > 1e-12:
            quantities[symbol] = quantities.get(symbol, 0.0) + quantity
    return [Holding(symbol, quantity) for symbol, quantity in sorted(quantities.items())]


def fetch_coinbase_holdings(client_factory: Callable | None = None) -> tuple[list[Holding], str]:
    """Fetch balances with Coinbase's official SDK and a read-only CDP key."""
    key = os.getenv("COINBASE_API_KEY_NAME") or os.getenv("COINBASE_API_KEY")
    secret = os.getenv("COINBASE_API_PRIVATE_KEY") or os.getenv("COINBASE_API_SECRET")
    if not key or not secret:
        raise RuntimeError("Coinbase read-only credentials are not configured")
    secret = secret.replace("\\n", "\n")
    if client_factory is None:
        from coinbase.rest import RESTClient
        client_factory = RESTClient
    client = client_factory(api_key=key, api_secret=secret, timeout=15)
    accounts: list[dict] = []
    cursor = None
    for _ in range(20):
        response = client.get_accounts(limit=250, cursor=cursor) if cursor else client.get_accounts(limit=250)
        payload = response.to_dict() if hasattr(response, "to_dict") else response
        accounts.extend(payload.get("accounts", []))
        if not payload.get("has_next") or not payload.get("cursor"):
            break
        cursor = payload["cursor"]
    holdings = holdings_from_accounts(accounts)
    if not holdings:
        raise RuntimeError("Coinbase returned no non-zero balances")
    LOG.info("Loaded %d non-zero Coinbase balances", len(holdings))
    fetched = datetime.now(timezone.utc)
    return holdings, f"Live quantities from Coinbase read-only API at {fetched:%Y-%m-%d %H:%M UTC}; cost basis and targets come from the portfolio issue when available."


def merge_holdings(coinbase: list[Holding], manual: list[Holding]) -> list[Holding]:
    """Coinbase controls matching quantities; the issue supplies metadata and external assets."""
    issue = {item.symbol: item for item in manual}
    merged = []
    seen = set()
    for item in coinbase:
        override = issue.get(item.symbol)
        merged.append(Holding(item.symbol, item.quantity,
                              override.average_cost if override else None,
                              override.target_allocation if override else None))
        seen.add(item.symbol)
    merged.extend(item for item in manual if item.symbol not in seen)
    return merged
