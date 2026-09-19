from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import httpx

LOG = logging.getLogger(__name__)
ISSUE_TITLE = "Portfolio Configuration — Edit This Issue"


@dataclass(frozen=True)
class Holding:
    symbol: str
    quantity: float
    average_cost: float | None = None
    target_allocation: float | None = None


def _number(value: str) -> float | None:
    cleaned = value.strip().replace("$", "").replace(",", "").replace("%", "")
    if not cleaned or cleaned.lower() in {"n/a", "na", "-", "optional"}:
        return None
    return float(cleaned)


def parse_portfolio_table(markdown: str) -> list[Holding]:
    """Parse the first portfolio-shaped Markdown table in an issue body."""
    lines = [line.strip() for line in markdown.splitlines() if line.strip().startswith("|")]
    for index, line in enumerate(lines):
        headers = [cell.strip().lower() for cell in line.strip("|").split("|")]
        if "symbol" not in headers or "quantity" not in headers or index + 1 >= len(lines):
            continue
        holdings: list[Holding] = []
        for row in lines[index + 2 :]:
            cells = [cell.strip() for cell in row.strip("|").split("|")]
            if len(cells) != len(headers):
                break
            values = dict(zip(headers, cells))
            enabled = values.get("enabled", "yes").strip().lower()
            if enabled not in {"yes", "y", "true", "1", "on"}:
                continue
            symbol = values.get("symbol", "").upper().replace("/USDT", "").strip()
            quantity = _number(values.get("quantity", ""))
            if not symbol or quantity is None or quantity <= 0:
                continue
            holdings.append(
                Holding(
                    symbol=symbol,
                    quantity=quantity,
                    average_cost=_number(values.get("average cost", values.get("avg cost", ""))),
                    target_allocation=_number(values.get("target %", values.get("target allocation", ""))),
                )
            )
        return holdings
    return []


def load_portfolio() -> tuple[list[Holding], str | None]:
    """Read portfolio configuration from a GitHub issue; fail open for reporting."""
    token = os.getenv("GITHUB_TOKEN")
    repository = os.getenv("GITHUB_REPOSITORY")
    if not token or not repository:
        return [], "Portfolio connection is not available in this run."
    headers = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}", "X-GitHub-Api-Version": "2022-11-28"}
    try:
        response = httpx.get(
            f"https://api.github.com/repos/{repository}/issues",
            headers=headers,
            params={"state": "all", "per_page": 100},
            timeout=15,
        )
        response.raise_for_status()
        issue = next((item for item in response.json() if "pull_request" not in item and item.get("title") == ISSUE_TITLE), None)
        if not issue:
            return [], f'Create a GitHub issue named "{ISSUE_TITLE}" to enable portfolio analysis.'
        holdings = parse_portfolio_table(issue.get("body") or "")
        if not holdings:
            return [], "The portfolio issue was found, but it contains no enabled holdings."
        return holdings, None
    except (httpx.HTTPError, ValueError) as exc:
        LOG.warning("Portfolio issue could not be read: %s", type(exc).__name__)
        return [], "Portfolio data could not be read; the market report still completed normally."
