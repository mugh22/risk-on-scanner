from __future__ import annotations

from datetime import datetime, timezone
from html import escape

from .scoring import CoinResult, regime


def _n(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value:,.{digits}f}"


def render(score: float, previous: float | None, coins: list[CoinResult], notes: list[str], context: dict) -> tuple[str, str]:
    title = f"RISK-ON SCORE: {score:.0f}/100 — {regime(score)}"
    delta = "First recorded scan" if previous is None else f"Previous: {previous:.0f} → Current: {score:.0f} ({score-previous:+.0f})"
    top = sorted(coins, key=lambda c: c.score, reverse=True)
    rows = "".join(f"<tr><td><b>{escape(c.symbol)}</b></td><td>{c.score:.0f}</td><td>{c.signal}</td><td>${_n(c.price, 4)}</td><td>{c.rel_7d:+.1f}%</td><td>{c.rel_30d:+.1f}%</td><td>{c.rsi:.1f}</td><td>{c.volume_ratio:.1f}x</td><td>{_n(c.entry_low,4)}–{_n(c.entry_high,4)}</td><td>{_n(c.target1,4)} / {_n(c.target2,4)}</td><td>{_n(c.invalidation,4)}</td></tr>" for c in top[:10])
    updates = "".join(f"<li>{escape(n)}</li>" for n in notes) or "<li>No material signal change.</li>"
    risks = []
    if context["btc_constructive"] is False: risks.append("BTC trend is not constructive")
    if context["eth_btc_positive"] is False: risks.append("ETH/BTC is weak")
    if context["breadth_20"] < 50: risks.append("Less than half the universe is above EMA20")
    risk_html = "".join(f"<li>{escape(x)}</li>" for x in risks) or "<li>No broad market risk flag triggered.</li>"
    css = "body{font-family:Arial,sans-serif;color:#17202a;max-width:900px;margin:auto}h1{font-size:23px}.badge{padding:10px;background:#eef4ff;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{background:#f5f6f7}@media(max-width:600px){table{font-size:11px}th,td{padding:5px}}"
    html = f"<!doctype html><html><head><style>{css}</style></head><body><h1>{title}</h1><div class='badge'>{delta}</div><h2>Market regime</h2><p>BTC trend: {'Constructive' if context['btc_constructive'] else 'Weak/mixed'}<br>ETH/BTC: {'Strengthening' if context['eth_btc_positive'] else 'Weakening'}<br>Breadth above EMA20: {context['breadth_20']:.0f}%<br>Breadth outperforming BTC (30D): {context['breadth_rel30']:.0f}%</p><h2>Top opportunities</h2><table><tr><th>Asset</th><th>Score</th><th>Signal</th><th>Price</th><th>7D/BTC</th><th>30D/BTC</th><th>RSI</th><th>Vol</th><th>Entry zone</th><th>Targets</th><th>Invalidation</th></tr>{rows}</table><h2>New signals</h2><ul>{updates}</ul><h2>Risk flags</h2><ul>{risk_html}</ul><p><small>Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}. Quantitative technical research only; not financial advice or a guarantee of outcomes.</small></p></body></html>"
    text = title + "\n" + delta + "\n\n" + "\n".join(f"{c.symbol}: {c.score:.0f} {c.signal} | 30D/BTC {c.rel_30d:+.1f}% | RSI {c.rsi:.1f}" for c in top[:10]) + "\n\nChanges:\n" + "\n".join(notes or ["No material signal change."]) + "\n\nTechnical research only; not financial advice."
    return html, text

