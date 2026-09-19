from __future__ import annotations

from datetime import datetime, timezone
from html import escape

from .exit_risk import ExitRiskResult
from .scoring import CoinResult, regime


def _n(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value:,.{digits}f}"


def _color(score: float) -> str:
    if score < 25: return "#16803c"
    if score < 45: return "#9a6700"
    if score < 60: return "#c45a00"
    return "#c62828"


def _bars(components: dict[str, float]) -> str:
    return "".join(f"<tr><td style='width:185px'>{escape(name)}</td><td><div class='track'><div class='fill' style='width:{value:.0f}%;background:{_color(value)}'></div></div></td><td style='width:36px;text-align:right'><b>{value:.0f}</b></td></tr>" for name, value in components.items())


def _history_chart(history: list[float]) -> str:
    if len(history) < 2:
        return "<p class='muted'>History begins with this scan. Trend will appear after the next run.</p>"
    cells = "".join(f"<td title='{value:.0f}/100' style='height:72px;vertical-align:bottom;padding:0 2px;border:0'><div style='height:{max(3,value*.65):.0f}px;background:{_color(value)};border-radius:3px 3px 0 0'></div></td>" for value in history[-20:])
    return f"<table class='spark' role='img' aria-label='Recent alt exit risk scores'><tr>{cells}</tr></table><div class='chart-label'><span>Older</span><span>Latest: {history[-1]:.0f}</span></div>"


def render(score: float, previous: float | None, coins: list[CoinResult], notes: list[str], context: dict, exit_risk: ExitRiskResult, history: list[float], dominance: dict | None = None) -> tuple[str, str]:
    title = f"RISK-ON SCORE: {score:.0f}/100 — {regime(score)}"
    delta = "First recorded scan" if previous is None else f"Previous: {previous:.0f} → Current: {score:.0f} ({score-previous:+.0f})"
    top = sorted(coins, key=lambda c: c.score, reverse=True)
    rows = "".join(f"<tr><td><b>{escape(c.symbol)}</b></td><td>{c.score:.0f}</td><td>{c.signal}</td><td>${_n(c.price, 4)}</td><td>{c.rel_7d:+.1f}%</td><td>{c.rel_30d:+.1f}%</td><td>{c.rsi:.1f}</td><td>{c.volume_ratio:.1f}x</td><td>{_n(c.entry_low,4)}–{_n(c.entry_high,4)}</td><td>{_n(c.target1,4)} / {_n(c.target2,4)}</td><td>{_n(c.invalidation,4)}</td></tr>" for c in top[:10])
    updates = "".join(f"<li>{escape(n)}</li>" for n in notes) or "<li>No material signal change.</li>"
    flags = "".join(f"<li>{escape(x)}</li>" for x in exit_risk.red_flags) or "<li>No confirmed broad exit flag.</li>"
    supports = "".join(f"<li>{escape(x)}</li>" for x in exit_risk.supports)
    dominance = dominance or {}
    dominance_html = ""
    if dominance:
        dominance_html = f"<h3>Capital rotation snapshot</h3><div class='metrics'><div><b>{dominance.get('btc_d',0):.1f}%</b><small>BTC.D</small></div><div><b>{dominance.get('eth_d',0):.1f}%</b><small>ETH.D</small></div><div><b>{dominance.get('stable_d',0):.1f}%</b><small>Stablecoin D.</small></div><div><b>{dominance.get('alt_ex_stable_btc',0):.2f}</b><small>Alts ex-stables / BTC</small></div><div><b>{dominance.get('concentration',0):.1f}%</b><small>BTC+ETH+USDT+USDC</small></div></div>"
    css = "body{font-family:Arial,sans-serif;color:#17202a;max-width:900px;margin:auto;padding:10px}h1{font-size:23px;margin-bottom:8px}h2{font-size:18px;margin-top:24px}h3{font-size:15px}.decision{padding:18px;border-radius:10px;color:#fff}.decision .call{font-size:23px;font-weight:700;margin:4px 0}.decision .score{font-size:14px}.badge{padding:10px;background:#eef4ff;border-radius:8px}.muted,small{color:#64748b}.compact{margin:7px 0;padding-left:20px}.track{height:10px;background:#edf0f3;border-radius:8px;overflow:hidden}.fill{height:10px;border-radius:8px}.components td{border:0;padding:5px}.spark{border-collapse:collapse;width:100%;max-width:420px}.chart-label{display:flex;justify-content:space-between;max-width:420px;font-size:11px;color:#64748b}.metrics{display:flex;flex-wrap:wrap;gap:8px}.metrics div{background:#f5f6f7;border-radius:7px;padding:9px;min-width:110px}.metrics small{display:block;margin-top:3px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{background:#f5f6f7}.scroll{overflow-x:auto}@media(max-width:600px){table{font-size:11px}th,td{padding:5px}.decision .call{font-size:20px}}"
    html = f"<!doctype html><html><head><meta name='viewport' content='width=device-width'><style>{css}</style></head><body><div class='decision' style='background:{_color(exit_risk.score)}'><div class='score'>ALT EXIT RISK: {exit_risk.score:.0f}/100 — {exit_risk.level}</div><div class='call'>{escape(exit_risk.call)}</div><div>{escape(exit_risk.summary)}</div></div><h2>Why this call</h2><ul class='compact'>{flags}{supports}</ul><table class='components'>{_bars(exit_risk.components)}</table><h3>Exit-risk trend</h3>{_history_chart(history)}{dominance_html}<h2>{title}</h2><div class='badge'>{delta}</div><p>BTC trend: {'Constructive' if context['btc_constructive'] else 'Weak/mixed'}<br>ETH/BTC: {'Strengthening' if context['eth_btc_positive'] else 'Weakening'}<br>Breadth above EMA20: {context['breadth_20']:.0f}%<br>Breadth outperforming BTC (30D): {context['breadth_rel30']:.0f}%</p><h2>Top opportunities</h2><div class='scroll'><table><tr><th>Asset</th><th>Score</th><th>Signal</th><th>Price</th><th>7D/BTC</th><th>30D/BTC</th><th>RSI</th><th>Vol</th><th>Entry zone</th><th>Targets</th><th>Invalidation</th></tr>{rows}</table></div><h2>New signals</h2><ul>{updates}</ul><p><small>Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}. The exit call requires agreement across independent market indicators; it is quantitative research, not financial advice or a guarantee.</small></p></body></html>"
    evidence = exit_risk.red_flags or exit_risk.supports or ["No confirmed broad exit flag."]
    text = f"ALT EXIT RISK: {exit_risk.score:.0f}/100 — {exit_risk.level}\nCALL: {exit_risk.call}\n{exit_risk.summary}\n" + "\n".join(f"- {x}" for x in evidence) + "\n\n" + title + "\n" + delta + "\n\n" + "\n".join(f"{c.symbol}: {c.score:.0f} {c.signal} | 30D/BTC {c.rel_30d:+.1f}% | RSI {c.rsi:.1f}" for c in top[:10]) + "\n\nChanges:\n" + "\n".join(notes or ["No material signal change."]) + "\n\nTechnical research only; not financial advice."
    return html, text
