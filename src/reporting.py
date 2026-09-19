from __future__ import annotations

from datetime import datetime, timezone
from html import escape

from .exit_risk import ExitRiskResult
from .scoring import CoinResult, regime


def _n(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value:,.{digits}f}"


def _risk_color(score: float) -> str:
    if score < 25: return "#16803c"
    if score < 45: return "#9a6700"
    if score < 60: return "#c45a00"
    return "#c62828"


def _heat_color(score: float) -> str:
    if score < 40: return "#16803c"
    if score < 60: return "#9a6700"
    if score < 75: return "#c45a00"
    return "#c62828"


def _bars(components: dict[str, float]) -> str:
    rows = []
    for name, value in components.items():
        width = max(0, min(100, value))
        rows.append(
            "<tr>"
            f"<td class='bar-label'>{escape(name)}</td>"
            "<td class='bar-cell'><table role='presentation' class='bar-track' cellpadding='0' cellspacing='0'><tr>"
            f"<td style='width:{width:.0f}%;background:{_risk_color(value)};height:12px;line-height:12px'>&nbsp;</td>"
            f"<td style='width:{100-width:.0f}%;background:#e2e8f0;height:12px;line-height:12px'>&nbsp;</td>"
            "</tr></table></td>"
            f"<td class='bar-value'><b>{value:.0f}</b></td></tr>"
        )
    return "".join(rows)


def _history_chart(history: list[float]) -> str:
    if len(history) < 2:
        return "<p class='muted'>History begins with this scan. Trend will appear after the next run.</p>"
    chips = "".join(
        f"<span class='trend-chip' style='border-color:{_risk_color(value)}'>{value:.0f}</span>"
        for value in history[-6:]
    )
    direction = history[-1] - history[-2]
    word = "rising" if direction > 1 else "falling" if direction < -1 else "stable"
    return f"<div>{chips}</div><div class='muted'>Oldest → latest · risk is <b>{word}</b></div>"


def _portfolio_html(portfolio_rows: list[dict], portfolio_note: str | None) -> str:
    if not portfolio_rows:
        note = portfolio_note or "Add holdings to the portfolio configuration issue."
        return f"<div class='soft'><b>Portfolio analysis not active.</b><br><span class='muted'>{escape(note)}</span></div>"
    urgent = [row for row in portfolio_rows if row["heat"] >= 60 or row["signal"] == "BUY"]
    actions = urgent or sorted(portfolio_rows, key=lambda row: row["heat"], reverse=True)[:3]
    cards = "".join(
        f"<div class='action-row'><b>{escape(row['symbol'])}</b><span>{escape(row['action'])}</span>"
        f"<small>Heat {row['heat']:.0f}/100 · {escape(row['signal'])} · allocation {row['allocation']:.1f}%</small></div>"
        for row in actions
    )
    rows = "".join(
        f"<tr><td><b>{escape(row['symbol'])}</b></td><td>{row['quantity']:,.6g}</td><td>${row['price']:,.4f}</td>"
        f"<td>${row['value']:,.2f}</td><td>{row['allocation']:.1f}%</td><td>{row['pnl']}</td>"
        f"<td>{escape(row['signal'])}</td><td><b>{row['heat']:.0f}</b></td><td>{escape(row['action'])}</td></tr>"
        for row in portfolio_rows
    )
    return f"{cards}<div class='scroll detail-table'><table><tr><th>Asset</th><th>Qty</th><th>Price</th><th>Value</th><th>Weight</th><th>P/L</th><th>Signal</th><th>Heat</th><th>Action</th></tr>{rows}</table></div>"


def render(
    score: float, previous: float | None, coins: list[CoinResult], notes: list[str],
    context: dict, exit_risk: ExitRiskResult, history: list[float],
    dominance: dict | None = None, portfolio_rows: list[dict] | None = None,
    portfolio_note: str | None = None, market_heat: dict | None = None,
) -> tuple[str, str]:
    portfolio_rows = portfolio_rows or []
    market_heat = market_heat or {"score": 0, "level": "LOW", "action": "HOLD"}
    title = f"RISK-ON SCORE: {score:.0f}/100 — {regime(score)}"
    delta = "New closed-candle model baseline" if previous is None else f"Previous: {previous:.0f} → Current: {score:.0f} ({score-previous:+.0f})"
    ranked = sorted(coins, key=lambda c: c.score, reverse=True)
    buys = [c for c in ranked if c.signal == "BUY"]
    others = [c for c in ranked if c.signal != "BUY"]
    top = buys + others[:max(0, 10-len(buys))]
    rows = "".join(f"<tr><td><b>{escape(c.symbol)}</b></td><td>{c.score:.0f}</td><td><b>{c.signal}</b></td><td>${_n(c.price, 4)}</td><td>{c.rel_30d:+.1f}%</td><td>{c.weekly_rel:+.1f}%</td><td>{c.daily_rel_confirmations}/3</td><td>{c.rsi:.1f}</td><td>{c.volume_ratio:.1f}x</td><td>{_n(c.entry_low,4)}–{_n(c.entry_high,4)}</td><td>{_n(c.target1,4)} / {_n(c.target2,4)}</td><td>{_n(c.invalidation,4)}</td></tr>" for c in top)
    updates = "".join(f"<li>{escape(n)}</li>" for n in notes) or "<li>No material signal change.</li>"
    evidence = exit_risk.red_flags or exit_risk.supports or ["No confirmed broad exit flag."]
    evidence_html = "".join(f"<li>{escape(item)}</li>" for item in evidence[:4])
    dominance = dominance or {}
    dominance_html = ""
    if dominance:
        dominance_html = f"<h3>Capital rotation</h3><div class='metrics'><div><b>{dominance.get('btc_d',0):.1f}%</b><small>BTC.D</small></div><div><b>{dominance.get('eth_d',0):.1f}%</b><small>ETH.D</small></div><div><b>{dominance.get('stable_d',0):.1f}%</b><small>Stablecoin D.</small></div><div><b>{dominance.get('alt_ex_stable_btc',0):.2f}</b><small>Alts/BTC</small></div><div><b>{dominance.get('concentration',0):.1f}%</b><small>Capital concentration</small></div></div>"
    css = "body{font-family:Arial,sans-serif;color:#17202a;max-width:900px;margin:auto;padding:10px}h1{font-size:23px;margin:0 0 8px}h2{font-size:19px;margin:24px 0 10px}h3{font-size:15px;margin:18px 0 8px}.decision{padding:16px;border-radius:10px;color:#fff;margin-bottom:10px}.decision .call{font-size:23px;font-weight:700;margin:4px 0}.decision .score{font-size:13px}.heat{padding:13px;border:2px solid;border-radius:10px;margin:10px 0}.heat b{font-size:18px}.badge,.soft{padding:11px;background:#eef4ff;border-radius:8px}.muted,small{color:#64748b}.compact{margin:7px 0;padding-left:20px}.components{width:100%;max-width:620px}.components td{border:0;padding:6px 4px}.bar-label{width:190px}.bar-cell{width:auto}.bar-track{width:100%;border-collapse:collapse;border:1px solid #cbd5e1}.bar-value{width:38px;text-align:right}.trend-chip{display:inline-block;border:2px solid;border-radius:14px;padding:4px 8px;margin:2px 5px 5px 0;font-weight:700}.metrics{display:flex;flex-wrap:wrap;gap:8px}.metrics div{background:#f5f6f7;border-radius:7px;padding:9px;min-width:110px}.metrics small,.action-row small{display:block;margin-top:3px}.action-row{border-left:5px solid #334155;background:#f8fafc;padding:10px 12px;margin:7px 0}.action-row span{float:right;font-weight:700}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{background:#f5f6f7}.scroll{overflow-x:auto}.detail-table{margin-top:12px}.details{border-top:1px solid #cbd5e1;margin-top:26px}@media(max-width:600px){table{font-size:11px}th,td{padding:5px}.decision .call{font-size:20px}.bar-label{width:145px}.action-row span{float:none;display:block;margin-top:3px}}"
    html = f"""<!doctype html><html><head><meta name='viewport' content='width=device-width'><style>{css}</style></head><body>
<div class='decision' style='background:{_risk_color(exit_risk.score)}'><div class='score'>ALT EXIT RISK: {exit_risk.score:.0f}/100 — {exit_risk.level}</div><div class='call'>{escape(exit_risk.call)}</div><div>{escape(exit_risk.summary)}</div></div>
<div class='heat' style='border-color:{_heat_color(market_heat['score'])}'><small>RALLY HEAT / PROFIT PROTECTION: {market_heat['score']:.0f}/100 — {escape(market_heat['level'])}</small><br><b>{escape(market_heat['action'])}</b></div>
<h2>Your portfolio — actions first</h2>{_portfolio_html(portfolio_rows, portfolio_note)}
<h2>What changed</h2><ul class='compact'>{updates}</ul>
<h2>Why this call</h2><ul class='compact'>{evidence_html}</ul>
<div class='details'><h2>Market evidence</h2><h3>Exit-risk components</h3><table role='presentation' class='components'>{_bars(exit_risk.components)}</table><h3>Exit-risk trend</h3>{_history_chart(history)}
<h3>{title}</h3><div class='badge'>{delta}</div><p><b>Closed-candle evidence:</b> 30-day regime {context['month_regime']:.0f}/100 · Weekly close {context['weekly_close']:.0f}/100 · Last 3 daily closes {context['last_3_closes']:.0f}/100</p><p>BTC trend: {'Constructive' if context['btc_constructive'] else 'Weak/mixed'}<br>ETH/BTC: {'Strengthening' if context['eth_btc_positive'] else 'Weakening'}<br>Breadth above EMA20: {context['breadth_20']:.0f}%<br>Breadth outperforming BTC (30D): {context['breadth_rel30']:.0f}%<br>Constructive completed weekly structures: {context['weekly_breadth']:.0f}%<br>Confirming 2 of last 3 daily closes vs BTC: {context['daily_confirmation_breadth']:.0f}%</p>{dominance_html}
<h2>Market opportunities</h2><div class='scroll'><table><tr><th>Asset</th><th>Score</th><th>Signal</th><th>Closed price</th><th>30D/BTC</th><th>Week/BTC</th><th>3D confirms</th><th>RSI</th><th>Vol</th><th>Entry zone</th><th>Targets</th><th>Invalidation</th></tr>{rows}</table></div></div>
<p><small>Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}. Signals use completed daily and weekly candles; intraday movement cannot flip a confirmed call. Quantitative research only; not financial advice or a guarantee.</small></p></body></html>"""
    portfolio_text = "\n".join(f"{r['symbol']}: {r['action']} | heat {r['heat']:.0f} | allocation {r['allocation']:.1f}%" for r in portfolio_rows) or (portfolio_note or "Portfolio not configured.")
    text = f"ALT EXIT RISK: {exit_risk.score:.0f}/100 — {exit_risk.level}\nCALL: {exit_risk.call}\nRALLY HEAT: {market_heat['score']:.0f}/100 — {market_heat['action']}\n\nYOUR PORTFOLIO\n{portfolio_text}\n\nCHANGES\n" + "\n".join(notes or ["No material signal change."]) + "\n\nMARKET EVIDENCE\n" + "\n".join(f"- {x}" for x in evidence) + "\n\n" + title + "\n" + delta + "\n\nTechnical research only; not financial advice."
    return html, text
