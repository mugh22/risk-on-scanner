from __future__ import annotations

from datetime import datetime, timezone
from html import escape

from .exit_risk import ExitRiskResult
from .scoring import CoinResult, regime
from .positioning import PositioningResult
from .weekly import WeeklySnapshot


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


def _bars(components: dict[str, float], directions: dict[str, str] | None = None) -> str:
    directions = directions or {}
    rows = []
    for name, value in components.items():
        width = max(0, min(100, value))
        fill = "" if width == 0 else f"<td style='width:{width:.0f}%;background:{_risk_color(value)};height:12px;line-height:12px'>&nbsp;</td>"
        remainder = 100 if width == 0 else 100 - width
        rows.append(
            "<tr>"
            f"<td class='bar-label'>{escape(name)}</td>"
            "<td class='bar-cell'><table role='presentation' class='bar-track' cellpadding='0' cellspacing='0'><tr>"
            f"{fill}<td style='width:{remainder:.0f}%;background:#e2e8f0;height:12px;line-height:12px'>&nbsp;</td>"
            "</tr></table></td>"
            f"<td class='bar-value'><b>{value:.0f}</b></td>"
            f"<td class='direction'>{escape(directions.get(name, 'Healthy · baseline'))}</td></tr>"
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
    source = f"<p class='muted'><small>{escape(portfolio_note)}</small></p>" if portfolio_note else ""
    return f"{cards}{source}<div class='scroll detail-table'><table><tr><th>Asset</th><th>Qty</th><th>Price</th><th>Value</th><th>Weight</th><th>P/L</th><th>Signal</th><th>Heat</th><th>Action</th></tr>{rows}</table></div>"


def render(
    score: float, previous: float | None, coins: list[CoinResult], notes: list[str],
    context: dict, exit_risk: ExitRiskResult, history: list[float],
    dominance: dict | None = None, portfolio_rows: list[dict] | None = None,
    portfolio_note: str | None = None, market_heat: dict | None = None,
    positioning: PositioningResult | None = None,
    quote_time: datetime | None = None, quote_source: str | None = None,
) -> tuple[str, str]:
    portfolio_rows = portfolio_rows or []
    market_heat = market_heat or {"score": 0, "level": "LOW", "action": "HOLD"}
    title = f"RISK-ON SCORE: {score:.0f}/100 — {regime(score)}"
    delta = "New closed-candle model baseline" if previous is None else f"Previous daily close: {previous:.1f} → Current: {score:.1f} ({score-previous:+.1f})"
    ranked = sorted(coins, key=lambda c: c.score, reverse=True)
    buys = [c for c in ranked if c.signal == "BUY"]
    others = [c for c in ranked if c.signal != "BUY"]
    top = buys + others[:max(0, 10-len(buys))]
    rows = "".join(f"<tr><td><b>{escape(c.symbol)}</b></td><td>{c.score:.0f}</td><td><b>{c.signal}</b></td><td>${_n(c.live_price, 4)}</td><td>${_n(c.price, 4)}</td><td>{c.rel_30d:+.1f}%</td><td>{c.weekly_rel:+.1f}%</td><td>{c.daily_rel_confirmations}/3</td><td>{c.rsi:.1f}</td><td>{c.volume_ratio:.1f}x</td><td>{_n(c.entry_low,4)}–{_n(c.entry_high,4)}</td><td>{_n(c.target1,4)} / {_n(c.target2,4)}</td><td>{_n(c.invalidation,4)}</td></tr>" for c in top)
    updates = "".join(f"<li>{escape(n)}</li>" for n in notes) or "<li>No material signal change.</li>"
    evidence = exit_risk.red_flags or exit_risk.supports or ["No confirmed broad exit flag."]
    evidence_html = "".join(f"<li>{escape(item)}</li>" for item in evidence[:4])
    dominance = dominance or {}
    dominance_html = ""
    if dominance:
        dominance_html = f"<h3>Capital rotation</h3><div class='metrics'><div><b>{dominance.get('btc_d',0):.1f}%</b><small>BTC.D</small></div><div><b>{dominance.get('eth_d',0):.1f}%</b><small>ETH.D</small></div><div><b>{dominance.get('stable_d',0):.1f}%</b><small>Stablecoin D.</small></div><div><b>{dominance.get('alt_ex_stable_btc',0):.2f}</b><small>Alts/BTC</small></div><div><b>{dominance.get('concentration',0):.1f}%</b><small>Capital concentration</small></div></div>"
    if positioning:
        positioning_html = (
            "<div class='positioning'><small>MARKET POSITIONING ENGINE</small>"
            f"<div class='position-row'><b>MARKET</b><span>{escape(positioning.market_regime)}</span></div>"
            f"<div class='position-row'><b>ALT ROTATION</b><span>{escape(positioning.rotation_phase)}</span></div>"
            f"<div class='position-row accent'><b>NEW CAPITAL</b><span>{escape(positioning.deployment_status)}</span></div>"
            f"<p>{escape(positioning.deployment_action)}</p>"
            f"<div class='position-row'><b>EXISTING POSITIONS</b><span>{escape(positioning.existing_action)}</span></div>"
            f"<div class='position-row'><b>RESERVE BAND</b><span>{escape(positioning.cash_band)}</span></div></div>"
        )
        positioning_evidence = "<h3>Positioning evidence</h3><ul class='compact'>" + "".join(
            f"<li>{escape(item)}</li>" for item in positioning.evidence
        ) + "</ul>"
    else:
        positioning_html = positioning_evidence = ""
    css = "body{font-family:Arial,sans-serif;color:#17202a;max-width:900px;margin:auto;padding:10px}h1{font-size:23px;margin:0 0 8px}h2{font-size:19px;margin:24px 0 10px}h3{font-size:15px;margin:18px 0 8px}.positioning{border:2px solid #334155;border-radius:10px;padding:13px;margin-bottom:12px;background:#f8fafc}.positioning>small{font-weight:700;color:#475569}.position-row{display:flex;gap:12px;justify-content:space-between;border-top:1px solid #dbe1e8;padding:8px 0}.position-row:first-of-type{margin-top:8px}.position-row b{min-width:130px}.position-row span{text-align:right;font-weight:700}.position-row.accent span{color:#155e75}.positioning p{margin:3px 0 8px;color:#475569}.decision{padding:16px;border-radius:10px;color:#fff;margin-bottom:10px}.decision .call{font-size:23px;font-weight:700;margin:4px 0}.decision .score{font-size:13px}.heat{padding:13px;border:2px solid;border-radius:10px;margin:10px 0}.heat b{font-size:18px}.badge,.soft{padding:11px;background:#eef4ff;border-radius:8px}.muted,small{color:#64748b}.compact{margin:7px 0;padding-left:20px}.components{width:100%;max-width:760px}.components td{border:0;padding:6px 4px}.bar-label{width:175px}.bar-cell{width:auto}.bar-track{width:100%;border-collapse:collapse;border:1px solid #cbd5e1}.bar-value{width:38px;text-align:right}.direction{width:145px;color:#475569;font-weight:700;white-space:nowrap}.trend-chip{display:inline-block;border:2px solid;border-radius:14px;padding:4px 8px;margin:2px 5px 5px 0;font-weight:700}.metrics{display:flex;flex-wrap:wrap;gap:8px}.metrics div{background:#f5f6f7;border-radius:7px;padding:9px;min-width:110px}.metrics small,.action-row small{display:block;margin-top:3px}.action-row{border-left:5px solid #334155;background:#f8fafc;padding:10px 12px;margin:7px 0}.action-row span{float:right;font-weight:700}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{background:#f5f6f7}.scroll{overflow-x:auto}.detail-table{margin-top:12px}.details{border-top:1px solid #cbd5e1;margin-top:26px}@media(max-width:600px){table{font-size:11px}th,td{padding:5px}.decision .call{font-size:20px}.bar-label{width:115px}.direction{width:108px;white-space:normal}.action-row span{float:none;display:block;margin-top:3px}.position-row{display:block}.position-row span{display:block;text-align:left;margin-top:3px}.position-row b{min-width:0}}"
    html = f"""<!doctype html><html><head><meta name='viewport' content='width=device-width'><style>{css}</style></head><body>
{positioning_html}
<div class='decision' style='background:{_risk_color(exit_risk.score)}'><div class='score'>ALT EXIT RISK: {exit_risk.score:.0f}/100 — {exit_risk.level}</div><div class='call'>{escape(exit_risk.call)}</div><div>{escape(exit_risk.summary)}</div></div>
<div class='heat' style='border-color:{_heat_color(market_heat['score'])}'><small>RALLY HEAT / PROFIT PROTECTION: {market_heat['score']:.0f}/100 — {escape(market_heat['level'])}</small><br><b>{escape(market_heat['action'])}</b></div>
<h2>Your portfolio — actions first</h2>{_portfolio_html(portfolio_rows, portfolio_note)}
<h2>What changed</h2><ul class='compact'>{updates}</ul>
<h2>Why this call</h2><ul class='compact'>{evidence_html}</ul>
<div class='details'><h2>Market evidence</h2>{positioning_evidence}<h3>Exit-risk components</h3><table role='presentation' class='components'><tr><th>Component</th><th>Risk bar</th><th>Score</th><th>Direction</th></tr>{_bars(exit_risk.components, exit_risk.directions)}</table><p class='muted'><small>Score is confirmed risk. Direction compares the underlying evidence with the previous scan and can warn before a score threshold is crossed.</small></p><h3>Exit-risk trend</h3>{_history_chart(history)}
<h3>{title}</h3><div class='badge'>{delta}</div><p><b>Closed-candle evidence:</b> 30-day regime {context['month_regime']:.0f}/100 · Weekly close {context['weekly_close']:.0f}/100 · Last 3 daily closes {context['last_3_closes']:.0f}/100</p><p>BTC trend: {'Constructive' if context['btc_constructive'] else 'Weak/mixed'}<br>ETH/BTC: {'Strengthening' if context['eth_btc_positive'] else 'Weakening'}<br>Breadth above EMA20: {context['breadth_20']:.0f}%<br>Breadth outperforming BTC (30D): {context['breadth_rel30']:.0f}%<br>Constructive completed weekly structures: {context['weekly_breadth']:.0f}%<br>Confirming 2 of last 3 daily closes vs BTC: {context['daily_confirmation_breadth']:.0f}%</p>{dominance_html}
<h2>Market opportunities</h2><p class='muted'>Live quotes fetched {quote_time or datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S UTC} from {escape(quote_source or 'live market source')}. Signal close is the latest completed daily candle used for scoring; execution levels use the live quote plus closed technical structure.</p><div class='scroll'><table><tr><th>Asset</th><th>Score</th><th>Signal</th><th>Live price</th><th>Signal close</th><th>30D/BTC</th><th>Week/BTC</th><th>3D confirms</th><th>RSI</th><th>Vol</th><th>Entry zone</th><th>Targets</th><th>Invalidation</th></tr>{rows}</table></div></div>
<p><small>Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}. Signals use completed daily and weekly candles; intraday movement cannot flip a confirmed call. Quantitative research only; not financial advice or a guarantee.</small></p></body></html>"""
    portfolio_text = "\n".join(f"{r['symbol']}: {r['action']} | heat {r['heat']:.0f} | allocation {r['allocation']:.1f}%" for r in portfolio_rows) or (portfolio_note or "Portfolio not configured.")
    positioning_text = "" if not positioning else f"MARKET: {positioning.market_regime}\nALT ROTATION: {positioning.rotation_phase}\nNEW CAPITAL: {positioning.deployment_status}\n{positioning.deployment_action}\nEXISTING POSITIONS: {positioning.existing_action}\nRESERVE BAND: {positioning.cash_band}\n\n"
    text = positioning_text + f"ALT EXIT RISK: {exit_risk.score:.0f}/100 — {exit_risk.level}\nCALL: {exit_risk.call}\nRALLY HEAT: {market_heat['score']:.0f}/100 — {market_heat['action']}\n\nYOUR PORTFOLIO\n{portfolio_text}\n\nCHANGES\n" + "\n".join(notes or ["No material signal change."]) + "\n\nMARKET EVIDENCE\n" + "\n".join(f"- {x}" for x in evidence) + "\n\n" + title + "\n" + delta + "\n\nTechnical research only; not financial advice."
    return html, text


def render_weekly(
    market: dict, snapshots: dict[str, WeeklySnapshot], portfolio_rows: list[dict],
    portfolio_note: str | None, exit_risk: ExitRiskResult, dominance: dict | None,
    quote_time: datetime | None, quote_source: str | None,
) -> tuple[str, str]:
    """Render a portfolio-first report using only completed weekly structure."""
    color = _risk_color(100 - market["score"])
    portfolio = []
    for row in sorted(portfolio_rows, key=lambda item: item["allocation"], reverse=True):
        item = row.get("weekly")
        if not item:
            portfolio.append(
                f"<tr><td><b>{escape(row['symbol'])}</b></td><td>{row['allocation']:.1f}%</td>"
                f"<td colspan='7'>Weekly candles unavailable — {escape(row.get('action','NOT SCORED'))}</td></tr>"
            )
            continue
        portfolio.append(
            f"<tr><td><b>{escape(row['symbol'])}</b></td><td>{row['allocation']:.1f}%</td>"
            f"<td><b>{escape(item.trend)}</b></td><td>{item.return_4w:+.1f}%</td><td>{item.return_12w:+.1f}%</td>"
            f"<td>{item.relative_4w:+.1f}%</td><td>{item.relative_12w:+.1f}%</td><td>{item.rsi:.0f}</td>"
            f"<td><b>{escape(row.get('weekly_action','HOLD'))}</b></td></tr>"
        )
    ranked = sorted((item for symbol, item in snapshots.items() if symbol != "BTC"), key=lambda x: x.relative_4w, reverse=True)
    leaders = "".join(
        f"<tr><td><b>{escape(item.symbol)}</b></td><td>{escape(item.trend)}</td><td>{item.return_4w:+.1f}%</td>"
        f"<td>{item.relative_4w:+.1f}%</td><td>{item.relative_12w:+.1f}%</td><td>{item.positive_weeks}/3</td>"
        f"<td>{item.drawdown_12w:.1f}%</td></tr>" for item in ranked[:10]
    )
    btc = snapshots.get("BTC")
    btc_line = "Unavailable" if not btc else f"{btc.trend} · 4W {btc.return_4w:+.1f}% · 12W {btc.return_12w:+.1f}% · weekly RSI {btc.rsi:.0f}"
    concentration = max((row["allocation"] for row in portfolio_rows), default=0)
    if market["score"] >= 65:
        base = "Hold weekly leaders; add only on support or confirmed weekly continuation."
        bull = "Breadth remains above 65% and BTC holds its 10-week EMA: allow winners room for another 2–6 weeks."
        bear = "Two weak weekly closes plus falling 4-week relative breadth: trim weaker alts and rebuild cash."
    else:
        base = "Protect capital and require weekly repair before adding risk."
        bull = "BTC reclaims its 4/10-week trend and alt relative breadth returns above 55%: selectively re-risk."
        bear = "BTC loses its 10-week EMA while distribution exceeds 50%: reduce broad alt exposure."
    dominance = dominance or {}
    css = "body{font-family:Arial,sans-serif;color:#17202a;max-width:920px;margin:auto;padding:12px}h1{font-size:24px;margin:0}h2{font-size:19px;margin:24px 0 9px}.hero{color:#fff;padding:17px;border-radius:10px}.hero b{font-size:24px}.sub{opacity:.9;margin-top:5px}.grid{display:flex;flex-wrap:wrap;gap:8px}.metric{background:#f1f5f9;border-radius:8px;padding:10px;min-width:125px}.metric b{font-size:19px;display:block}.call{border-left:5px solid #334155;background:#f8fafc;padding:11px;margin:8px 0}.scenario{padding:11px;border-radius:8px;background:#f8fafc;margin:7px 0}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:7px;border-bottom:1px solid #dbe1e8;text-align:left}th{background:#f1f5f9}.scroll{overflow-x:auto}.muted,small{color:#64748b}@media(max-width:600px){table{font-size:10px}th,td{padding:5px}.hero b{font-size:20px}}"
    html = f"""<!doctype html><html><head><meta name='viewport' content='width=device-width'><style>{css}</style></head><body>
<div class='hero' style='background:{color}'><small>WEEKLY CRYPTO OUTLOOK · COMPLETED CANDLES ONLY</small><br><b>{escape(market['posture'])} — {market['score']:.0f}/100</b><div class='sub'>{escape(market['horizon'])}</div></div>
<h2>What to do this week</h2><div class='call'><b>Base plan:</b> {escape(base)}</div><div class='call'><b>Portfolio concentration:</b> largest position {concentration:.1f}% · broad exit risk {exit_risk.score:.0f}/100 ({escape(exit_risk.level)})</div>
<h2>Your portfolio — weekly decisions</h2>{f"<p class='muted'><small>{escape(portfolio_note)}</small></p>" if portfolio_note else ''}<div class='scroll'><table><tr><th>Asset</th><th>Weight</th><th>Weekly trend</th><th>4W</th><th>12W</th><th>4W/BTC</th><th>12W/BTC</th><th>W-RSI</th><th>Action</th></tr>{''.join(portfolio)}</table></div>
<h2>Multi-week market structure</h2><div class='grid'><div class='metric'><b>{market['above4']:.0f}%</b><small>Above 4W EMA</small></div><div class='metric'><b>{market['above10']:.0f}%</b><small>Above 10W EMA</small></div><div class='metric'><b>{market['rel4']:.0f}%</b><small>Beat BTC over 4W</small></div><div class='metric'><b>{market['rel12']:.0f}%</b><small>Beat BTC over 12W</small></div><div class='metric'><b>{market['distribution']:.0f}%</b><small>Weekly weakening</small></div></div><p><b>BTC weekly:</b> {escape(btc_line)}</p><p><b>Rotation:</b> BTC.D {dominance.get('btc_d',0):.1f}% · ETH.D {dominance.get('eth_d',0):.1f}% · Stablecoin dominance {dominance.get('stable_d',0):.1f}%</p>
<h2>2–6 week scenario map</h2><div class='scenario'><b>Base:</b> {escape(base)}</div><div class='scenario'><b>Bull confirmation:</b> {escape(bull)}</div><div class='scenario'><b>Bear / protection trigger:</b> {escape(bear)}</div>
<h2>Weekly relative-strength leaders</h2><div class='scroll'><table><tr><th>Asset</th><th>Trend</th><th>4W</th><th>4W/BTC</th><th>12W/BTC</th><th>Positive weeks</th><th>From 12W high</th></tr>{leaders}</table></div>
<p class='muted'><small>Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}. Live portfolio prices fetched {quote_time or datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} from {escape(quote_source or 'live source')}; trend decisions use completed weekly candles. The 2–6 week window is scenario analysis, not a price prediction or guarantee.</small></p></body></html>"""
    actions = "\n".join(f"{row['symbol']}: {row.get('weekly_action', row.get('action','NOT SCORED'))}" for row in portfolio_rows)
    text = f"WEEKLY CRYPTO OUTLOOK: {market['posture']} ({market['score']:.0f}/100)\n{market['horizon']}\n\nBASE PLAN\n{base}\n\nPORTFOLIO\n{actions or portfolio_note or 'Not configured'}\n\nSCENARIOS\nBull: {bull}\nBear: {bear}\n\nCompleted weekly-candle research; not financial advice."
    return html, text
