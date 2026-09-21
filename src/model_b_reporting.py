from __future__ import annotations

from html import escape

from .model_b import AdaptivePrediction, ModelQuality


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def render_model_b(predictions: list[AdaptivePrediction], quality: ModelQuality,
                   portfolio_rows: list[dict], portfolio_note: str | None,
                   generated_at: str, model_version: str) -> tuple[str, str]:
    by_symbol = {item.symbol: item for item in predictions}
    ranked = sorted(predictions, key=lambda item: (
        item.upside_probability + item.outperformance_probability - item.drawdown_probability
    ), reverse=True)
    entries = [item for item in ranked if item.action == "CONSIDER STAGED ENTRY"]
    defensive = [item for item in ranked if item.action == "PROTECT / DO NOT ADD"]
    headline = "DEFENSIVE CONDITIONS" if len(defensive) >= max(3, len(predictions) // 3) else (
        "SELECTIVE OPPORTUNITY" if entries else "NO HIGH-CONVICTION EDGE"
    )
    portfolio_body = []
    text_portfolio = []
    for row in portfolio_rows:
        item = by_symbol.get(row["symbol"])
        if item is None:
            continue
        portfolio_body.append(
            f"<tr><td><b>{escape(item.symbol)}</b></td><td>{row['allocation']:.1f}%</td>"
            f"<td>{_pct(item.upside_probability)}</td><td>{_pct(item.outperformance_probability)}</td>"
            f"<td>{_pct(item.drawdown_probability)}</td><td><b>{escape(item.action)}</b></td></tr>"
        )
        text_portfolio.append(
            f"{item.symbol}: {item.action} | upside {_pct(item.upside_probability)} | "
            f"beat BTC {_pct(item.outperformance_probability)} | drawdown {_pct(item.drawdown_probability)}"
        )
    opportunity_body = []
    text_opportunities = []
    for item in ranked[:12]:
        opportunity_body.append(
            f"<tr><td><b>{escape(item.symbol)}</b></td><td>${item.price:,.4f}</td>"
            f"<td>{_pct(item.upside_probability)}</td><td>{_pct(item.outperformance_probability)}</td>"
            f"<td>{_pct(item.drawdown_probability)}</td><td>{escape(item.confidence)}</td>"
            f"<td><b>{escape(item.action)}</b><br><small>{escape(' · '.join(item.drivers))}</small></td></tr>"
        )
        text_opportunities.append(
            f"{item.symbol}: {item.action}; upside {_pct(item.upside_probability)}, "
            f">BTC {_pct(item.outperformance_probability)}, drawdown {_pct(item.drawdown_probability)}"
        )
    metrics = " · ".join(
        f"{name.replace('_', ' ')} BA {value:.2f}" for name, value in quality.balanced_accuracy.items() if value is not None
    ) or "Validation unavailable"
    note = f"<p class='muted'>{escape(portfolio_note)}</p>" if portfolio_note else ""
    html = f"""<!doctype html><html><head><meta name='viewport' content='width=device-width'>
<style>body{{font-family:Arial,sans-serif;max-width:820px;margin:auto;padding:16px;color:#111827}}h1{{font-size:25px}}h2{{font-size:20px;margin-top:28px}}.hero{{background:#172554;color:white;padding:18px;border-radius:12px}}.tag{{font-size:13px;color:#bfdbfe;text-transform:uppercase}}.cards{{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}}.card{{background:#eff6ff;padding:10px;border-radius:8px;min-width:130px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:left;padding:8px 6px;border-bottom:1px solid #e5e7eb;vertical-align:top}}th{{background:#f3f4f6}}small,.muted{{color:#64748b}}.scroll{{overflow-x:auto}}.warning{{border-left:4px solid #d97706;padding:10px;background:#fffbeb}}</style></head>
<body><div class='hero'><div class='tag'>MoQuant Adaptive · Model B</div><h1>{escape(headline)}</h1>
<p>Independent probabilistic challenger. Model A remains unchanged.</p></div>
<div class='cards'><div class='card'><b>{len(entries)}</b><br>staged-entry candidates</div><div class='card'><b>{len(defensive)}</b><br>protect signals</div><div class='card'><b>{quality.samples:,}</b><br>training observations</div></div>
<div class='warning'><b>Probability, not certainty.</b> A 65% estimate still permits the opposite outcome. Position sizing and invalidation remain necessary.</div>
<h2>Your portfolio</h2>{note}<div class='scroll'><table><tr><th>Asset</th><th>Weight</th><th>≥10% upside</th><th>Beat BTC</th><th>≥15% drawdown</th><th>Model B action</th></tr>{''.join(portfolio_body) or '<tr><td colspan="6">No supported portfolio assets.</td></tr>'}</table></div>
<h2>Model B ranking</h2><div class='scroll'><table><tr><th>Asset</th><th>Close</th><th>≥10% upside</th><th>Beat BTC</th><th>≥15% drawdown</th><th>Confidence</th><th>Action / drivers</th></tr>{''.join(opportunity_body)}</table></div>
<h2>Validation snapshot</h2><p>{escape(metrics)}</p><p class='muted'>Balanced accuracy is measured on the latest time-ordered validation block. This is an engineering diagnostic, not evidence of future profitability.</p>
<p class='muted'>Generated {escape(generated_at)} · {escape(model_version)} · completed daily candles only · separate from MoQuant Sentinel</p></body></html>"""
    text = "\n".join([
        f"MOQUANT ADAPTIVE — MODEL B: {headline}",
        "Independent report; Model A remains unchanged.", "", "YOUR PORTFOLIO",
        *(text_portfolio or ["No supported portfolio assets."]), "", "MODEL B RANKING",
        *text_opportunities, "", f"Training observations: {quality.samples:,}", metrics,
        f"Generated {generated_at} · {model_version}",
    ])
    return html, text
