# Crypto Risk-On Scanner

A selective, transparent Python 3.12 scanner that measures crypto risk appetite, ranks established altcoins by BTC-relative strength, and emails a mobile-readable report at noon and 8 PM U.S. Central time.

## What it does

- Downloads daily OHLCV candles from Binance's public API; no market-data key is required.
- Produces a 0–100 market Risk-On Score from BTC trend, ETH/BTC trend, breadth, and aggregate alt/BTC momentum.
- Produces a 0–100 Alt Strength Score from 7D/30D BTC-relative returns, trend, momentum, volume, and breakout confirmation.
- Classifies each asset as `BUY`, `WATCH`, or `NO SIGNAL`. A BUY requires a risk-on regime and at least six independent confirmations.
- Suppresses BUY signals when RSI, distance above EMA20, or the daily move indicates chasing.
- Calculates ATR/market-structure-aware entry, targets, invalidation, and reward/risk levels.
- Restores lightweight state with GitHub Actions cache and highlights regime/signal changes without automated commits.

The initial universe is configured in [`config.yaml`](config.yaml): ETH, SOL, XRP, BNB, ADA, DOGE, AVAX, LINK, NEAR, ARB, OP, SUI, APT, INJ, RENDER, TAO, AAVE, UNI, ONDO, and SEI. Unsupported Binance pairs are logged and skipped without aborting the report.

## Scores and regimes

Risk-On component weights: BTC trend 25%, ETH/BTC 25%, breadth 35%, aggregate alt/BTC momentum 15%. Regimes are: 0–34 Risk-Off, 35–54 Neutral, 55–69 Early Risk-On, 70–84 Risk-On, and 85–100 Strong Risk-On.

Alt Strength defaults: 30D vs BTC 30%, 7D vs BTC 15%, trend 25%, momentum 15%, volume 8%, breakout 7%. All weights and thresholds are centralized in `config.yaml`.

## Local setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
pytest -q
python -m src.scanner --no-email
```

The HTML and text reports are written to `reports/`. Remove `--no-email` after configuring `.env`.

## GitHub Actions and secrets

In **Settings → Secrets and variables → Actions**, create exactly these repository secrets:

| Secret | Value |
|---|---|
| `EMAIL_USER` | Gmail/SMTP sender address |
| `EMAIL_PASSWORD` | Gmail App Password, not the normal account password |
| `EMAIL_TO` | Destination email address |

For Gmail, enable 2-Step Verification and create an App Password. Secret values are never logged. The defaults use `smtp.gmail.com:465`; local runs can override `SMTP_HOST` and `SMTP_PORT`.

The workflow can be launched from **Actions → Crypto risk-on scanner → Run workflow**. Scheduled runs occur at 12:00 PM and 8:00 PM `America/Chicago`. GitHub schedules are UTC-only, so four UTC trigger hours plus an in-job timezone gate handle CST/CDT correctly. The two nonmatching UTC invocations exit without scanning.

## Customization

- Add/remove tickers under `symbols` in `config.yaml`.
- Adjust score weights under `weights`; values need not sum to 100 because calculations normalize them.
- Adjust selectivity and overextension thresholds under `signals`.
- Run `python -m src.scanner --help` for path overrides.

Example headline: `RISK-ON SCORE: 74/100 — RISK-ON`. Ranked rows include price, signal, relative returns, RSI, volume ratio, entry zone, targets, and invalidation. Live values are always calculated and never hardcoded.

## Limitations

Daily candles can miss intraday changes. Exchange availability and symbol mapping vary. The first run has no comparison history. GitHub cache eviction may also reset history. BTC dominance is intentionally omitted until a reliable credential-free historical feed is configured; no missing indicator is fabricated. Signals are systematic technical research, not financial advice or guaranteed outcomes.

