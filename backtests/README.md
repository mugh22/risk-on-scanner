# Historical decision replay

This branch-only harness replays selected critical dates using only candles that were complete at the simulated timestamp. Future candles are loaded separately and used only to grade the frozen call over 7, 30, and 90 days.

Run:

```bash
python -m src.backtest --output backtests/output
```

The output is intentionally described as a selected-event case study rather than a statistically representative backtest. It excludes historical market-cap dominance because the production dominance endpoint supplies only a current snapshot.
