# Champion vs candidate validation

Production logic is the champion. A candidate is promoted only when it improves
the intended metric without materially worsening false entries, drawdown, or
recent-market behavior.

## Candidate accepted in this round

- Persist a confirmed `REDUCE / EXIT` state through a short relief bounce.
- Block contradictory new-capital actions while that defensive state is active.
- Give the weekly report separate **holder** and **new-capital** actions.
- Add weekly entry decisions only for non-extended advancing/constructive assets.

## Candidate rejected in this round

The first daily momentum-probe rule reduced selected-case entry accuracy. It was
removed and is not part of the promoted candidate.

## Results

| Test set | Baseline | Candidate | Decision |
|---|---|---|---|
| 7 selected turning points | 47.4% entry hit rate; 0 bad adds | 47.4%; 0 bad adds | No regression |
| 61 daily closes in Apr-2024 and Jan-2025 | 51 good exits; 137 missed exits; 20 bad entries | 73 good exits; 115 missed exits; 19 bad entries | Improved |
| Latest 14 daily closes | 84.9% entry hit rate; 5 bad entries | 85.2%; 5 bad entries | Slight improvement |
| 78 completed weeks | 166 missed-upside observations; 84 good exits; 66 missed exits | 158 missed upside; 15 good vs 3 bad explicit entries; 97 good exits; 58 missed exits | Improved |

Daily and weekly rows overlap and must not be interpreted as independent trades
or guaranteed forward performance. They are regression and decision-quality
tests. Future changes must be evaluated against these stored champion outputs.
