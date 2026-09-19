from __future__ import annotations

import json
from pathlib import Path


def load(path: str) -> dict:
    file = Path(path)
    if not file.exists(): return {}
    try: return json.loads(file.read_text())
    except (json.JSONDecodeError, OSError): return {}


def save(path: str, data: dict) -> None:
    Path(path).write_text(json.dumps(data, indent=2))


def changes(previous: dict, score: float, signals: dict[str, str]) -> list[str]:
    notes = []
    if "risk_score" in previous: notes.append(f"Risk-On Score {previous['risk_score']:.0f} → {score:.0f} ({score-previous['risk_score']:+.0f})")
    old = previous.get("signals", {})
    for symbol, signal in signals.items():
        if signal == "BUY" and old.get(symbol) != "BUY": notes.append(f"NEW BUY SIGNAL: {symbol}")
        elif old.get(symbol) and old[symbol] != signal: notes.append(f"{symbol}: {old[symbol]} → {signal}")
    return notes

