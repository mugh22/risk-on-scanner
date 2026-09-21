from __future__ import annotations


def persistent_exit_call(current_score: float, prior_scores: list[float], raw_call: str) -> tuple[str, bool]:
    """Keep confirmed alt-risk reduction active through a short relief bounce."""
    recent = prior_scores[-3:]
    if current_score >= 75:
        return "EXIT MOST ALT RISK", False
    if current_score >= 60:
        return "REDUCE ALT RISK", False
    if recent and max(recent) >= 75 and current_score >= 35:
        return "REDUCE ALT RISK — DEFENSIVE STATE", True
    if recent and max(recent) >= 60 and current_score >= 45:
        return "REDUCE ALT RISK — DEFENSIVE STATE", True
    return raw_call, False
