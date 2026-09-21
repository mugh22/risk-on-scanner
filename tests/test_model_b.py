import numpy as np
import pandas as pd
import pytest
import json

from src.model_b import AdaptivePrediction, FEATURES, ModelQuality, current_market_features, feature_frame, labeled_examples, predict, train_models
from src.model_b_reporting import render_model_b
from src.model_b_scanner import historical_daily


def _frame(periods=180, start=100, step=.5):
    close = pd.Series(start + step * np.arange(periods), dtype=float)
    return pd.DataFrame({
        "time": pd.date_range("2025-01-01", periods=periods, freq="D", tz="UTC"),
        "open": close - 1, "high": close + 2, "low": close - 2,
        "close": close, "volume": 1000 + np.arange(periods) * 2,
    })


def test_feature_frame_is_point_in_time_and_complete():
    features = feature_frame(_frame(), _frame(start=200, step=.4))
    derived_features = {name for name in FEATURES if name.startswith(("market_", "prior_"))}
    assert (set(FEATURES) - derived_features).issubset(features.columns)
    assert len(features) == 180
    assert features.iloc[-1].relative_30d > 0
    assert features.iloc[-1].low < features.iloc[-1].close


def test_labels_remove_unknown_future_rows():
    cfg = {"horizon_days": 14, "upside_threshold_pct": 10,
           "btc_outperformance_threshold_pct": 5, "drawdown_threshold_pct": -15}
    examples = labeled_examples("ALT", _frame(), _frame(start=200, step=.4), cfg)
    assert examples.tail(14)["upside"].isna().all()
    assert examples.iloc[-15]["upside"] in {0, 1}


def test_upside_label_uses_intrawindow_high_not_only_final_close():
    asset, btc = _frame(), _frame(start=200, step=.4)
    asset.loc[101, "high"] = asset.loc[100, "close"] * 1.20
    asset.loc[101:, "close"] = asset.loc[100, "close"]
    cfg = {"horizon_days": 14, "upside_threshold_pct": 10,
           "btc_outperformance_threshold_pct": 5, "drawdown_threshold_pct": -15}
    examples = labeled_examples("ALT", asset, btc, cfg)
    assert examples.loc[100, "upside"] == 1


def test_training_and_prediction_are_deterministic():
    rng = np.random.default_rng(42)
    rows = 500
    dataset = pd.DataFrame(rng.normal(size=(rows, len(FEATURES))), columns=FEATURES)
    dataset["time"] = pd.date_range("2024-01-01", periods=rows, freq="D")
    dataset["upside"] = (dataset["relative_30d"] + dataset["return_7d"] > 0).astype(int)
    dataset["outperform_btc"] = (dataset["relative_7d"] > 0).astype(int)
    dataset["drawdown"] = (dataset["volatility_20d"] > 0).astype(int)
    for name in FEATURES:
        if name not in dataset:
            dataset[name] = .5
    first, quality = train_models(dataset, .2)
    second, _ = train_models(dataset, .2)
    frame, btc = _frame(), _frame(start=200, step=.4)
    market = current_market_features({"ALT": frame}, btc)
    outcomes = {name: .5 for name in FEATURES if name.startswith(("prior_", "market_prior_"))}
    a = predict("ALT", frame, btc, first, {"protect_drawdown_probability": .45,
                "minimum_upside_probability": .6, "minimum_outperformance_probability": .55,
                "maximum_drawdown_probability": .3}, market, outcome_context=outcomes)
    b = predict("ALT", frame, btc, second, {"protect_drawdown_probability": .45,
                "minimum_upside_probability": .6, "minimum_outperformance_probability": .55,
                "maximum_drawdown_probability": .3}, market, outcome_context=outcomes)
    assert a.symbol == b.symbol and a.action == b.action and a.confidence == b.confidence
    assert a.upside_probability == pytest.approx(b.upside_probability)
    assert a.outperformance_probability == pytest.approx(b.outperformance_probability)
    assert a.drawdown_probability == pytest.approx(b.drawdown_probability)
    assert quality.samples == rows


def test_model_b_report_is_explicitly_separate_and_probability_labeled():
    item = AdaptivePrediction("ALT", 123.45, .66, .58, .21, "CONSIDER STAGED ENTRY", "MEDIUM",
                              ("30D leadership vs BTC",))
    quality = ModelQuality(1000, 200, {"upside": .61, "outperform_btc": .59, "drawdown": .64},
                           {"upside": .2, "outperform_btc": .21, "drawdown": .18},
                           {"upside": .60, "outperform_btc": .58, "drawdown": .62},
                           {"upside": "extra_trees", "outperform_btc": "logistic", "drawdown": "random_forest"},
                           {"upside": .2, "outperform_btc": .25, "drawdown": .18})
    html, text = render_model_b([item], quality, [{"symbol": "ALT", "allocation": 100}], None,
                                "2026-09-21 00:00 UTC", "test-v1")
    assert "Model A remains unchanged" in html
    assert "≥10% upside" in html and "≥15% drawdown" in html
    assert "MOQUANT ADAPTIVE — MODEL B" in text


def test_unvalidated_model_cannot_emit_actionable_language():
    rng = np.random.default_rng(7)
    rows = 300
    dataset = pd.DataFrame(rng.normal(size=(rows, len(FEATURES))), columns=FEATURES)
    dataset["time"] = pd.date_range("2024-01-01", periods=rows, freq="D")
    for target in ("upside", "outperform_btc", "drawdown"):
        dataset[target] = np.arange(rows) % 2
    models, _ = train_models(dataset, .2)
    weak = ModelQuality(rows, 60, {name: .5 for name in ("upside", "outperform_btc", "drawdown")},
                        {name: .25 for name in ("upside", "outperform_btc", "drawdown")},
                        {name: .5 for name in ("upside", "outperform_btc", "drawdown")},
                        {name: "logistic" for name in ("upside", "outperform_btc", "drawdown")},
                        {name: .5 for name in ("upside", "outperform_btc", "drawdown")})
    frame, btc = _frame(), _frame(start=200, step=.4)
    outcomes = {name: .5 for name in FEATURES if name.startswith(("prior_", "market_prior_"))}
    result = predict("ALT", frame, btc, models, {"protect_drawdown_probability": .45,
                     "minimum_upside_probability": .6, "minimum_outperformance_probability": .55,
                     "maximum_drawdown_probability": .3}, current_market_features({"ALT": frame}, btc), weak,
                     outcome_context=outcomes)
    assert result.action == "NO VALIDATED EDGE — IGNORE"


def test_model_b_history_paginates_without_changing_model_a_client():
    class Response:
        def __init__(self, params): self.params = params
        def raise_for_status(self): return None
        def json(self):
            limit = self.params["limit"]
            finish = pd.Timestamp(self.params.get("endTime", pd.Timestamp("2026-01-01", tz="UTC").timestamp()*1000), unit="ms", tz="UTC")
            rows = []
            for index, stamp in enumerate(pd.date_range(end=finish.floor("D"), periods=limit, freq="D", tz="UTC")):
                value = float(100 + index)
                rows.append([int(stamp.timestamp()*1000), value, value+2, value-2, value, 1000,
                             0, 0, 0, 0, 0, 0])
            return rows
    class Session:
        def __init__(self): self.calls = 0
        def get(self, url, params): self.calls += 1; return Response(params)
    class FakeClient:
        def __init__(self): self.client = Session()
    client = FakeClient()
    result = historical_daily(client, "BTCUSDT", 2200)
    assert len(result) == 2200
    assert client.client.calls == 3
    assert result.time.is_monotonic_increasing


def test_quality_is_serializable_for_reproducible_experiment_artifacts():
    quality = ModelQuality(100, 20, {name: .6 for name in ("upside", "outperform_btc", "drawdown")},
                           {name: .2 for name in ("upside", "outperform_btc", "drawdown")},
                           {name: .58 for name in ("upside", "outperform_btc", "drawdown")},
                           {name: "logistic" for name in ("upside", "outperform_btc", "drawdown")},
                           {name: .4 for name in ("upside", "outperform_btc", "drawdown")})
    from dataclasses import asdict
    payload = json.loads(json.dumps(asdict(quality)))
    assert payload["samples"] == 100
    assert payload["balanced_accuracy"]["upside"] == .6
