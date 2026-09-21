import numpy as np
import pandas as pd

from src.model_b import AdaptivePrediction, FEATURES, ModelQuality, feature_frame, labeled_examples, predict, train_models
from src.model_b_reporting import render_model_b


def _frame(periods=180, start=100, step=.5):
    close = pd.Series(start + step * np.arange(periods), dtype=float)
    return pd.DataFrame({
        "time": pd.date_range("2025-01-01", periods=periods, freq="D", tz="UTC"),
        "open": close - 1, "high": close + 2, "low": close - 2,
        "close": close, "volume": 1000 + np.arange(periods) * 2,
    })


def test_feature_frame_is_point_in_time_and_complete():
    features = feature_frame(_frame(), _frame(start=200, step=.4))
    assert set(FEATURES).issubset(features.columns)
    assert len(features) == 180
    assert features.iloc[-1].relative_30d > 0
    assert features.iloc[-1].low < features.iloc[-1].close


def test_labels_remove_unknown_future_rows():
    cfg = {"horizon_days": 14, "upside_threshold_pct": 10,
           "btc_outperformance_threshold_pct": 5, "drawdown_threshold_pct": -15}
    examples = labeled_examples("ALT", _frame(), _frame(start=200, step=.4), cfg)
    assert examples.tail(14)["upside"].isna().all()
    assert examples.iloc[-15]["upside"] in {0, 1}


def test_training_and_prediction_are_deterministic():
    rng = np.random.default_rng(42)
    rows = 500
    dataset = pd.DataFrame(rng.normal(size=(rows, len(FEATURES))), columns=FEATURES)
    dataset["time"] = pd.date_range("2024-01-01", periods=rows, freq="D")
    dataset["upside"] = (dataset["relative_30d"] + dataset["return_7d"] > 0).astype(int)
    dataset["outperform_btc"] = (dataset["relative_7d"] > 0).astype(int)
    dataset["drawdown"] = (dataset["volatility_20d"] > 0).astype(int)
    first, quality = train_models(dataset, .2)
    second, _ = train_models(dataset, .2)
    frame, btc = _frame(), _frame(start=200, step=.4)
    a = predict("ALT", frame, btc, first, {"protect_drawdown_probability": .45,
                "minimum_upside_probability": .6, "minimum_outperformance_probability": .55,
                "maximum_drawdown_probability": .3})
    b = predict("ALT", frame, btc, second, {"protect_drawdown_probability": .45,
                "minimum_upside_probability": .6, "minimum_outperformance_probability": .55,
                "maximum_drawdown_probability": .3})
    assert a == b
    assert quality.samples == rows


def test_model_b_report_is_explicitly_separate_and_probability_labeled():
    item = AdaptivePrediction("ALT", 123.45, .66, .58, .21, "CONSIDER STAGED ENTRY", "MEDIUM",
                              ("30D leadership vs BTC",))
    quality = ModelQuality(1000, 200, {"upside": .61, "outperform_btc": .59, "drawdown": .64},
                           {"upside": .2, "outperform_btc": .21, "drawdown": .18})
    html, text = render_model_b([item], quality, [{"symbol": "ALT", "allocation": 100}], None,
                                "2026-09-21 00:00 UTC", "test-v1")
    assert "Model A remains unchanged" in html
    assert "≥10% upside" in html and "≥15% drawdown" in html
    assert "MOQUANT ADAPTIVE — MODEL B" in text
