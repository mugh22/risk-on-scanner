from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .indicators import ema, rsi


FEATURES = [
    "return_3d", "return_7d", "return_14d", "return_30d",
    "relative_7d", "relative_30d", "distance_ema20", "distance_ema50",
    "rsi14", "volatility_20d", "drawdown_60d", "volume_ratio_20d",
    "btc_return_7d", "btc_return_30d", "btc_distance_ema20",
]
TARGETS = ("upside", "outperform_btc", "drawdown")


@dataclass(frozen=True)
class ModelQuality:
    samples: int
    validation_samples: int
    balanced_accuracy: dict[str, float | None]
    brier_score: dict[str, float | None]


@dataclass(frozen=True)
class AdaptivePrediction:
    symbol: str
    price: float
    upside_probability: float
    outperformance_probability: float
    drawdown_probability: float
    action: str
    confidence: str
    drivers: tuple[str, ...]


def _pct(series: pd.Series, periods: int) -> pd.Series:
    return series.pct_change(periods) * 100


def feature_frame(asset: pd.DataFrame, btc: pd.DataFrame) -> pd.DataFrame:
    """Build point-in-time features using only information known at each close."""
    merged = asset.merge(
        btc[["time", "close"]].rename(columns={"close": "btc_close"}),
        on="time", how="inner",
    ).copy()
    close = merged["close"].astype(float)
    btc_close = merged["btc_close"].astype(float)
    result = pd.DataFrame({"time": merged["time"], "close": close, "low": merged["low"].astype(float)})
    for days in (3, 7, 14, 30):
        result[f"return_{days}d"] = _pct(close, days)
    result["relative_7d"] = result["return_7d"] - _pct(btc_close, 7)
    result["relative_30d"] = result["return_30d"] - _pct(btc_close, 30)
    result["distance_ema20"] = (close / ema(close, 20) - 1) * 100
    result["distance_ema50"] = (close / ema(close, 50) - 1) * 100
    result["rsi14"] = rsi(close)
    result["volatility_20d"] = close.pct_change().rolling(20).std() * np.sqrt(365) * 100
    result["drawdown_60d"] = (close / close.rolling(60).max() - 1) * 100
    result["volume_ratio_20d"] = merged["volume"] / merged["volume"].rolling(20).mean()
    result["btc_return_7d"] = _pct(btc_close, 7)
    result["btc_return_30d"] = _pct(btc_close, 30)
    result["btc_distance_ema20"] = (btc_close / ema(btc_close, 20) - 1) * 100
    result["btc_close"] = btc_close
    return result


def labeled_examples(symbol: str, asset: pd.DataFrame, btc: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    horizon = int(cfg["horizon_days"])
    features = feature_frame(asset, btc)
    future_close = features["close"].shift(-horizon)
    future_btc = features["btc_close"].shift(-horizon)
    forward_return = (future_close / features["close"] - 1) * 100
    btc_forward_return = (future_btc / features["btc_close"] - 1) * 100
    # Minimum low during the next horizon, excluding the signal close itself.
    future_low = pd.concat(
        [features["low"].shift(-offset) for offset in range(1, horizon + 1)], axis=1
    ).min(axis=1)
    forward_drawdown = (future_low / features["close"] - 1) * 100
    features["symbol"] = symbol
    features["upside"] = (forward_return >= float(cfg["upside_threshold_pct"])).astype(int)
    features["outperform_btc"] = (
        forward_return - btc_forward_return >= float(cfg["btc_outperformance_threshold_pct"])
    ).astype(int)
    features["drawdown"] = (forward_drawdown <= float(cfg["drawdown_threshold_pct"])).astype(int)
    # Rows without a fully known future horizon must never enter training.
    features.loc[future_close.isna() | future_btc.isna(), list(TARGETS)] = np.nan
    return features


def training_dataset(frames: dict[str, pd.DataFrame], btc: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    examples = [labeled_examples(symbol, frame, btc, cfg) for symbol, frame in frames.items()]
    if not examples:
        raise ValueError("No asset history is available for Model B")
    dataset = pd.concat(examples, ignore_index=True)
    minimum = int(cfg.get("minimum_history_days", 90))
    dataset = dataset.loc[dataset.groupby("symbol").cumcount() >= minimum]
    return dataset.dropna(subset=[*FEATURES, *TARGETS]).sort_values("time").reset_index(drop=True)


def _pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def train_models(dataset: pd.DataFrame, validation_fraction: float = .2) -> tuple[dict[str, Pipeline], ModelQuality]:
    if len(dataset) < 200:
        raise ValueError(f"Model B needs at least 200 labeled observations; received {len(dataset)}")
    # Split by timestamp, not row number: no asset observed on a validation date
    # may leak into training through another row from the same date.
    dates = pd.Series(dataset["time"].drop_duplicates().sort_values().to_list())
    split = max(1, min(len(dates) - 1, int(len(dates) * (1 - validation_fraction))))
    cutoff = dates.iloc[split]
    train = dataset.loc[dataset["time"] < cutoff]
    validation = dataset.loc[dataset["time"] >= cutoff]
    models: dict[str, Pipeline] = {}
    accuracies: dict[str, float | None] = {}
    briers: dict[str, float | None] = {}
    for target in TARGETS:
        if train[target].nunique() < 2:
            raise ValueError(f"Training history contains only one {target} class")
        candidate = _pipeline().fit(train[FEATURES], train[target].astype(int))
        if validation[target].nunique() >= 2:
            probability = candidate.predict_proba(validation[FEATURES])[:, 1]
            accuracies[target] = round(float(balanced_accuracy_score(validation[target], probability >= .5)), 3)
            briers[target] = round(float(brier_score_loss(validation[target], probability)), 3)
        else:
            accuracies[target], briers[target] = None, None
        models[target] = _pipeline().fit(dataset[FEATURES], dataset[target].astype(int))
    quality = ModelQuality(len(dataset), len(validation), accuracies, briers)
    return models, quality


def _drivers(row: pd.Series) -> tuple[str, ...]:
    candidates: list[str] = []
    if row["relative_30d"] >= 5: candidates.append("30D leadership vs BTC")
    elif row["relative_30d"] <= -5: candidates.append("30D weakness vs BTC")
    if row["distance_ema20"] >= 10 or row["rsi14"] >= 74: candidates.append("extended above trend")
    elif row["distance_ema20"] > 0: candidates.append("above 20D trend")
    else: candidates.append("below 20D trend")
    if row["volume_ratio_20d"] >= 1.25: candidates.append("volume confirmation")
    if row["btc_distance_ema20"] < 0: candidates.append("BTC below 20D trend")
    return tuple(candidates[:3])


def predict(symbol: str, frame: pd.DataFrame, btc: pd.DataFrame, models: dict[str, Pipeline], decision: dict) -> AdaptivePrediction:
    row = feature_frame(frame, btc).dropna(subset=FEATURES).iloc[-1]
    sample = pd.DataFrame([{name: row[name] for name in FEATURES}])
    probabilities = {target: float(models[target].predict_proba(sample)[0, 1]) for target in TARGETS}
    if probabilities["drawdown"] >= float(decision["protect_drawdown_probability"]):
        action = "PROTECT / DO NOT ADD"
    elif (probabilities["upside"] >= float(decision["minimum_upside_probability"])
          and probabilities["outperform_btc"] >= float(decision["minimum_outperformance_probability"])
          and probabilities["drawdown"] <= float(decision["maximum_drawdown_probability"])):
        action = "CONSIDER STAGED ENTRY"
    elif probabilities["drawdown"] <= float(decision["maximum_drawdown_probability"]):
        action = "HOLD / WATCH"
    else:
        action = "WAIT — MIXED RISK"
    distance = min(abs(probabilities[target] - .5) for target in TARGETS)
    confidence = "HIGH" if distance >= .22 else "MEDIUM" if distance >= .10 else "LOW"
    return AdaptivePrediction(symbol, float(row["close"]), probabilities["upside"],
                              probabilities["outperform_btc"], probabilities["drawdown"],
                              action, confidence, _drivers(row))
