from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .indicators import ema, rsi


FEATURES = [
    "return_1d", "return_3d", "return_7d", "return_14d", "return_30d", "return_60d",
    "relative_7d", "relative_14d", "relative_30d", "relative_60d",
    "distance_ema20", "distance_ema50", "ema20_slope_5d", "rsi14",
    "volatility_7d", "volatility_20d", "volatility_60d",
    "drawdown_30d", "drawdown_60d", "drawdown_90d",
    "volume_ratio_20d", "range_pct",
    "btc_return_7d", "btc_return_30d", "btc_return_60d",
    "btc_distance_ema20", "btc_ema20_slope_5d", "btc_volatility_20d", "btc_drawdown_60d",
    "market_breadth_ema20", "market_breadth_relative_7d", "market_median_relative_30d",
]
TARGETS = ("upside", "outperform_btc", "drawdown")


@dataclass(frozen=True)
class ModelQuality:
    samples: int
    validation_samples: int
    balanced_accuracy: dict[str, float | None]
    brier_score: dict[str, float | None]
    cross_validation_accuracy: dict[str, float | None]
    selected_model: dict[str, str]
    positive_rate: dict[str, float]


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
    result = pd.DataFrame({"time": merged["time"], "close": close,
                           "high": merged["high"].astype(float), "low": merged["low"].astype(float)})
    for days in (1, 3, 7, 14, 30, 60):
        result[f"return_{days}d"] = _pct(close, days)
    result["relative_7d"] = result["return_7d"] - _pct(btc_close, 7)
    result["relative_14d"] = result["return_14d"] - _pct(btc_close, 14)
    result["relative_30d"] = result["return_30d"] - _pct(btc_close, 30)
    result["relative_60d"] = result["return_60d"] - _pct(btc_close, 60)
    ema20 = ema(close, 20)
    btc_ema20 = ema(btc_close, 20)
    result["distance_ema20"] = (close / ema20 - 1) * 100
    result["distance_ema50"] = (close / ema(close, 50) - 1) * 100
    result["ema20_slope_5d"] = _pct(ema20, 5)
    result["rsi14"] = rsi(close)
    result["volatility_7d"] = close.pct_change().rolling(7).std() * np.sqrt(365) * 100
    result["volatility_20d"] = close.pct_change().rolling(20).std() * np.sqrt(365) * 100
    result["volatility_60d"] = close.pct_change().rolling(60).std() * np.sqrt(365) * 100
    for days in (30, 60, 90):
        result[f"drawdown_{days}d"] = (close / close.rolling(days).max() - 1) * 100
    result["volume_ratio_20d"] = merged["volume"] / merged["volume"].rolling(20).mean()
    result["range_pct"] = (merged["high"] - merged["low"]) / close * 100
    result["btc_return_7d"] = _pct(btc_close, 7)
    result["btc_return_30d"] = _pct(btc_close, 30)
    result["btc_return_60d"] = _pct(btc_close, 60)
    result["btc_distance_ema20"] = (btc_close / btc_ema20 - 1) * 100
    result["btc_ema20_slope_5d"] = _pct(btc_ema20, 5)
    result["btc_volatility_20d"] = btc_close.pct_change().rolling(20).std() * np.sqrt(365) * 100
    result["btc_drawdown_60d"] = (btc_close / btc_close.rolling(60).max() - 1) * 100
    result["btc_close"] = btc_close
    return result


def labeled_examples(symbol: str, asset: pd.DataFrame, btc: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    horizon = int(cfg["horizon_days"])
    features = feature_frame(asset, btc)
    future_close = features["close"].shift(-horizon)
    future_btc = features["btc_close"].shift(-horizon)
    future_high = pd.concat(
        [features["high"].shift(-offset) for offset in range(1, horizon + 1)], axis=1
    ).max(axis=1)
    forward_return = (future_close / features["close"] - 1) * 100
    maximum_upside = (future_high / features["close"] - 1) * 100
    btc_forward_return = (future_btc / features["btc_close"] - 1) * 100
    # Minimum low during the next horizon, excluding the signal close itself.
    future_low = pd.concat(
        [features["low"].shift(-offset) for offset in range(1, horizon + 1)], axis=1
    ).min(axis=1)
    forward_drawdown = (future_low / features["close"] - 1) * 100
    features["symbol"] = symbol
    features["upside"] = (maximum_upside >= float(cfg["upside_threshold_pct"])).astype(int)
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
    # Cross-sectional conditions describe whether strength is isolated or broad.
    market = dataset.groupby("time").agg(
        market_breadth_ema20=("distance_ema20", lambda values: float((values > 0).mean()) * 100),
        market_breadth_relative_7d=("relative_7d", lambda values: float((values > 0).mean()) * 100),
        market_median_relative_30d=("relative_30d", "median"),
    ).reset_index()
    dataset = dataset.merge(market, on="time", how="left")
    return dataset.dropna(subset=[*FEATURES, *TARGETS]).sort_values("time").reset_index(drop=True)


def _pipeline(model) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", model),
    ])


def _candidates() -> dict[str, Pipeline]:
    return {
        "logistic": _pipeline(LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        "random_forest": _pipeline(RandomForestClassifier(
            n_estimators=150, max_depth=8, min_samples_leaf=20,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1,
        )),
        "extra_trees": _pipeline(ExtraTreesClassifier(
            n_estimators=150, max_depth=10, min_samples_leaf=15,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )),
        "hist_gradient_boosting": _pipeline(HistGradientBoostingClassifier(
            max_iter=150, learning_rate=.05, max_leaf_nodes=15,
            min_samples_leaf=30, l2_regularization=1.0, class_weight="balanced", random_state=42,
        )),
    }


def _time_folds(dataset: pd.DataFrame, folds: int = 3) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    dates = pd.Series(dataset["time"].drop_duplicates().sort_values().to_list())
    boundaries = np.linspace(.4, 1.0, folds + 1)
    result = []
    for index in range(folds):
        train_end = dates.iloc[min(len(dates) - 1, int(len(dates) * boundaries[index]))]
        valid_end_index = min(len(dates), int(len(dates) * boundaries[index + 1]))
        valid_end = dates.iloc[valid_end_index - 1]
        train = dataset.loc[dataset["time"] < train_end]
        valid = dataset.loc[(dataset["time"] >= train_end) & (dataset["time"] <= valid_end)]
        if len(train) and len(valid):
            result.append((train, valid))
    return result


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
    development, holdout = train, validation
    models: dict[str, Pipeline] = {}
    accuracies: dict[str, float | None] = {}
    briers: dict[str, float | None] = {}
    cv_accuracies: dict[str, float | None] = {}
    selected: dict[str, str] = {}
    positive_rates = {target: round(float(dataset[target].mean()), 3) for target in TARGETS}
    for target in TARGETS:
        if development[target].nunique() < 2:
            raise ValueError(f"Training history contains only one {target} class")
        tournament: list[tuple[float, str]] = []
        for name in _candidates():
            fold_scores = []
            for fold_train, fold_valid in _time_folds(development):
                if fold_train[target].nunique() < 2 or fold_valid[target].nunique() < 2:
                    continue
                contender = _candidates()[name].fit(fold_train[FEATURES], fold_train[target].astype(int))
                fold_probability = contender.predict_proba(fold_valid[FEATURES])[:, 1]
                fold_scores.append(balanced_accuracy_score(fold_valid[target], fold_probability >= .5))
            tournament.append((float(np.mean(fold_scores)) if fold_scores else 0.0, name))
        cv_score, winner = max(tournament)
        selected[target] = winner
        cv_accuracies[target] = round(cv_score, 3)
        candidate = _candidates()[winner].fit(development[FEATURES], development[target].astype(int))
        if holdout[target].nunique() >= 2:
            probability = candidate.predict_proba(holdout[FEATURES])[:, 1]
            accuracies[target] = round(float(balanced_accuracy_score(holdout[target], probability >= .5)), 3)
            briers[target] = round(float(brier_score_loss(holdout[target], probability)), 3)
        else:
            accuracies[target], briers[target] = None, None
        models[target] = _candidates()[winner].fit(dataset[FEATURES], dataset[target].astype(int))
    quality = ModelQuality(len(dataset), len(holdout), accuracies, briers, cv_accuracies, selected, positive_rates)
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


def current_market_features(frames: dict[str, pd.DataFrame], btc: pd.DataFrame) -> dict[str, float]:
    rows = [feature_frame(frame, btc).iloc[-1] for frame in frames.values()]
    return {
        "market_breadth_ema20": float(np.mean([row["distance_ema20"] > 0 for row in rows])) * 100,
        "market_breadth_relative_7d": float(np.mean([row["relative_7d"] > 0 for row in rows])) * 100,
        "market_median_relative_30d": float(np.median([row["relative_30d"] for row in rows])),
    }


def predict(symbol: str, frame: pd.DataFrame, btc: pd.DataFrame, models: dict[str, Pipeline], decision: dict,
            market_features: dict[str, float] | None = None, quality: ModelQuality | None = None,
            minimum_validated_accuracy: float = .60) -> AdaptivePrediction:
    features = feature_frame(frame, btc)
    for name, value in (market_features or {}).items():
        features[name] = value
    row = features.dropna(subset=FEATURES).iloc[-1]
    sample = pd.DataFrame([{name: row[name] for name in FEATURES}])
    probabilities = {target: float(models[target].predict_proba(sample)[0, 1]) for target in TARGETS}
    upside_valid = quality is None or (quality.balanced_accuracy.get("upside") or 0) >= minimum_validated_accuracy
    relative_valid = quality is None or (quality.balanced_accuracy.get("outperform_btc") or 0) >= minimum_validated_accuracy
    drawdown_valid = quality is None or (quality.balanced_accuracy.get("drawdown") or 0) >= minimum_validated_accuracy
    if not any((upside_valid, relative_valid, drawdown_valid)):
        action = "NO VALIDATED EDGE — IGNORE"
    elif drawdown_valid and probabilities["drawdown"] >= float(decision["protect_drawdown_probability"]):
        action = "PROTECT / DO NOT ADD"
    elif (upside_valid and relative_valid
          and probabilities["upside"] >= float(decision["minimum_upside_probability"])
          and probabilities["outperform_btc"] >= float(decision["minimum_outperformance_probability"])
          and probabilities["drawdown"] <= float(decision["maximum_drawdown_probability"])):
        action = "CONSIDER STAGED ENTRY"
    elif drawdown_valid and probabilities["drawdown"] <= float(decision["maximum_drawdown_probability"]):
        action = "HOLD / WATCH"
    else:
        action = "UNVALIDATED — OBSERVE ONLY"
    distance = min(abs(probabilities[target] - .5) for target in TARGETS)
    confidence = "HIGH" if distance >= .22 else "MEDIUM" if distance >= .10 else "LOW"
    return AdaptivePrediction(symbol, float(row["close"]), probabilities["upside"],
                              probabilities["outperform_btc"], probabilities["drawdown"],
                              action, confidence, _drivers(row))
