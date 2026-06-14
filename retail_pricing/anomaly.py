from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _add_past_only_statistics(
    data: pd.DataFrame,
    window: int,
    min_periods: int,
) -> pd.DataFrame:
    out = data.sort_values(["item_id", "date"]).reset_index(drop=True).copy()
    sales_by_item = out.groupby("item_id", sort=False)["sales"]

    out["local_mean_28"] = sales_by_item.transform(
        lambda values: values.shift(1).rolling(window, min_periods=min_periods).mean()
    )
    out["local_median_28"] = sales_by_item.transform(
        lambda values: values.shift(1).rolling(window, min_periods=min_periods).median()
    )
    out["local_std_28"] = sales_by_item.transform(
        lambda values: values.shift(1).rolling(window, min_periods=min_periods).std()
    )
    out["local_nonzero_rate_28"] = sales_by_item.transform(
        lambda values: values.shift(1).rolling(window, min_periods=min_periods).apply(
            lambda history: np.mean(history > 0), raw=True
        )
    )
    return out


def _expanding_thresholds(
    scored: pd.DataFrame,
    quantile: float,
    minimum_scores: int,
) -> dict[pd.Timestamp, float]:
    """Calibrate each date using scores observed strictly before that date."""
    history: list[float] = []
    thresholds: dict[pd.Timestamp, float] = {}

    for date, daily_rows in scored.sort_values("date").groupby("date", sort=True):
        finite_history = np.asarray(history, dtype=float)
        finite_history = finite_history[np.isfinite(finite_history)]
        thresholds[date] = (
            float(np.quantile(finite_history, quantile))
            if len(finite_history) >= minimum_scores
            else np.inf
        )
        history.extend(daily_rows["spike_score"].dropna().tolist())
    return thresholds


def score_anomalies(
    data: pd.DataFrame,
    window: int = 28,
    min_periods: int = 14,
    spike_quantile: float = 0.999,
    minimum_calibration_scores: int = 2_000,
) -> pd.DataFrame:
    """Detect sales spikes and unexpected zero-sales observations."""
    out = _add_past_only_statistics(data, window, min_periods)
    mean = out["local_mean_28"].clip(lower=0.1)
    scale = out["local_std_28"].fillna(0.0) + np.sqrt(mean + 1.0)
    residual = out["sales"] - out["local_mean_28"]

    out["spike_score"] = residual.clip(lower=0.0) / scale
    out["drop_score"] = (-residual).clip(lower=0.0) / scale
    out["anomaly_score"] = np.maximum(out["spike_score"], out["drop_score"])

    thresholds = _expanding_thresholds(
        out,
        quantile=spike_quantile,
        minimum_scores=minimum_calibration_scores,
    )
    out["spike_threshold"] = out["date"].map(thresholds)
    out["spike_flag"] = out["spike_score"] >= out["spike_threshold"]
    out["high_confidence_spike_flag"] = out["spike_flag"] & (
        out["promo_active"].eq(0)
        | (out["spike_score"] >= 10 * out["spike_threshold"])
    )
    out["zero_drop_flag"] = (
        out["sales"].eq(0)
        & out["local_mean_28"].ge(15)
        & out["local_nonzero_rate_28"].ge(0.90)
    )
    out["is_high_confidence_anomaly"] = out["high_confidence_spike_flag"]
    out["is_candidate_anomaly"] = (
        out["is_high_confidence_anomaly"] | out["zero_drop_flag"]
    )
    out["anomaly_type"] = np.select(
        [
            out["high_confidence_spike_flag"],
            out["zero_drop_flag"],
            out["spike_flag"] & out["promo_active"].eq(1),
        ],
        ["technical_spike", "unexpected_zero_review", "explained_promo_spike"],
        default="normal",
    )

    finite_threshold = out["spike_threshold"].replace(np.inf, np.nan)
    zero_review_level = finite_threshold.fillna(finite_threshold.median())
    promo_adjusted_spike = out["spike_score"] / (1 + 9 * out["promo_active"])
    out["ranking_score"] = np.maximum(
        promo_adjusted_spike.fillna(-1),
        np.where(
            out["zero_drop_flag"],
            zero_review_level + out["drop_score"],
            -1,
        ),
    )
    out.attrs["latest_spike_threshold"] = thresholds[max(thresholds)]
    return out


def detector_metrics(
    scored: pd.DataFrame,
    top_ks: Iterable[int] = (20, 50, 100, 200),
) -> pd.DataFrame:
    """Evaluate the detector when the optional hidden label is available."""
    if "is_anomaly_label" not in scored:
        return pd.DataFrame()

    y_true = scored["is_anomaly_label"].to_numpy(dtype=int)
    scores = scored["ranking_score"].fillna(-1).to_numpy(dtype=float)

    rows = [
        {"metric": "average_precision", "value": average_precision_score(y_true, scores)},
        {"metric": "roc_auc", "value": roc_auc_score(y_true, scores)},
    ]
    for tier, column in (
        ("high_confidence", "is_high_confidence_anomaly"),
        ("review", "is_candidate_anomaly"),
    ):
        flags = scored[column].to_numpy(dtype=int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, flags, average="binary", zero_division=0
        )
        rows.extend(
            [
                {"metric": f"{tier}_precision", "value": precision},
                {"metric": f"{tier}_recall", "value": recall},
                {"metric": f"{tier}_f1", "value": f1},
                {"metric": f"{tier}_rows", "value": int(flags.sum())},
            ]
        )
    order = np.argsort(-scores)
    positives = max(int(y_true.sum()), 1)
    for k in top_ks:
        k = min(k, len(scored))
        hits = int(y_true[order[:k]].sum())
        rows.extend(
            [
                {"metric": f"precision_at_{k}", "value": hits / k},
                {"metric": f"recall_at_{k}", "value": hits / positives},
            ]
        )
    return pd.DataFrame(rows)
