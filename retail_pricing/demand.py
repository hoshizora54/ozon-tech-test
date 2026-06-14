from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from .config import MODEL_FEATURES, SEED


@dataclass
class DemandModelResult:
    model: XGBRegressor
    metrics: pd.DataFrame
    predictions: pd.DataFrame


def create_demand_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=750,
        learning_rate=0.035,
        max_depth=5,
        min_child_weight=8,
        subsample=0.85,
        colsample_bytree=0.90,
        reg_alpha=0.05,
        reg_lambda=5.0,
        objective="reg:squarederror",
        eval_metric="mae",
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=4,
    )


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    absolute_error = np.abs(y_true - y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "WAPE": absolute_error.sum() / max(np.abs(y_true).sum(), 1.0),
        "SMAPE": np.mean(
            2 * absolute_error / np.maximum(np.abs(y_true) + np.abs(y_pred), 1.0)
        ),
    }


def _append_metrics(rows, cutoff, model_name, scope, actual, prediction) -> None:
    row = {"fold_start": cutoff.date(), "model": model_name, "scope": scope}
    row.update(regression_metrics(actual, prediction))
    rows.append(row)


def walk_forward_validation(
    model_frame: pd.DataFrame,
    cutoffs: Iterable[str | pd.Timestamp] = ("2023-04-01", "2023-07-01", "2023-10-01"),
    validation_days: int = 90,
) -> DemandModelResult:
    """Evaluate the one-day demand model on expanding time folds."""
    metric_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    last_model = None
    ready = model_frame.dropna(subset=MODEL_FEATURES).copy()

    for cutoff_value in cutoffs:
        cutoff = pd.Timestamp(cutoff_value)
        end = cutoff + pd.Timedelta(days=validation_days - 1)
        train = ready[(ready["date"] < cutoff) & ~ready["is_candidate_anomaly"]]
        valid = ready[ready["date"].between(cutoff, end)]
        if train.empty or valid.empty:
            continue

        model = create_demand_model()
        model.fit(train[MODEL_FEATURES], np.log1p(train["sales"]))
        prediction = np.clip(np.expm1(model.predict(valid[MODEL_FEATURES])), 0, None)
        baseline = valid["roll_mean_30_clean"].clip(lower=0).to_numpy()

        for model_name, values in (
            ("rolling_mean_30", baseline),
            ("xgboost_global", prediction),
        ):
            _append_metrics(
                metric_rows, cutoff, model_name, "all_rows", valid["sales"], values
            )
            clean = ~valid["is_candidate_anomaly"].to_numpy()
            _append_metrics(
                metric_rows,
                cutoff,
                model_name,
                "detector_clean_rows",
                valid.loc[clean, "sales"],
                values[clean],
            )

        fold_predictions = valid[
            ["date", "item_id", "sales", "is_candidate_anomaly"]
        ].copy()
        fold_predictions["prediction"] = prediction
        fold_predictions["fold_start"] = cutoff
        prediction_rows.append(fold_predictions)
        last_model = model

    if last_model is None:
        raise ValueError("No validation fold could be constructed")
    return DemandModelResult(
        model=last_model,
        metrics=pd.DataFrame(metric_rows),
        predictions=pd.concat(prediction_rows, ignore_index=True),
    )


def fit_final_demand_model(model_frame: pd.DataFrame) -> XGBRegressor:
    ready = model_frame.dropna(subset=MODEL_FEATURES)
    train = ready[~ready["is_candidate_anomaly"]]
    model = create_demand_model()
    model.fit(train[MODEL_FEATURES], np.log1p(train["sales"]))
    return model
