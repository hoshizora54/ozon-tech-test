import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor

from .config import FEATURE_GROUPS, MODEL_FEATURES


def _group_feature_contributions(contributions: pd.DataFrame) -> pd.DataFrame:
    grouped = pd.DataFrame(index=contributions.index)
    for group_name, columns in FEATURE_GROUPS.items():
        grouped[group_name] = contributions[columns].sum(axis=1)
    return grouped


def root_cause_report(
    model: XGBRegressor,
    model_frame: pd.DataFrame,
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """Explain expected demand and separate it from unexplained residuals."""
    selected = model_frame.loc[rows.index].copy()
    matrix = xgb.DMatrix(selected[MODEL_FEATURES], enable_categorical=True)
    shap_values = model.get_booster().predict(matrix, pred_contribs=True)

    contributions = pd.DataFrame(
        shap_values[:, :-1], columns=MODEL_FEATURES, index=selected.index
    )
    grouped = _group_feature_contributions(contributions)
    predicted_sales = np.clip(shap_values.sum(axis=1), 0, None)

    context_groups = [name for name in FEATURE_GROUPS if name != "identity"]
    top_driver = grouped[context_groups].abs().idxmax(axis=1)
    top_effect = [
        grouped.loc[index, driver]
        for index, driver in zip(grouped.index, top_driver)
    ]
    log_residual = np.log1p(selected["sales"].to_numpy()) - np.log1p(predicted_sales)
    diagnosis = np.select(
        [log_residual > 2.0, log_residual < -2.0],
        ["unexplained_positive_residual", "unexplained_negative_residual"],
        default="mostly_explained_by_context",
    )

    report = selected[
        [
            "date",
            "item_id",
            "category",
            "sales",
            "ranking_score",
            "anomaly_type",
        ]
    ].copy()
    report["expected_sales"] = predicted_sales
    report["actual_to_expected"] = (
        report["sales"] / np.maximum(report["expected_sales"], 1.0)
    )
    report["largest_context_driver"] = top_driver.to_numpy()
    report["driver_sales_effect"] = top_effect
    report["diagnosis"] = diagnosis
    return report.sort_values("ranking_score", ascending=False)
