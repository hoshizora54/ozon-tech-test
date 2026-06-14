import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def _fit_item_elasticity(item_data: pd.DataFrame) -> dict[str, float]:
    features = pd.DataFrame(
        {
            "log_price": np.log(item_data["price"].clip(lower=1e-6)),
            "promo_active": item_data["promo_active"].astype(float),
            "log_competitor": np.log(item_data["competitor_price"].clip(lower=1e-6)),
            "log_ad_spend": np.log1p(item_data["ad_spend"].clip(lower=0)),
            "log_lag_1": np.log1p(item_data["lag_1_clean"].clip(lower=0)),
            "log_lag_7": np.log1p(item_data["lag_7_clean"].clip(lower=0)),
            "weather_index": item_data["weather_index"],
            "is_weekend": item_data["is_weekend"].astype(float),
        },
        index=item_data.index,
    )
    calendar = pd.get_dummies(
        item_data[["month", "day_of_week"]].astype(str),
        drop_first=True,
        dtype=float,
    )
    design = pd.concat([features, calendar], axis=1).fillna(0.0)
    target = np.log1p(item_data["sales"].to_numpy(dtype=float))

    scaler = StandardScaler()
    scaled_design = scaler.fit_transform(design)
    model = Ridge(alpha=10.0).fit(scaled_design, target)
    price_index = design.columns.get_loc("log_price")
    return {
        "raw_elasticity": model.coef_[price_index] / scaler.scale_[price_index],
        "price_log_std": design["log_price"].std(),
        "model_r2": model.score(scaled_design, target),
    }


def _regularize_elasticities(estimates: pd.DataFrame) -> pd.DataFrame:
    """Enforce demand monotonicity and shrink noisy item estimates."""
    result = estimates.copy()
    result["sign_constrained_elasticity"] = result["raw_elasticity"].clip(-5.0, -0.05)
    category_prior = (
        result.groupby("category", observed=True)["sign_constrained_elasticity"]
        .transform("median")
        .clip(-2.0, -0.30)
    )
    information = result["n_observations"] * result["price_log_std"].pow(2)
    result["shrinkage_weight"] = information / (information + 40.0)
    result["peer_prior"] = category_prior
    item_supported = (
        result["raw_elasticity"].lt(-0.05)
        & result["n_unique_prices"].ge(5)
        & result["model_r2"].ge(0.30)
    )
    result["elasticity_source"] = np.where(
        item_supported, "item_supported", "category_prior_driven"
    )
    result["elasticity"] = (
        result["shrinkage_weight"] * result["sign_constrained_elasticity"]
        + (1 - result["shrinkage_weight"]) * result["peer_prior"]
    ).clip(-5.0, -0.05)
    return result


def estimate_elasticities(model_frame: pd.DataFrame) -> pd.DataFrame:
    """Estimate regularized price elasticity separately for every item."""
    clean = model_frame[
        ~model_frame["is_candidate_anomaly"]
        & model_frame["lag_1_clean"].notna()
        & model_frame["roll_mean_30_clean"].notna()
    ].copy()
    median_prices = clean.groupby("item_id", observed=True)["price"].median()
    segments = pd.qcut(
        median_prices.rank(method="first"),
        q=5,
        labels=["Budget", "Economy", "Standard", "Premium", "Luxury"],
    )

    rows = []
    for item_id, item_data in clean.groupby("item_id", observed=True):
        rows.append(
            {
                "item_id": item_id,
                "category": item_data["category"].iloc[0],
                "price_segment": str(segments.loc[item_id]),
                "n_observations": len(item_data),
                "n_unique_prices": item_data["price"].nunique(),
                **_fit_item_elasticity(item_data),
            }
        )

    result = _regularize_elasticities(pd.DataFrame(rows))
    return result.sort_values("item_id").reset_index(drop=True)
