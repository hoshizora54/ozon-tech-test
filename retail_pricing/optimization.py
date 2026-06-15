import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .config import CATEGORICAL_FEATURES, MODEL_FEATURES


def make_next_day_context(model_frame: pd.DataFrame) -> pd.DataFrame:
    """Create a conservative no-promo scenario for the next day."""
    next_date = model_frame["date"].max() + pd.Timedelta(days=1)
    records = []

    for _, item_data in model_frame.groupby("item_id", observed=True, sort=True):
        item_data = item_data.sort_values("date")
        latest = item_data.iloc[-1].copy()
        regular = item_data[item_data["promo_active"].eq(0)]
        if regular.empty:
            regular = item_data

        latest["date"] = next_date
        latest["price"] = regular.iloc[-1]["price"]
        latest["promo_active"] = 0
        latest["competitor_price"] = item_data.tail(7)["competitor_price"].median()
        latest["ad_spend"] = item_data.tail(30)["ad_spend"].median()
        latest["weather_index"] = item_data.tail(30)["weather_index"].median()
        latest["stock_level"] = item_data.iloc[-1]["stock_level"]
        for lag in (1, 2, 3, 7, 14, 21, 28):
            latest[f"lag_{lag}_clean"] = item_data.iloc[-lag]["sales_clean"]
        latest["roll_mean_7_clean"] = item_data.tail(7)["sales_clean"].mean()
        latest["roll_mean_30_clean"] = item_data.tail(30)["sales_clean"].mean()
        latest["roll_std_30_clean"] = item_data.tail(30)["sales_clean"].std()
        latest["roll_median_7_clean"] = item_data.tail(7)["sales_clean"].median()
        latest["roll_median_30_clean"] = item_data.tail(30)["sales_clean"].median()
        for window in (7, 30):
            latest[f"ewm_mean_{window}_clean"] = (
                item_data["sales_clean"]
                .ewm(span=window, adjust=False, min_periods=max(2, window // 4))
                .mean()
                .iloc[-1]
            )
        latest["same_weekday_mean_4"] = np.mean(
            [latest[f"lag_{lag}_clean"] for lag in (7, 14, 21, 28)]
        )
        latest["short_trend"] = latest["roll_mean_7_clean"] / (
            latest["roll_mean_30_clean"] + 1
        )
        latest["lag_week_ratio"] = latest["lag_1_clean"] / (
            latest["lag_7_clean"] + 1
        )
        latest["item_promo_mean"] = regular["sales_clean"].mean()
        latest["regular_price_90"] = item_data.tail(90)["price"].median()
        records.append(latest.to_dict())

    context = pd.DataFrame(records)
    context["price_cost_ratio"] = context["price"] / context["cost"].clip(lower=1e-6)
    context["price_competitor_ratio"] = (
        context["price"] / context["competitor_price"].clip(lower=1e-6)
    )
    context["price_vs_regular_90"] = (
        context["price"] / context["regular_price_90"].clip(lower=1e-6)
    )
    context["log_ad_spend"] = np.log1p(context["ad_spend"].clip(lower=0))
    context["month"] = next_date.month
    context["day_of_week"] = next_date.dayofweek
    context["is_weekend"] = int(next_date.dayofweek >= 5)
    context["day_of_year_sin"] = np.sin(2 * np.pi * next_date.dayofyear / 365.25)
    context["day_of_year_cos"] = np.cos(2 * np.pi * next_date.dayofyear / 365.25)
    for column in CATEGORICAL_FEATURES:
        context[column] = pd.Categorical(
            context[column], categories=model_frame[column].cat.categories
        )
    return context.sort_values("item_id").reset_index(drop=True)


def _price_bounds(item_data, current_price, cost, change_limit, minimum_margin):
    regular_prices = item_data.loc[item_data["promo_active"].eq(0), "price"]
    if regular_prices.empty:
        regular_prices = item_data["price"]
    lower = max(
        cost * (1 + minimum_margin),
        current_price * (1 - change_limit),
        float(regular_prices.quantile(0.05)),
    )
    upper = min(
        current_price * (1 + change_limit),
        float(regular_prices.quantile(0.95)),
    )
    historical_range_conflict = lower > upper
    if historical_range_conflict:
        lower = upper = max(cost * (1 + minimum_margin), current_price)
    return lower, upper, historical_range_conflict


def _objective_curves(base_demand, current_price, elasticity, cost, prices):
    demand = base_demand * (prices / current_price) ** elasticity
    return demand, prices * demand, (prices - cost) * demand


def optimize_prices(
    model: XGBRegressor,
    model_frame: pd.DataFrame,
    elasticities: pd.DataFrame,
    price_change_limit: float = 0.15,
    minimum_margin: float = 0.05,
    grid_size: int = 61,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select sales-, GMV- and profit-optimal prices under guardrails."""
    context = make_next_day_context(model_frame)
    context["base_demand"] = np.clip(model.predict(context[MODEL_FEATURES]), 0, None)
    elasticity_by_item = elasticities.set_index("item_id")
    recommendation_rows, curve_rows = [], []

    for _, row in context.iterrows():
        item_id = row["item_id"]
        current_price, cost = float(row["price"]), float(row["cost"])
        margin_floor = cost * (1 + minimum_margin)
        margin_floor_required = current_price < margin_floor
        item_history = model_frame[model_frame["item_id"] == item_id]
        elasticity_info = elasticity_by_item.loc[item_id]
        elasticity = float(elasticity_info["elasticity"])
        elasticity_source = elasticity_info["elasticity_source"]
        applied_change_limit = (
            price_change_limit
            if elasticity_source == "item_supported"
            else min(price_change_limit, 0.05)
        )
        lower, upper, historical_range_conflict = _price_bounds(
            item_history,
            current_price,
            cost,
            applied_change_limit,
            minimum_margin,
        )
        prices = np.linspace(lower, upper, grid_size)
        demand, gmv, profit = _objective_curves(
            row["base_demand"], current_price, elasticity, cost, prices
        )
        curve_rows.append(
            pd.DataFrame(
                {
                    "item_id": item_id,
                    "candidate_price": prices,
                    "predicted_sales": demand,
                    "predicted_gmv": gmv,
                    "predicted_profit": profit,
                }
            )
        )

        if row["base_demand"] < 1.0:
            hold_index = int(np.argmin(np.abs(prices - current_price)))
            sales_index = gmv_index = profit_index = hold_index
            status = "low_demand_hold"
        else:
            sales_index = int(np.argmax(demand))
            gmv_index = int(np.argmax(gmv))
            profit_index = int(np.argmax(profit))
            status = "optimized"

        if margin_floor_required:
            status = "margin_floor_correction"
        elif historical_range_conflict:
            status = "historical_range_conflict_hold"

        recommendation_rows.append(
            {
                "date": row["date"],
                "item_id": item_id,
                "category": row["category"],
                "current_regular_price": current_price,
                "cost": cost,
                "margin_floor_required": margin_floor_required,
                "historical_range_conflict": historical_range_conflict,
                "elasticity": elasticity,
                "elasticity_source": elasticity_source,
                "applied_price_change_limit": applied_change_limit,
                "base_demand": row["base_demand"],
                "sales_optimal_price": prices[sales_index],
                "gmv_optimal_price": prices[gmv_index],
                "profit_optimal_price": prices[profit_index],
                "profit_optimal_sales": demand[profit_index],
                "profit_at_optimum": profit[profit_index],
                "lower_guardrail": lower,
                "upper_guardrail": upper,
                "recommendation_status": status,
            }
        )

    recommendations = pd.DataFrame(recommendation_rows).sort_values("item_id")
    curves = pd.concat(curve_rows, ignore_index=True)
    return recommendations.reset_index(drop=True), curves
