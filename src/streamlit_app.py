

from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import heapq
from collections import defaultdict

import plotly.graph_objects as go


ROUTE_OBJECTIVES = {
    "Balanced": "balanced_weight",
    "Shortest distance": "distance_miles",
    "Fastest time": "transit_hours",
    "Lowest cost": "estimated_cost_usd",
    "Most reliable": "reliability_weight",
}

ORIGIN_CITY_OPTIONS = [
    "Kansas City",
    "Columbus",
    "Chicago",
    "Philadelphia",
    "Houston",
]

DESTINATION_CITY_OPTIONS = [
    "Los Angeles",
    "Portland",
    "Seattle",
    "Denver",
    "Indianapolis",
]


APP_DIR = Path(__file__).resolve().parent
# Hugging Face runs this file from src/streamlit_app.py, while datasets live in the repository-root data/ folder.
REPO_ROOT = APP_DIR.parent if APP_DIR.name == "src" else APP_DIR
DATA_DIR = REPO_ROOT / "data"


st.set_page_config(
    page_title="Supply Chain Analytics",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# st.set_page_config(
#     page_title="Retail Demand Forecasting",
#     page_icon="📈",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem;}
        [data-testid="stMetricValue"] {font-size: 1.55rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


COLUMN_ALIASES = {
    "store": {"store", "store_id", "storeid"},
    "department": {"department", "dept", "department_id", "dept_id"},
    "date": {"date", "week", "week_date", "sales_date"},
    "weekly_sales": {"weekly_sales", "weeklysales", "sales", "demand"},
    "is_holiday": {"is_holiday", "isholiday", "holiday"},
    "store_type": {"store_type", "type"},
    "store_size": {"store_size", "size"},
    "region": {"region", "store_region"},
    "temperature": {"temperature", "temp"},
    "fuel_price": {"fuel_price", "fuelprice"},
    "markdown_1": {"markdown_1", "markdown1", "mark_down1"},
    "markdown_2": {"markdown_2", "markdown2", "mark_down2"},
    "markdown_3": {"markdown_3", "markdown3", "mark_down3"},
    "markdown_4": {"markdown_4", "markdown4", "mark_down4"},
    "markdown_5": {"markdown_5", "markdown5", "mark_down5"},
    "cpi": {"cpi", "consumer_price_index"},
    "unemployment": {"unemployment", "unemployment_rate"},
    "holiday_name": {"holiday_name", "holiday_event", "event_name"},
    "season": {"season", "season_name"},
}

NUMERIC_EXTERNAL_COLUMNS = [
    "temperature",
    "fuel_price",
    "markdown_1",
    "markdown_2",
    "markdown_3",
    "markdown_4",
    "markdown_5",
    "cpi",
    "unemployment",
]
CATEGORICAL_COLUMNS = ["store_type", "region", "season", "holiday_name"]
LAG_WEEKS = [1, 2, 4, 8, 13, 26]
ROLLING_WINDOWS = [4, 8, 13]


def normalise_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def metric_row(items: list[tuple[str, str, str | None]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value, help_text) in zip(columns, items):
        column.metric(label, value, help=help_text)

def canonicalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output.columns = [normalise_name(column) for column in output.columns]

    rename_map: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for column in output.columns:
            if column in aliases:
                rename_map[column] = canonical
                break

    output = output.rename(columns=rename_map)
    return output.loc[:, ~output.columns.duplicated()]


@st.cache_data(show_spinner=False)
def read_csv_bytes(content: bytes) -> pd.DataFrame:
    buffer = io.BytesIO(content)
    try:
        data = pd.read_csv(buffer, low_memory=False)
    except UnicodeDecodeError:
        buffer.seek(0)
        data = pd.read_csv(buffer, encoding="latin-1", low_memory=False)
    return canonicalise_columns(data)


@st.cache_data(show_spinner=False)
def read_local_csv(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    return read_csv_bytes(path.read_bytes())


def find_local_file(filename: str) -> Path | None:
    candidates = [DATA_DIR / filename, REPO_ROOT / filename, APP_DIR / filename]
    return next((path for path in candidates if path.exists()), None)


def load_dataset(uploaded_file, filename: str) -> tuple[pd.DataFrame | None, str]:
    if uploaded_file is not None:
        try:
            return read_csv_bytes(uploaded_file.getvalue()), f"Uploaded: {uploaded_file.name}"
        except Exception as exc:
            st.error(f"Could not read {uploaded_file.name}: {exc}")
            return None, "Read error"

    local_path = find_local_file(filename)
    if local_path is not None:
        try:
            return read_local_csv(str(local_path)), f"Bundled: {local_path}"
        except Exception as exc:
            st.error(f"Could not read {local_path}: {exc}")
            return None, "Read error"

    return None, "Not loaded"


def parse_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    text = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
        "y": 1,
        "n": 0,
        "1": 1,
        "0": 0,
    }
    mapped = text.map(mapping)
    numeric = pd.to_numeric(series, errors="coerce")
    return mapped.fillna(numeric).fillna(0).astype(int).clip(0, 1)


def validate_columns(df: pd.DataFrame, required: set[str], dataset_name: str) -> bool:
    missing = sorted(required - set(df.columns))
    if missing:
        st.error(
            f"{dataset_name} is missing required columns: {', '.join(missing)}. "
            f"Detected columns: {', '.join(df.columns)}"
        )
        return False
    return True


def clean_sales(sales: pd.DataFrame) -> pd.DataFrame:
    output = sales.copy().reset_index(drop=True)
    output["_source_row"] = np.arange(len(output))
    output["date"] = pd.to_datetime(output["date"], errors="coerce")
    output["store"] = pd.to_numeric(output["store"], errors="coerce")
    output["weekly_sales"] = pd.to_numeric(output["weekly_sales"], errors="coerce")
    output["department"] = output["department"].astype(str).str.strip()

    if "is_holiday" in output:
        output["is_holiday"] = parse_boolean(output["is_holiday"])
    else:
        output["is_holiday"] = 0

    output = output.dropna(subset=["store", "date", "weekly_sales"])
    output["store"] = output["store"].astype(int)
    return output.sort_values(["date", "store", "department"]).reset_index(drop=True)


def clean_stores(stores: pd.DataFrame) -> pd.DataFrame:
    output = stores.copy()
    output["store"] = pd.to_numeric(output["store"], errors="coerce")
    output = output.dropna(subset=["store"])
    output["store"] = output["store"].astype(int)

    if "store_size" in output:
        output["store_size"] = pd.to_numeric(output["store_size"], errors="coerce")
    for column in ["store_type", "region"]:
        if column in output:
            output[column] = output[column].astype(str).str.strip()

    keep = [column for column in ["store", "store_type", "store_size", "region"] if column in output]
    return output[keep].drop_duplicates("store", keep="last")


def aggregate_feature_rows(features: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    output = features.copy()
    for column in NUMERIC_EXTERNAL_COLUMNS:
        if column in output:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    if "is_holiday" in output:
        output["is_holiday"] = parse_boolean(output["is_holiday"])

    aggregations: dict[str, str] = {}
    for column in output.columns:
        if column in keys:
            continue
        if column in NUMERIC_EXTERNAL_COLUMNS:
            aggregations[column] = "mean"
        elif column == "is_holiday":
            aggregations[column] = "max"
        elif column in {"season", "holiday_name"}:
            aggregations[column] = "last"

    if not aggregations:
        return output[keys].drop_duplicates(keys)
    return output.groupby(keys, as_index=False).agg(aggregations)


def attach_features(
    sales: pd.DataFrame,
    features: pd.DataFrame | None,
    allow_row_alignment: bool,
) -> tuple[pd.DataFrame, str]:
    if features is None:
        return sales.copy(), "Not used — sales history and calendar lags only"

    feature_frame = features.copy().reset_index(drop=True)
    feature_columns = [
        column
        for column in NUMERIC_EXTERNAL_COLUMNS + ["season", "holiday_name", "is_holiday"]
        if column in feature_frame
    ]
    if not feature_columns:
        return sales.copy(), "Ignored — no recognised forecasting columns"

    if "date" in feature_frame:
        feature_frame["date"] = pd.to_datetime(feature_frame["date"], errors="coerce")

    if {"store", "date"}.issubset(feature_frame.columns):
        feature_frame["store"] = pd.to_numeric(feature_frame["store"], errors="coerce")
        feature_frame = feature_frame.dropna(subset=["store", "date"])
        feature_frame["store"] = feature_frame["store"].astype(int)
        keyed = aggregate_feature_rows(feature_frame, ["store", "date"])
        if "is_holiday" in keyed and "is_holiday" in sales:
            keyed = keyed.drop(columns="is_holiday")
        merged = sales.merge(keyed, on=["store", "date"], how="left", validate="many_to_one")
        return merged, "Merged safely using store_id + date"

    if "date" in feature_frame:
        feature_frame = feature_frame.dropna(subset=["date"])
        keyed = aggregate_feature_rows(feature_frame, ["date"])
        if "is_holiday" in keyed and "is_holiday" in sales:
            keyed = keyed.drop(columns="is_holiday")
        merged = sales.merge(keyed, on="date", how="left", validate="many_to_one")
        return merged, "Merged using date"

    if allow_row_alignment and len(feature_frame) == len(sales):
        aligned = feature_frame[feature_columns].copy()
        if "is_holiday" in aligned and "is_holiday" in sales:
            aligned = aligned.drop(columns="is_holiday")
        combined = pd.concat([sales.reset_index(drop=True), aligned.reset_index(drop=True)], axis=1)
        return combined, "Aligned row-by-row because row counts matched"

    if len(feature_frame) == len(sales):
        return sales.copy(), "Available but not used — enable row-order alignment in the sidebar"

    return sales.copy(), (
        "Ignored safely — features.csv has no store_id/date key and its row count does not match sales.csv"
    )


def prepare_history(
    sales: pd.DataFrame,
    stores: pd.DataFrame | None,
    features: pd.DataFrame | None,
    selected_department: str,
    allow_row_alignment: bool,
) -> tuple[pd.DataFrame, str]:
    enriched, feature_status = attach_features(sales, features, allow_row_alignment)

    if selected_department != "All departments":
        enriched = enriched[enriched["department"] == selected_department].copy()

    aggregations: dict[str, str] = {
        "weekly_sales": "sum",
        "is_holiday": "max",
    }
    for column in NUMERIC_EXTERNAL_COLUMNS:
        if column in enriched:
            aggregations[column] = "mean"
    for column in ["season", "holiday_name"]:
        if column in enriched:
            aggregations[column] = "last"

    history = (
        enriched.groupby(["store", "date"], as_index=False)
        .agg(aggregations)
        .sort_values(["store", "date"])
    )

    if stores is not None:
        history = history.merge(stores, on="store", how="left", validate="many_to_one")

    return history.sort_values(["store", "date"]).reset_index(drop=True), feature_status


def add_calendar_season(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    month = pd.to_datetime(output["date"]).dt.month
    season = np.select(
        [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
        ["Winter", "Spring", "Summer"],
        default="Autumn",
    )
    output["calendar_season"] = season
    return output


def build_model_features(
    frame: pd.DataFrame,
    category_maps: dict[str, dict[str, int]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    data = add_calendar_season(frame.sort_values(["store", "date"]).copy())
    dates = pd.to_datetime(data["date"], errors="coerce")
    iso_week = dates.dt.isocalendar().week.astype(float)

    matrix = pd.DataFrame(index=data.index)
    matrix["store"] = pd.to_numeric(data["store"], errors="coerce")
    matrix["year"] = dates.dt.year
    matrix["month"] = dates.dt.month
    matrix["quarter"] = dates.dt.quarter
    matrix["week_of_year"] = iso_week
    matrix["weeks_since_start"] = (dates - dates.min()).dt.days / 7.0
    matrix["week_sin"] = np.sin(2 * np.pi * iso_week / 52.18)
    matrix["week_cos"] = np.cos(2 * np.pi * iso_week / 52.18)
    matrix["month_sin"] = np.sin(2 * np.pi * dates.dt.month / 12.0)
    matrix["month_cos"] = np.cos(2 * np.pi * dates.dt.month / 12.0)
    matrix["is_holiday"] = parse_boolean(data["is_holiday"]) if "is_holiday" in data else 0

    grouped_sales = data.groupby("store", sort=False)["weekly_sales"]
    for lag in LAG_WEEKS:
        matrix[f"sales_lag_{lag}"] = grouped_sales.shift(lag)
    for window in ROLLING_WINDOWS:
        matrix[f"sales_roll_mean_{window}"] = grouped_sales.transform(
            lambda values, w=window: values.shift(1).rolling(w, min_periods=1).mean()
        )
        matrix[f"sales_roll_std_{window}"] = grouped_sales.transform(
            lambda values, w=window: values.shift(1).rolling(w, min_periods=2).std()
        )

    for column in NUMERIC_EXTERNAL_COLUMNS + ["store_size"]:
        if column in data:
            matrix[column] = pd.to_numeric(data[column], errors="coerce")

    category_maps = {} if category_maps is None else {key: value.copy() for key, value in category_maps.items()}
    for column in CATEGORICAL_COLUMNS + ["calendar_season"]:
        if column not in data:
            continue
        values = data[column].fillna("Unknown").astype(str).str.strip()
        if column not in category_maps:
            unique_values = sorted(values.unique().tolist())
            category_maps[column] = {value: index for index, value in enumerate(unique_values)}
        matrix[f"{column}_code"] = values.map(category_maps[column]).fillna(-1)

    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    return matrix.astype(float), category_maps


def train_backtest(
    history: pd.DataFrame,
    validation_weeks: int,
) -> tuple[HistGradientBoostingRegressor, dict[str, float], pd.DataFrame, list[str], dict[str, dict[str, int]]]:
    unique_dates = np.array(sorted(history["date"].dropna().unique()))
    if len(unique_dates) < 16:
        raise ValueError("At least 16 historical weeks are required.")

    validation_weeks = min(validation_weeks, max(4, len(unique_dates) // 3))
    cutoff = pd.Timestamp(unique_dates[-validation_weeks])

    matrix, category_maps = build_model_features(history)
    usable = matrix["sales_lag_1"].notna()
    train_mask = usable & (history["date"] < cutoff)
    valid_mask = usable & (history["date"] >= cutoff)

    if train_mask.sum() < 50 or valid_mask.sum() < 10:
        raise ValueError("Not enough usable rows remain after creating lag features and the validation split.")

    feature_columns = matrix.columns.tolist()
    model = HistGradientBoostingRegressor(
        learning_rate=0.055,
        max_iter=350,
        max_leaf_nodes=31,
        min_samples_leaf=15,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.12,
        random_state=42,
    )
    model.fit(matrix.loc[train_mask, feature_columns], history.loc[train_mask, "weekly_sales"])
    predicted = model.predict(matrix.loc[valid_mask, feature_columns]).clip(min=0)
    actual = history.loc[valid_mask, "weekly_sales"].to_numpy()

    validation = history.loc[valid_mask, ["store", "date", "weekly_sales", "is_holiday"]].copy()
    validation["predicted_weekly_sales"] = predicted
    validation["absolute_error"] = np.abs(validation["weekly_sales"] - predicted)

    holiday_weights = np.where(parse_boolean(validation["is_holiday"]).to_numpy() == 1, 5.0, 1.0)
    metrics = {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(mean_squared_error(actual, predicted) ** 0.5),
        "R2": float(r2_score(actual, predicted)),
        "WMAE": float(np.average(np.abs(actual - predicted), weights=holiday_weights)),
    }
    return model, metrics, validation, feature_columns, category_maps


def fit_final_model(
    history: pd.DataFrame,
    feature_columns: list[str],
    category_maps: dict[str, dict[str, int]],
) -> HistGradientBoostingRegressor:
    matrix, _ = build_model_features(history, category_maps)
    matrix = matrix.reindex(columns=feature_columns)
    usable = matrix["sales_lag_1"].notna()

    model = HistGradientBoostingRegressor(
        learning_rate=0.055,
        max_iter=350,
        max_leaf_nodes=31,
        min_samples_leaf=15,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.12,
        random_state=42,
    )
    model.fit(matrix.loc[usable], history.loc[usable, "weekly_sales"])
    return model


def infer_next_date(last_date: pd.Timestamp, unique_dates: pd.Series) -> pd.Timestamp:
    sorted_dates = pd.Series(pd.to_datetime(unique_dates).dropna().unique()).sort_values()
    differences = sorted_dates.diff().dropna().dt.days
    step_days = int(differences.mode().iloc[0]) if not differences.empty else 7
    if step_days <= 0 or step_days > 31:
        step_days = 7
    return pd.Timestamp(last_date) + pd.Timedelta(days=step_days)


def recursive_forecast(
    model: HistGradientBoostingRegressor,
    history: pd.DataFrame,
    horizon: int,
    feature_columns: list[str],
    category_maps: dict[str, dict[str, int]],
) -> pd.DataFrame:
    working = history.copy().sort_values(["store", "date"]).reset_index(drop=True)
    forecasts: list[pd.DataFrame] = []
    last_date = pd.Timestamp(working["date"].max())
    next_date = infer_next_date(last_date, working["date"])

    carry_columns = [
        column
        for column in NUMERIC_EXTERNAL_COLUMNS + ["store_type", "store_size", "region", "season"]
        if column in working
    ]
    latest_by_store = working.sort_values("date").groupby("store", as_index=False).tail(1)

    for step in range(horizon):
        future_rows = latest_by_store[["store"] + carry_columns].copy()
        future_rows["date"] = next_date
        future_rows["weekly_sales"] = np.nan
        future_rows["is_holiday"] = 0
        if "holiday_name" in working:
            future_rows["holiday_name"] = "None"

        combined = pd.concat([working, future_rows], ignore_index=True, sort=False)
        matrix, _ = build_model_features(combined, category_maps)
        future_index = combined.index[-len(future_rows):]
        x_future = matrix.loc[future_index].reindex(columns=feature_columns)
        predictions = model.predict(x_future).clip(min=0)

        future_rows["weekly_sales"] = predictions
        future_rows["forecast_weekly_sales"] = predictions
        forecasts.append(future_rows.copy())

        working = pd.concat(
            [working, future_rows.drop(columns="forecast_weekly_sales")],
            ignore_index=True,
            sort=False,
        )
        latest_by_store = future_rows.drop(columns="forecast_weekly_sales").copy()
        next_date = infer_next_date(next_date, working["date"])

    output = pd.concat(forecasts, ignore_index=True)
    keep = [
        column
        for column in ["store", "date", "forecast_weekly_sales", "is_holiday", "store_type", "region"]
        if column in output
    ]
    return output[keep].sort_values(["date", "store"]).reset_index(drop=True)


# Add these imports with the imports at the top of streamlit_app.py



##################################################################################################################################################################



def _logistics_file(filename: str) -> Path | None:
    """Find a logistics CSV under the usual Kaggle extraction locations."""
    candidates = [
        DATA_DIR / "logistics" / filename,
        DATA_DIR / "logistics_data" / filename,
        DATA_DIR / filename,
        REPO_ROOT / "logistics_data" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path

    matches = list(DATA_DIR.rglob(filename)) if DATA_DIR.exists() else []
    return matches[0] if matches else None


@st.cache_data(show_spinner=False)
def _read_logistics_csv(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    try:
        frame = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        frame = pd.read_csv(path, encoding="latin-1", low_memory=False)
    frame.columns = [normalise_name(column) for column in frame.columns]
    return frame.loc[:, ~frame.columns.duplicated()]


def _load_logistics_csv(filename: str) -> tuple[pd.DataFrame | None, str]:
    path = _logistics_file(filename)
    if path is None:
        return None, "Not found"
    try:
        return _read_logistics_csv(str(path)), str(path)
    except Exception as exc:
        return None, f"Read error: {exc}"


def _first_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _number(frame: pd.DataFrame, candidates: list[str], default=np.nan) -> pd.Series:
    column = _first_column(frame, candidates)
    if column is None:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _binary(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    text = series.astype(str).str.strip().str.lower()
    mapped = text.map(
        {
            "true": 1.0,
            "false": 0.0,
            "yes": 1.0,
            "no": 0.0,
            "y": 1.0,
            "n": 0.0,
            "1": 1.0,
            "0": 0.0,
            "on time": 1.0,
            "late": 0.0,
        }
    )
    return mapped.fillna(pd.to_numeric(series, errors="coerce"))


def _haversine_miles(lat1, lon1, lat2, lon2):
    radius = 3958.7613
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    value = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(value))


def _prepare_facilities(frame: pd.DataFrame) -> pd.DataFrame:
    id_col = _first_column(frame, ["facility_id", "terminal_id", "warehouse_id", "location_id"])
    lat_col = _first_column(frame, ["latitude", "lat", "facility_latitude"])
    lon_col = _first_column(frame, ["longitude", "lon", "lng", "facility_longitude"])

    if not id_col or not lat_col or not lon_col:
        raise ValueError(
            "facilities.csv must contain facility_id, latitude and longitude columns."
        )

    name_col = _first_column(frame, ["facility_name", "terminal_name", "warehouse_name", "name"])
    city_col = _first_column(frame, ["city", "facility_city", "location_city"])
    state_col = _first_column(frame, ["state", "facility_state", "location_state"])
    type_col = _first_column(frame, ["facility_type", "terminal_type", "warehouse_type"])

    output = pd.DataFrame(
        {
            "facility_id": frame[id_col].astype(str).str.strip(),
            "facility_name": frame[name_col].astype(str).str.strip() if name_col else frame[id_col].astype(str),
            "city": frame[city_col].fillna("").astype(str).str.strip() if city_col else "",
            "state": frame[state_col].fillna("").astype(str).str.strip() if state_col else "",
            "facility_type": frame[type_col].fillna("").astype(str).str.strip() if type_col else "",
            "latitude": pd.to_numeric(frame[lat_col], errors="coerce"),
            "longitude": pd.to_numeric(frame[lon_col], errors="coerce"),
        }
    )
    output = output.dropna(subset=["facility_id", "latitude", "longitude"])
    output = output[
        output["latitude"].between(-90, 90)
        & output["longitude"].between(-180, 180)
    ]
    return output.drop_duplicates("facility_id", keep="last").reset_index(drop=True)


def _resolve_route_endpoint(
    value: object,
    state: object,
    facility_ids: set[str],
    id_lookup: dict[str, str],
    name_lookup: dict[str, str],
    city_lookup: dict[str, str],
    city_state_lookup: dict[str, str],
) -> str | None:
    raw = str(value).strip()
    if raw in facility_ids:
        return raw

    key = normalise_name(raw)
    state_key = normalise_name(state)
    if key and state_key and f"{key}|{state_key}" in city_state_lookup:
        return city_state_lookup[f"{key}|{state_key}"]
    return id_lookup.get(key) or name_lookup.get(key) or city_lookup.get(key)


def _prepare_routes(frame: pd.DataFrame, facilities: pd.DataFrame) -> pd.DataFrame:
    route_id_col = _first_column(frame, ["route_id", "lane_id"])
    origin_col = _first_column(
        frame,
        [
            "origin_facility_id",
            "origin_terminal_id",
            "origin_location_id",
            "origin_city",
            "origin",
            "from_facility_id",
            "source",
        ],
    )
    destination_col = _first_column(
        frame,
        [
            "destination_facility_id",
            "destination_terminal_id",
            "destination_location_id",
            "destination_city",
            "destination",
            "to_facility_id",
            "target",
        ],
    )
    if not origin_col or not destination_col:
        raise ValueError("routes.csv must contain origin and destination columns.")

    origin_state_col = _first_column(frame, ["origin_state", "from_state"])
    destination_state_col = _first_column(frame, ["destination_state", "to_state"])

    facility_ids = set(facilities["facility_id"])
    id_lookup = {normalise_name(value): value for value in facilities["facility_id"]}
    name_lookup = {
        normalise_name(name): facility_id
        for facility_id, name in facilities[["facility_id", "facility_name"]].itertuples(index=False)
    }
    city_lookup: dict[str, str] = {}
    city_state_lookup: dict[str, str] = {}
    for row in facilities.itertuples(index=False):
        city_key = normalise_name(row.city)
        state_key = normalise_name(row.state)
        if city_key and city_key not in city_lookup:
            city_lookup[city_key] = row.facility_id
        if city_key and state_key:
            city_state_lookup[f"{city_key}|{state_key}"] = row.facility_id

    origin_states = frame[origin_state_col] if origin_state_col else pd.Series("", index=frame.index)
    destination_states = (
        frame[destination_state_col] if destination_state_col else pd.Series("", index=frame.index)
    )

    output = pd.DataFrame(index=frame.index)
    output["route_id"] = (
        frame[route_id_col].astype(str).str.strip()
        if route_id_col
        else [f"ROUTE_{index + 1:04d}" for index in range(len(frame))]
    )
    output["origin_facility_id"] = [
        _resolve_route_endpoint(
            value,
            state,
            facility_ids,
            id_lookup,
            name_lookup,
            city_lookup,
            city_state_lookup,
        )
        for value, state in zip(frame[origin_col], origin_states)
    ]
    output["destination_facility_id"] = [
        _resolve_route_endpoint(
            value,
            state,
            facility_ids,
            id_lookup,
            name_lookup,
            city_lookup,
            city_state_lookup,
        )
        for value, state in zip(frame[destination_col], destination_states)
    ]
    output = output.dropna(subset=["origin_facility_id", "destination_facility_id"])
    output = output[output["origin_facility_id"] != output["destination_facility_id"]].copy()

    coordinates = facilities.set_index("facility_id")[["latitude", "longitude"]]
    origin_coordinates = coordinates.reindex(output["origin_facility_id"])
    destination_coordinates = coordinates.reindex(output["destination_facility_id"])
    straight_line_distance = pd.Series(
        _haversine_miles(
            origin_coordinates["latitude"].to_numpy(),
            origin_coordinates["longitude"].to_numpy(),
            destination_coordinates["latitude"].to_numpy(),
            destination_coordinates["longitude"].to_numpy(),
        ),
        index=output.index,
    )

    distance = _number(
        frame,
        ["distance_miles", "route_distance_miles", "planned_miles", "standard_miles", "distance", "miles"],
    )
    output["distance_miles"] = distance.reindex(output.index).fillna(straight_line_distance)

    transit_hours = _number(
        frame,
        ["typical_transit_hours", "standard_transit_hours", "transit_hours", "travel_time_hours", "planned_hours"],
    ).reindex(output.index)
    transit_minutes = _number(frame, ["transit_minutes", "travel_time_minutes"]).reindex(output.index)
    transit_days = _number(frame, ["transit_days", "standard_transit_days"]).reindex(output.index)
    output["transit_hours"] = transit_hours.fillna(transit_minutes / 60).fillna(transit_days * 24)
    output["transit_hours"] = output["transit_hours"].fillna(output["distance_miles"] / 50).clip(lower=0.1)

    direct_cost = _number(
        frame,
        ["route_cost_usd", "route_cost", "estimated_cost", "flat_rate", "linehaul_rate", "total_rate"],
    ).reindex(output.index)
    rate_per_mile = _number(
        frame,
        ["base_rate_per_mile", "rate_per_mile", "contract_rate_per_mile"],
    ).reindex(output.index)
    toll_cost = _number(frame, ["toll_cost", "toll_cost_usd", "tolls"]).reindex(output.index).fillna(0)
    output["estimated_cost_usd"] = direct_cost.fillna(rate_per_mile * output["distance_miles"])
    output["estimated_cost_usd"] = output["estimated_cost_usd"].fillna(output["distance_miles"] * 2.50)
    output["estimated_cost_usd"] += toll_cost

    reliability = _number(
        frame,
        ["on_time_pct", "on_time_percentage", "on_time_rate", "reliability_pct"],
        default=90,
    ).reindex(output.index)
    reliability = reliability.where(reliability > 1, reliability * 100)
    output["on_time_pct"] = reliability.fillna(90).clip(1, 100)

    output = output[output["distance_miles"].gt(0)]
    return output.drop_duplicates("route_id", keep="last").reset_index(drop=True)


def _enrich_routes(
    routes: pd.DataFrame,
    trips: pd.DataFrame | None,
    loads: pd.DataFrame | None,
    delivery_events: pd.DataFrame | None,
) -> pd.DataFrame:
    output = routes.copy()
    output["trip_count"] = 0
    output["load_count"] = 0
    output["average_revenue_usd"] = np.nan

    trip_route = pd.DataFrame()
    load_route = pd.DataFrame()

    if trips is not None and "route_id" in trips.columns:
        trip_frame = trips.copy()
        trip_frame["route_id"] = trip_frame["route_id"].astype(str).str.strip()

        trip_id_col = _first_column(trip_frame, ["trip_id"])
        load_id_col = _first_column(trip_frame, ["load_id", "shipment_id"])
        if trip_id_col:
            trip_route = trip_frame[[trip_id_col, "route_id"]].dropna().drop_duplicates(trip_id_col)
            trip_route = trip_route.rename(columns={trip_id_col: "trip_id"})
        if load_id_col:
            load_route = trip_frame[[load_id_col, "route_id"]].dropna().drop_duplicates(load_id_col)
            load_route = load_route.rename(columns={load_id_col: "load_id"})

        actual_distance = _number(
            trip_frame,
            ["actual_distance_miles", "actual_miles", "total_miles", "miles_driven", "trip_miles"],
        )
        actual_cost = _number(
            trip_frame,
            ["total_trip_cost", "trip_cost_usd", "trip_cost", "operating_cost", "total_cost"],
        )
        departure_col = _first_column(
            trip_frame,
            ["actual_departure_datetime", "actual_departure", "departure_datetime", "trip_start_datetime"],
        )
        arrival_col = _first_column(
            trip_frame,
            ["actual_arrival_datetime", "actual_arrival", "arrival_datetime", "trip_end_datetime"],
        )
        actual_hours = _number(
            trip_frame,
            ["actual_duration_hours", "actual_transit_hours", "trip_duration_hours"],
        )
        if departure_col and arrival_col:
            departure = pd.to_datetime(trip_frame[departure_col], errors="coerce")
            arrival = pd.to_datetime(trip_frame[arrival_col], errors="coerce")
            actual_hours = actual_hours.fillna((arrival - departure).dt.total_seconds() / 3600)

        trip_summary_frame = pd.DataFrame(
            {
                "route_id": trip_frame["route_id"],
                "actual_distance": actual_distance,
                "actual_hours": actual_hours,
                "actual_cost": actual_cost,
            }
        )
        on_time_col = _first_column(trip_frame, ["on_time_flag", "on_time", "delivered_on_time"])
        if on_time_col:
            trip_summary_frame["trip_on_time"] = _binary(trip_frame[on_time_col])

        aggregations = {
            "trip_count": ("route_id", "size"),
            "historical_distance": ("actual_distance", "median"),
            "historical_hours": ("actual_hours", "median"),
            "historical_cost": ("actual_cost", "median"),
        }
        if "trip_on_time" in trip_summary_frame:
            aggregations["trip_on_time"] = ("trip_on_time", "mean")

        trip_summary = trip_summary_frame.groupby("route_id", as_index=False).agg(**aggregations)
        output = output.drop(columns="trip_count").merge(trip_summary, on="route_id", how="left")
        output["trip_count"] = output["trip_count"].fillna(0).astype(int)
        output["distance_miles"] = output["historical_distance"].where(
            output["historical_distance"].gt(0), output["distance_miles"]
        )
        output["transit_hours"] = output["historical_hours"].where(
            output["historical_hours"].gt(0), output["transit_hours"]
        )
        output["estimated_cost_usd"] = output["historical_cost"].where(
            output["historical_cost"].gt(0), output["estimated_cost_usd"]
        )
        if "trip_on_time" in output:
            output["on_time_pct"] = output["trip_on_time"].mul(100).where(
                output["trip_on_time"].notna(), output["on_time_pct"]
            )

    if loads is not None:
        load_frame = loads.copy()
        if "route_id" not in load_frame.columns and not load_route.empty and "load_id" in load_frame.columns:
            load_frame = load_frame.merge(load_route, on="load_id", how="left")
        if "route_id" in load_frame.columns:
            load_frame["route_id"] = load_frame["route_id"].astype(str).str.strip()
            revenue = _number(
                load_frame,
                ["revenue_usd", "load_revenue", "total_revenue", "linehaul_revenue", "revenue"],
            )
            load_summary = pd.DataFrame(
                {"route_id": load_frame["route_id"], "revenue": revenue}
            ).groupby("route_id", as_index=False).agg(
                load_count=("route_id", "size"),
                average_revenue_usd=("revenue", "mean"),
            )
            output = output.drop(columns=["load_count", "average_revenue_usd"]).merge(
                load_summary, on="route_id", how="left"
            )
            output["load_count"] = output["load_count"].fillna(0).astype(int)

    if delivery_events is not None:
        event_frame = delivery_events.copy()
        if "route_id" not in event_frame.columns and not trip_route.empty and "trip_id" in event_frame.columns:
            event_frame = event_frame.merge(trip_route, on="trip_id", how="left")
        if "route_id" not in event_frame.columns and not load_route.empty and "load_id" in event_frame.columns:
            event_frame = event_frame.merge(load_route, on="load_id", how="left")
        if "route_id" in event_frame.columns:
            on_time_col = _first_column(event_frame, ["on_time_flag", "on_time", "delivered_on_time"])
            detention_col = _first_column(event_frame, ["detention_minutes", "dwell_minutes"])
            event_summary_frame = pd.DataFrame({"route_id": event_frame["route_id"].astype(str).str.strip()})
            if on_time_col:
                event_summary_frame["event_on_time"] = _binary(event_frame[on_time_col])
            if detention_col:
                event_summary_frame["detention_minutes"] = pd.to_numeric(
                    event_frame[detention_col], errors="coerce"
                )

            aggregations = {}
            if "event_on_time" in event_summary_frame:
                aggregations["event_on_time"] = ("event_on_time", "mean")
            if "detention_minutes" in event_summary_frame:
                aggregations["detention_minutes"] = ("detention_minutes", "median")
            if aggregations:
                event_summary = event_summary_frame.groupby("route_id", as_index=False).agg(**aggregations)
                output = output.merge(event_summary, on="route_id", how="left")
                if "event_on_time" in output:
                    output["on_time_pct"] = output["event_on_time"].mul(100).where(
                        output["event_on_time"].notna(), output["on_time_pct"]
                    )
                if "detention_minutes" in output:
                    output["transit_hours"] += output["detention_minutes"].fillna(0) / 60

    output["on_time_pct"] = output["on_time_pct"].fillna(90).clip(1, 100)
    output["reliability_weight"] = -np.log(output["on_time_pct"] / 100)

    def minmax(series: pd.Series) -> pd.Series:
        minimum, maximum = series.min(), series.max()
        if pd.isna(minimum) or pd.isna(maximum) or minimum == maximum:
            return pd.Series(0.0, index=series.index)
        return (series - minimum) / (maximum - minimum)

    output["balanced_weight"] = (
        0.30 * minmax(output["distance_miles"])
        + 0.30 * minmax(output["transit_hours"])
        + 0.25 * minmax(output["estimated_cost_usd"])
        + 0.15 * minmax(output["reliability_weight"])
        + 0.000001
    )
    return output


# def _route_graph(routes: pd.DataFrame) -> dict[str, list[dict]]:
#     graph: dict[str, list[dict]] = defaultdict(list)
#     existing_pairs = set(zip(routes["origin_facility_id"], routes["destination_facility_id"]))

#     for edge in routes.to_dict("records"):
#         graph[edge["origin_facility_id"]].append(edge)
#         reverse_pair = (edge["destination_facility_id"], edge["origin_facility_id"])
#         if reverse_pair not in existing_pairs:
#             reverse = edge.copy()
#             reverse["origin_facility_id"] = edge["destination_facility_id"]
#             reverse["destination_facility_id"] = edge["origin_facility_id"]
#             reverse["route_id"] = f"{edge['route_id']} · reverse"
#             graph[reverse["origin_facility_id"]].append(reverse)
#     return graph

def _route_graph(routes: pd.DataFrame) -> dict[str, list[dict]]:
    """Build a directed graph using only actual lanes from routes.csv."""
    graph: dict[str, list[dict]] = defaultdict(list)

    for edge in routes.to_dict("records"):
        graph[edge["origin_facility_id"]].append(edge)

    return graph

def _shortest_path(
    graph: dict[str, list[dict]],
    origin: str,
    destination: str,
    weight_column: str,
) -> list[dict] | None:
    queue: list[tuple[float, str, list[dict]]] = [(0.0, origin, [])]
    best = {origin: 0.0}

    while queue:
        cost, node, path = heapq.heappop(queue)
        if node == destination:
            return path
        if cost > best.get(node, float("inf")):
            continue

        for edge in graph.get(node, []):
            weight = float(edge.get(weight_column, np.nan))
            if not np.isfinite(weight) or weight < 0:
                continue
            next_node = edge["destination_facility_id"]
            next_cost = cost + weight
            if next_cost < best.get(next_node, float("inf")):
                best[next_node] = next_cost
                heapq.heappush(queue, (next_cost, next_node, path + [edge]))
    return None


def _path_summary(path: list[dict]) -> dict[str, object]:
    facility_sequence = [path[0]["origin_facility_id"]] + [
        edge["destination_facility_id"] for edge in path
    ]
    return {
        "segments": len(path),
        "distance_miles": float(sum(edge["distance_miles"] for edge in path)),
        "transit_hours": float(sum(edge["transit_hours"] for edge in path)),
        "estimated_cost_usd": float(sum(edge["estimated_cost_usd"] for edge in path)),
        "on_time_pct": float(np.prod([edge["on_time_pct"] / 100 for edge in path]) * 100),
        "facility_sequence": facility_sequence,
    }


def _route_map(path: list[dict], facilities: pd.DataFrame) -> go.Figure:
    sequence = _path_summary(path)["facility_sequence"]
    points = facilities.set_index("facility_id").reindex(sequence).reset_index()
    points["label"] = points.apply(
        lambda row: f"{row['facility_name']}<br>{row['city']}, {row['state']}", axis=1
    )

    figure = go.Figure(
        go.Scattermap(
            lat=points["latitude"],
            lon=points["longitude"],
            mode="lines+markers",
            line={"width": 4},
            marker={"size": 12},
            text=points["label"],
            hovertemplate="%{text}<extra></extra>",
        )
    )
    figure.update_layout(
        map={
            "style": "open-street-map",
            "center": {
                "lat": float(points["latitude"].mean()),
                "lon": float(points["longitude"].mean()),
            },
            "zoom": 3,
        },
        height=520,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        showlegend=False,
    )
    return figure

def _available_route_objectives(
    routes: pd.DataFrame,
) -> dict[str, str]:
    """
    Return only optimization objectives supported by meaningful,
    non-equivalent route values.
    """

    available: dict[str, str] = {
        "Shortest distance": "distance_miles",
    }

    distance = pd.to_numeric(
        routes["distance_miles"],
        errors="coerce",
    )

    transit = pd.to_numeric(
        routes["transit_hours"],
        errors="coerce",
    )

    cost = pd.to_numeric(
        routes["estimated_cost_usd"],
        errors="coerce",
    )

    reliability = pd.to_numeric(
        routes["on_time_pct"],
        errors="coerce",
    )

    valid_distance = distance.gt(0)

    # Time is useful only when speed differs meaningfully between lanes.
    effective_speed = distance / transit.replace(0, np.nan)
    speed_variation = (
        effective_speed[valid_distance].std()
        / effective_speed[valid_distance].mean()
        if effective_speed[valid_distance].notna().sum() > 1
        else 0
    )

    has_meaningful_time = (
        transit.nunique(dropna=True) > 1
        and np.isfinite(speed_variation)
        and speed_variation > 0.05
    )

    if has_meaningful_time:
        available["Fastest time"] = "transit_hours"

    # Cost is useful only when cost per mile varies across lanes.
    cost_per_mile = cost / distance.replace(0, np.nan)
    cost_variation = (
        cost_per_mile[valid_distance].std()
        / cost_per_mile[valid_distance].mean()
        if cost_per_mile[valid_distance].notna().sum() > 1
        else 0
    )

    has_meaningful_cost = (
        cost.nunique(dropna=True) > 1
        and np.isfinite(cost_variation)
        and cost_variation > 0.05
    )

    if has_meaningful_cost:
        available["Lowest cost"] = "estimated_cost_usd"

    # Reliability is useful only when lanes have different values.
    has_meaningful_reliability = (
        reliability.nunique(dropna=True) > 1
        and reliability.std(skipna=True) > 0.5
    )

    if has_meaningful_reliability:
        available["Most reliable"] = "reliability_weight"

    # Balanced is meaningful only when at least two independent
    # non-distance criteria are available.
    independent_criteria = sum(
        [
            has_meaningful_time,
            has_meaningful_cost,
            has_meaningful_reliability,
        ]
    )

    if independent_criteria >= 2:
        available = {
            "Balanced": "balanced_weight",
            **available,
        }

    return available


def _route_metric_diagnostics(
    routes: pd.DataFrame,
) -> pd.DataFrame:
    distance = pd.to_numeric(
        routes["distance_miles"],
        errors="coerce",
    )

    transit = pd.to_numeric(
        routes["transit_hours"],
        errors="coerce",
    )

    cost = pd.to_numeric(
        routes["estimated_cost_usd"],
        errors="coerce",
    )

    reliability = pd.to_numeric(
        routes["on_time_pct"],
        errors="coerce",
    )

    speed = distance / transit.replace(0, np.nan)
    cost_per_mile = cost / distance.replace(0, np.nan)

    return pd.DataFrame(
        [
            {
                "metric": "Distance",
                "unique_values": distance.nunique(),
                "minimum": distance.min(),
                "maximum": distance.max(),
                "variation": distance.std(),
                "interpretation": "Always available",
            },
            {
                "metric": "Effective speed",
                "unique_values": speed.round(2).nunique(),
                "minimum": speed.min(),
                "maximum": speed.max(),
                "variation": speed.std(),
                "interpretation": (
                    "Fastest differs from shortest only when "
                    "effective speeds vary"
                ),
            },
            {
                "metric": "Cost per mile",
                "unique_values": cost_per_mile.round(2).nunique(),
                "minimum": cost_per_mile.min(),
                "maximum": cost_per_mile.max(),
                "variation": cost_per_mile.std(),
                "interpretation": (
                    "Lowest cost differs from shortest only when "
                    "cost per mile varies"
                ),
            },
            {
                "metric": "On-time percentage",
                "unique_values": reliability.round(2).nunique(),
                "minimum": reliability.min(),
                "maximum": reliability.max(),
                "variation": reliability.std(),
                "interpretation": (
                    "Most reliable requires differing reliability values"
                ),
            },
        ]
    )



def _city_facility_map(
    facilities: pd.DataFrame,
    routes: pd.DataFrame,
    allowed_cities: list[str],
) -> dict[str, str]:
    """
    Map each permitted city to its most connected facility.

    This keeps the dropdown city-based while still supplying a facility_id
    to the route optimizer.
    """
    connection_counts = pd.concat(
        [
            routes["origin_facility_id"],
            routes["destination_facility_id"],
        ],
        ignore_index=True,
    ).value_counts()

    candidates = facilities.copy()
    candidates["_city_key"] = candidates["city"].map(normalise_name)
    candidates["_connections"] = (
        candidates["facility_id"].map(connection_counts).fillna(0)
    )

    result: dict[str, str] = {}

    for city in allowed_cities:
        matches = candidates[
            candidates["_city_key"] == normalise_name(city)
        ].copy()

        if matches.empty:
            continue

        matches = matches.sort_values(
            ["_connections", "facility_name"],
            ascending=[False, True],
        )

        result[city] = str(matches.iloc[0]["facility_id"])

    return result


def _find_direct_path(
    routes: pd.DataFrame,
    origin: str,
    destination: str,
    weight_column: str,
) -> list[dict] | None:
    """
    Return the best actual one-lane route between the selected facilities.

    No reverse route is manufactured here.
    """
    direct = routes[
        (routes["origin_facility_id"] == origin)
        & (routes["destination_facility_id"] == destination)
    ].copy()

    if direct.empty:
        return None

    direct[weight_column] = pd.to_numeric(
        direct[weight_column],
        errors="coerce",
    )

    direct = direct.dropna(subset=[weight_column])

    if direct.empty:
        return None

    best_edge = direct.sort_values(weight_column).iloc[0].to_dict()
    return [best_edge]


def _route_segments_frame(
    path: list[dict],
    labels: dict[str, str],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "segment": index,
                "route_id": edge["route_id"],
                "origin": labels.get(
                    edge["origin_facility_id"],
                    edge["origin_facility_id"],
                ),
                "destination": labels.get(
                    edge["destination_facility_id"],
                    edge["destination_facility_id"],
                ),
                "distance_miles": round(
                    float(edge["distance_miles"]),
                    2,
                ),
                "transit_hours": round(
                    float(edge["transit_hours"]),
                    2,
                ),
                "estimated_cost_usd": round(
                    float(edge["estimated_cost_usd"]),
                    2,
                ),
                "on_time_pct": round(
                    float(edge["on_time_pct"]),
                    2,
                ),
                "historical_trips": int(
                    edge.get("trip_count", 0) or 0
                ),
                "historical_loads": int(
                    edge.get("load_count", 0) or 0
                ),
            }
            for index, edge in enumerate(path, start=1)
        ]
    )


def _comparison_row(
    route_type: str,
    path: list[dict],
    labels: dict[str, str],
) -> dict[str, object]:
    summary = _path_summary(path)

    route_names = [
        labels.get(facility_id, facility_id).split(" — ")[0]
        for facility_id in summary["facility_sequence"]
    ]

    return {
        "route_type": route_type,
        "segments": summary["segments"],
        "distance_miles": round(summary["distance_miles"], 2),
        "transit_hours": round(summary["transit_hours"], 2),
        "estimated_cost_usd": round(
            summary["estimated_cost_usd"],
            2,
        ),
        "estimated_on_time_pct": round(
            summary["on_time_pct"],
            2,
        ),
        "path": " → ".join(route_names),
    }

def _reachable_nodes(
    graph: dict[str, list[dict]],
    start: str,
) -> set[str]:
    """Return all facilities reachable from the starting facility."""
    visited = {start}
    stack = [start]

    while stack:
        current = stack.pop()

        for edge in graph.get(current, []):
            next_node = edge["destination_facility_id"]

            if next_node not in visited:
                visited.add(next_node)
                stack.append(next_node)

    return visited
    


def route_optimization_page() -> None:
    st.title("🗺️ Route Optimization")
    # st.caption(
    #     "Compare the direct logistics lane with the optimized route "
    #     "using actual lanes from routes.csv."
    # )

    facilities_raw, facilities_source = _load_logistics_csv(
        "facilities.csv"
    )
    routes_raw, routes_source = _load_logistics_csv("routes.csv")
    trips_raw, trips_source = _load_logistics_csv("trips.csv")
    loads_raw, loads_source = _load_logistics_csv("loads.csv")
    events_raw, events_source = _load_logistics_csv(
        "delivery_events.csv"
    )

    status = pd.DataFrame(
        [
            {
                "file": "facilities.csv",
                "required": "Yes",
                "status": (
                    "Loaded"
                    if facilities_raw is not None
                    else "Not found"
                ),
                "source": facilities_source,
            },
            {
                "file": "routes.csv",
                "required": "Yes",
                "status": (
                    "Loaded"
                    if routes_raw is not None
                    else "Not found"
                ),
                "source": routes_source,
            },
            {
                "file": "trips.csv",
                "required": "No",
                "status": (
                    "Loaded"
                    if trips_raw is not None
                    else "Not found"
                ),
                "source": trips_source,
            },
            {
                "file": "loads.csv",
                "required": "No",
                "status": (
                    "Loaded"
                    if loads_raw is not None
                    else "Not found"
                ),
                "source": loads_source,
            },
            {
                "file": "delivery_events.csv",
                "required": "No",
                "status": (
                    "Loaded"
                    if events_raw is not None
                    else "Not found"
                ),
                "source": events_source,
            },
        ]
    )

    # st.dataframe(
    #     status,
    #     width="stretch",
    #     hide_index=True,
    # )

    if facilities_raw is None or routes_raw is None:
        st.error(
            "facilities.csv and routes.csv are required under "
            "data/logistics/."
        )
        return

    try:
        facilities = _prepare_facilities(facilities_raw)
        routes = _prepare_routes(routes_raw, facilities)
        routes = _enrich_routes(
            routes,
            trips_raw,
            loads_raw,
            events_raw,
        )
    except Exception as exc:
        st.error(f"Could not prepare the route network: {exc}")
        return

    if routes.empty:
        st.error("No routes could be matched to facilities.csv.")
        return

    connected_ids = set(routes["origin_facility_id"]) | set(
        routes["destination_facility_id"]
    )

    facilities = facilities[
        facilities["facility_id"].isin(connected_ids)
    ].copy()

    labels = {
        row.facility_id: (
            f"{row.facility_name} — {row.city}, {row.state}"
            if row.city or row.state
            else row.facility_name
        )
        for row in facilities.itertuples(index=False)
    }

    origin_map = _city_facility_map(
        facilities,
        routes,
        ORIGIN_CITY_OPTIONS,
    )

    destination_map = _city_facility_map(
        facilities,
        routes,
        DESTINATION_CITY_OPTIONS,
    )

    # missing_origins = [
    #     city
    #     for city in ORIGIN_CITY_OPTIONS
    #     if city not in origin_map
    # ]

    # missing_destinations = [
    #     city
    #     for city in DESTINATION_CITY_OPTIONS
    #     if city not in destination_map
    # ]

    # if missing_origins:
    #     st.warning(
    #         "These origin cities were not matched in facilities.csv: "
    #         + ", ".join(missing_origins)
    #     )

    # if missing_destinations:
    #     st.warning(
    #         "These destination cities were not matched in facilities.csv: "
    #         + ", ".join(missing_destinations)
    #     )

    if not origin_map or not destination_map:
        st.error(
            "None of the configured origin or destination cities "
            "could be matched to facilities.csv."
        )
        return

    # metric_row(
    #     [
    #         ("Facilities", f"{len(facilities):,}", None),
    #         ("Route lanes", f"{len(routes):,}", None),
    #         (
    #             "Historical trips",
    #             f"{int(routes['trip_count'].sum()):,}",
    #             None,
    #         ),
    #         (
    #             "Historical loads",
    #             f"{int(routes['load_count'].sum()):,}",
    #             None,
    #         ),
    #     ]
    # )

    # left, right = st.columns(2)

    # origin_city = left.selectbox(
    #     "Origin city",
    #     list(origin_map.keys()),
    # )

    # destination_city = right.selectbox(
    #     "Destination city",
    #     list(destination_map.keys()),
    # )

    # origin = origin_map[origin_city]
    # destination = destination_map[destination_city]
    
    
    
    # Build the directed route network before showing the dropdowns.
    graph = _route_graph(routes)
    
    left, right = st.columns(2)
    
    origin_city = left.selectbox(
        "Origin city",
        list(origin_map.keys()),
        key="route_origin_city",
    )
    
    origin = origin_map[origin_city]
    
    # Determine which facilities can actually be reached from this origin.
    reachable_facilities = _reachable_nodes(
        graph,
        origin,
    )
    
    # Show only configured destinations that are reachable.
    reachable_destination_map = {
        city: facility_id
        for city, facility_id in destination_map.items()
        if facility_id in reachable_facilities
        and facility_id != origin
    }
    
    if not reachable_destination_map:
        right.selectbox(
            "Destination city",
            ["No reachable destinations"],
            disabled=True,
            key=f"route_destination_unavailable_{origin}",
        )
    
        st.warning(
            f"None of the configured destination cities can be reached "
            f"from {origin_city} using the lanes in routes.csv."
        )
        return
    
    destination_city = right.selectbox(
        "Destination city",
        list(reachable_destination_map.keys()),
        key="route_destination_city",
    )
    
    destination = reachable_destination_map[destination_city]


    # --------------------------------------------------------------
    
    available_objectives = _available_route_objectives(routes)
    
    objective = st.radio(
        "Optimization objective",
        list(available_objectives),
        horizontal=True,
    )
    
    with st.expander("Available route-metric values"):
        diagnostics = _route_metric_diagnostics(routes)
    
        numeric_columns = [
            "minimum",
            "maximum",
            "variation",
        ]
    
        diagnostics[numeric_columns] = diagnostics[
            numeric_columns
        ].round(2)
    
        st.dataframe(
            diagnostics,
            width="stretch",
            hide_index=True,
        )
    
        unavailable = [
            name
            for name in ROUTE_OBJECTIVES
            if name not in available_objectives
        ]
    
        if unavailable:
            st.info(
                "Hidden objectives because their values are missing, "
                "constant or effectively derived from distance: "
                + ", ".join(unavailable)
            )
    
    # objective = st.radio(
    #     "Optimization objective",
    #     list(ROUTE_OBJECTIVES),
    #     horizontal=True,
    # )

    if st.button(
        "Compare routes",
        type="primary",
        width="stretch",
    ):
        # weight_column = ROUTE_OBJECTIVES[objective]
        weight_column = available_objectives[objective]

        # graph = _route_graph(routes)

        suggested_path = _shortest_path(
            graph,
            origin,
            destination,
            weight_column,
        )

        direct_path = _find_direct_path(
            routes,
            origin,
            destination,
            weight_column,
        )

        # Safety rule: the direct route must remain available to the
        # shortest-distance optimizer whenever a direct lane exists.
        if (
            objective == "Shortest distance"
            and direct_path is not None
            and suggested_path is not None
        ):
            direct_distance = _path_summary(
                direct_path
            )["distance_miles"]

            suggested_distance = _path_summary(
                suggested_path
            )["distance_miles"]

            if direct_distance < suggested_distance:
                suggested_path = direct_path

        if suggested_path is None:
            st.session_state.pop(
                "route_optimization_result",
                None,
            )
            st.error(
                "No connected route was found between the selected cities."
            )
        else:
            st.session_state["route_optimization_result"] = {
                "origin_city": origin_city,
                "destination_city": destination_city,
                "origin": origin,
                "destination": destination,
                "objective": objective,
                "suggested_path": suggested_path,
                "direct_path": direct_path,
                "facilities": facilities,
                "labels": labels,
            }

    result = st.session_state.get(
        "route_optimization_result"
    )

    if not result:
        st.info(
            "Choose the origin, destination and objective, "
            "then select **Compare routes**."
        )
        return

    if (
        result["origin_city"] != origin_city
        or result["destination_city"] != destination_city
        or result["objective"] != objective
    ):
        st.info(
            "Select **Compare routes** to calculate the new selection."
        )
        return

    suggested_path = result["suggested_path"]
    direct_path = result["direct_path"]

    comparison_rows = [
        _comparison_row(
            "Suggested route",
            suggested_path,
            labels,
        )
    ]

    if direct_path is not None:
        comparison_rows.insert(
            0,
            _comparison_row(
                "Direct route",
                direct_path,
                labels,
            ),
        )

    st.subheader("Direct route vs suggested route")

    comparison = pd.DataFrame(comparison_rows)

    st.dataframe(
        comparison,
        width="stretch",
        hide_index=True,
    )

    if direct_path is None:
        st.warning(
            f"No direct {origin_city} → {destination_city} lane "
            "exists in routes.csv. The suggested route therefore "
            "uses connected intermediate facilities."
        )
    else:
        direct_summary = _path_summary(direct_path)
        suggested_summary = _path_summary(suggested_path)

        same_sequence = (
            direct_summary["facility_sequence"]
            == suggested_summary["facility_sequence"]
        )

        if same_sequence:
            st.success(
                "The optimizer selected the direct route."
            )
        elif objective == "Shortest distance":
            st.warning(
                "The dataset reports that a multi-stop route is "
                "shorter than the direct lane. Review the distance "
                "values shown in the comparison table."
            )

    suggested_tab, direct_tab = st.tabs(
        [
            "Suggested route",
            "Direct route",
        ]
    )

    with suggested_tab:
        suggested_summary = _path_summary(suggested_path)

        metric_row(
            [
                (
                    "Distance",
                    f"{suggested_summary['distance_miles']:,.0f} mi",
                    None,
                ),
                (
                    "Transit time",
                    f"{suggested_summary['transit_hours']:,.1f} hr",
                    None,
                ),
                (
                    "Estimated cost",
                    f"${suggested_summary['estimated_cost_usd']:,.0f}",
                    None,
                ),
                (
                    "On-time probability",
                    f"{suggested_summary['on_time_pct']:.1f}%",
                    None,
                ),
            ]
        )

        
        st.plotly_chart(
            _route_map(
                suggested_path,
                result["facilities"],
            ),
            width="stretch",
            key=f"suggested_route_map_{origin}_{destination}_{objective}",
        )

        
        # st.plotly_chart(
        #     _route_map(
        #         suggested_path,
        #         result["facilities"],
        #     ),
        #     width="stretch",
        # )

        suggested_segments = _route_segments_frame(
            suggested_path,
            labels,
        )

        st.dataframe(
            suggested_segments,
            width="stretch",
            hide_index=True,
        )

    with direct_tab:
        if direct_path is None:
            st.info(
                "No direct lane exists for this city pair."
            )
        else:
            direct_summary = _path_summary(direct_path)

            metric_row(
                [
                    (
                        "Distance",
                        f"{direct_summary['distance_miles']:,.0f} mi",
                        None,
                    ),
                    (
                        "Transit time",
                        f"{direct_summary['transit_hours']:,.1f} hr",
                        None,
                    ),
                    (
                        "Estimated cost",
                        f"${direct_summary['estimated_cost_usd']:,.0f}",
                        None,
                    ),
                    (
                        "On-time probability",
                        f"{direct_summary['on_time_pct']:.1f}%",
                        None,
                    ),
                ]
            )

            
            st.plotly_chart(
                _route_map(
                    direct_path,
                    result["facilities"],
                ),
                width="stretch",
                key=f"direct_route_map_{origin}_{destination}_{objective}",
            )

            
            # st.plotly_chart(
            #     _route_map(
            #         direct_path,
            #         result["facilities"],
            #     ),
            #     width="stretch",
            # )

            direct_segments = _route_segments_frame(
                direct_path,
                labels,
            )

            st.dataframe(
                direct_segments,
                width="stretch",
                hide_index=True,
            )

    suggested_segments = _route_segments_frame(
        suggested_path,
        labels,
    )

    st.download_button(
        "Download suggested route CSV",
        data=suggested_segments.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="suggested_route.csv",
        mime="text/csv",
        width="stretch",
    )

    
# def route_optimization_page() -> None:
#     st.title("🗺️ Route Optimization")
#     st.caption(
#         "Optimize facility-to-facility lanes by distance, transit time, operating cost or historical reliability."
#     )

#     facilities_raw, facilities_source = _load_logistics_csv("facilities.csv")
#     routes_raw, routes_source = _load_logistics_csv("routes.csv")
#     trips_raw, trips_source = _load_logistics_csv("trips.csv")
#     loads_raw, loads_source = _load_logistics_csv("loads.csv")
#     events_raw, events_source = _load_logistics_csv("delivery_events.csv")

#     status = pd.DataFrame(
#         [
#             {"file": "facilities.csv", "required": "Yes", "status": "Loaded" if facilities_raw is not None else "Not found", "source": facilities_source},
#             {"file": "routes.csv", "required": "Yes", "status": "Loaded" if routes_raw is not None else "Not found", "source": routes_source},
#             {"file": "trips.csv", "required": "No", "status": "Loaded" if trips_raw is not None else "Not found", "source": trips_source},
#             {"file": "loads.csv", "required": "No", "status": "Loaded" if loads_raw is not None else "Not found", "source": loads_source},
#             {"file": "delivery_events.csv", "required": "No", "status": "Loaded" if events_raw is not None else "Not found", "source": events_source},
#         ]
#     )
#     st.dataframe(status, width="stretch", hide_index=True)

#     if facilities_raw is None or routes_raw is None:
#         st.error(
#             "Upload facilities.csv and routes.csv under data/logistics/. "
#             "trips.csv, loads.csv and delivery_events.csv are optional."
#         )
#         return

#     try:
#         facilities = _prepare_facilities(facilities_raw)
#         routes = _prepare_routes(routes_raw, facilities)
#         routes = _enrich_routes(routes, trips_raw, loads_raw, events_raw)
#     except Exception as exc:
#         st.error(f"Could not prepare the route network: {exc}")
#         return

#     if routes.empty:
#         st.error("No routes could be matched to facilities.csv.")
#         return

#     connected_ids = sorted(
#         set(routes["origin_facility_id"]) | set(routes["destination_facility_id"])
#     )
#     facilities = facilities[facilities["facility_id"].isin(connected_ids)].copy()
#     labels = {
#         row.facility_id: (
#             f"{row.facility_name} — {row.city}, {row.state}"
#             if row.city or row.state
#             else row.facility_name
#         )
#         for row in facilities.itertuples(index=False)
#     }

#     metric_row(
#         [
#             ("Facilities", f"{len(facilities):,}", None),
#             ("Route lanes", f"{len(routes):,}", None),
#             ("Historical trips", f"{int(routes['trip_count'].sum()):,}", None),
#             ("Historical loads", f"{int(routes['load_count'].sum()):,}", None),
#         ]
#     )

#     left, right = st.columns(2)
#     origin = left.selectbox(
#         "Origin facility",
#         connected_ids,
#         format_func=lambda facility_id: labels.get(facility_id, facility_id),
#     )
#     destination = right.selectbox(
#         "Destination facility",
#         [facility_id for facility_id in connected_ids if facility_id != origin],
#         format_func=lambda facility_id: labels.get(facility_id, facility_id),
#     )
#     objective = st.radio(
#         "Optimization objective",
#         list(ROUTE_OBJECTIVES),
#         horizontal=True,
#     )

#     if st.button("Optimize route", type="primary", width="stretch"):
#         graph = _route_graph(routes)
#         alternatives = []
#         selected_path = None
#         seen_paths: set[tuple[str, ...]] = set()

#         for objective_name, weight_column in ROUTE_OBJECTIVES.items():
#             path = _shortest_path(graph, origin, destination, weight_column)
#             if path is None:
#                 continue
#             summary = _path_summary(path)
#             sequence = tuple(summary["facility_sequence"])
#             if objective_name == objective:
#                 selected_path = path
#             if sequence in seen_paths:
#                 continue
#             seen_paths.add(sequence)
#             alternatives.append(
#                 {
#                     "objective": objective_name,
#                     "segments": summary["segments"],
#                     "distance_miles": summary["distance_miles"],
#                     "transit_hours": summary["transit_hours"],
#                     "estimated_cost_usd": summary["estimated_cost_usd"],
#                     "estimated_on_time_pct": summary["on_time_pct"],
#                     "path": " → ".join(
#                         labels.get(facility_id, facility_id).split(" — ")[0]
#                         for facility_id in summary["facility_sequence"]
#                     ),
#                 }
#             )

#         if selected_path is None:
#             st.session_state.pop("route_optimization_result", None)
#             st.error("No connected path was found between the selected facilities.")
#         else:
#             st.session_state["route_optimization_result"] = {
#                 "origin": origin,
#                 "destination": destination,
#                 "objective": objective,
#                 "path": selected_path,
#                 "alternatives": pd.DataFrame(alternatives),
#                 "facilities": facilities,
#                 "labels": labels,
#             }

#     result = st.session_state.get("route_optimization_result")
#     if not result:
#         st.info("Select the route details and click **Optimize route**.")
#         return
#     if result["origin"] != origin or result["destination"] != destination:
#         st.info("Click **Optimize route** to calculate the newly selected route.")
#         return

#     path = result["path"]
#     summary = _path_summary(path)

#     st.subheader(f"Recommended route — {result['objective']}")
#     metric_row(
#         [
#             ("Distance", f"{summary['distance_miles']:,.0f} mi", None),
#             ("Transit time", f"{summary['transit_hours']:,.1f} hr", None),
#             ("Estimated cost", f"${summary['estimated_cost_usd']:,.0f}", None),
#             ("On-time probability", f"{summary['on_time_pct']:.1f}%", "Combined segment reliability"),
#         ]
#     )
#     st.plotly_chart(_route_map(path, result["facilities"]), width="stretch")

#     segments = pd.DataFrame(
#         [
#             {
#                 "segment": index,
#                 "route_id": edge["route_id"],
#                 "origin": result["labels"].get(edge["origin_facility_id"], edge["origin_facility_id"]),
#                 "destination": result["labels"].get(edge["destination_facility_id"], edge["destination_facility_id"]),
#                 "distance_miles": round(edge["distance_miles"], 1),
#                 "transit_hours": round(edge["transit_hours"], 2),
#                 "estimated_cost_usd": round(edge["estimated_cost_usd"], 2),
#                 "on_time_pct": round(edge["on_time_pct"], 1),
#                 "historical_trips": int(edge.get("trip_count", 0)),
#                 "historical_loads": int(edge.get("load_count", 0)),
#             }
#             for index, edge in enumerate(path, start=1)
#         ]
#     )

#     st.subheader("Route segments")
#     st.dataframe(segments, width="stretch", hide_index=True)

#     alternatives = result["alternatives"].copy()
#     if not alternatives.empty:
#         for column in [
#             "distance_miles",
#             "transit_hours",
#             "estimated_cost_usd",
#             "estimated_on_time_pct",
#         ]:
#             alternatives[column] = alternatives[column].round(2)
#         st.subheader("Alternative routes")
#         st.dataframe(alternatives, width="stretch", hide_index=True)

#     st.download_button(
#         "Download optimized route CSV",
#         data=segments.to_csv(index=False).encode("utf-8"),
#         file_name="optimized_route.csv",
#         mime="text/csv",
#         width="stretch",
#     )

#     st.caption(
#         "This is strategic lane-network optimization using routes.csv, not live turn-by-turn navigation."
#     )




# def main() -> None:
#     st.title("📈 Retail Demand Forecasting")


def demand_forecasting_page() -> None:
    st.title("📈 Retail Demand Forecasting")
    st.caption(
        "Forecast future weekly sales from the CSV files stored in the repository-root data/ folder. "
        "sales.csv is the historical demand dataset; stores.csv and features.csv are optional enrichments."
    )
    # Fixed Demand Forecasting defaults
    
    validation_weeks = 12
    horizon = 12
    allow_row_alignment = False
    
    
    # with st.sidebar:
    #     st.header("Dataset location")
    #     st.code("data/sales.csv\ndata/stores.csv\ndata/features.csv", language=None)
    #     st.caption(f"Resolved folder: {DATA_DIR}")

    #     st.divider()
    #     validation_weeks = st.slider(
    #         "Validation window (weeks)", 4, 26, 12,
    #         help="The final historical weeks are held out for chronological testing.",
    #     )
    #     horizon = st.slider(
    #         "Forecast horizon (weeks)", 4, 52, 12,
    #         help="Number of future weeks to forecast.",
    #     )
    
    #     with st.expander("Advanced feature alignment"):
    #         allow_row_alignment = st.checkbox(
    #             "Align features.csv by row order",
    #             value=False,
    #             help=(
    #                 "Enable only when features.csv has exactly one row for every sales.csv row "
    #                 "and both files are in precisely the same row order. A date/store_id join is safer."
    #             ),
    #         )
    
    # Files are loaded automatically from the repository-root data/ directory.
    sales, sales_source = load_dataset(None, "sales.csv")
    stores, stores_source = load_dataset(None, "stores.csv")
    features, features_source = load_dataset(None, "features.csv")

    status_rows = []
    for filename, required, frame, source in [
        ("sales.csv", "Yes", sales, sales_source),
        ("stores.csv", "No", stores, stores_source),
        ("features.csv", "No", features, features_source),
    ]:
        status_rows.append({
            "file": filename,
            "required": required,
            "status": "Loaded" if frame is not None else "Not found",
            "rows": 0 if frame is None else len(frame),
            "source": source,
        })

    # st.subheader("Dataset status")
    # # st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)
    # st.dataframe(pd.DataFrame(status_rows), width="stretch", hide_index=True)

    if sales is None:
        st.error(
            "sales.csv was not found. Upload it in the Hugging Face Files tab at data/sales.csv. "
            "Required columns: store_id, department, date, weekly_sales, is_holiday."
        )
        return

    if not validate_columns(sales, {"store", "department", "date", "weekly_sales"}, "sales.csv"):
        return

    sales = clean_sales(sales)
    if stores is not None:
        if validate_columns(stores, {"store"}, "stores.csv"):
            stores = clean_stores(stores)
        else:
            stores = None

    departments = sorted(sales["department"].dropna().astype(str).unique().tolist())
    selected_department = st.selectbox(
        "Forecast scope",
        ["All departments"] + departments,
        help=(
            "All departments forecasts total demand for each store. Selecting one department "
            "forecasts only that department."
        ),
    )

    history, feature_status = prepare_history(
        sales,
        stores,
        features,
        selected_department,
        allow_row_alignment,
    )

    # if features is not None:
    #     st.info(f"features.csv handling: {feature_status}")
    #     if not ({"store", "date"}.issubset(features.columns) or "date" in features.columns):
    #         st.warning(
    #             "features.csv has no store_id or date, so it cannot be joined using business keys. "
    #             "The forecast will still use sales.csv correctly. Enable row-order alignment only when "
    #             "the two files have identical row counts and identical row ordering."
    #         )

    if history.empty:
        st.error("No usable sales rows remain for the selected forecast scope.")
        return

    unique_weeks = history["date"].nunique()
    metric_row(
        [
            ("Historical rows", f"{len(history):,}", "One row per store and week"),
            ("Stores", f"{history['store'].nunique():,}", None),
            ("Historical weeks", f"{unique_weeks:,}", None),
            ("Date range", f"{history['date'].min():%Y-%m-%d} → {history['date'].max():%Y-%m-%d}", None),
        ]
    )

    overview = history.groupby("date", as_index=False)["weekly_sales"].sum()
    st.plotly_chart(
        px.line(overview, x="date", y="weekly_sales", markers=True, title="Historical weekly demand"),
        width="stretch",
    )

    if st.button("Train model and forecast", type="primary", width="stretch"):
        try:
            with st.spinner("Training the demand model and generating forecasts..."):
                _, metrics, validation, feature_columns, category_maps = train_backtest(
                    history,
                    validation_weeks,
                )
                final_model = fit_final_model(history, feature_columns, category_maps)
                forecast = recursive_forecast(
                    final_model,
                    history,
                    horizon,
                    feature_columns,
                    category_maps,
                )

            st.session_state["demand_results"] = {
                "scope": selected_department,
                "metrics": metrics,
                "validation": validation,
                "forecast": forecast,
                "history": history,
                "feature_columns": feature_columns,
            }
        except Exception as exc:
            st.error(f"Model training failed: {exc}")

    results = st.session_state.get("demand_results")
    if not results or results.get("scope") != selected_department:
        st.info("Select **Train model and forecast** to generate the forecast.")
        return

    metrics = results["metrics"]
    validation = results["validation"]
    forecast = results["forecast"]
    result_history = results["history"]

    st.subheader("Backtest performance")
    metric_row(
        [
            ("MAE", f"{metrics['MAE']:,.0f}", "Average absolute store-week error"),
            ("RMSE", f"{metrics['RMSE']:,.0f}", "Penalises larger errors"),
            ("Holiday WMAE", f"{metrics['WMAE']:,.0f}", "Holiday errors receive 5× weight"),
            ("R²", f"{metrics['R2']:.3f}", "Closer to 1 is better"),
        ]
    )

    validation_weekly = (
        validation.groupby("date", as_index=False)[["weekly_sales", "predicted_weekly_sales"]]
        .sum()
        .rename(columns={"weekly_sales": "Actual", "predicted_weekly_sales": "Predicted"})
    )
    validation_chart = validation_weekly.melt("date", var_name="Series", value_name="Weekly sales")
    st.plotly_chart(
        px.line(
            validation_chart,
            x="date",
            y="Weekly sales",
            color="Series",
            markers=True,
            title="Chronological validation: actual versus predicted",
        ),
        width="stretch",
    )

    st.subheader("Demand forecast")
    stores_available = sorted(forecast["store"].dropna().astype(int).unique().tolist())
    selected_store = st.selectbox("View forecast", ["All stores"] + [str(value) for value in stores_available])

    if selected_store == "All stores":
        historical_view = result_history.groupby("date", as_index=False)["weekly_sales"].sum()
        forecast_view = forecast.groupby("date", as_index=False)["forecast_weekly_sales"].sum()
        chart_title = "Total weekly demand forecast"
    else:
        store_id = int(selected_store)
        historical_view = result_history[result_history["store"] == store_id][["date", "weekly_sales"]]
        forecast_view = forecast[forecast["store"] == store_id][["date", "forecast_weekly_sales"]]
        chart_title = f"Weekly demand forecast — Store {store_id}"

    historical_view = historical_view.tail(52).rename(columns={"weekly_sales": "Actual"})
    forecast_view = forecast_view.rename(columns={"forecast_weekly_sales": "Forecast"})
    chart_frame = pd.concat(
        [
            historical_view.set_index("date")[["Actual"]],
            forecast_view.set_index("date")[["Forecast"]],
        ],
        axis=0,
    ).reset_index()
    chart_frame = chart_frame.melt("date", var_name="Series", value_name="Weekly sales").dropna()

    st.plotly_chart(
        px.line(chart_frame, x="date", y="Weekly sales", color="Series", markers=True, title=chart_title),
        width="stretch",
    )

    forecast_total = forecast["forecast_weekly_sales"].sum()
    weekly_totals = forecast.groupby("date", as_index=False)["forecast_weekly_sales"].sum()
    peak = weekly_totals.nlargest(1, "forecast_weekly_sales").iloc[0]
    metric_row(
        [
            ("Forecast total", f"{forecast_total:,.0f}", None),
            # ("Average week", f"{weekly_totals['forecast_weekly_sales'].mean():,.0f}", None),
            ("Peak week", f"{pd.Timestamp(peak['date']):%Y-%m-%d}", f"{peak['forecast_weekly_sales']:,.0f}"),
            ("Forecast weeks", f"{forecast['date'].nunique():,}", None),
        ]
    )

    display_forecast = forecast.copy()
    display_forecast["date"] = pd.to_datetime(display_forecast["date"]).dt.strftime("%Y-%m-%d")
    st.dataframe(display_forecast, width="stretch", hide_index=True)
    st.download_button(
        "Download demand forecast CSV",
        data=display_forecast.to_csv(index=False).encode("utf-8"),
        file_name="demand_forecast.csv",
        mime="text/csv",
        width="stretch",
    )

    with st.expander("How this forecast works"):
        st.write(
            "The model learns from historical weekly sales, store identifiers, calendar seasonality, "
            "lagged sales and rolling averages. stores.csv adds store metadata. features.csv is optional "
            "and is used only when it can be aligned safely."
        )
        st.write("Model inputs:", ", ".join(results["feature_columns"]))

def main() -> None:
    selected_page = st.sidebar.radio(
        "Page",
        [
            "Demand Forecasting",
            "Route Optimization",
        ],
        label_visibility="collapsed",
        key="main_page_navigation",
    )

    if selected_page == "Demand Forecasting":
        demand_forecasting_page()

    elif selected_page == "Route Optimization":
        route_optimization_page()

        
if __name__ == "__main__":
    main()

