"""
Predictive Analytics Using Historical Data
===========================================
Pipeline:
  1. Load raw (messy) historical data
  2. Clean & preprocess (parse dates, dedupe, handle missing values, cap outliers,
     fill gaps to a continuous daily series)
  3. Feature engineering (calendar features, lags, rolling stats)
  4. Time-based train/test split (no shuffling -- this is a time series)
  5. Fit a seasonal-naive baseline, two regression models (scikit-learn), and a
     classical time-series model (Holt-Winters exponential smoothing, statsmodels)
  6. Evaluate every model on the held-out test window (MAE, RMSE, MAPE, R^2)
  7. Forecast the next 30 days with the best model and export everything

Run:
    python predictive_model.py
Requires: pandas, numpy, scikit-learn, statsmodels
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing

pd.set_option("display.width", 120)

# ----------------------------------------------------------------------
# 1. Load raw data
# ----------------------------------------------------------------------
raw = pd.read_csv("historical_sales_raw.csv")
print(f"Raw rows: {len(raw)}, nulls in sales: {raw['sales'].isna().sum()}, "
      f"duplicate rows: {raw.duplicated().sum()}")

# ----------------------------------------------------------------------
# 2. Clean & preprocess
# ----------------------------------------------------------------------
df = raw.copy()

# Parse mixed date formats (some rows are MM/DD/YYYY, some YYYY-MM-DD)
df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=False)

# Drop exact duplicate rows (same date+value double-reported)
before = len(df)
df = df.drop_duplicates()
print(f"Dropped {before - len(df)} exact duplicate rows")

# If the same date appears more than once with different values, keep the mean
df = df.groupby("date", as_index=False).agg({"sales": "mean", "promo_flag": "max"})

# Cap obvious data-entry errors using an IQR-based outlier rule rather than
# deleting rows outright (preserves the calendar and avoids amplifying gaps)
q1, q3 = df["sales"].quantile([0.25, 0.75])
iqr = q3 - q1
low, high = q1 - 3 * iqr, q3 + 3 * iqr
n_outliers = ((df["sales"] < low) | (df["sales"] > high)).sum()
df["sales"] = df["sales"].clip(lower=max(low, 0), upper=high)
# Negative sales are never valid regardless of the IQR bound
df.loc[df["sales"] < 0, "sales"] = np.nan
print(f"Capped/flagged {n_outliers} outlier values outside "
      f"[{max(low,0):.0f}, {high:.0f}]")

# Reindex to a continuous daily calendar so gaps (missing days) become
# explicit NaNs rather than silently shifting the seasonal pattern
full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
df = df.set_index("date").reindex(full_range)
df.index.name = "date"
n_gap_days = df["sales"].isna().sum()
print(f"Missing values after reindexing to a continuous calendar: {n_gap_days} "
      f"({n_gap_days / len(df):.1%} of days)")

# Fill missing values with time-aware interpolation (linear on a short series
# like this is a reasonable default; for longer gaps you'd want seasonal fill)
df["sales"] = df["sales"].interpolate(method="linear").bfill().ffill()
df["promo_flag"] = df["promo_flag"].fillna(0).astype(int)

print(f"\nClean series: {len(df)} days, {df.index.min().date()} to {df.index.max().date()}")

# ----------------------------------------------------------------------
# 3. Feature engineering
# ----------------------------------------------------------------------
df["day_index"] = np.arange(len(df))  # linear trend term
df["day_of_week"] = df.index.dayofweek
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["month"] = df.index.month
# cyclical encoding of day-of-year captures annual seasonality without
# creating an artificial jump between Dec 31 and Jan 1
doy = df.index.dayofyear
df["doy_sin"] = np.sin(2 * np.pi * doy / 365)
df["doy_cos"] = np.cos(2 * np.pi * doy / 365)
df["lag_7"] = df["sales"].shift(7)
df["lag_14"] = df["sales"].shift(14)
df["rolling_7_mean"] = df["sales"].shift(1).rolling(7).mean()
df["rolling_30_mean"] = df["sales"].shift(1).rolling(30).mean()
df = df.dropna()  # drop the first 30 rows that can't have a full rolling window

df.to_csv("historical_sales_clean.csv")

# ----------------------------------------------------------------------
# 4. Time-based train/test split (last 30 days held out -- never shuffle time series)
# ----------------------------------------------------------------------
TEST_DAYS = 30
train = df.iloc[:-TEST_DAYS]
test = df.iloc[-TEST_DAYS:]
print(f"\nTrain: {len(train)} days ({train.index.min().date()} to {train.index.max().date()})")
print(f"Test:  {len(test)} days ({test.index.min().date()} to {test.index.max().date()})")

FEATURES = ["day_index", "day_of_week", "is_weekend", "month", "doy_sin", "doy_cos",
            "lag_7", "lag_14", "rolling_7_mean", "rolling_30_mean", "promo_flag"]
X_train, y_train = train[FEATURES], train["sales"]
X_test, y_test = test[FEATURES], test["sales"]

# ----------------------------------------------------------------------
# 5. Models
# ----------------------------------------------------------------------
results = {}

# 5a. Seasonal-naive baseline: "today = same weekday last week"
naive_pred = test["lag_7"].values
results["Seasonal Naive"] = naive_pred

# 5b. Linear Regression
lin = LinearRegression()
lin.fit(X_train, y_train)
results["Linear Regression"] = lin.predict(X_test)

# 5c. Random Forest Regressor
rf = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
rf.fit(X_train, y_train)
results["Random Forest"] = rf.predict(X_test)

importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
importances.to_csv("feature_importance.csv", header=["importance"])
print("\nRandom Forest feature importances:")
print(importances.round(3))

# 5d. Holt-Winters Exponential Smoothing (classical time-series model,
# additive trend + weekly seasonality, fit on the raw series only -- no
# engineered features, this is the "time-series" approach vs. the
# "regression on features" approach above)
hw = ExponentialSmoothing(
    train["sales"], trend="add", seasonal="add", seasonal_periods=7
).fit()
results["Holt-Winters"] = hw.forecast(TEST_DAYS).values

# ----------------------------------------------------------------------
# 6. Evaluate
# ----------------------------------------------------------------------
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # MAPE is undefined when actual=0; a handful of near-zero days can occur
    # after cleaning (e.g. a promo/outlier day clipped near the floor), so
    # exclude those from the MAPE denominator rather than reporting inf.
    nonzero = y_true != 0
    mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

metrics = {name: evaluate(y_test.values, pred) for name, pred in results.items()}
metrics_df = pd.DataFrame(metrics).T.round(3)
metrics_df = metrics_df.sort_values("RMSE")
print("\nModel comparison on 30-day held-out test window:")
print(metrics_df)

best_model_name = metrics_df.index[0]
print(f"\nBest model by RMSE: {best_model_name}")

# Save per-day predictions from every model against actuals
pred_export = test[["sales"]].rename(columns={"sales": "actual"}).copy()
for name, pred in results.items():
    pred_export[name.replace(" ", "_")] = pred
pred_export.to_csv("test_predictions.csv")
metrics_df.to_csv("model_metrics.csv")

# ----------------------------------------------------------------------
# 7. Forecast the next 30 days with the best model
# ----------------------------------------------------------------------
FORECAST_DAYS = 30
future_dates = pd.date_range(df.index.max() + pd.Timedelta(days=1), periods=FORECAST_DAYS, freq="D")

# Refit the winning model on the FULL cleaned history (train+test) before
# forecasting genuinely unseen future days.
if best_model_name == "Holt-Winters":
    hw_full = ExponentialSmoothing(
        df["sales"], trend="add", seasonal="add", seasonal_periods=7
    ).fit()
    future_point = hw_full.forecast(FORECAST_DAYS).values
    resid_std = np.std(df["sales"].values[-90:] - hw_full.fittedvalues.values[-90:])
else:
    # Rebuild future feature rows iteratively since lag/rolling features
    # depend on previously forecast values
    history = df["sales"].copy()
    model = rf if best_model_name == "Random Forest" else lin
    if best_model_name != "Holt-Winters" and best_model_name not in ("Random Forest", "Linear Regression"):
        model = lin  # seasonal naive has no iterative model; fall back safely
    model.fit(df[FEATURES], df["sales"])  # refit on full history

    future_point = []
    day_idx = df["day_index"].iloc[-1]
    for d in future_dates:
        day_idx += 1
        doy = d.dayofyear
        row = {
            "day_index": day_idx,
            "day_of_week": d.dayofweek,
            "is_weekend": int(d.dayofweek >= 5),
            "month": d.month,
            "doy_sin": np.sin(2 * np.pi * doy / 365),
            "doy_cos": np.cos(2 * np.pi * doy / 365),
            "lag_7": history.iloc[-7],
            "lag_14": history.iloc[-14],
            "rolling_7_mean": history.iloc[-7:].mean(),
            "rolling_30_mean": history.iloc[-30:].mean(),
            "promo_flag": 0,
        }
        x_row = pd.DataFrame([row])[FEATURES]
        yhat = float(model.predict(x_row)[0])
        future_point.append(yhat)
        history.loc[d] = yhat  # feed forecast back in for next day's lags

    future_point = np.array(future_point)
    resid_std = np.std(y_test.values - results[best_model_name])

forecast_df = pd.DataFrame({
    "date": future_dates,
    "forecast": future_point,
    "lower_80": future_point - 1.28 * resid_std,
    "upper_80": future_point + 1.28 * resid_std,
})
forecast_df.to_csv("forecast_future.csv", index=False)

print(f"\nForecast saved for {FORECAST_DAYS} future days using {best_model_name}")
print(forecast_df.head())
