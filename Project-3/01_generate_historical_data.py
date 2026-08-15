"""
Generates 3 years of daily sales history with a genuine trend, weekly and
annual seasonality, promo spikes, and realistic data-quality problems
(missing days, duplicate rows, typo'd outliers, unsorted order) so the
predictive-modeling script has real cleaning work to do -- not just a
tidy CSV pretending to be raw data.
"""
import numpy as np
import pandas as pd
from datetime import date, timedelta

rng = np.random.default_rng(20260814)

N_DAYS = 3 * 365
start = date.today() - timedelta(days=N_DAYS - 1)
dates = [start + timedelta(days=i) for i in range(N_DAYS)]

rows = []
for i, d in enumerate(dates):
    day_of_year = d.timetuple().tm_yday
    # long-run trend: gentle linear growth
    trend = 400 + i * 0.28
    # annual seasonality: peak around late Nov (holiday season), trough in Feb
    annual = 90 * np.sin(2 * np.pi * (day_of_year - 305) / 365)
    # weekly seasonality: weekends higher for a retail-style business
    dow = d.weekday()  # 0=Mon
    weekly = {0: -20, 1: -15, 2: -10, 3: -5, 4: 15, 5: 45, 6: 30}[dow]
    # occasional promo days: random 3% of days, big spike
    promo = 1 if rng.random() < 0.03 else 0
    promo_effect = 220 * promo
    noise = rng.normal(0, 28)
    value = max(0, trend + annual + weekly + promo_effect + noise)
    rows.append({"date": d, "sales": round(value, 2), "promo_flag": promo})

df = pd.DataFrame(rows)

# ---- inject real-world messiness ----
# 1. Missing days entirely (simulate gaps in reporting) -- drop ~2.5% of rows
drop_idx = rng.choice(df.index, size=int(len(df) * 0.025), replace=False)
df = df.drop(index=drop_idx).reset_index(drop=True)

# 2. Some missing sales values (kept as NaN, reported but not filled)
nan_idx = rng.choice(df.index, size=int(len(df) * 0.02), replace=False)
df.loc[nan_idx, "sales"] = np.nan

# 3. A handful of duplicate rows (double-reported days)
dupe_rows = df.sample(n=15, random_state=7)
df = pd.concat([df, dupe_rows], ignore_index=True)

# 4. A few garbled outliers / data-entry errors (e.g. misplaced decimal, stray negative)
err_idx = rng.choice(df.index, size=10, replace=False)
for idx in err_idx:
    choice = rng.integers(0, 3)
    if choice == 0:
        df.loc[idx, "sales"] = df.loc[idx, "sales"] * 10  # decimal error
    elif choice == 1:
        df.loc[idx, "sales"] = -abs(df.loc[idx, "sales"])  # impossible negative
    else:
        df.loc[idx, "sales"] = df.loc[idx, "sales"] * 0.02  # near-zero glitch

# 5. Shuffle row order (real exports are rarely pre-sorted) and mix date string formats
df = df.sample(frac=1, random_state=3).reset_index(drop=True)
def fmt_date(d, i):
    # ~15% of rows use a different (still unambiguous) date format, common when
    # data is stitched together from multiple source systems
    if i % 7 == 0:
        return pd.Timestamp(d).strftime("%m/%d/%Y")
    return pd.Timestamp(d).strftime("%Y-%m-%d")
df["date"] = [fmt_date(d, i) for i, d in enumerate(df["date"])]

df.to_csv("/home/claude/forecast/historical_sales_raw.csv", index=False)
print(df.shape)
print(df.head())
print("nulls:", df["sales"].isna().sum())
