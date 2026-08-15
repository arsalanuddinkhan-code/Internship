"""
01_generate_data.py
--------------------
Generates a realistic synthetic e-commerce customer dataset combining
DEMOGRAPHICS + BEHAVIORAL (RFM-style) features for the segmentation project.

In a real internship setting, this file would be replaced by your company's
actual transaction/customer export. The generation logic here creates
believable, internally-consistent patterns (e.g. younger customers buy more
electronics online, high-income customers have higher AOV) so that the
clustering step produces meaningful, explainable segments.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 1200  # number of customers

# ---------------------------------------------------------------
# 1. DEMOGRAPHICS
# ---------------------------------------------------------------
customer_id = [f"CUST{100000+i}" for i in range(N)]
age = np.clip(np.random.normal(35, 11, N), 18, 70).astype(int)
gender = np.random.choice(["Male", "Female"], N, p=[0.52, 0.48])
city_tier = np.random.choice(["Tier 1", "Tier 2", "Tier 3"], N, p=[0.45, 0.35, 0.20])

# Annual income (correlated loosely with age, city tier)
base_income = np.random.normal(55000, 18000, N)
city_boost = np.where(city_tier == "Tier 1", 15000, np.where(city_tier == "Tier 2", 5000, 0))
annual_income = np.clip(base_income + city_boost + (age - 35) * 400, 15000, 180000).astype(int)

membership_years = np.clip(np.random.exponential(2.2, N), 0, 12).round(1)

# ---------------------------------------------------------------
# 2. BEHAVIORAL / RFM FEATURES
# ---------------------------------------------------------------
# Recency: days since last purchase (lower = more recently active)
recency_days = np.clip(np.random.exponential(45, N), 1, 365).astype(int)

# Purchase frequency (orders in last 12 months) - loosely tied to income & membership
frequency = np.clip(
    np.random.poisson(3 + membership_years * 1.1 + annual_income / 40000, N), 0, 60
)

# Average order value - tied to income
avg_order_value = np.clip(
    np.random.normal(20 + annual_income / 1200, 15, N), 8, 500
).round(2)

total_spend = np.round(frequency * avg_order_value * np.random.uniform(0.85, 1.15, N), 2)

preferred_category = np.random.choice(
    ["Electronics", "Fashion", "Groceries", "Home & Furniture", "Beauty", "Sports"],
    N,
    p=[0.22, 0.25, 0.20, 0.12, 0.13, 0.08],
)

discount_sensitivity = np.clip(np.random.beta(2, 3, N), 0, 1).round(2)  # 0=low,1=high
online_engagement_score = np.clip(
    np.random.normal(50 + frequency * 2 - recency_days * 0.1, 12, N), 0, 100
).round(1)  # app/site visits, clicks, wishlist activity (0-100 index)

cart_abandon_rate = np.clip(np.random.beta(2, 5, N) + discount_sensitivity * 0.1, 0, 1).round(2)

returns_rate = np.clip(np.random.beta(1.5, 10, N), 0, 0.6).round(2)

support_tickets = np.random.poisson(0.6, N)

churn_risk = np.where(
    (recency_days > 120) & (frequency < 3), "High",
    np.where((recency_days > 60) | (frequency < 6), "Medium", "Low")
)

df = pd.DataFrame({
    "CustomerID": customer_id,
    "Age": age,
    "Gender": gender,
    "CityTier": city_tier,
    "AnnualIncome": annual_income,
    "MembershipYears": membership_years,
    "RecencyDays": recency_days,
    "Frequency": frequency,
    "AvgOrderValue": avg_order_value,
    "TotalSpend": total_spend,
    "PreferredCategory": preferred_category,
    "DiscountSensitivity": discount_sensitivity,
    "OnlineEngagementScore": online_engagement_score,
    "CartAbandonRate": cart_abandon_rate,
    "ReturnsRate": returns_rate,
    "SupportTickets": support_tickets,
    "ChurnRisk": churn_risk,
})

out_path = "/home/claude/customer_segmentation/data/customer_data.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows -> {out_path}")
print(df.head())
