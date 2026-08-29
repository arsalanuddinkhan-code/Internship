"""
Generates a messy e-commerce orders export with the kinds of problems a
real data-cleaning automation pipeline actually has to handle:
  - inconsistent text casing/spacing/typos in categorical fields
  - mixed date formats
  - currency values stored as strings with symbols/commas
  - inconsistent boolean representations
  - missing values scattered across several columns
  - exact and near-duplicate rows
  - a few out-of-range / impossible numeric values
"""
import numpy as np
import pandas as pd
from datetime import date, timedelta

rng = np.random.default_rng(20260817)

N = 1600
CATEGORY_CANON = ["Electronics", "Apparel", "Home & Garden", "Sports", "Beauty"]
# realistic messy variants of each canonical category a pipeline must standardize
CATEGORY_VARIANTS = {
    "Electronics": ["Electronics", "electronics", "ELECTRONICS", "Electronic", "Electronics "],
    "Apparel": ["Apparel", "apparel", "Appreal", "APPAREL", " Apparel"],
    "Home & Garden": ["Home & Garden", "home and garden", "Home&Garden", "HOME & GARDEN", "Home & garden"],
    "Sports": ["Sports", "sports", "SPORTS", "Sport", "Sports "],
    "Beauty": ["Beauty", "beauty", "BEAUTY", "Beuaty", " Beauty"],
}
COUNTRY_CANON = ["United States", "Canada", "United Kingdom", "Australia", "Germany"]
COUNTRY_VARIANTS = {
    "United States": ["United States", "USA", "U.S.A.", "us", "United states", "America"],
    "Canada": ["Canada", "CANADA", "canada", "CA"],
    "United Kingdom": ["United Kingdom", "UK", "U.K.", "england", "United kingdom"],
    "Australia": ["Australia", "australia", "AUS", "AUSTRALIA"],
    "Germany": ["Germany", "germany", "GERMANY", "DE"],
}
PAYMENT_METHODS = ["Credit Card", "PayPal", "Debit Card", "Gift Card"]
BOOL_TRUE_VARIANTS = ["Yes", "yes", "Y", "TRUE", "true", "1"]
BOOL_FALSE_VARIANTS = ["No", "no", "N", "FALSE", "false", "0"]

start = date.today() - timedelta(days=540)

rows = []
for i in range(N):
    cat = rng.choice(CATEGORY_CANON)
    cat_messy = rng.choice(CATEGORY_VARIANTS[cat])
    country = rng.choice(COUNTRY_CANON)
    country_messy = rng.choice(COUNTRY_VARIANTS[country])

    d = start + timedelta(days=int(rng.integers(0, 540)))
    # mix date formats across rows, as if exported from different systems
    fmt_choice = i % 4
    if fmt_choice == 0:
        date_str = d.strftime("%Y-%m-%d")
    elif fmt_choice == 1:
        date_str = d.strftime("%m/%d/%Y")
    elif fmt_choice == 2:
        date_str = d.strftime("%d-%b-%Y")
    else:
        date_str = d.strftime("%B %d, %Y")

    qty = int(rng.integers(1, 6))
    unit_price = round(float(rng.uniform(8, 220)), 2)
    # currency sometimes stored as a formatted string, sometimes a plain float
    if rng.random() < 0.4:
        price_str = f"${unit_price:,.2f}"
    else:
        price_str = unit_price

    discount = round(float(rng.uniform(0, 0.3)), 2)
    # a few impossible discounts (data entry errors)
    if rng.random() < 0.01:
        discount = round(float(rng.uniform(1.5, 3)), 2)  # >100% discount, invalid

    returned_bool = rng.random() < 0.08
    returned_str = rng.choice(BOOL_TRUE_VARIANTS if returned_bool else BOOL_FALSE_VARIANTS)

    payment = rng.choice(PAYMENT_METHODS)
    customer = f"Customer {1000 + (i % 900)}"
    email = f"  customer{1000 + (i % 900)}@example.com " if rng.random() < 0.3 else f"customer{1000 + (i % 900)}@example.com"

    rows.append({
        "OrderID": f"ORD-{20000 + i}",
        "OrderDate": date_str,
        "CustomerName": customer,
        "Email": email,
        "Country": country_messy,
        "ProductCategory": cat_messy,
        "Quantity": qty,
        "UnitPrice": price_str,
        "Discount": discount,
        "PaymentMethod": payment,
        "IsReturned": returned_str,
    })

df = pd.DataFrame(rows)

# ---- missing values scattered across several columns ----
for col, rate in [("Email", 0.04), ("Country", 0.03), ("ProductCategory", 0.02),
                   ("Discount", 0.03), ("PaymentMethod", 0.02), ("Quantity", 0.015)]:
    idx = rng.choice(df.index, size=int(len(df) * rate), replace=False)
    df.loc[idx, col] = np.nan

# ---- exact duplicate rows (re-submitted orders) ----
dupes = df.sample(n=22, random_state=5)
df = pd.concat([df, dupes], ignore_index=True)

# ---- shuffle so duplicates aren't conveniently adjacent ----
df = df.sample(frac=1, random_state=11).reset_index(drop=True)

df.to_csv("/home/claude/clean_auto/orders_raw.csv", index=False)
print(df.shape)
print(df.head())
print("nulls per column:\n", df.isna().sum())
print("exact duplicate rows:", df.duplicated().sum())
