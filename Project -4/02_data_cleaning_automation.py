"""
Data Cleaning & Reporting Automation
=====================================
A reusable pipeline that takes any raw CSV export, cleans it, and produces an
automated report -- with a full audit log of every action taken, so nothing
happens silently. This is meant to be re-run on new data drops (e.g. a
scheduled job), not a one-off script -- run_pipeline() is the entry point and
can be called on any similarly-shaped orders export.

Run:
    python data_cleaning_automation.py orders_raw.csv
Requires: pandas, numpy
"""
import re
import sys
import json
from datetime import datetime
import numpy as np
import pandas as pd


class CleaningLog:
    """Collects a timestamped, human-readable audit trail of every cleaning
    action taken, plus machine-readable counts for the report."""

    def __init__(self):
        self.entries = []

    def add(self, stage, message, count=None):
        self.entries.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "stage": stage,
            "message": message,
            "count": count,
        })
        suffix = f" ({count})" if count is not None else ""
        print(f"[{stage}] {message}{suffix}")

    def to_dataframe(self):
        return pd.DataFrame(self.entries)

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(self.entries, f, indent=2)


# ----------------------------------------------------------------------
# Canonical value maps -- in production these would live in a config file
# and grow as new variants are seen. Matching is case/whitespace-insensitive.
# ----------------------------------------------------------------------
CATEGORY_MAP = {
    "electronics": "Electronics", "electronic": "Electronics",
    "apparel": "Apparel", "appreal": "Apparel",
    "home & garden": "Home & Garden", "home and garden": "Home & Garden", "home&garden": "Home & Garden",
    "sports": "Sports", "sport": "Sports",
    "beauty": "Beauty", "beuaty": "Beauty",
}
COUNTRY_MAP = {
    "united states": "United States", "usa": "United States", "u.s.a.": "United States",
    "us": "United States", "america": "United States",
    "canada": "Canada", "ca": "Canada",
    "united kingdom": "United Kingdom", "uk": "United Kingdom", "u.k.": "United Kingdom",
    "england": "United Kingdom",
    "australia": "Australia", "aus": "Australia",
    "germany": "Germany", "de": "Germany",
}
BOOL_TRUE = {"yes", "y", "true", "1"}
BOOL_FALSE = {"no", "n", "false", "0"}


def standardize_categorical(series, mapping, log, column_name):
    """Lowercase/trim, map known variants to a canonical label, and log how
    many values were changed and how many still fall outside the known map."""
    raw = series.astype(str)
    key = raw.str.strip().str.lower()
    mapped = key.map(mapping)
    changed = (mapped.notna() & (mapped != raw)).sum()
    unmapped = mapped.isna() & series.notna()
    unmapped_values = sorted(set(raw[unmapped]) - {"nan"})
    result = mapped.where(mapped.notna(), series)
    log.add("Standardize", f"'{column_name}': normalized {int(changed)} inconsistent values to canonical labels", int(changed))
    if unmapped_values:
        log.add("Standardize", f"'{column_name}': {len(unmapped_values)} unrecognized value(s) left as-is: {unmapped_values[:5]}")
    return result


def parse_mixed_dates(series, log, column_name):
    parsed = pd.to_datetime(series, format="mixed", errors="coerce")
    n_failed = parsed.isna().sum() - series.isna().sum()
    log.add("Standardize", f"'{column_name}': parsed {len(series) - series.isna().sum()} mixed-format date strings into a single format")
    if n_failed > 0:
        log.add("Standardize", f"'{column_name}': {n_failed} value(s) could not be parsed as dates", int(n_failed))
    return parsed


def clean_currency(series, log, column_name):
    """Strip $ and , from string-formatted currency and coerce to float."""
    def parse_one(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, float)):
            return float(v)
        s = re.sub(r"[^0-9.\-]", "", str(v))
        return float(s) if s not in ("", "-", ".") else np.nan

    was_string = series.apply(lambda v: isinstance(v, str)).sum()
    result = series.apply(parse_one)
    log.add("Standardize", f"'{column_name}': converted {int(was_string)} string-formatted currency values to numeric", int(was_string))
    return result


def normalize_boolean(series, log, column_name):
    def parse_one(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().lower()
        if s in BOOL_TRUE:
            return True
        if s in BOOL_FALSE:
            return False
        return np.nan
    result = series.apply(parse_one)
    n_variants = series.astype(str).str.strip().str.lower().nunique()
    log.add("Standardize", f"'{column_name}': collapsed {n_variants} representations (Yes/Y/true/1, etc.) into True/False")
    return result


def run_pipeline(input_path, output_dir="."):
    log = CleaningLog()
    log.add("Ingest", f"Loading raw file: {input_path}")
    raw = pd.read_csv(input_path)
    n_raw = len(raw)
    log.add("Ingest", f"Loaded {n_raw} rows, {raw.shape[1]} columns")

    df = raw.copy()

    # ------------------------------------------------------------------
    # 1. Validate / profile before cleaning
    # ------------------------------------------------------------------
    missing_before = df.isna().sum()
    dupes_before = df.duplicated().sum()
    log.add("Validate", "Missing values by column (before): " +
            ", ".join(f"{c}={int(v)}" for c, v in missing_before[missing_before > 0].items()))
    log.add("Validate", f"Exact duplicate rows found", int(dupes_before))

    # ------------------------------------------------------------------
    # 2. Remove duplicates
    # ------------------------------------------------------------------
    df = df.drop_duplicates()
    log.add("Clean", f"Removed exact duplicate rows", int(dupes_before))

    # ------------------------------------------------------------------
    # 3. Standardize text/categorical/date/currency/boolean fields
    # ------------------------------------------------------------------
    df["Email"] = df["Email"].astype(str).str.strip().replace("nan", np.nan)
    df["ProductCategory"] = standardize_categorical(df["ProductCategory"], CATEGORY_MAP, log, "ProductCategory")
    df["Country"] = standardize_categorical(df["Country"], COUNTRY_MAP, log, "Country")
    df["OrderDate"] = parse_mixed_dates(df["OrderDate"], log, "OrderDate")
    df["UnitPrice"] = clean_currency(df["UnitPrice"], log, "UnitPrice")
    df["IsReturned"] = normalize_boolean(df["IsReturned"], log, "IsReturned")

    # ------------------------------------------------------------------
    # 4. Fix out-of-range values
    # ------------------------------------------------------------------
    bad_discount = (df["Discount"] > 1) | (df["Discount"] < 0)
    n_bad_discount = bad_discount.sum()
    df.loc[bad_discount, "Discount"] = np.nan
    log.add("Clean", f"Flagged and nulled {int(n_bad_discount)} impossible Discount value(s) (outside 0-100%)", int(n_bad_discount))

    # ------------------------------------------------------------------
    # 5. Handle missing values (column-appropriate strategy, logged per column)
    # ------------------------------------------------------------------
    cat_fill = df["ProductCategory"].mode()[0]
    n = df["ProductCategory"].isna().sum()
    df["ProductCategory"] = df["ProductCategory"].fillna(cat_fill)
    log.add("Impute", f"'ProductCategory': filled {int(n)} missing value(s) with mode ('{cat_fill}')", int(n))

    country_fill = df["Country"].mode()[0]
    n = df["Country"].isna().sum()
    df["Country"] = df["Country"].fillna(country_fill)
    log.add("Impute", f"'Country': filled {int(n)} missing value(s) with mode ('{country_fill}')", int(n))

    n = df["PaymentMethod"].isna().sum()
    df["PaymentMethod"] = df["PaymentMethod"].fillna("Unknown")
    log.add("Impute", f"'PaymentMethod': filled {int(n)} missing value(s) with 'Unknown'", int(n))

    qty_median = df["Quantity"].median()
    n = df["Quantity"].isna().sum()
    df["Quantity"] = df["Quantity"].fillna(qty_median)
    log.add("Impute", f"'Quantity': filled {int(n)} missing value(s) with median ({qty_median})", int(n))

    discount_median = df["Discount"].median()
    n = df["Discount"].isna().sum()
    df["Discount"] = df["Discount"].fillna(discount_median)
    log.add("Impute", f"'Discount': filled {int(n)} missing value(s) (incl. flagged outliers) with median ({discount_median:.2f})", int(n))

    n = df["Email"].isna().sum()
    df["Email"] = df["Email"].fillna("not_provided@unknown.com")
    log.add("Impute", f"'Email': filled {int(n)} missing value(s) with placeholder", int(n))

    # ------------------------------------------------------------------
    # 6. Derived fields for reporting
    # ------------------------------------------------------------------
    df["LineTotal"] = (df["Quantity"] * df["UnitPrice"] * (1 - df["Discount"])).round(2)

    # ------------------------------------------------------------------
    # 7. Post-clean validation
    # ------------------------------------------------------------------
    missing_after = df.isna().sum()
    dupes_after = df.duplicated().sum()
    log.add("Validate", f"Missing values remaining (after)", int(missing_after.sum()))
    log.add("Validate", f"Duplicate rows remaining (after)", int(dupes_after))

    def quality_score(missing_total, cells_total, dupes, n_rows):
        completeness = 1 - (missing_total / cells_total)
        uniqueness = 1 - (dupes / max(n_rows, 1))
        return round((completeness * 0.7 + uniqueness * 0.3) * 100, 1)

    cells_before = n_raw * raw.shape[1]
    cells_after = len(df) * df.shape[1]
    score_before = quality_score(missing_before.sum(), cells_before, dupes_before, n_raw)
    score_after = quality_score(missing_after.sum(), cells_after, dupes_after, len(df))
    log.add("Report", f"Data quality score before cleaning: {score_before}/100")
    log.add("Report", f"Data quality score after cleaning: {score_after}/100")

    # ------------------------------------------------------------------
    # 8. Export cleaned data, log, and summary report
    # ------------------------------------------------------------------
    df.to_csv(f"{output_dir}/orders_clean.csv", index=False)
    log.to_dataframe().to_csv(f"{output_dir}/cleaning_log.csv", index=False)
    log.to_json(f"{output_dir}/cleaning_log.json")

    summary = {
        "rows_before": int(n_raw),
        "rows_after": int(len(df)),
        "duplicates_removed": int(dupes_before),
        "missing_before": int(missing_before.sum()),
        "missing_after": int(missing_after.sum()),
        "quality_score_before": score_before,
        "quality_score_after": score_after,
        "total_revenue": round(float(df["LineTotal"].sum()), 2),
        "avg_order_value": round(float(df["LineTotal"].mean()), 2),
        "return_rate": round(float(df["IsReturned"].mean()), 3),
    }
    with open(f"{output_dir}/report_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # category / country breakdown for the automated visual summary
    df.groupby("ProductCategory")["LineTotal"].agg(["sum", "count"]).reset_index().rename(
        columns={"sum": "revenue", "count": "orders"}
    ).to_csv(f"{output_dir}/report_by_category.csv", index=False)
    df.groupby("Country")["LineTotal"].agg(["sum", "count"]).reset_index().rename(
        columns={"sum": "revenue", "count": "orders"}
    ).to_csv(f"{output_dir}/report_by_country.csv", index=False)

    missing_before[missing_before > 0].rename("missing_before").to_frame().join(
        missing_after.rename("missing_after"), how="outer"
    ).fillna(0).astype(int).reset_index().rename(columns={"index": "column"}).to_csv(
        f"{output_dir}/report_missing_by_column.csv", index=False
    )

    log.add("Report", "Exported: orders_clean.csv, cleaning_log.csv/json, report_summary.json, "
                       "report_by_category.csv, report_by_country.csv, report_missing_by_column.csv")
    print("\n=== Pipeline complete ===")
    print(json.dumps(summary, indent=2))
    return df, log, summary


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "orders_raw.csv"
    run_pipeline(input_path)
