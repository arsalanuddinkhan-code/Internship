"""
02_segmentation_analysis.py
----------------------------
Customer Segmentation Project - Full Pipeline
1. Load & explore data (EDA)
2. Feature engineering + preprocessing (encoding, scaling)
3. Find optimal number of clusters (Elbow Method + Silhouette Score)
4. Apply K-Means clustering
5. Reduce dimensions with PCA for 2D visualization
6. Profile each segment (demographics + behavior)
7. Export labeled dataset + all plots + segment summary table
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

DATA_PATH = "/home/claude/customer_segmentation/data/customer_data.csv"
PLOTS_DIR = "/home/claude/customer_segmentation/plots"
OUT_DIR = "/home/claude/customer_segmentation/outputs"

# =================================================================
# 1. LOAD DATA
# =================================================================
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print(df.info())
print(df.describe())

# =================================================================
# 2. EDA - a few key distribution / relationship plots
# =================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

sns.histplot(df["Age"], bins=25, kde=True, ax=axes[0, 0], color="#4C72B0")
axes[0, 0].set_title("Age Distribution")

sns.histplot(df["TotalSpend"], bins=30, kde=True, ax=axes[0, 1], color="#DD8452")
axes[0, 1].set_title("Total Spend Distribution")

sns.boxplot(x="CityTier", y="AnnualIncome", hue="CityTier", data=df, ax=axes[1, 0], palette="Set2", legend=False)
axes[1, 0].set_title("Annual Income by City Tier")

cat_counts = df["PreferredCategory"].value_counts()
axes[1, 1].pie(cat_counts, labels=cat_counts.index, autopct="%1.0f%%",
               colors=sns.color_palette("pastel"))
axes[1, 1].set_title("Preferred Category Share")

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/01_eda_overview.png", bbox_inches="tight")
plt.close()

# Correlation heatmap of numeric behavioral features
numeric_cols = ["Age", "AnnualIncome", "MembershipYears", "RecencyDays", "Frequency",
                 "AvgOrderValue", "TotalSpend", "DiscountSensitivity",
                 "OnlineEngagementScore", "CartAbandonRate", "ReturnsRate"]
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap - Behavioral & Demographic Features")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/02_correlation_heatmap.png", bbox_inches="tight")
plt.close()

# =================================================================
# 3. FEATURE ENGINEERING + PREPROCESSING
# =================================================================
model_df = df.copy()

le_gender = LabelEncoder()
le_city = LabelEncoder()
le_cat = LabelEncoder()
model_df["Gender_enc"] = le_gender.fit_transform(model_df["Gender"])
model_df["CityTier_enc"] = le_city.fit_transform(model_df["CityTier"])
model_df["PreferredCategory_enc"] = le_cat.fit_transform(model_df["PreferredCategory"])

# NOTE: feature selection matters a lot for cluster quality. Demographic /
# secondary behavioral fields (Gender, CityTier, ReturnsRate, CartAbandonRate,
# PreferredCategory) are kept in the dataframe for *profiling* the segments
# afterwards, but were tested and found to add noise rather than structure
# to the clustering itself (lower silhouette score) - a common finding in
# real RFM-style segmentation. The core RFM + value features below gave the
# best-separated, most business-interpretable clusters.
feature_cols = [
    "AnnualIncome", "MembershipYears", "RecencyDays", "Frequency",
    "AvgOrderValue", "TotalSpend", "OnlineEngagementScore",
]

X = model_df[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =================================================================
# 4. FIND OPTIMAL K - Elbow Method + Silhouette Score
# =================================================================
inertias, sil_scores = [], []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(list(K_range), inertias, marker="o", color="#4C72B0")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia (WCSS)")
axes[0].set_title("Elbow Method")

axes[1].plot(list(K_range), sil_scores, marker="o", color="#55A868")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score by k")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/03_optimal_k.png", bbox_inches="tight")
plt.close()

best_k = list(K_range)[int(np.argmax(sil_scores))]
print(f"\nSuggested optimal k (by silhouette score): {best_k}")
# Business-friendly override: keep k=4 unless silhouette strongly prefers otherwise,
# since 4 segments (e.g. Champions/Loyal/At-Risk/New) is a standard, actionable size.
FINAL_K = best_k if best_k in (3, 4, 5) else 4
print(f"Using FINAL_K = {FINAL_K} for the business-facing segmentation")

# =================================================================
# 5. FINAL K-MEANS MODEL
# =================================================================
kmeans = KMeans(n_clusters=FINAL_K, random_state=42, n_init=10)
model_df["Segment"] = kmeans.fit_predict(X_scaled)
final_sil = silhouette_score(X_scaled, model_df["Segment"])
print(f"Final silhouette score (k={FINAL_K}): {final_sil:.3f}")

# =================================================================
# 6. PCA FOR 2D VISUALIZATION
# =================================================================
pca = PCA(n_components=2, random_state=42)
pcs = pca.fit_transform(X_scaled)
model_df["PC1"], model_df["PC2"] = pcs[:, 0], pcs[:, 1]
print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.round(3)}")

plt.figure(figsize=(9, 7))
palette = sns.color_palette("Set1", FINAL_K)
sns.scatterplot(data=model_df, x="PC1", y="PC2", hue="Segment", palette=palette,
                 s=45, alpha=0.8, edgecolor="white", linewidth=0.3)
plt.title(f"Customer Segments Visualized via PCA (k={FINAL_K})")
plt.legend(title="Segment")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/04_pca_clusters.png", bbox_inches="tight")
plt.close()

# =================================================================
# 7. SEGMENT PROFILING
# =================================================================
profile_numeric = model_df.groupby("Segment")[
    ["Age", "AnnualIncome", "MembershipYears", "RecencyDays", "Frequency",
     "AvgOrderValue", "TotalSpend", "DiscountSensitivity",
     "OnlineEngagementScore", "CartAbandonRate", "ReturnsRate"]
].mean().round(1)
profile_numeric["SegmentSize"] = model_df["Segment"].value_counts().sort_index()
profile_numeric["SegmentSharePct"] = (profile_numeric["SegmentSize"] / len(model_df) * 100).round(1)

# most common categorical values per segment
mode_cat = model_df.groupby("Segment")[["Gender", "CityTier", "PreferredCategory", "ChurnRisk"]] \
    .agg(lambda x: x.value_counts().index[0])

profile = profile_numeric.join(mode_cat)
profile.to_csv(f"{OUT_DIR}/segment_profile_summary.csv")
print("\n=== SEGMENT PROFILE SUMMARY ===")
print(profile)

# Bar chart comparison of key metrics across segments
key_metrics = ["TotalSpend", "Frequency", "AvgOrderValue", "OnlineEngagementScore", "RecencyDays"]
fig, axes = plt.subplots(1, len(key_metrics), figsize=(22, 4.5))
for i, m in enumerate(key_metrics):
    sns.barplot(x=profile_numeric.index, y=profile_numeric[m], hue=profile_numeric.index, ax=axes[i], palette=palette, legend=False)
    axes[i].set_title(m)
    axes[i].set_xlabel("Segment")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/05_segment_metric_comparison.png", bbox_inches="tight")
plt.close()

# Segment size pie chart
plt.figure(figsize=(6, 6))
sizes = model_df["Segment"].value_counts().sort_index()
plt.pie(sizes, labels=[f"Segment {i}" for i in sizes.index], autopct="%1.1f%%",
        colors=palette, startangle=90)
plt.title("Customer Distribution Across Segments")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/06_segment_sizes.png", bbox_inches="tight")
plt.close()

# Preferred category composition within each segment (stacked bar)
comp = pd.crosstab(model_df["Segment"], model_df["PreferredCategory"], normalize="index") * 100
comp.plot(kind="bar", stacked=True, figsize=(9, 6), colormap="tab20")
plt.ylabel("% of customers")
plt.title("Preferred Category Mix by Segment")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Category")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/07_category_mix_by_segment.png", bbox_inches="tight")
plt.close()

# =================================================================
# 8. EXPORT LABELED DATASET
# =================================================================
export_cols = ["CustomerID", "Age", "Gender", "CityTier", "AnnualIncome",
               "MembershipYears", "RecencyDays", "Frequency", "AvgOrderValue",
               "TotalSpend", "PreferredCategory", "DiscountSensitivity",
               "OnlineEngagementScore", "CartAbandonRate", "ReturnsRate",
               "SupportTickets", "ChurnRisk", "Segment", "PC1", "PC2"]
model_df[export_cols].to_csv(f"{OUT_DIR}/customers_with_segments.csv", index=False)

print("\nAll outputs saved to:", OUT_DIR, "and", PLOTS_DIR)
