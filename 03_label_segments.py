"""
03_label_segments.py
---------------------
Turns the numeric cluster IDs (0,1,2,3) into business-friendly segment
names based on the profile summary, and saves a final labeled export
that a marketing/business stakeholder can act on directly.
"""
import pandas as pd

OUT_DIR = "/home/claude/customer_segmentation/outputs"

profile = pd.read_csv(f"{OUT_DIR}/segment_profile_summary.csv", index_col="Segment")
customers = pd.read_csv(f"{OUT_DIR}/customers_with_segments.csv")

# Rank segments to assign names programmatically (robust if re-run with
# different random data), based on TotalSpend and RecencyDays:
#   - High spend, high frequency, low recency-days      -> "Champions / Loyal High-Value"
#   - High/med spend, very recent, largest group         -> "Mainstream Regulars"
#   - Low spend, low AOV, price-sensitive                -> "Budget / Price-Sensitive"
#   - High RecencyDays (long time since last purchase)   -> "At-Risk / Dormant"
name_map = {}
dormant_seg = profile["RecencyDays"].idxmax()
champion_seg = profile["TotalSpend"].idxmax()
remaining = [s for s in profile.index if s not in (dormant_seg, champion_seg)]
budget_seg = profile.loc[remaining, "AvgOrderValue"].idxmin()
mainstream_seg = [s for s in remaining if s != budget_seg][0]

name_map[dormant_seg] = "At-Risk / Dormant"
name_map[champion_seg] = "Champions (High-Value Loyal)"
name_map[budget_seg] = "Budget-Conscious / New"
name_map[mainstream_seg] = "Mainstream Regulars"

profile["SegmentName"] = profile.index.map(name_map)
customers["SegmentName"] = customers["Segment"].map(name_map)

profile.to_csv(f"{OUT_DIR}/segment_profile_summary.csv")
customers.to_csv(f"{OUT_DIR}/customers_with_segments.csv", index=False)

print(profile[["SegmentName", "SegmentSize", "SegmentSharePct", "TotalSpend",
               "Frequency", "AvgOrderValue", "RecencyDays", "OnlineEngagementScore"]])
