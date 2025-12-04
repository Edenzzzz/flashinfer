"""
Script to analyze correlations in SM performance data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Read the CSV file
df = pd.read_csv("sm_performance.csv")

print("=" * 80)
print("SM Performance Correlation Analysis")
print("=" * 80)
print(f"\nTotal SMs: {len(df)}")
print(f"\nData columns: {list(df.columns)}")
print("\n" + "=" * 80)

# Calculate correlations
correlations = {}

# 1. Correlation between prefill_qo_len and prefill_time_ms (if available)
if "prefill_qo_len" in df.columns:
    if "prefill_time_ms" in df.columns:
        corr = df["prefill_qo_len"].corr(df["prefill_time_ms"])
        correlations["prefill_qo_len vs prefill_time_ms"] = corr
        print(f"\n1. Correlation (prefill_qo_len vs prefill_time_ms): {corr:.4f}")
        pearson_r, pearson_p = stats.pearsonr(df["prefill_qo_len"], df["prefill_time_ms"])
        print(f"   Pearson r: {pearson_r:.4f}, p-value: {pearson_p:.4e}")
    else:
        print("\n1. prefill_time_ms not found in CSV, skipping prefill_qo_len vs prefill_time_ms")

# 2. Correlation between prefill_qo_len and sm_time
if "prefill_qo_len" in df.columns and "sm_time" in df.columns:
    corr = df["prefill_qo_len"].corr(df["sm_time"])
    correlations["prefill_qo_len vs sm_time"] = corr
    print(f"\n2. Correlation (prefill_qo_len vs sm_time): {corr:.4f}")
    pearson_r, pearson_p = stats.pearsonr(df["prefill_qo_len"], df["sm_time"])
    print(f"   Pearson r: {pearson_r:.4f}, p-value: {pearson_p:.4e}")

# 3. Correlation between decode_kv_len and decode_time_ms (if available)
if "decode_kv_len" in df.columns:
    if "decode_time_ms" in df.columns:
        corr = df["decode_kv_len"].corr(df["decode_time_ms"])
        correlations["decode_kv_len vs decode_time_ms"] = corr
        print(f"\n3. Correlation (decode_kv_len vs decode_time_ms): {corr:.4f}")
        pearson_r, pearson_p = stats.pearsonr(df["decode_kv_len"], df["decode_time_ms"])
        print(f"   Pearson r: {pearson_r:.4f}, p-value: {pearson_p:.4e}")
    else:
        print("\n3. decode_time_ms not found in CSV, skipping decode_kv_len vs decode_time_ms")

# 4. Correlation between decode_qo_len and decode_time_ms (if available)
if "decode_qo_len" in df.columns:
    if "decode_time_ms" in df.columns:
        corr = df["decode_qo_len"].corr(df["decode_time_ms"])
        correlations["decode_qo_len vs decode_time_ms"] = corr
        print(f"\n4. Correlation (decode_qo_len vs decode_time_ms): {corr:.4f}")
        pearson_r, pearson_p = stats.pearsonr(df["decode_qo_len"], df["decode_time_ms"])
        print(f"   Pearson r: {pearson_r:.4f}, p-value: {pearson_p:.4e}")
    else:
        print("\n4. decode_time_ms not found in CSV, skipping decode_qo_len vs decode_time_ms")

# 5. Correlation between decode_qo_len and sm_time
if "decode_qo_len" in df.columns and "sm_time" in df.columns:
    corr = df["decode_qo_len"].corr(df["sm_time"])
    correlations["decode_qo_len vs sm_time"] = corr
    print(f"\n5. Correlation (decode_qo_len vs sm_time): {corr:.4f}")
    pearson_r, pearson_p = stats.pearsonr(df["decode_qo_len"], df["sm_time"])
    print(f"   Pearson r: {pearson_r:.4f}, p-value: {pearson_p:.4e}")

# 6. Correlation between decode_kv_len and sm_time
if "decode_kv_len" in df.columns and "sm_time" in df.columns:
    corr = df["decode_kv_len"].corr(df["sm_time"])
    correlations["decode_kv_len vs sm_time"] = corr
    print(f"\n6. Correlation (decode_kv_len vs sm_time): {corr:.4f}")
    pearson_r, pearson_p = stats.pearsonr(df["decode_kv_len"], df["sm_time"])
    print(f"   Pearson r: {pearson_r:.4f}, p-value: {pearson_p:.4e}")

# Additional: Combined decode metrics
if all(col in df.columns for col in ["decode_qo_len", "decode_kv_len", "sm_time"]):
    # Create a combined metric (sum or product)
    df["decode_total_len"] = df["decode_qo_len"] + df["decode_kv_len"]
    corr = df["decode_total_len"].corr(df["sm_time"])
    correlations["decode_total_len vs sm_time"] = corr
    print(f"\n7. Correlation (decode_qo_len + decode_kv_len vs sm_time): {corr:.4f}")
    pearson_r, pearson_p = stats.pearsonr(df["decode_total_len"], df["sm_time"])
    print(f"   Pearson r: {pearson_r:.4f}, p-value: {pearson_p:.4e}")

print("\n" + "=" * 80)
print("Summary Statistics")
print("=" * 80)
print(df.describe())

# Create visualizations
print("\n" + "=" * 80)
print("Generating correlation plots...")
print("=" * 80)

# Set up the plotting style
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

plot_idx = 0

# 1. prefill_qo_len vs prefill_time_ms (if available)
if "prefill_qo_len" in df.columns and "prefill_time_ms" in df.columns:
    axes[plot_idx].scatter(df["prefill_qo_len"], df["prefill_time_ms"], alpha=0.6)
    axes[plot_idx].set_xlabel("Prefill QO Length")
    axes[plot_idx].set_ylabel("Prefill Time (ms)")
    axes[plot_idx].set_title(f"Prefill QO Len vs Prefill Time\n(r={correlations.get('prefill_qo_len vs prefill_time_ms', 0):.3f})")
    z = np.polyfit(df["prefill_qo_len"], df["prefill_time_ms"], 1)
    p = np.poly1d(z)
    axes[plot_idx].plot(df["prefill_qo_len"], p(df["prefill_qo_len"]), "r--", alpha=0.8)
    plot_idx += 1
elif "prefill_qo_len" in df.columns:
    # If prefill_time_ms not available, skip this plot
    pass

# 2. prefill_qo_len vs sm_time
if "prefill_qo_len" in df.columns and "sm_time" in df.columns:
    axes[plot_idx].scatter(df["prefill_qo_len"], df["sm_time"], alpha=0.6)
    axes[plot_idx].set_xlabel("Prefill QO Length")
    axes[plot_idx].set_ylabel("SM Time (ms)")
    axes[plot_idx].set_title(f"Prefill QO Len vs SM Time\n(r={correlations.get('prefill_qo_len vs sm_time', 0):.3f})")
    z = np.polyfit(df["prefill_qo_len"], df["sm_time"], 1)
    p = np.poly1d(z)
    axes[plot_idx].plot(df["prefill_qo_len"], p(df["prefill_qo_len"]), "r--", alpha=0.8)
    plot_idx += 1

# 3. decode_kv_len vs decode_time_ms (if available)
if "decode_kv_len" in df.columns and "decode_time_ms" in df.columns:
    axes[plot_idx].scatter(df["decode_kv_len"], df["decode_time_ms"], alpha=0.6)
    axes[plot_idx].set_xlabel("Decode KV Length")
    axes[plot_idx].set_ylabel("Decode Time (ms)")
    axes[plot_idx].set_title(f"Decode KV Len vs Decode Time\n(r={correlations.get('decode_kv_len vs decode_time_ms', 0):.3f})")
    z = np.polyfit(df["decode_kv_len"], df["decode_time_ms"], 1)
    p = np.poly1d(z)
    axes[plot_idx].plot(df["decode_kv_len"], p(df["decode_kv_len"]), "r--", alpha=0.8)
    plot_idx += 1
elif "decode_kv_len" in df.columns:
    # If decode_time_ms not available, skip this plot
    pass

# 4. decode_qo_len vs decode_time_ms (if available)
if "decode_qo_len" in df.columns and "decode_time_ms" in df.columns:
    axes[plot_idx].scatter(df["decode_qo_len"], df["decode_time_ms"], alpha=0.6)
    axes[plot_idx].set_xlabel("Decode QO Length")
    axes[plot_idx].set_ylabel("Decode Time (ms)")
    axes[plot_idx].set_title(f"Decode QO Len vs Decode Time\n(r={correlations.get('decode_qo_len vs decode_time_ms', 0):.3f})")
    z = np.polyfit(df["decode_qo_len"], df["decode_time_ms"], 1)
    p = np.poly1d(z)
    axes[plot_idx].plot(df["decode_qo_len"], p(df["decode_qo_len"]), "r--", alpha=0.8)
    plot_idx += 1
elif "decode_qo_len" in df.columns:
    # If decode_time_ms not available, skip this plot
    pass

# 5. decode_qo_len vs sm_time
if "decode_qo_len" in df.columns and "sm_time" in df.columns:
    axes[plot_idx].scatter(df["decode_qo_len"], df["sm_time"], alpha=0.6)
    axes[plot_idx].set_xlabel("Decode QO Length")
    axes[plot_idx].set_ylabel("SM Time (ms)")
    axes[plot_idx].set_title(f"Decode QO Len vs SM Time\n(r={correlations.get('decode_qo_len vs sm_time', 0):.3f})")
    z = np.polyfit(df["decode_qo_len"], df["sm_time"], 1)
    p = np.poly1d(z)
    axes[plot_idx].plot(df["decode_qo_len"], p(df["decode_qo_len"]), "r--", alpha=0.8)
    plot_idx += 1

# 6. decode_kv_len vs sm_time
if "decode_kv_len" in df.columns and "sm_time" in df.columns:
    axes[plot_idx].scatter(df["decode_kv_len"], df["sm_time"], alpha=0.6)
    axes[plot_idx].set_xlabel("Decode KV Length")
    axes[plot_idx].set_ylabel("SM Time (ms)")
    axes[plot_idx].set_title(f"Decode KV Len vs SM Time\n(r={correlations.get('decode_kv_len vs sm_time', 0):.3f})")
    z = np.polyfit(df["decode_kv_len"], df["sm_time"], 1)
    p = np.poly1d(z)
    axes[plot_idx].plot(df["decode_kv_len"], p(df["decode_kv_len"]), "r--", alpha=0.8)
    plot_idx += 1

# Hide unused subplots
for i in range(plot_idx, len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.savefig("sm_performance_correlations.png", dpi=150, bbox_inches="tight")
print("Saved correlation plots to sm_performance_correlations.png")

# Create a correlation heatmap
print("\nGenerating correlation heatmap...")
time_cols = [col for col in df.columns if "time" in col.lower()]
len_cols = [col for col in df.columns if "len" in col.lower()]
corr_cols = time_cols + len_cols

if len(corr_cols) > 1:
    corr_matrix = df[corr_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", center=0, square=True)
    plt.title("Correlation Heatmap: Lengths vs Times")
    plt.tight_layout()
    plt.savefig("sm_performance_heatmap.png", dpi=150, bbox_inches="tight")
    print("Saved correlation heatmap to sm_performance_heatmap.png")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)

