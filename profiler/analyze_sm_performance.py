"""
Script to analyze correlations in SM performance data.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def main(args):
    # Read the CSV file
    df = pd.read_csv(args.csv)

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
            pearson_r, pearson_p = stats.pearsonr(
                df["prefill_qo_len"], df["prefill_time_ms"]
            )
            print(f"   Pearson r: {pearson_r:.4f}, p-value: {pearson_p:.4e}")
        else:
            print(
                "\n1. prefill_time_ms not found in CSV, skipping prefill_qo_len vs prefill_time_ms"
            )

    # 2. Correlation between prefill_qo_len and sm_time
    if "prefill_qo_len" in df.columns and "sm_time" in df.columns:
        corr = df["prefill_qo_len"].corr(df["sm_time"])
        correlations["prefill_qo_len vs sm_time"] = corr
        print(f"\n2. Correlation (prefill_qo_len vs sm_time): {corr:.4f}")
        pearson_r, pearson_p = stats.pearsonr(df["prefill_qo_len"], df["sm_time"])
        print(f"   Pearson r: {pearson_r:.4f}, p-value: {pearson_p:.4e}")

    # 3. Correlation between prefill_kv_len and prefill_time_ms (if available)
    if "prefill_kv_len" in df.columns:
        if "prefill_time_ms" in df.columns:
            corr = df["prefill_kv_len"].corr(df["prefill_time_ms"])
            correlations["prefill_kv_len vs prefill_time_ms"] = corr
            print(f"\n3. Correlation (prefill_kv_len vs prefill_time_ms): {corr:.4f}")
            pearson_r, pearson_p = stats.pearsonr(
                df["prefill_kv_len"], df["prefill_time_ms"]
            )
            print(f"   Pearson r: {pearson_r:.4f}, p-value: {pearson_p:.4e}")
        else:
            print(
                "\n3. prefill_time_ms not found in CSV, skipping prefill_kv_len vs prefill_time_ms"
            )

    # 4. Correlation between decode_kv_len and decode_time_ms (if available)
    if "decode_kv_len" in df.columns:
        if "decode_time_ms" in df.columns:
            corr = df["decode_kv_len"].corr(df["decode_time_ms"])
            correlations["decode_kv_len vs decode_time_ms"] = corr
            print(f"\n3. Correlation (decode_kv_len vs decode_time_ms): {corr:.4f}")
            pearson_r, pearson_p = stats.pearsonr(
                df["decode_kv_len"], df["decode_time_ms"]
            )
            print(f"   Pearson r: {pearson_r:.4f}, p-value: {pearson_p:.4e}")
        else:
            print(
                "\n3. decode_time_ms not found in CSV, skipping decode_kv_len vs decode_time_ms"
            )

    # 6. Correlation between decode_qo_len and decode_time_ms (if available)
    if "decode_qo_len" in df.columns:
        if "decode_time_ms" in df.columns:
            corr = df["decode_qo_len"].corr(df["decode_time_ms"])
            correlations["decode_qo_len vs decode_time_ms"] = corr
            print(f"\n6. Correlation (decode_qo_len vs decode_time_ms): {corr:.4f}")
            pearson_r, pearson_p = stats.pearsonr(
                df["decode_qo_len"], df["decode_time_ms"]
            )
            print(f"   Pearson r: {pearson_r:.4f}, p-value: {pearson_p:.4e}")
        else:
            print(
                "\n6. decode_time_ms not found in CSV, skipping decode_qo_len vs decode_time_ms"
            )

    # 7. Correlation between decode_qo_len and sm_time
    if "decode_qo_len" in df.columns and "sm_time" in df.columns:
        corr = df["decode_qo_len"].corr(df["sm_time"])
        correlations["decode_qo_len vs sm_time"] = corr
        print(f"\n7. Correlation (decode_qo_len vs sm_time): {corr:.4f}")
        pearson_r, pearson_p = stats.pearsonr(df["decode_qo_len"], df["sm_time"])
        print(f"   Pearson r: {pearson_r:.4f}, p-value: {pearson_p:.4e}")

    # 8. Correlation between decode_kv_len and sm_time
    if "decode_kv_len" in df.columns and "sm_time" in df.columns:
        corr = df["decode_kv_len"].corr(df["sm_time"])
        correlations["decode_kv_len vs sm_time"] = corr
        print(f"\n8. Correlation (decode_kv_len vs sm_time): {corr:.4f}")
        pearson_r, pearson_p = stats.pearsonr(df["decode_kv_len"], df["sm_time"])
        print(f"   Pearson r: {pearson_r:.4f}, p-value: {pearson_p:.4e}")

    # Additional: Combined decode metrics
    if all(col in df.columns for col in ["decode_qo_len", "decode_kv_len", "sm_time"]):
        # Create a combined metric (sum or product)
        df["decode_total_len"] = df["decode_qo_len"] + df["decode_kv_len"]
        corr = df["decode_total_len"].corr(df["sm_time"])
        correlations["decode_total_len vs sm_time"] = corr
        print(
            f"\n9. Correlation (decode_qo_len + decode_kv_len vs sm_time): {corr:.4f}"
        )
        pearson_r, pearson_p = stats.pearsonr(df["decode_total_len"], df["sm_time"])
        print(f"   Pearson r: {pearson_r:.4f}, p-value: {pearson_p:.4e}")

    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(df.describe())

    # Group by 4-tuple and analyze variance
    print("\n" + "=" * 80)
    print(
        "Grouping by 4-tuple (prefill_qo_len, prefill_kv_len, decode_qo_len, decode_kv_len)"
    )
    print("=" * 80)

    if all(
        col in df.columns
        for col in [
            "prefill_qo_len",
            "prefill_kv_len",
            "decode_qo_len",
            "decode_kv_len",
            "sm_time",
        ]
    ):
        # Group by the 4-tuple
        grouped = df.groupby(
            ["prefill_qo_len", "prefill_kv_len", "decode_qo_len", "decode_kv_len"]
        )

        # Calculate statistics for each group
        group_stats = (
            grouped["sm_time"]
            .agg(["count", "mean", "std", "var", "min", "max"])
            .reset_index()
        )
        group_stats.columns = [
            "prefill_qo_len",
            "prefill_kv_len",
            "decode_qo_len",
            "decode_kv_len",
            "count",
            "mean",
            "std",
            "variance",
            "min",
            "max",
        ]

        # Calculate coefficient of variation (CV) = std / mean
        group_stats["cv"] = (
            group_stats["std"] / (group_stats["mean"] + 1e-10) * 100
        )  # as percentage

        # Sort by variance (descending) to see which groups have highest variance
        group_stats_sorted = group_stats.sort_values("variance", ascending=False)

        print(f"\nTotal unique configurations: {len(group_stats)}")
        print(f"Total data points: {len(df)}")
        print(f"Average samples per configuration: {len(df) / len(group_stats):.2f}")

        print("\nTop 20 configurations by variance:")
        print("=" * 120)
        print(group_stats_sorted.head(20).to_string(index=False))

        print("\nTop 20 configurations by coefficient of variation (CV = std/mean):")
        print("=" * 120)
        group_stats_sorted_cv = group_stats.sort_values("cv", ascending=False)
        print(group_stats_sorted_cv.head(20).to_string(index=False))

        print("\nOverall statistics across all groups:")
        print("=" * 80)
        print(f"Mean variance: {group_stats['variance'].mean():.6f}")
        print(f"Median variance: {group_stats['variance'].median():.6f}")
        print(f"Max variance: {group_stats['variance'].max():.6f}")
        print(f"Mean CV: {group_stats['cv'].mean():.2f}%")
        print(f"Median CV: {group_stats['cv'].median():.2f}%")
        print(f"Max CV: {group_stats['cv'].max():.2f}%")

        # Save group statistics to CSV
        group_stats_sorted.to_csv("sm_performance_group_stats.csv", index=False)
        print("\nSaved group statistics to sm_performance_group_stats.csv")

        # Create visualization of variance distribution
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Variance distribution
        axes[0, 0].hist(group_stats["variance"], bins=50, alpha=0.7, edgecolor="black")
        axes[0, 0].set_xlabel("Variance of sm_time")
        axes[0, 0].set_ylabel("Number of Configurations")
        axes[0, 0].set_title("Distribution of Variance Across Configurations")
        axes[0, 0].set_yscale("log")
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: CV distribution
        axes[0, 1].hist(group_stats["cv"], bins=50, alpha=0.7, edgecolor="black")
        axes[0, 1].set_xlabel("Coefficient of Variation (%)")
        axes[0, 1].set_ylabel("Number of Configurations")
        axes[0, 1].set_title("Distribution of CV Across Configurations")
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Mean vs Variance
        axes[1, 0].scatter(group_stats["mean"], group_stats["variance"], alpha=0.6)
        axes[1, 0].set_xlabel("Mean sm_time (ms)")
        axes[1, 0].set_ylabel("Variance of sm_time")
        axes[1, 0].set_title("Mean vs Variance")
        axes[1, 0].set_xscale("log")
        axes[1, 0].set_yscale("log")
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Count vs Variance
        axes[1, 1].scatter(group_stats["count"], group_stats["variance"], alpha=0.6)
        axes[1, 1].set_xlabel("Number of Samples in Group")
        axes[1, 1].set_ylabel("Variance of sm_time")
        axes[1, 1].set_title("Sample Count vs Variance")
        axes[1, 1].set_yscale("log")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            "sm_performance_variance_analysis.png", dpi=150, bbox_inches="tight"
        )
        print("Saved variance analysis plots to sm_performance_variance_analysis.png")
    else:
        print(
            "Missing required columns for 4-tuple grouping. Need: prefill_qo_len, prefill_kv_len, decode_qo_len, decode_kv_len, sm_time"
        )

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
        axes[plot_idx].set_title(
            f"Prefill QO Len vs Prefill Time\n(r={correlations.get('prefill_qo_len vs prefill_time_ms', 0):.3f})"
        )
        z = np.polyfit(df["prefill_qo_len"], df["prefill_time_ms"], 1)
        p = np.poly1d(z)
        axes[plot_idx].plot(
            df["prefill_qo_len"], p(df["prefill_qo_len"]), "r--", alpha=0.8
        )
        plot_idx += 1
    elif "prefill_qo_len" in df.columns:
        # If prefill_time_ms not available, skip this plot
        pass

    # 2. prefill_qo_len vs sm_time
    if "prefill_qo_len" in df.columns and "sm_time" in df.columns:
        axes[plot_idx].scatter(df["prefill_qo_len"], df["sm_time"], alpha=0.6)
        axes[plot_idx].set_xlabel("Prefill QO Length")
        axes[plot_idx].set_ylabel("SM Time (ms)")
        axes[plot_idx].set_title(
            f"Prefill QO Len vs SM Time\n(r={correlations.get('prefill_qo_len vs sm_time', 0):.3f})"
        )
        z = np.polyfit(df["prefill_qo_len"], df["sm_time"], 1)
        p = np.poly1d(z)
        axes[plot_idx].plot(
            df["prefill_qo_len"], p(df["prefill_qo_len"]), "r--", alpha=0.8
        )
        plot_idx += 1

    # 3. prefill_kv_len vs prefill_time_ms (if available)
    if "prefill_kv_len" in df.columns and "prefill_time_ms" in df.columns:
        axes[plot_idx].scatter(df["prefill_kv_len"], df["prefill_time_ms"], alpha=0.6)
        axes[plot_idx].set_xlabel("Prefill KV Length")
        axes[plot_idx].set_ylabel("Prefill Time (ms)")
        axes[plot_idx].set_title(
            f"Prefill KV Len vs Prefill Time\n(r={correlations.get('prefill_kv_len vs prefill_time_ms', 0):.3f})"
        )
        z = np.polyfit(df["prefill_kv_len"], df["prefill_time_ms"], 1)
        p = np.poly1d(z)
        axes[plot_idx].plot(
            df["prefill_kv_len"], p(df["prefill_kv_len"]), "r--", alpha=0.8
        )
        plot_idx += 1
    elif "prefill_kv_len" in df.columns:
        # If prefill_time_ms not available, skip this plot
        pass

    # 4. decode_kv_len vs decode_time_ms (if available)
    if "decode_kv_len" in df.columns and "decode_time_ms" in df.columns:
        axes[plot_idx].scatter(df["decode_kv_len"], df["decode_time_ms"], alpha=0.6)
        axes[plot_idx].set_xlabel("Decode KV Length")
        axes[plot_idx].set_ylabel("Decode Time (ms)")
        axes[plot_idx].set_title(
            f"Decode KV Len vs Decode Time\n(r={correlations.get('decode_kv_len vs decode_time_ms', 0):.3f})"
        )
        z = np.polyfit(df["decode_kv_len"], df["decode_time_ms"], 1)
        p = np.poly1d(z)
        axes[plot_idx].plot(
            df["decode_kv_len"], p(df["decode_kv_len"]), "r--", alpha=0.8
        )
        plot_idx += 1
    elif "decode_kv_len" in df.columns:
        # If decode_time_ms not available, skip this plot
        pass

    # 6. decode_qo_len vs decode_time_ms (if available)
    if "decode_qo_len" in df.columns and "decode_time_ms" in df.columns:
        axes[plot_idx].scatter(df["decode_qo_len"], df["decode_time_ms"], alpha=0.6)
        axes[plot_idx].set_xlabel("Decode QO Length")
        axes[plot_idx].set_ylabel("Decode Time (ms)")
        axes[plot_idx].set_title(
            f"Decode QO Len vs Decode Time\n(r={correlations.get('decode_qo_len vs decode_time_ms', 0):.3f})"
        )
        z = np.polyfit(df["decode_qo_len"], df["decode_time_ms"], 1)
        p = np.poly1d(z)
        axes[plot_idx].plot(
            df["decode_qo_len"], p(df["decode_qo_len"]), "r--", alpha=0.8
        )
        plot_idx += 1
    elif "decode_qo_len" in df.columns:
        # If decode_time_ms not available, skip this plot
        pass

    # 7. decode_qo_len vs sm_time
    if "decode_qo_len" in df.columns and "sm_time" in df.columns:
        axes[plot_idx].scatter(df["decode_qo_len"], df["sm_time"], alpha=0.6)
        axes[plot_idx].set_xlabel("Decode QO Length")
        axes[plot_idx].set_ylabel("SM Time (ms)")
        axes[plot_idx].set_title(
            f"Decode QO Len vs SM Time\n(r={correlations.get('decode_qo_len vs sm_time', 0):.3f})"
        )
        z = np.polyfit(df["decode_qo_len"], df["sm_time"], 1)
        p = np.poly1d(z)
        axes[plot_idx].plot(
            df["decode_qo_len"], p(df["decode_qo_len"]), "r--", alpha=0.8
        )
        plot_idx += 1

    # 8. decode_kv_len vs sm_time
    if (
        "decode_kv_len" in df.columns
        and "sm_time" in df.columns
        and plot_idx < len(axes)
    ):
        axes[plot_idx].scatter(df["decode_kv_len"], df["sm_time"], alpha=0.6)
        axes[plot_idx].set_xlabel("Decode KV Length")
        axes[plot_idx].set_ylabel("SM Time (ms)")
        axes[plot_idx].set_title(
            f"Decode KV Len vs SM Time\n(r={correlations.get('decode_kv_len vs sm_time', 0):.3f})"
        )
        z = np.polyfit(df["decode_kv_len"], df["sm_time"], 1)
        p = np.poly1d(z)
        axes[plot_idx].plot(
            df["decode_kv_len"], p(df["decode_kv_len"]), "r--", alpha=0.8
        )
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
        sns.heatmap(
            corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", center=0, square=True
        )
        plt.title("Correlation Heatmap: Lengths vs Times")
        plt.tight_layout()
        plt.savefig("sm_performance_heatmap.png", dpi=150, bbox_inches="tight")
        print("Saved correlation heatmap to sm_performance_heatmap.png")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze correlations in SM performance data"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="sm_performance.csv",
        help="Path to the CSV file containing SM performance data",
    )
    args = parser.parse_args()
    main(args)
