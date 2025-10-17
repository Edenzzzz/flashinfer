import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

save_dir = "bench_plots/persistent"


def plot_persistent_trends(df_all):
    """Generate line graphs showing bandwidth trends across decode lengths for each prefill length."""

    # Clean up potential duplicated header rows and ensure numeric types
    if "scheduler" in df_all.columns:
        df_all = df_all[df_all["scheduler"] != "scheduler"]
    df_all["seq_cfg_id"] = pd.to_numeric(df_all["seq_cfg_id"], errors="coerce")
    df_all["bandwidth_GB_s"] = pd.to_numeric(df_all["bandwidth_GB_s"], errors="coerce")
    df_all = df_all.dropna(subset=["seq_cfg_id", "bandwidth_GB_s"])
    df_all["seq_cfg_id"] = df_all["seq_cfg_id"].astype(int)

    # Get unique prefill lengths and decode lengths
    prefill_lengths = sorted(df_all["prefill_len"].unique())
    # decode_lengths = sorted(df_all["decode_len"].unique())

    # Color and style mapping for schedulers (same as in plot_persistent_individual.py)
    scheduler_config = {
        "BatchAttentionWrapper (flipped)": {
            "color": "#1f77b4",
            "label": "Persistent (flipped)",
            "marker": "o",
        },
        "BatchAttentionWrapper": {
            "color": "#2ca02c",
            "label": "Persistent (original)",
            "marker": "s",
        },
        "BatchPrefillWithPagedKVCacheWrapper": {
            "color": "#ff7f0e",
            "label": "Batch Prefill (SGLang)",
            "marker": "^",
        },
        "Decode + Prefill": {
            "color": "#9467bd",
            "label": "Decode + Prefill (vLLM)",
            "marker": "d",
        },
    }

    # Create a plot for each prefill length
    for prefill_len in prefill_lengths:
        if prefill_len == 0:  # Skip decode-only cases
            continue

        plt.figure(figsize=(12, 8))

        # Filter data for this prefill length
        prefill_data = df_all[df_all["prefill_len"] == prefill_len]

        # Get schedulers in the desired legend/order if present in data
        desired_order = [
            "BatchAttentionWrapper (flipped)",
            "BatchAttentionWrapper",
            "BatchPrefillWithPagedKVCacheWrapper",
            "Decode + Prefill",
        ]
        present = set(prefill_data["scheduler"].unique())
        schedulers = [s for s in desired_order if s in present]

        # Plot lines for each scheduler
        for scheduler in schedulers:
            if scheduler not in scheduler_config:
                continue

            config = scheduler_config[scheduler]

            # Get data for this scheduler
            scheduler_data = prefill_data[prefill_data["scheduler"] == scheduler]

            # Group by decode length and calculate mean bandwidth
            decode_bandwidth = (
                scheduler_data.groupby("decode_len")["bandwidth_GB_s"]
                .mean()
                .reset_index()
            )

            # Sort by decode length
            decode_bandwidth = decode_bandwidth.sort_values("decode_len")

            if len(decode_bandwidth) > 0:
                # Plot the line
                plt.plot(
                    decode_bandwidth["decode_len"] // 1024,  # Convert to k
                    decode_bandwidth["bandwidth_GB_s"],
                    color=config["color"],
                    label=config["label"],
                    marker=config["marker"],
                    linewidth=2,
                    markersize=8,
                    markerfacecolor=config["color"],
                    markeredgecolor="white",
                    markeredgewidth=1,
                )

                # Add bandwidth values as text labels at each point
                for _, row in decode_bandwidth.iterrows():
                    plt.annotate(
                        f"{row['bandwidth_GB_s']:.1f}",
                        (row["decode_len"] // 1024, row["bandwidth_GB_s"]),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                        color=config["color"],
                    )

        # Get num_decode_reqs for filename (should be consistent across all data)
        num_decode_reqs = (
            df_all["num_decode_reqs"].iloc[0]
            if "num_decode_reqs" in df_all.columns
            else 128
        )
        # Customize the plot
        plt.xlabel("Decode Length (K tokens)", fontsize=12)
        plt.ylabel("Average Bandwidth (GB/s)", fontsize=12)
        plt.title(
            f"Bandwidth Trends - {prefill_len // 1024}K Prefill, {num_decode_reqs} Decode Reqs",
            fontsize=14,
            fontweight="bold",
        )

        # Set x-axis ticks based on actual decode lengths present for this prefill length
        decode_ticks = sorted(
            {int(d) // 1024 for d in prefill_data["decode_len"].unique() if d > 0}
        )
        if decode_ticks:
            plt.xticks(decode_ticks)

        # Add grid for better readability
        plt.grid(True, alpha=0.3, linestyle="--")

        # Set y-axis to start from 0
        plt.ylim(bottom=0)

        # Add legend
        plt.legend(fontsize=10, loc="best", framealpha=0.9)

        # Improve layout
        plt.tight_layout()

        # Save the plot
        plot_filename = f"{save_dir}/persistent_trends_{prefill_len // 1024}k_prefill_{num_decode_reqs}_decode_reqs.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        print(f"Trend plot saved as: {plot_filename}")
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot persistent attention bandwidth trends"
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default="bench_batch_attention.csv",
        help="CSV file to load data from",
    )
    args = parser.parse_args()

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Load CSV data
    df_all = pd.read_csv(args.csv_file)

    # Generate trend plots
    plot_persistent_trends(df_all)


if __name__ == "__main__":
    main()
