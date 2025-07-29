import matplotlib.pyplot as plt
import pandas as pd

cases = {"1": "decode-only", "2": "prefill-only", "3": "hybrid", "4": "hybrid-chunked"}
# Load CSVs
df_reverse = pd.read_csv("bench_batch_attention_flipped.csv")
df = pd.read_csv("bench_batch_attention.csv")
for case in cases:
    # Compute averages
    persistent_flipped_avg = df_reverse[
        (df_reverse["scheduler"] == "BatchAttentionWrapper")
        & (df_reverse["seq_cfg_id"] == case)
    ]["bandwidth_GB_s"].mean()
    batch_prefill_avg = df[
        (df["scheduler"] == "BatchPrefillWithPagedKVCacheWrapper")
        & (df["seq_cfg_id"] == case)
    ]["bandwidth_GB_s"].mean()
    persistent_original_avg = df[
        (df["scheduler"] == "BatchAttentionWrapper") & (df["seq_cfg_id"] == case)
    ]["bandwidth_GB_s"].mean()

    # Prepare data for bar plot
    labels = ["Persistent (flipped)", "Batch Prefill", "Persistent (original)"]
    values = [persistent_flipped_avg, batch_prefill_avg, persistent_original_avg]

    # Plot
    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, values, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.ylabel("Average Bandwidth (GB/s)")
    plt.title(f"Average Bandwidth Comparison ({cases[case]})")
    plt.ylim(0, max(values) * 1.1)
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 10,
            f"{yval:.1f}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(f"persistent_{cases[case]}.png")
    plt.close()
