import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

cases = {"1": "decode-only", "2": "prefill-only", "3": "hybrid", "4": "hybrid-chunked"}
# Load CSVs
df_reverse = pd.read_csv("bench_batch_attention_flipped.csv")
df = pd.read_csv("bench_batch_attention.csv")

# Collect data for all cases
case_names = []
persistent_flipped_values = []
batch_prefill_values = []
persistent_original_values = []

for case in cases:
    # Compute averages
    persistent_flipped_avg = df_reverse[
        (df_reverse["scheduler"] == "BatchAttentionWrapper")
        & (df_reverse["seq_cfg_id"] == int(case))
    ]["bandwidth_GB_s"].mean()
    batch_prefill_avg = df[
        (df["scheduler"] == "BatchPrefillWithPagedKVCacheWrapper")
        & (df["seq_cfg_id"] == int(case))
    ]["bandwidth_GB_s"].mean()
    persistent_original_avg = df[
        (df["scheduler"] == "BatchAttentionWrapper") & (df["seq_cfg_id"] == int(case))
    ]["bandwidth_GB_s"].mean()

    case_names.append(cases[case])
    persistent_flipped_values.append(persistent_flipped_avg)
    batch_prefill_values.append(batch_prefill_avg)
    persistent_original_values.append(persistent_original_avg)

# Create grouped bar plot
x = np.arange(len(case_names))
width = 0.25

plt.figure(figsize=(12, 6))
bars1 = plt.bar(
    x - width,
    persistent_flipped_values,
    width,
    label="Persistent (flipped)",
    color="#1f77b4",
)
bars2 = plt.bar(x, batch_prefill_values, width, label="Batch Prefill", color="#ff7f0e")
bars3 = plt.bar(
    x + width,
    persistent_original_values,
    width,
    label="Persistent (original)",
    color="#2ca02c",
)

plt.xlabel("Case Type")
plt.ylabel("Average Bandwidth (GB/s)")
plt.title("Average Bandwidth Comparison Across All Cases")
plt.xticks(x, case_names)
plt.legend()


# Add value labels on bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 10,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)

plt.tight_layout()
plt.savefig("persistent_comparison_all_cases.png", dpi=300, bbox_inches="tight")
plt.show()
