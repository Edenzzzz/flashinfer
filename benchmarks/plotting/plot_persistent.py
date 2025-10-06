import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

cases = {"1": "chunked"}
# Load consolidated CSV (contains flipped, original, prefill, and decode+prefill)
df_all = pd.read_csv("bench_batch_attention_flipped.csv")

# Clean up potential duplicated header rows after appends and ensure numeric types
if "scheduler" in df_all.columns:
    df_all = df_all[df_all["scheduler"] != "scheduler"]
df_all["seq_cfg_id"] = pd.to_numeric(df_all["seq_cfg_id"], errors="coerce")
df_all["bandwidth_GB_s"] = pd.to_numeric(df_all["bandwidth_GB_s"], errors="coerce")
df_all = df_all.dropna(subset=["seq_cfg_id", "bandwidth_GB_s"])  # keep only valid rows
df_all["seq_cfg_id"] = df_all["seq_cfg_id"].astype(int)
repeats = df_all["num_repeats"].unique()

# Pick an available case (prefer 4: hybrid-chunked). Avoid empty plots when requested case is missing
available_cases = sorted(df_all["seq_cfg_id"].dropna().unique().tolist())
if not available_cases:
    raise ValueError("No rows found in bench_batch_attention_flipped.csv")
preferred_case = 1 if 1 in available_cases else available_cases[0]
selected_case_label = cases.get(str(preferred_case), f"case {preferred_case}")

for repeat in repeats:
    # Collect data for all cases
    case_names = []
    persistent_flipped_values = []
    batch_prefill_values = []
    persistent_original_values = []
    decode_prefill_values = []

    for case in [str(preferred_case)]:
        # Compute averages for this specific repeat count
        persistent_flipped_avg = df_all[
            (df_all["scheduler"] == "BatchAttentionWrapper (flipped)")
            & (df_all["seq_cfg_id"] == int(case))
            & (df_all["num_repeats"] == repeat)
        ]["bandwidth_GB_s"].mean()
        batch_prefill_avg = df_all[
            (df_all["scheduler"] == "BatchPrefillWithPagedKVCacheWrapper")
            & (df_all["seq_cfg_id"] == int(case))
            & (df_all["num_repeats"] == repeat)
        ]["bandwidth_GB_s"].mean()
        decode_prefill_avg = df_all[
            (df_all["scheduler"] == "Decode + Prefill")
            & (df_all["seq_cfg_id"] == int(case))
            & (df_all["num_repeats"] == repeat)
        ]["bandwidth_GB_s"].mean()
        persistent_original_avg = df_all[
            (df_all["scheduler"] == "BatchAttentionWrapper")
            & (df_all["seq_cfg_id"] == int(case))
            & (df_all["num_repeats"] == repeat)
        ]["bandwidth_GB_s"].mean()

        case_names.append(selected_case_label)
        # Replace NaNs with zeros to ensure bars render even if some schedulers are missing
        persistent_flipped_values.append(np.nan_to_num(persistent_flipped_avg, nan=0.0))
        batch_prefill_values.append(np.nan_to_num(batch_prefill_avg, nan=0.0))
        decode_prefill_values.append(np.nan_to_num(decode_prefill_avg, nan=0.0))
        persistent_original_values.append(
            np.nan_to_num(persistent_original_avg, nan=0.0)
        )

    # Create grouped bar plot
    x = np.arange(len(case_names))
    width = 0.2
    # Center four bars around each category to avoid large empty margins when only one case
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(
        x + offsets[0],
        persistent_flipped_values,
        width,
        label="Persistent (flipped)",
        color="#1f77b4",
    )
    bars2 = plt.bar(
        x + offsets[1],
        batch_prefill_values,
        width,
        label="Batch Prefill",
        color="#ff7f0e",
    )
    bars3 = plt.bar(
        x + offsets[2],
        decode_prefill_values,
        width,
        label="Decode + Prefill",
        color="#9467bd",
    )
    bars4 = plt.bar(
        x + offsets[3],
        persistent_original_values,
        width,
        label="Persistent (original)",
        color="#2ca02c",
    )

    plt.xlabel("Case Type")
    plt.ylabel("Average Bandwidth (GB/s)")
    plt.title(f"Average Bandwidth Comparison ({selected_case_label}, {repeat} repeats)")
    plt.xticks(x, case_names)
    plt.legend()

    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height == 0:
                continue
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
    add_value_labels(bars4)

    plt.tight_layout()
    plt.savefig(
        f"persistent_comparison_{repeat}_repeats.png", dpi=300, bbox_inches="tight"
    )
    plt.show()
