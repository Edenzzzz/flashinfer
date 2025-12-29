#!/usr/bin/env python3
"""
Simple tile-based SM cost model:

  T_sm ≈ c1 * P + c2 * D + c3 * min(P, D) + c0

where
  P = ceil(prefill_qo_len / 128) * ceil(prefill_kv_len / 64)
  D = ceil(decode_qo_len / 16)  * ceil(decode_kv_len / 128)

Expected CSV columns:
  decode_kv_len, decode_qo_len, prefill_kv_len, prefill_qo_len,
  decode_time_ms, prefill_time_ms, sm_time
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ceil_div(a, b):
    a = np.asarray(a, dtype=np.int64)
    return (a + b - 1) // b


def compute_tiles(df: pd.DataFrame):
    q_p = df["prefill_qo_len"].to_numpy(np.int64)
    k_p = df["prefill_kv_len"].to_numpy(np.int64)
    q_d = df["decode_qo_len"].to_numpy(np.int64)
    k_d = df["decode_kv_len"].to_numpy(np.int64)

    # Prefill: q tile 128, k tile 64
    P = ceil_div(q_p, 128) * ceil_div(k_p, 64)

    # Decode: q tile 16, kv tile 128
    D = ceil_div(q_d, 16) * ceil_div(k_d, 128)

    M = np.minimum(P, D)

    return P.astype(np.float64), D.astype(np.float64), M.astype(np.float64)


def fit_linear(X, y):
    """OLS with intercept: returns theta, where y ≈ X_design @ theta."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    ones = np.ones((X.shape[0], 1), dtype=np.float64)
    X_design = np.concatenate([X, ones], axis=1)
    theta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    y_pred = X_design @ theta
    return theta, y_pred


def compute_metrics(y, y_pred, label=""):
    y = np.asarray(y, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    res = y - y_pred

    mae = np.mean(np.abs(res))
    mape = np.mean(np.abs(res) / np.clip(y, 1e-8, None)) * 100.0
    ss_res = np.sum(res**2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    print(f"\n=== {label} ===")
    print(f"MAE:   {mae:.6f} ms")
    print(f"MAPE:  {mape:.3f} %")
    print(f"R^2:   {r2:.5f}")

    return {
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "residuals": res,
        "y": y,
        "y_pred": y_pred,
    }


def main(args):
    df = pd.read_csv(args.csv_path)
    print("Data shape:", df.shape)
    print("Columns:", list(df.columns))

    # ----------------------------------------------------------------------
    # 1) Per-sample model
    # ----------------------------------------------------------------------
    P, D, M = compute_tiles(df)
    y = df["sm_time"].to_numpy(np.float64)

    # optionally drop pure-zero work to avoid silly % errors
    mask = (P + D) > 0
    P, D, M, y = P[mask], D[mask], M[mask], y[mask]

    X = np.stack([P, D, M], axis=1)
    theta, y_pred = fit_linear(X, y)

    print("\nFitted coefficients (per-sample):")
    print(f"c1 (P):           {theta[0]:.6e}")
    print(f"c2 (D):           {theta[1]:.6e}")
    print(f"c3 (min(P,D)):    {theta[2]:.6e}")
    print(f"c0 (intercept):   {theta[3]:.6e}")

    # metrics_sample = compute_metrics(y, y_pred, "Per-sample")

    # ----------------------------------------------------------------------
    # 2) Grouped-by-mean model (one point per length config)
    # ----------------------------------------------------------------------
    keys = ["decode_kv_len", "decode_qo_len", "prefill_kv_len", "prefill_qo_len"]
    grouped = df.groupby(keys).agg({"sm_time": "mean"}).reset_index()

    P_g, D_g, M_g = compute_tiles(grouped)
    y_g = grouped["sm_time"].to_numpy(np.float64)

    mask_g = (P_g + D_g) > 0
    P_g, D_g, M_g, y_g = P_g[mask_g], D_g[mask_g], M_g[mask_g], y_g[mask_g]

    X_g = np.stack([P_g, D_g, M_g], axis=1)
    theta_g, y_pred_g = fit_linear(X_g, y_g)

    print("\nFitted coefficients (grouped means):")
    print(f"c1_g (P):         {theta_g[0]:.6e}")
    print(f"c2_g (D):         {theta_g[1]:.6e}")
    print(f"c3_g (min(P,D)):  {theta_g[2]:.6e}")
    print(f"c0_g (intercept): {theta_g[3]:.6e}")

    metrics_group = compute_metrics(y_g, y_pred_g, "Grouped-by-mean")

    # ----------------------------------------------------------------------
    # 3) Quick plots for the grouped model (usually more stable)
    # ----------------------------------------------------------------------
    plt.figure(figsize=(12, 5))

    # Pred vs actual
    plt.subplot(1, 2, 1)
    plt.scatter(y_g, y_pred_g, alpha=0.6)
    lo, hi = y_g.min(), y_g.max()
    plt.plot([lo, hi], [lo, hi], "r--", lw=2)
    plt.xlabel("Actual grouped sm_time (ms)")
    plt.ylabel("Predicted sm_time (ms)")
    plt.title(f"Grouped: Pred vs Actual (R²={metrics_group['r2']:.3f})")
    plt.grid(alpha=0.3)

    # Residuals
    plt.subplot(1, 2, 2)
    res_g = metrics_group["residuals"]
    plt.scatter(y_pred_g, res_g, alpha=0.6)
    plt.axhline(0.0, color="r", linestyle="--", lw=2)
    plt.xlabel("Predicted sm_time (ms)")
    plt.ylabel("Residual (ms)")
    plt.title("Grouped: Residuals vs Predicted")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("sm_cost_model_simple_tiles.png", dpi=150, bbox_inches="tight")
    print("\nSaved plot to sm_cost_model_simple_tiles.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple tile-based SM cost model (no R_pd, no logs)."
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="sm_performance_final_flipped.csv",
        help="Path to per-SM CSV.",
    )
    args = parser.parse_args()
    main(args)
