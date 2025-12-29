#!/usr/bin/env python3
"""
Polynomial (degree-2) regression for SM execution time:

  Inputs:  decode_kv_len, decode_qo_len, prefill_kv_len, prefill_qo_len
  Target:  sm_time

We:
  1) Fit per-sample model
  2) Fit grouped-by-mean model (one point per length config)

No logs, no PD ratio, no tiling – just poly(degree=2) + ridge.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score


def compute_mape(y_true, y_pred, eps=1e-8, min_y_for_mape=0.0):
    """Mean absolute percentage error, optionally ignoring very small y."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mask = y_true > min_y_for_mape
    if not np.any(mask):
        return np.nan

    rel = np.abs(y_pred[mask] - y_true[mask]) / np.maximum(np.abs(y_true[mask]), eps)
    return rel.mean() * 100.0


def fit_and_eval(X, y, degree=2, alpha=1e-3, label=""):
    """
    Fit Poly(degree) + Ridge(alpha) and print metrics.

    Returns (y_pred, model, scaler, poly).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Standardize inputs
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    Xp = poly.fit_transform(Xs)

    # Ridge regression
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(Xp, y)

    y_pred = model.predict(Xp)

    mae = mean_absolute_error(y, y_pred)
    mape_all = compute_mape(y, y_pred, min_y_for_mape=0.0)
    # more meaningful MAPE: ignore tiny times (e.g. <0.02 ms)
    mape_big = compute_mape(y, y_pred, min_y_for_mape=0.02)
    r2 = r2_score(y, y_pred)

    print(f"\n=== {label} ===")
    print(f"Samples:          {len(y)}")
    print(f"Degree:           {degree}")
    print(f"Ridge alpha:      {alpha:g}")
    print(f"MAE:              {mae:.6f} ms")
    print(f"MAPE (all):       {mape_all:.2f} %")
    print(f"MAPE (y>=0.02ms): {mape_big:.2f} %")
    print(f"R^2:              {r2:.5f}")

    return (
        y_pred,
        model,
        scaler,
        poly,
        {
            "mae": mae,
            "mape_all": mape_all,
            "mape_big": mape_big,
            "r2": r2,
        },
    )


def main(args):
    df = pd.read_csv(args.csv_path)
    print("Data shape:", df.shape)
    print("Columns:", list(df.columns))

    # ------------------------------------------------------------------
    # 1) Per-sample model
    # ------------------------------------------------------------------
    X = df[
        ["decode_kv_len", "decode_qo_len", "prefill_kv_len", "prefill_qo_len"]
    ].to_numpy(float)
    y = df["sm_time"].to_numpy(float)

    # optionally drop pure-zero work rows
    work_mask = (X.sum(axis=1) > 0) & (y > 0)
    X_samp = X[work_mask]
    y_samp = y[work_mask]

    y_pred_samp, model_samp, scaler_samp, poly_samp, metrics_samp = fit_and_eval(
        X_samp,
        y_samp,
        degree=args.degree,
        alpha=args.alpha,
        label="Per-sample poly regression",
    )

    # ------------------------------------------------------------------
    # 2) Grouped-by-mean model
    # ------------------------------------------------------------------
    keys = ["decode_kv_len", "decode_qo_len", "prefill_kv_len", "prefill_qo_len"]
    grouped = df.groupby(keys).agg({"sm_time": "mean"}).reset_index()
    Xg = grouped[keys].to_numpy(float)
    yg = grouped["sm_time"].to_numpy(float)

    maskg = (Xg.sum(axis=1) > 0) & (yg > 0)
    Xg = Xg[maskg]
    yg = yg[maskg]

    y_pred_g, model_g, scaler_g, poly_g, metrics_g = fit_and_eval(
        Xg,
        yg,
        degree=args.degree,
        alpha=args.alpha,
        label="Grouped-by-mean poly regression",
    )

    # ------------------------------------------------------------------
    # 3) Plots for grouped model (clearer shape)
    # ------------------------------------------------------------------
    plt.figure(figsize=(12, 5))

    # Pred vs actual
    plt.subplot(1, 2, 1)
    plt.scatter(yg, y_pred_g, alpha=0.6)
    lo, hi = yg.min(), yg.max()
    plt.plot([lo, hi], [lo, hi], "r--", lw=2)
    plt.xlabel("Actual grouped sm_time (ms)")
    plt.ylabel("Predicted sm_time (ms)")
    plt.title(f"Grouped: Pred vs Actual (R²={metrics_g['r2']:.3f})")
    plt.grid(alpha=0.3)

    # Residuals
    plt.subplot(1, 2, 2)
    residuals_g = yg - y_pred_g
    plt.scatter(y_pred_g, residuals_g, alpha=0.6)
    plt.axhline(0.0, color="r", linestyle="--", lw=2)
    plt.xlabel("Predicted sm_time (ms)")
    plt.ylabel("Residual (ms)")
    plt.title("Grouped: Residuals vs Predicted")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("sm_cost_model_poly2.png", dpi=150, bbox_inches="tight")
    print("Saved plot to sm_cost_model_poly2.png")

    # Optionally save predictions for inspection
    out = grouped.loc[maskg, :].copy()
    out["sm_time_pred"] = y_pred_g
    out["abs_err_ms"] = np.abs(out["sm_time"] - out["sm_time_pred"])
    out.to_csv("sm_cost_model_poly2_grouped_predictions.csv", index=False)
    print("Saved grouped predictions to sm_cost_model_poly2_grouped_predictions.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit degree-2 polynomial SM cost model with ridge regression."
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="sm_performance_final_flipped.csv",
        help="Path to the CSV file.",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=2,
        help="Polynomial degree (default: 2).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-3,
        help="Ridge regularization strength (default: 1e-3).",
    )
    args = parser.parse_args()
    main(args)
