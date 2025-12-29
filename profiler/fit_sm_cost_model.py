#!/usr/bin/env python3
"""
Script to fit a regression model for SM execution time prediction.

Model:
  T = Speedup * (θ₁*x₁ + θ₂*x₂ + θ₃*x₃ + θ₄*x₄ + θ₅*x₅) + β
  where Speedup = α₁*R_{pd} + α₂*R_{pd}² + α₃

Where:
  - x₁ = sum_i (r_{i,p} * n_{i,p}) = prefill_kv_len * prefill_qo_len
  - x₂ = sum_i r_{i,p} = prefill_kv_len
  - x₃ = sum_i n_{i,p} = prefill_qo_len
  - x₄ = sum_i r_{i,d}^2 = decode_kv_len^2
  - x₅ = sum_i r_{i,d} = decode_kv_len
  - R_{pd} = S_d / (S_p + S_d)
  - S_p = sum_i n_{i,p} * (sum_i n_{i,p} / sum_i(n_{i,p} + r_{i,p}))
  - S_d = sum_i r_{i,d}
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class SMCostModel(nn.Module):
    """Performance model for predicting SM execution time.

    Model: T = Speedup * (θ₁*x₁ + θ₂*x₂ + θ₃*x₃ + θ₄*x₄ + θ₅*x₅) + β
    where Speedup = α₁*R_{pd} + α₂*R_{pd}² + α₃
    """

    def __init__(self, standardize: bool = True, use_speedup: bool = False):
        super().__init__()
        # Model parameters
        self.theta_1 = nn.Parameter(
            torch.randn(1, dtype=torch.float32)
        )  # r_p*n_p coefficient
        self.theta_2 = nn.Parameter(
            torch.randn(1, dtype=torch.float32)
        )  # r_p coefficient
        self.theta_3 = nn.Parameter(
            torch.randn(1, dtype=torch.float32)
        )  # n_p coefficient
        self.theta_4 = nn.Parameter(
            torch.randn(1, dtype=torch.float32)
        )  # r_d² coefficient
        self.theta_5 = nn.Parameter(
            torch.randn(1, dtype=torch.float32)
        )  # r_d coefficient
        self.beta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))  # bias term

        # Speedup parameters
        self.alpha_1 = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float32)
        )  # R_pd coefficient
        self.alpha_2 = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float32)
        )  # R_pd² coefficient
        self.alpha_3 = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float32)
        )  # constant term

        # Feature normalization stats
        self.register_buffer("feat_mean", torch.zeros(5, dtype=torch.float32))
        self.register_buffer("feat_std", torch.ones(5, dtype=torch.float32))
        self.standardize = standardize
        self.use_speedup = use_speedup

    def forward(self, term1, term2, term3, term4, term5, R_pd):
        """Forward pass.

        Args:
            term1: r_p * n_p
            term2: r_p
            term3: n_p
            term4: r_d²
            term5: r_d
            R_pd: R_{pd} = S_d / (S_p + S_d) for speedup calculation

        Returns:
            Predicted time in milliseconds
        """
        # Standardize features if enabled
        if self.standardize and not self.training:
            features = torch.stack([term1, term2, term3, term4, term5], dim=0)
            mean = self.feat_mean.view(-1, 1)
            std = self.feat_std.view(-1, 1)
            features = (features - mean) / std
            term1, term2, term3, term4, term5 = features

        # Calculate weighted features
        weighted_features = (
            self.theta_1 * term1
            + self.theta_2 * term2
            + self.theta_3 * term3
            + self.theta_4 * term4
            + self.theta_5 * term5
        )
        # Calculate speedup (or use constant 1.0 if speedup is disabled)
        if self.use_speedup:
            speedup = self.alpha_1 * R_pd + self.alpha_2 * (R_pd**2) + self.alpha_3
            pred = speedup * weighted_features + self.beta

        else:
            pred = weighted_features + self.beta
        return pred

    def get_params(self):
        """Get model parameters as a dictionary."""
        return {
            "alpha_1": self.alpha_1.item(),
            "alpha_2": self.alpha_2.item(),
            "alpha_3": self.alpha_3.item(),
            "theta_1": self.theta_1.item(),
            "theta_2": self.theta_2.item(),
            "theta_3": self.theta_3.item(),
            "theta_4": self.theta_4.item(),
            "theta_5": self.theta_5.item(),
            "beta": self.beta.item(),
            "feat_mean_term1": float(self.feat_mean[0].item()),
            "feat_mean_term2": float(self.feat_mean[1].item()),
            "feat_mean_term3": float(self.feat_mean[2].item()),
            "feat_mean_term4": float(self.feat_mean[3].item()),
            "feat_mean_term5": float(self.feat_mean[4].item()),
            "feat_std_term1": float(self.feat_std[0].item()),
            "feat_std_term2": float(self.feat_std[1].item()),
            "feat_std_term3": float(self.feat_std[2].item()),
            "feat_std_term4": float(self.feat_std[3].item()),
            "feat_std_term5": float(self.feat_std[4].item()),
        }


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute model features from raw profiling data.

    Args:
        df: DataFrame with columns [prefill_qo_len, prefill_kv_len, decode_kv_len, sm_time]

    Returns:
        DataFrame with added feature columns [term1, term2, term3, term4, term5, R_pd]
    """
    df = df.copy()

    # Extract variables
    n_p = df["prefill_qo_len"].values  # n_{i,p}
    r_p = df["prefill_kv_len"].values  # r_{i,p}
    r_d = df["decode_kv_len"].values  # r_{i,d}

    # Compute model terms
    df["term1"] = r_p * n_p  # r_{i,p} * n_{i,p}
    df["term2"] = r_p  # r_{i,p}
    df["term3"] = n_p  # n_{i,p}
    df["term4"] = r_d**2  # r_{i,d}^2
    df["term5"] = r_d  # r_{i,d}

    # Calculate S_p, S_d, R_{pd}
    # S_p = sum_i n_{i,p} * (sum_i n_{i,p} / sum_i(n_{i,p} + r_{i,p}))
    # For a single SM, if we have aggregated values:
    # sum_i n_{i,p} = prefill_qo_len (total)
    # sum_i r_{i,p} = prefill_kv_len (total)
    # sum_i r_{i,d} = decode_kv_len (total)
    sum_n_p = n_p
    sum_r_p = r_p
    sum_r_d = r_d

    S_p = sum_n_p * (sum_n_p / (sum_n_p + sum_r_p + 1e-10))
    S_d = sum_r_d
    df["R_pd"] = S_d / (S_p + S_d + 1e-10)

    return df


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model: SMCostModel,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    standardize: bool = True,
    use_speedup: bool = True,
):
    """Train the performance model using least squares.

    Args:
        train_df: Training data with columns [term1, term2, term3, term4, term5, R_pd, sm_time]
        val_df: Validation data with columns [term1, term2, term3, term4, term5, R_pd, sm_time]
        model: Model to train
        device: Device to train on
        standardize: Whether to standardize features

    Returns:
        Trained model and training history
    """
    # Prepare data
    train_features_raw = torch.tensor(
        train_df[["term1", "term2", "term3", "term4", "term5"]].values,
        dtype=torch.float32,
    ).to(device)
    train_R_pd = torch.tensor(train_df["R_pd"].values, dtype=torch.float32).to(device)
    # Transform targets to log scale
    train_targets_raw = torch.tensor(
        train_df["sm_time"].values, dtype=torch.float32
    ).to(device)
    train_targets = torch.log(
        train_targets_raw + 1e-10
    )  # Add small epsilon to avoid log(0)

    val_features = torch.tensor(
        val_df[["term1", "term2", "term3", "term4", "term5"]].values,
        dtype=torch.float32,
    ).to(device)
    val_R_pd = torch.tensor(val_df["R_pd"].values, dtype=torch.float32).to(device)
    # Transform targets to log scale
    val_targets_raw = torch.tensor(val_df["sm_time"].values, dtype=torch.float32).to(
        device
    )
    val_targets = torch.log(
        val_targets_raw + 1e-10
    )  # Add small epsilon to avoid log(0)

    # Feature standardization
    if standardize:
        # Compute stats from training set
        feat_mean = train_features_raw.mean(dim=0)
        feat_std = train_features_raw.std(dim=0)
        # Avoid extremely small std
        eps = 1e-6
        feat_std = torch.where(feat_std < eps, torch.ones_like(feat_std), feat_std)
        # Store stats in the model
        with torch.no_grad():
            model.feat_mean.copy_(feat_mean)
            model.feat_std.copy_(feat_std)

        # Standardize train features
        train_features = (train_features_raw - feat_mean) / feat_std
    else:
        train_features = train_features_raw

    # Solve using least squares
    # Model: T = Speedup * (θ₁*x₁ + θ₂*x₂ + θ₃*x₃ + θ₄*x₄ + θ₅*x₅) + β
    # where Speedup = α₁*R + α₂*R² + α₃
    # This expands to:
    # T = α₁*R*(θ₁*x₁ + ...) + α₂*R²*(θ₁*x₁ + ...) + α₃*(θ₁*x₁ + ...) + β
    # This is nonlinear, so we need to solve iteratively or use a different approach

    # Approach: Solve for θ's and α's together using least squares on the expanded form
    # For each sample i:
    #   T_i = α₁*R_i*(θ₁*x₁ᵢ + θ₂*x₂ᵢ + θ₃*x₃ᵢ + θ₄*x₄ᵢ + θ₅*x₅ᵢ)
    #        + α₂*R_i²*(θ₁*x₁ᵢ + θ₂*x₂ᵢ + θ₃*x₃ᵢ + θ₄*x₄ᵢ + θ₅*x₅ᵢ)
    #        + α₃*(θ₁*x₁ᵢ + θ₂*x₂ᵢ + θ₃*x₃ᵢ + θ₄*x₄ᵢ + θ₅*x₅ᵢ)
    #        + β

    # This can be rewritten as:
    # T_i = (α₁*R_i + α₂*R_i² + α₃) * (θ₁*x₁ᵢ + θ₂*x₂ᵢ + θ₃*x₃ᵢ + θ₄*x₄ᵢ + θ₅*x₅ᵢ) + β

    # We'll use an iterative approach: first solve for θ's assuming speedup=1, then solve for α's

    # Step 1: Solve for θ's assuming no speedup (speedup = 1.0)
    term1_t, term2_t, term3_t, term4_t, term5_t = train_features.T
    X_base = torch.stack([term1_t, term2_t, term3_t, term4_t, term5_t], dim=1).to(
        torch.float64
    )
    ones = torch.ones((X_base.shape[0], 1), dtype=torch.float64, device=device)
    X_base = torch.cat([X_base, ones], dim=1)  # [N, 6]
    y_base = train_targets.to(torch.float64).unsqueeze(1)  # [N, 1]

    solution_base = torch.linalg.lstsq(X_base, y_base)
    w_base = solution_base.solution.squeeze(1)  # [6]

    # Assign base parameters
    with torch.no_grad():
        model.theta_1.copy_(w_base[0].to(torch.float32).unsqueeze(0))
        model.theta_2.copy_(w_base[1].to(torch.float32).unsqueeze(0))
        model.theta_3.copy_(w_base[2].to(torch.float32).unsqueeze(0))
        model.theta_4.copy_(w_base[3].to(torch.float32).unsqueeze(0))
        model.theta_5.copy_(w_base[4].to(torch.float32).unsqueeze(0))
        model.beta.copy_(w_base[5].to(torch.float32))

    # Step 2: Solve for α's using fixed θ's (only if speedup is enabled)
    if use_speedup:
        # T = (α₁*R + α₂*R² + α₃) * (θ₁*x₁ + θ₂*x₂ + θ₃*x₃ + θ₄*x₄ + θ₅*x₅) + β
        # Rearranging: T - β = (α₁*R + α₂*R² + α₃) * weighted_features
        # Let w = θ₁*x₁ + θ₂*x₂ + θ₃*x₃ + θ₄*x₄ + θ₅*x₅
        # Then: (T - β) / w = α₁*R + α₂*R² + α₃ (if w != 0)

        theta_1 = model.theta_1.to(torch.float64)
        theta_2 = model.theta_2.to(torch.float64)
        theta_3 = model.theta_3.to(torch.float64)
        theta_4 = model.theta_4.to(torch.float64)
        theta_5 = model.theta_5.to(torch.float64)
        beta = model.beta.to(torch.float64)

        if standardize:
            feat_mean_f64 = model.feat_mean.to(device).to(torch.float64)
            feat_std_f64 = model.feat_std.to(device).to(torch.float64)
            train_features_std = (
                train_features_raw.to(torch.float64) - feat_mean_f64
            ) / feat_std_f64
        else:
            train_features_std = train_features_raw.to(torch.float64)

        term1_n, term2_n, term3_n, term4_n, term5_n = train_features_std.T
        R = train_R_pd.to(torch.float64)
        y = train_targets.to(torch.float64)

        # Compute weighted features
        w = (
            theta_1 * term1_n
            + theta_2 * term2_n
            + theta_3 * term3_n
            + theta_4 * term4_n
            + theta_5 * term5_n
        )

        # Avoid division by zero
        w_safe = torch.where(torch.abs(w) < 1e-10, torch.ones_like(w), w)

        # Solve for α's: (T - β) / w = α₁*R + α₂*R² + α₃
        target = (y - beta) / w_safe
        X_alpha = torch.stack([R, R**2, torch.ones_like(R)], dim=1)  # [N, 3]

        solution_alpha = torch.linalg.lstsq(X_alpha, target.unsqueeze(1))
        alpha = solution_alpha.solution.squeeze(1)  # [3]

        with torch.no_grad():
            model.alpha_1.copy_(alpha[0].to(torch.float32))
            model.alpha_2.copy_(alpha[1].to(torch.float32))
            model.alpha_3.copy_(alpha[2].to(torch.float32))
    else:
        # If speedup is disabled, set alpha parameters so speedup = 1.0
        with torch.no_grad():
            model.alpha_1.copy_(torch.tensor(0.0, dtype=torch.float32))
            model.alpha_2.copy_(torch.tensor(0.0, dtype=torch.float32))
            model.alpha_3.copy_(torch.tensor(1.0, dtype=torch.float32))

    # Compute training and validation losses
    model.eval()
    with torch.no_grad():
        # Training loss (on log scale)
        if standardize:
            term1_t, term2_t, term3_t, term4_t, term5_t = train_features.T
        else:
            term1_t, term2_t, term3_t, term4_t, term5_t = train_features_raw.T
        pred_train_log = model(term1_t, term2_t, term3_t, term4_t, term5_t, train_R_pd)
        train_mse = torch.mean((pred_train_log - train_targets) ** 2).item()

        # Validation loss (on log scale)
        if standardize:
            val_features_std = (val_features - feat_mean) / feat_std
            term1_v, term2_v, term3_v, term4_v, term5_v = val_features_std.T
        else:
            term1_v, term2_v, term3_v, term4_v, term5_v = val_features.T
        pred_val_log = model(term1_v, term2_v, term3_v, term4_v, term5_v, val_R_pd)
        val_mse = torch.mean((pred_val_log - val_targets) ** 2).item()

    history = {
        "train_loss": [train_mse],
        "val_loss": [val_mse],
    }

    print(f"Fitted parameters (least squares): {model.get_params()}")
    print(f"Train MSE: {train_mse:.6f}")
    print(f"Val MSE:   {val_mse:.6f}")

    return model, history


def evaluate_model(model: nn.Module, df: pd.DataFrame, device: str = "cpu"):
    """Evaluate model on a dataset.

    Args:
        model: Trained model
        df: DataFrame with columns [term1, term2, term3, term4, term5, R_pd, sm_time]
        device: Device to evaluate on

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()

    features = torch.tensor(
        df[["term1", "term2", "term3", "term4", "term5"]].values,
        dtype=torch.float32,
    ).to(device)
    R_pd = torch.tensor(df["R_pd"].values, dtype=torch.float32).to(device)
    targets = torch.tensor(df["sm_time"].values, dtype=torch.float32).to(device)

    with torch.no_grad():
        term1, term2, term3, term4, term5 = features.T
        # Model predicts in log scale, so exponentiate to get back to original scale
        pred_log = model(term1, term2, term3, term4, term5, R_pd)
        pred = torch.exp(pred_log).cpu().numpy()
        targets_np = targets.cpu().numpy()

    mse = np.mean((pred - targets_np) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred - targets_np))

    # R²
    ss_res = np.sum((targets_np - pred) ** 2)
    ss_tot = np.sum((targets_np - np.mean(targets_np)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Mean deviation percentage
    nonzero_mask = np.abs(targets_np) > 1e-10
    if np.any(nonzero_mask):
        mean_deviation_pct = (
            np.mean(
                np.abs(
                    (pred[nonzero_mask] - targets_np[nonzero_mask])
                    / targets_np[nonzero_mask]
                )
            )
            * 100
        )
    else:
        mean_deviation_pct = np.nan

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mean_deviation_pct": mean_deviation_pct,
        "predictions": pred,
        "targets": targets_np,
    }


def main(args):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Read the SM performance data
    print(f"Loading data from {args.csv_path}...")
    df = pd.read_csv(args.csv_path)

    print("=" * 80)
    print("SM Cost Model Fitting")
    print("=" * 80)
    print(f"\nData shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())

    # Compute features
    print("\nComputing features...")
    df = compute_features(df)
    print(
        f"Features computed. R_pd range: [{df['R_pd'].min():.6f}, {df['R_pd'].max():.6f}]"
    )

    # Split into train/val
    if args.val_split > 0:
        n_val = int(len(df) * args.val_split)
        val_df = df.iloc[:n_val].copy()
        train_df = df.iloc[n_val:].copy()
    else:
        train_df = df.copy()
        val_df = df.iloc[: min(100, len(df))].copy()  # Small validation set

    print(f"\nTrain samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")

    # Create model
    model = SMCostModel(standardize=args.standardize, use_speedup=args.speedup).to(
        device
    )

    # Train model
    print("\n" + "=" * 80)
    print("Training model (least squares)...")
    if args.speedup:
        print("Speedup term: ENABLED")
    else:
        print("Speedup term: DISABLED")
    print("=" * 80)
    model, history = train_model(
        train_df,
        val_df,
        model,
        device=device,
        standardize=args.standardize,
        use_speedup=args.speedup,
    )

    # Evaluate
    print("\n" + "=" * 80)
    print("Evaluation Results:")
    print("=" * 80)
    metrics = evaluate_model(model, df, device=device)
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f} ms")
    print(f"MAE: {metrics['mae']:.6f} ms")
    print(f"R²: {metrics['r2']:.6f}")
    print(f"Mean Deviation %: {metrics['mean_deviation_pct']:.4f}%")

    # Create visualization
    plt.figure(figsize=(12, 5))

    # Plot 1: Predicted vs Actual
    plt.subplot(1, 2, 1)
    plt.scatter(metrics["targets"], metrics["predictions"], alpha=0.6)
    plt.plot(
        [metrics["targets"].min(), metrics["targets"].max()],
        [metrics["targets"].min(), metrics["targets"].max()],
        "r--",
        lw=2,
    )
    plt.xlabel("Actual SM Time (ms)")
    plt.ylabel("Predicted SM Time (ms)")
    plt.title(f"Predicted vs Actual (R² = {metrics['r2']:.3f})")
    plt.grid(True, alpha=0.3)

    # Plot 2: Residuals
    plt.subplot(1, 2, 2)
    residuals = metrics["targets"] - metrics["predictions"]
    plt.scatter(metrics["predictions"], residuals, alpha=0.6)
    plt.axhline(y=0, color="r", linestyle="--", lw=2)
    plt.xlabel("Predicted SM Time (ms)")
    plt.ylabel("Residuals (ms)")
    plt.title("Residual Plot")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("sm_cost_model_fit.png", dpi=150, bbox_inches="tight")
    print("\nSaved plot to sm_cost_model_fit.png")

    # Save model parameters
    params_df = pd.DataFrame(
        {
            "parameter": [
                "alpha_1",
                "alpha_2",
                "alpha_3",
                "theta_1",
                "theta_2",
                "theta_3",
                "theta_4",
                "theta_5",
                "beta",
            ],
            "value": [
                model.alpha_1.item(),
                model.alpha_2.item(),
                model.alpha_3.item(),
                model.theta_1.item(),
                model.theta_2.item(),
                model.theta_3.item(),
                model.theta_4.item(),
                model.theta_5.item(),
                model.beta.item(),
            ],
            "description": [
                "R_pd coefficient",
                "R_pd² coefficient",
                "Speedup intercept",
                "r_p*n_p coefficient",
                "r_p coefficient",
                "n_p coefficient",
                "r_d² coefficient",
                "r_d coefficient",
                "Bias term",
            ],
        }
    )
    params_df.to_csv("sm_cost_model_parameters.csv", index=False)
    print("Saved model parameters to sm_cost_model_parameters.csv")

    # Save predictions
    results_df = pd.DataFrame(
        {
            "prefill_qo_len": df["prefill_qo_len"].values,
            "prefill_kv_len": df["prefill_kv_len"].values,
            "decode_kv_len": df["decode_kv_len"].values,
            "actual_time": metrics["targets"],
            "predicted_time": metrics["predictions"],
            "residual": residuals,
            "R_pd": df["R_pd"].values,
        }
    )
    results_df.to_csv("sm_cost_model_predictions.csv", index=False)
    print("Saved predictions to sm_cost_model_predictions.csv")

    # Save model checkpoint
    if args.save_model:
        checkpoint_path = "sm_cost_model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "params": model.get_params(),
                "history": history,
                "metrics": metrics,
                "standardize": args.standardize,
                "use_speedup": args.speedup,
            },
            checkpoint_path,
        )
        print(f"Saved model checkpoint to {checkpoint_path}")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit regression model for SM execution time prediction"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="sm_performance_final.csv",
        help="Path to the CSV file containing SM performance data",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.0,
        help="Fraction of data to use for validation (0.0 = use all for training, use first 100 for val)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Standardize features during training and evaluation",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save model checkpoint",
    )
    parser.add_argument(
        "--speedup",
        action="store_true",
        help="Enable speedup term in the model (default: disabled)",
    )
    args = parser.parse_args()
    main(args)
