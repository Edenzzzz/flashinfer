#!/usr/bin/env python3
"""
Script to fit a 3-layer MLP for SM execution time prediction.

Uses a neural network to learn the mapping from features to execution time.
Features: [term1, term2, term3, term4, term5, R_pd]
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm


class SMCostMLP(nn.Module):
    """3-layer MLP for predicting SM execution time."""

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim_1: int = 64,
        hidden_dim_2: int = 128,
        standardize: bool = True,
    ):
        super().__init__()
        # 3-layer MLP: input -> hidden -> hidden -> output
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, 1),
        )

        # Initialize weights with smaller values to prevent explosion
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        # Feature normalization stats (for 6 features: term1-5 + R_pd)
        self.register_buffer("feat_mean", torch.zeros(input_dim, dtype=torch.float32))
        self.register_buffer("feat_std", torch.ones(input_dim, dtype=torch.float32))
        self.standardize = standardize

    def forward(self, features):
        """Forward pass.

        Args:
            features: Tensor of shape [batch_size, 6] containing [term1, term2, term3, term4, term5, R_pd]

        Returns:
            Predicted time in log scale
        """
        # Standardize features if enabled (apply during both training and evaluation)
        if self.standardize:
            features = (features - self.feat_mean) / self.feat_std

        # Pass through MLP
        pred = self.net(features).squeeze(-1)
        return pred


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
    model: SMCostMLP,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    standardize: bool = True,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
):
    """Train the MLP model using gradient descent.

    Args:
        train_df: Training data with columns [term1, term2, term3, term4, term5, R_pd, sm_time]
        val_df: Validation data with columns [term1, term2, term3, term4, term5, R_pd, sm_time]
        model: Model to train
        device: Device to train on
        standardize: Whether to standardize features
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate

    Returns:
        Trained model and training history
    """
    # Prepare data
    train_features = torch.tensor(
        train_df[["term1", "term2", "term3", "term4", "term5", "R_pd"]].values,
        dtype=torch.float32,
    ).to(device)
    # Transform targets to log scale
    train_targets_raw = torch.tensor(
        train_df["sm_time"].values, dtype=torch.float32
    ).to(device)
    train_targets = torch.log(train_targets_raw + 1e-10)

    val_features = torch.tensor(
        val_df[["term1", "term2", "term3", "term4", "term5", "R_pd"]].values,
        dtype=torch.float32,
    ).to(device)
    val_targets_raw = torch.tensor(val_df["sm_time"].values, dtype=torch.float32).to(
        device
    )
    val_targets = torch.log(val_targets_raw + 1e-10)

    # Feature standardization
    if standardize:
        # Compute stats from training set
        feat_mean = train_features.mean(dim=0)
        feat_std = train_features.std(dim=0)
        # Avoid extremely small std
        eps = 1e-6
        feat_std = torch.where(feat_std < eps, torch.ones_like(feat_std), feat_std)
        # Store stats in the model
        with torch.no_grad():
            model.feat_mean.copy_(feat_mean)
            model.feat_std.copy_(feat_std)

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(train_features, train_targets)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Learning rate scheduler to reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    # Training loop with early stopping
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    patience = 50

    model.train()
    for epoch in tqdm(range(epochs), desc="Training"):
        epoch_train_loss = 0.0
        num_batches = 0

        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()

            # Forward pass
            pred = model(batch_features)
            loss = criterion(pred, batch_targets)

            # Check for NaN/Inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(
                    f"Warning: NaN/Inf loss detected at epoch {epoch + 1}, skipping batch"
                )
                continue

            # Backward pass
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_train_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_train_loss / num_batches if num_batches > 0 else 0.0

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(val_features)
            # Clip predictions to prevent overflow (log scale: reasonable range is roughly -20 to 20)
            val_pred = torch.clamp(val_pred, min=-20.0, max=20.0)
            val_loss = criterion(val_pred, val_targets).item()

            # Check for NaN/Inf
            if np.isnan(val_loss) or np.isinf(val_loss):
                print(f"Warning: NaN/Inf validation loss at epoch {epoch + 1}")
                val_loss = float("inf")
        model.train()

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)

        # Update learning rate scheduler
        scheduler.step(val_loss)

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1  # 1-indexed for reporting
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:
            print(
                f"\nEarly stopping at epoch {epoch + 1} (no improvement for {patience} epochs)"
            )
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()

    print(
        f"\nTraining completed. Best validation loss: {best_val_loss:.6f} (at epoch {best_epoch})"
    )
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final val loss: {history['val_loss'][-1]:.6f}")

    return model, history, best_val_loss, best_epoch


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
        df[["term1", "term2", "term3", "term4", "term5", "R_pd"]].values,
        dtype=torch.float32,
    ).to(device)
    targets = torch.tensor(df["sm_time"].values, dtype=torch.float32).to(device)

    with torch.no_grad():
        # Model predicts in log scale, so exponentiate to get back to original scale
        pred_log = model(features)
        # Clip predictions to prevent overflow (log scale: reasonable range is roughly -20 to 20)
        pred_log = torch.clamp(pred_log, min=-20.0, max=20.0)
        pred = torch.exp(pred_log).cpu().numpy()
        targets_np = targets.cpu().numpy()

        # Filter out any remaining NaN/Inf values
        valid_mask = (
            np.isfinite(pred) & np.isfinite(targets_np) & (pred > 0) & (targets_np > 0)
        )
        if not np.all(valid_mask):
            print(
                f"Warning: {np.sum(~valid_mask)} invalid predictions/targets filtered out"
            )
            pred = pred[valid_mask]
            targets_np = targets_np[valid_mask]

    if len(pred) == 0:
        print("Error: All predictions are invalid!")
        return {
            "mse": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "mean_deviation_pct": np.nan,
            "predictions": np.array([]),
            "targets": np.array([]),
        }

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
    print(f"Loading data from {args.csv}...")
    df = pd.read_csv(args.csv)

    print("=" * 80)
    print("SM Cost Model Fitting (3-Layer MLP)")
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
        val_df = df.sample(n=80)
        train_df = df.drop(val_df.index)

    print(f"\nTrain samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")

    # Create model
    model = SMCostMLP(
        input_dim=6,
        hidden_dim_1=args.hidden_dim_1,
        hidden_dim_2=args.hidden_dim_2,
        standardize=args.standardize,
    ).to(device)

    # Train model
    print("\n" + "=" * 80)
    print("Training model (3-layer MLP)...")
    print("=" * 80)
    model, history, best_val_loss, best_epoch = train_model(
        train_df,
        val_df,
        model,
        device=device,
        standardize=args.standardize,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
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
    plt.figure(figsize=(15, 5))

    # Plot 1: Predicted vs Actual
    plt.subplot(1, 3, 1)
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
    plt.subplot(1, 3, 2)
    residuals = metrics["targets"] - metrics["predictions"]
    plt.scatter(metrics["predictions"], residuals, alpha=0.6)
    plt.axhline(y=0, color="r", linestyle="--", lw=2)
    plt.xlabel("Predicted SM Time (ms)")
    plt.ylabel("Residuals (ms)")
    plt.title("Residual Plot")
    plt.grid(True, alpha=0.3)

    # Plot 3: Training history
    plt.subplot(1, 3, 3)
    plt.plot(history["train_loss"], label="Train Loss", alpha=0.7)
    plt.plot(history["val_loss"], label="Val Loss", alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (log scale)")
    plt.title("Training History")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")

    plt.tight_layout()
    plt.savefig("sm_cost_model_mlp_fit.png", dpi=150, bbox_inches="tight")
    print("\nSaved plot to sm_cost_model_mlp_fit.png")

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
    results_df.to_csv("sm_cost_model_mlp_predictions.csv", index=False)
    print("Saved predictions to sm_cost_model_mlp_predictions.csv")

    # Save model checkpoint (with best validation loss)
    if args.save_model:
        checkpoint_path = "sm_cost_model_mlp.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "history": history,
                "metrics": metrics,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "standardize": args.standardize,
                "hidden_dim_1": args.hidden_dim_1,
                "hidden_dim_2": args.hidden_dim_2,
            },
            checkpoint_path,
        )
        print(
            f"Saved model checkpoint to {checkpoint_path} (best val loss: {best_val_loss:.6f} at epoch {best_epoch})"
        )

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit 3-layer MLP for SM execution time prediction"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="profiler/sm_performance_final_flipped.csv",
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
        "--hidden-dim-1",
        type=int,
        default=64,
        help="Hidden dimension for MLP layers",
    )
    parser.add_argument(
        "--hidden-dim-2",
        type=int,
        default=128,
        help="Hidden dimension for MLP layers",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    args = parser.parse_args()
    main(args)
