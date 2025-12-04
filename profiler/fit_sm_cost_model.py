"""
Script to fit a regression model for SM execution time prediction.
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def main(args):
    # Read the SM performance data
    df = pd.read_csv(args.csv_path)

    print("=" * 80)
    print("SM Cost Model Fitting")
    print("=" * 80)
    print(f"\nData shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())

    # Extract variables
    # n_{i,p} = prefill_qo_len
    # r_{i,p} = prefill_kv_len
    # r_{i,d} = decode_kv_len
    n_p = df['prefill_qo_len'].values
    r_p = df['prefill_kv_len'].values
    r_d = df['decode_kv_len'].values

    # Target: total execution time
    y = df['sm_time'].values

    print(f"\nData ranges:")
    print(f"prefill_qo_len: [{n_p.min()}, {n_p.max()}]")
    print(f"prefill_kv_len: [{r_p.min()}, {r_p.max()}]")
    print(f"decode_kv_len: [{r_d.min()}, {r_d.max()}]")
    print(f"sm_time: [{y.min()}, {y.max()}]")

    # Calculate intermediate variables for each SM
    # For each SM, we sum over its work items
    # S_p = sum_i n_{i,p} * (sum_i n_{i,p} / sum_i(n_{i,p} + r_{i,p}))
    # S_d = sum_i r_{i,d}
    # R_{pd} = S_d / (S_p + S_d)

    # For a single SM, if we have aggregated values:
    # sum_i n_{i,p} = prefill_qo_len (total)
    # sum_i r_{i,p} = prefill_kv_len (total)
    # sum_i r_{i,d} = decode_kv_len (total)

    sum_n_p = n_p  # Total prefill qo length for this SM
    sum_r_p = r_p  # Total prefill kv length for this SM
    sum_r_d = r_d  # Total decode kv length for this SM

    # Calculate S_p, S_d, R_{pd} for each SM
    S_p = sum_n_p * (sum_n_p / (sum_n_p + sum_r_p + 1e-10))  # Add small epsilon to avoid division by zero
    S_d = sum_r_d
    R_pd = S_d / (S_p + S_d + 1e-10)  # Add small epsilon to avoid division by zero

    # Calculate features for the regression
    # Features in the model:
    # 1. sum_i (r_{i,p} * n_{i,p}) = prefill_kv_len * prefill_qo_len
    # 2. sum_i r_{i,p} = prefill_kv_len
    # 3. sum_i n_{i,p} = prefill_qo_len
    # 4. sum_i r_{i,d}^2 = decode_kv_len^2
    # 5. sum_i r_{i,d} = decode_kv_len

    feature_rp_np = sum_r_p * sum_n_p  # r_{i,p} * n_{i,p}
    feature_rp = sum_r_p
    feature_np = sum_n_p
    feature_rd_sq = sum_r_d ** 2
    feature_rd = sum_r_d

    # Build feature matrix X
    # The model is: T = Speedup * (θ₁*feature1 + θ₂*feature2 + θ₃*feature3 + θ₄*feature4 + θ₅*feature5) + β
    # where Speedup = α₁*R_{pd} + α₂*R_{pd}² + α₃

    # We need to solve for: α₁, α₂, α₃, θ₁, θ₂, θ₃, θ₄, θ₅, β
    # This is a nonlinear regression problem

    # Standardize features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_features = np.column_stack([
        feature_rp_np,
        feature_rp,
        feature_np,
        feature_rd_sq,
        feature_rd,
    ])

    X_scaled = scaler_X.fit_transform(X_features)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    print(f"\nFeature matrix shape: {X_scaled.shape}")

    # Define the model function
    def model(params, X, R_pd):
        """
        Model: T = Speedup * (θ₁*x₁ + θ₂*x₂ + θ₃*x₃ + θ₄*x₄ + θ₅*x₅) + β
        where Speedup = α₁*R_{pd} + α₂*R_{pd}² + α₃
        """
        alpha1, alpha2, alpha3, theta1, theta2, theta3, theta4, theta5, beta = params
        
        # Calculate speedup for each sample
        speedup = alpha1 * R_pd + alpha2 * (R_pd ** 2) + alpha3
        
        # Calculate weighted features
        weighted_features = (
            theta1 * X[:, 0] +  # r_{i,p} * n_{i,p}
            theta2 * X[:, 1] +  # r_{i,p}
            theta3 * X[:, 2] +  # n_{i,p}
            theta4 * X[:, 3] +  # r_{i,d}^2
            theta5 * X[:, 4]    # r_{i,d}
        )
        
        # Apply speedup and add bias
        y_pred = speedup * weighted_features + beta
        return y_pred

    # Define loss function (mean squared error)
    def loss_function(params, X, y, R_pd):
        y_pred = model(params, X, R_pd)
        mse = np.mean((y - y_pred) ** 2)
        return mse

    # Initial parameter guess
    # [alpha1, alpha2, alpha3, theta1, theta2, theta3, theta4, theta5, beta]
    initial_params = np.array([1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])

    print("\nFitting regression model...")
    print("Initial parameters:", initial_params)

    # Fit the model using scipy.optimize
    result = minimize(
        loss_function,
        initial_params,
        args=(X_scaled, y_scaled, R_pd),
        method='L-BFGS-B',
        options={'maxiter': 1000, 'disp': True}
    )

    if result.success:
        alpha1, alpha2, alpha3, theta1, theta2, theta3, theta4, theta5, beta = result.x
        
        print("\n" + "=" * 80)
        print("Fitted Parameters:")
        print("=" * 80)
        print(f"α₁ (R_pd coefficient): {alpha1:.6f}")
        print(f"α₂ (R_pd² coefficient): {alpha2:.6f}")
        print(f"α₃ (speedup intercept): {alpha3:.6f}")
        print(f"θ₁ (r_p*n_p coefficient): {theta1:.6f}")
        print(f"θ₂ (r_p coefficient): {theta2:.6f}")
        print(f"θ₃ (n_p coefficient): {theta3:.6f}")
        print(f"θ₄ (r_d² coefficient): {theta4:.6f}")
        print(f"θ₅ (r_d coefficient): {theta5:.6f}")
        print(f"β (bias): {beta:.6f}")
        
        # Calculate predictions
        y_pred = model(result.x, X_scaled, R_pd)
        
        # Unscale predictions and actual values
        y_pred_unscaled = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        y_actual_unscaled = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
        
        # Calculate metrics
        mse = np.mean((y_actual_unscaled - y_pred_unscaled) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_actual_unscaled - y_pred_unscaled))
        r2 = 1 - np.sum((y_actual_unscaled - y_pred_unscaled) ** 2) / np.sum((y_actual_unscaled - np.mean(y_actual_unscaled)) ** 2)
        
        print("\n" + "=" * 80)
        print("Model Performance:")
        print("=" * 80)
        print(f"MSE: {mse:.6f}")
        print(f"RMSE: {rmse:.6f} ms")
        print(f"MAE: {mae:.6f} ms")
        print(f"R²: {r2:.6f}")
        
        # Create visualization
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Predicted vs Actual
        plt.subplot(1, 2, 1)
        plt.scatter(y_actual_unscaled, y_pred_unscaled, alpha=0.6)
        plt.plot([y_actual_unscaled.min(), y_actual_unscaled.max()], 
                 [y_actual_unscaled.min(), y_actual_unscaled.max()], 'r--', lw=2)
        plt.xlabel('Actual SM Time (ms)')
        plt.ylabel('Predicted SM Time (ms)')
        plt.title(f'Predicted vs Actual (R² = {r2:.3f})')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Residuals
        plt.subplot(1, 2, 2)
        residuals = y_actual_unscaled - y_pred_unscaled
        plt.scatter(y_pred_unscaled, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        plt.xlabel('Predicted SM Time (ms)')
        plt.ylabel('Residuals (ms)')
        plt.title('Residual Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("sm_cost_model_fit.png", dpi=150, bbox_inches="tight")
        print("\nSaved plot to sm_cost_model_fit.png")
        
        # Save model parameters
        params_df = pd.DataFrame({
            'parameter': ['alpha1', 'alpha2', 'alpha3', 'theta1', 'theta2', 'theta3', 'theta4', 'theta5', 'beta'],
            'value': result.x,
            'description': [
                'R_pd coefficient',
                'R_pd² coefficient',
                'Speedup intercept',
                'r_p*n_p coefficient',
                'r_p coefficient',
                'n_p coefficient',
                'r_d² coefficient',
                'r_d coefficient',
                'Bias term'
            ]
        })
        params_df.to_csv("sm_cost_model_parameters.csv", index=False)
        print("Saved model parameters to sm_cost_model_parameters.csv")
        
        # Save predictions
        results_df = pd.DataFrame({
            'prefill_qo_len': n_p,
            'prefill_kv_len': r_p,
            'decode_kv_len': r_d,
            'actual_time': y_actual_unscaled,
            'predicted_time': y_pred_unscaled,
            'residual': residuals,
            'R_pd': R_pd,
        })
        results_df.to_csv("sm_cost_model_predictions.csv", index=False)
        print("Saved predictions to sm_cost_model_predictions.csv")
        
    else:
        print(f"\nOptimization failed: {result.message}")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit regression model for SM execution time prediction")
    parser.add_argument(
        "--csv-path",
        type=str,
        default="sm_performance_final.csv",
        help="Path to the CSV file containing SM performance data",
    )
    args = parser.parse_args()
    main(args)

