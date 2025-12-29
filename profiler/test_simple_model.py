import numpy as np
import pandas as pd

df = pd.read_csv("profiler/sm_performance_final_flipped.csv")

X = df[["decode_kv_len", "decode_qo_len", "prefill_kv_len", "prefill_qo_len"]].to_numpy(
    float
)
y = df["sm_time"].to_numpy(float)

keys = ["decode_kv_len", "decode_qo_len", "prefill_kv_len", "prefill_qo_len"]
g = df.groupby(keys)["sm_time"]
mean_df = g.mean().reset_index()
Xg = mean_df[keys].to_numpy(float)
yg = mean_df["sm_time"].to_numpy(float)

X_design_g = np.concatenate([Xg, np.ones((Xg.shape[0], 1))], axis=1)
theta_g, *_ = np.linalg.lstsq(X_design_g, yg, rcond=None)
yg_pred = X_design_g @ theta_g

res_g = yg - yg_pred
mae_g = np.mean(np.abs(res_g))
mape_g = np.mean(np.abs(res_g) / np.clip(yg, 1e-8, None)) * 100
ss_res_g = np.sum(res_g**2)
ss_tot_g = np.sum((yg - yg.mean()) ** 2)
r2_g = 1 - ss_res_g / ss_tot_g

print("Grouped MAE:", mae_g, "ms")
print("Grouped MAPE:", mape_g, "%")
print("Grouped R^2:", r2_g)
