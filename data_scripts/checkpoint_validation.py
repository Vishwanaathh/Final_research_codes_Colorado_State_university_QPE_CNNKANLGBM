import numpy as np
data = np.load("../data/dataset/processed_T4_improved/checkpoints/000049.npz", allow_pickle=False)
X = data["X"]
y = data["y"]
print("X shape:", X.shape)      # should be (69, 4, 9, 9)
print("y shape:", y.shape)      # should be (69,)
print("y min/max:", y.min(), y.max())   # should be > 1.0 mm/h
print("X[0] channel means:", X[0].mean(axis=(1,2)))  # Z_low, ZDR_low, Z_high, ZDR_high
print("Any NaN in X:", np.isnan(X).any())   # should be False