import numpy as np, json

X_train = np.load("../data/dataset/processed_T4_improved/X_train.npy")
y_train = np.load("../data/dataset/processed_T4_improved/y_train.npy")
X_val   = np.load("../data/dataset/processed_T4_improved/X_val.npy")
y_val   = np.load("../data/dataset/processed_T4_improved/y_val.npy")

with open("../data/dataset/processed_T4_improved/dataset_stats.json") as f:
    stats = json.load(f)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val:  ", X_val.shape)
print("y_val:  ", y_val.shape)
print()
print("y_train  min=%.2f  max=%.2f  mean=%.2f" % (y_train.min(), y_train.max(), y_train.mean()))
print("y_val    min=%.2f  max=%.2f  mean=%.2f" % (y_val.min(),   y_val.max(),   y_val.mean()))
print()
print("Channel stats:")
for ch in ["Z_low","ZDR_low","Z_high","ZDR_high"]:
    print(f"  {ch:10s}  mean={stats[ch]['mean']:.4f}  std={stats[ch]['std']:.4f}")
print()
print("Build info:", stats["build_info"])