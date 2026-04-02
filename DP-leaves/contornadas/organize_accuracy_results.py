import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CONFIG -------------------------------------------------------------
CSV_PATH   = "results_filled_leaves.csv"
OUT_FOLDER = "graphs"
OUT_FILE   = "Global_Average_Accuracy.pdf"
# ------------------------------------------------------------------------

os.makedirs(OUT_FOLDER, exist_ok=True)
df = pd.read_csv(CSV_PATH)

# Convert columns to numeric
df["Noise Scale"] = pd.to_numeric(df["Noise Scale"], errors="coerce")
df["Accuracy"]    = pd.to_numeric(df["Accuracy"], errors="coerce") / 100  # normalize to [0,1]

# Group by Noise Scale (global average across all classes/seeds)
grouped = df.groupby("Noise Scale")["Accuracy"]
means   = grouped.mean()
stds    = 1.96 * grouped.std() / grouped.count().pow(0.5)

# Plot
plt.figure(figsize=(8, 5))
plt.errorbar(
    means.index, means.values, yerr=stds.values,
    fmt='o-', capsize=5, linewidth=2.5
)
plt.xlabel("Noise Scale", fontsize=16)
plt.ylabel("Global Average Accuracy", fontsize=16)
plt.ylim(0, 1)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()

out_path = os.path.join(OUT_FOLDER, OUT_FILE)
plt.savefig(out_path, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")
