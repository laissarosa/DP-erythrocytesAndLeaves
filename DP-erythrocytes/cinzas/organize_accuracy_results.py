import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CONFIG -------------------------------------------------------------
CSV_PATH   = "new_results_erythrocytes.csv"
OUT_FOLDER = "graficos"
# ------------------------------------------------------------------------

os.makedirs(OUT_FOLDER, exist_ok=True)
df = pd.read_csv(CSV_PATH)

# Filtra round=50 e método loss, sem filtrar por classe
filtered = df[
    (df['Round']  == 50) &
    (df['Method'] == 'loss')
].copy()

filtered['Accuracy'] = pd.to_numeric(filtered['Accuracy'], errors='coerce') / 100

# Agrupa por Noise Scale — média sobre todas as classes e seeds
grouped = filtered.groupby('Noise Scale')['Accuracy']
means   = grouped.mean()
stds    = 1.96 * grouped.std() / grouped.count().pow(0.5)

plt.figure(figsize=(8, 5))
plt.errorbar(
    means.index, means.values, yerr=stds.values,
    fmt='o-', capsize=5, linewidth=2.5
)
plt.xlabel('Ruído',          fontsize=16)
plt.ylabel('Acurácia média', fontsize=16)
plt.ylim(0, 1)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()

out_path = os.path.join(OUT_FOLDER, 'Acuracia_global_por_ruido.pdf')
plt.savefig(out_path, bbox_inches='tight')
plt.close()
print(f"Salvo: {out_path}")