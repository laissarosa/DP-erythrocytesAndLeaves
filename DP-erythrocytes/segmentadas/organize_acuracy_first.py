import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CONFIG -------------------------------------------------------------
CSV_PATH   = "acuracia_primeira_camada.csv"
OUT_FOLDER = "graficos"
# ------------------------------------------------------------------------

os.makedirs(OUT_FOLDER, exist_ok=True)
df = pd.read_csv(CSV_PATH)

# Filtra round=49 (última rodada, índice começa em 0)
# e agrupa por noise_scale — média global sobre todas as classes e seeds
filtered = df[df['round_num'] == 49].copy()

filtered['accuracy_cliente'] = pd.to_numeric(filtered['accuracy_cliente'], errors='coerce') / 100

grouped = filtered.groupby('noise_scale')['accuracy_cliente']
means   = grouped.mean()
stds    = 1.96 * grouped.std() / grouped.count().pow(0.5)

plt.figure(figsize=(8, 5))
plt.errorbar(
    means.index, means.values, yerr=stds.values,
    fmt='o-', capsize=5, linewidth=2.5
)
plt.xlabel('Ruído',                        fontsize=16)
plt.ylabel('Acurácia média (1ª camada)',   fontsize=16)
plt.ylim(0, 1)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()

out_path = os.path.join(OUT_FOLDER, 'Acuracia_global_primeira_camada.pdf')
plt.savefig(out_path, bbox_inches='tight')
plt.close()
print(f"Salvo: {out_path}")