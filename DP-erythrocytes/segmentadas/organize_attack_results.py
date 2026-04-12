import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CONFIG -------------------------------------------------------------
CSV_PATH   = "new_results_erythrocytes.csv"
OUT_FOLDER = "graficos"

CLASS_ORDER   = ["round", "sickle", "others"]  # ordem desejada para as classes
NAME_MAPPING  = {'loss': 'Gradiente', 'naive': 'Ingênua'}
STYLE_MAPPING = {'loss': '-',         'naive': '--'}
# ------------------------------------------------------------------------

os.makedirs(OUT_FOLDER, exist_ok=True)
df = pd.read_csv(CSV_PATH)

# =============================================================================
# Gráficos de BARRA — métrica por classe, ruído = 0.0, round = 50
# =============================================================================

def plot_metric_by_class(data, metric, method, filename):
    filtered = data[
        (data['Round']       == 50) &
        (data['Method']      == method) &
        (data['Noise Scale'] == 0.0)
    ]
    grouped = filtered.groupby('Label')[metric].mean()
    grouped = grouped.reindex(CLASS_ORDER)   # garante ordem: circular, falciforme, outras

    plt.figure()
    grouped.plot(kind='bar', rot=0)
    plt.xlabel('Classe', fontsize=25)
    plt.ylabel(metric,   fontsize=25)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Salvo: {filename}")

# =============================================================================
# Gráficos de LINHA — métrica por escala de ruído, round = 50
# =============================================================================

def plot_metric_over_noise(data, metric, filename):
    filtered = data[data['Round'] == 50]   # todas as noise scales

    grouped = filtered.groupby(['Noise Scale', 'Method'])[metric].mean().reset_index()

    plt.figure()
    for method in grouped['Method'].unique():
        method_data = grouped[grouped['Method'] == method]
        plt.plot(
            method_data['Noise Scale'],
            method_data[metric],
            label=NAME_MAPPING.get(method, method),
            linestyle=STYLE_MAPPING.get(method, '-'),
            linewidth=2.5
        )
    plt.xlabel('Ruído', fontsize=25)
    plt.ylabel(metric,  fontsize=25)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Salvo: {filename}")

# =============================================================================
# Geração dos gráficos
# =============================================================================

for metric in ['MSE', 'PSNR', 'SSIM']:
    plot_metric_by_class(df, metric=metric, method='loss',
                         filename=f'{OUT_FOLDER}/{metric}_por_classe.pdf')
    plot_metric_over_noise(df, metric=metric,
                           filename=f'{OUT_FOLDER}/{metric}_por_ruido_medio.pdf')

print("\nTodos os gráficos gerados.")