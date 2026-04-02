import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CONFIG -------------------------------------------------------------
CSV_PATH   = "results_filled_leaves.csv"
OUT_FOLDER = "graphs"
# ------------------------------------------------------------------------

os.makedirs(OUT_FOLDER, exist_ok=True)
df = pd.read_csv(CSV_PATH)

# --- MSE by class (Round 50, Noise Scale = 0.0, Method = loss) ---
def plot_MSE(data, method, filename):
    filtered = data[(data['Round'] == 50) & (data['Method'] == method) & (data['Noise Scale'] == 0.0)]
    grouped = filtered.groupby('Label')['MSE'].mean()

    plt.figure()
    grouped.plot(kind='bar', rot=0)
    plt.xlabel('Class', fontsize=20)
    plt.ylabel('MSE', fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

plot_MSE(df, method='loss', filename=os.path.join(OUT_FOLDER, 'MSE_by_class.pdf'))

# --- MSE over Noise Scales (global average) ---
def plot_MSE_over_Noise_Scales(data, filename):
    filtered = data[(data['Round'] == 50) & (data['Noise Scale'] < 0.06)]
    grouped = filtered.groupby(['Noise Scale','Method'])['MSE'].mean().reset_index()

    style_mapping = {'loss': '-', 'naive': '--'}

    plt.figure()
    for method in grouped['Method'].unique():
        method_data = grouped[grouped['Method'] == method]
        plt.plot(
            method_data['Noise Scale'],
            method_data['MSE'],
            label=method,
            linestyle=style_mapping.get(method, '-'),
            linewidth=2.5
        )
    plt.xlabel('Noise Scale', fontsize=20)
    plt.ylabel('Average MSE', fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

plot_MSE_over_Noise_Scales(df, filename=os.path.join(OUT_FOLDER, 'MSE_over_noise.pdf'))

# --- PSNR by class ---
def plot_PSNR(data, method, filename):
    filtered = data[(data['Round'] == 50) & (data['Method'] == method) & (data['Noise Scale'] == 0.0)]
    grouped = filtered.groupby('Label')['PSNR'].mean()

    plt.figure()
    grouped.plot(kind='bar', rot=0)
    plt.xlabel('Class', fontsize=20)
    plt.ylabel('PSNR', fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

plot_PSNR(df, method='loss', filename=os.path.join(OUT_FOLDER, 'PSNR_by_class.pdf'))

# --- PSNR over Noise Scales ---
def plot_PSNR_over_Noise_Scales(data, filename):
    filtered = data[(data['Round'] == 50) & (data['Noise Scale'] < 0.06)]
    grouped = filtered.groupby(['Noise Scale','Method'])['PSNR'].mean().reset_index()

    style_mapping = {'loss': '-', 'naive': '--'}

    plt.figure()
    for method in grouped['Method'].unique():
        method_data = grouped[grouped['Method'] == method]
        plt.plot(
            method_data['Noise Scale'],
            method_data['PSNR'],
            label=method,
            linestyle=style_mapping.get(method, '-'),
            linewidth=2.5
        )
    plt.xlabel('Noise Scale', fontsize=20)
    plt.ylabel('Average PSNR', fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

plot_PSNR_over_Noise_Scales(df, filename=os.path.join(OUT_FOLDER, 'PSNR_over_noise.pdf'))

# --- SSIM by class ---
def plot_SSIM(data, method, filename):
    filtered = data[(data['Round'] == 50) & (data['Method'] == method) & (data['Noise Scale'] == 0.0)]
    grouped = filtered.groupby('Label')['SSIM'].mean()

    plt.figure()
    grouped.plot(kind='bar', rot=0)
    plt.xlabel('Class', fontsize=20)
    plt.ylabel('SSIM', fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

plot_SSIM(df, method='loss', filename=os.path.join(OUT_FOLDER, 'SSIM_by_class.pdf'))

# --- SSIM over Noise Scales ---
def plot_SSIM_over_Noise_Scales(data, filename):
    filtered = data[(data['Round'] == 50) & (data['Noise Scale'] < 0.06)]
    grouped = filtered.groupby(['Noise Scale','Method'])['SSIM'].mean().reset_index()

    style_mapping = {'loss': '-', 'naive': '--'}

    plt.figure()
    for method in grouped['Method'].unique():
        method_data = grouped[grouped['Method'] == method]
        plt.plot(
            method_data['Noise Scale'],
            method_data['SSIM'],
            label=method,
            linestyle=style_mapping.get(method, '-'),
            linewidth=2.5
        )
    plt.xlabel('Noise Scale', fontsize=20)
    plt.ylabel('Average SSIM', fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

plot_SSIM_over_Noise_Scales(df, filename=os.path.join(OUT_FOLDER, 'SSIM_over_noise.pdf'))
