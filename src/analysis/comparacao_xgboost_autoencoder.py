import os
import re
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from xgboost import XGBClassifier

DRILL_PATH = r"C:\Users\beatr\OneDrive\Desktop\Projeto_Brocas_AE\data\segmented"
SAVE_PATH = r"C:\Users\beatr\OneDrive\Desktop\Projeto_Brocas_AE\src\plots_todos_testes"
CANAL_ALVO = "4"
N_NORMAL = 5
N_MFCC = 20

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

def extract_features_46d(path):
    try:
        y, sr = librosa.load(path, sr=None, mono=True)
        rms = librosa.feature.rms(y=y)[0]
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

        return np.hstack([
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
            np.mean(rms), np.std(rms),
            np.mean(centroid), np.std(centroid),
            np.mean(bandwidth), np.std(bandwidth)
        ])
    except: return None

processed_data = []
print("Iniciando carregamento de todas as brocas...")

pastas = [p for p in os.listdir(DRILL_PATH) if "drill_4mm" in p.lower()]

for pasta in pastas:
    drill_match = re.search(r"drill_4mm_(\d+)", pasta.lower())
    if drill_match:
        drill_num = int(drill_match.group(1))
        caminho_pasta = os.path.join(DRILL_PATH, pasta)
        temp_holes = []

        for root, _, files in os.walk(caminho_pasta):
            for f in files:
                if f.lower().endswith(".wav") and f"tr{CANAL_ALVO}" in f.lower():
                    h_match = re.search(r"hole(\d+)", f.lower())
                    if h_match:
                        temp_holes.append((int(h_match.group(1)), os.path.join(root, f)))

        if temp_holes:
            temp_holes.sort()
            n_total = len(temp_holes)
            ponto_corte = int(n_total * 0.8)

            print(f"Lendo Drill {drill_num:02d} - {n_total} furos encontrados.")
            for i, (h_idx, path) in enumerate(temp_holes):
                feat = extract_features_46d(path)
                if feat is not None:
                    label = 1 if i >= ponto_corte else 0
                    processed_data.append({'drill': drill_num, 'hole': h_idx, 'features': feat, 'label': label})

df = pd.DataFrame(processed_data)
lista_brocas = sorted(df['drill'].unique())

print(f"\nIniciando geração de gráficos para {len(lista_brocas)} brocas...")

for TEST_DRILL in lista_brocas:
    df_train = df[df['drill'] != TEST_DRILL]
    df_test = df[df['drill'] == TEST_DRILL].sort_values('hole')

    scaler = StandardScaler()
    X_train_raw = np.stack(df_train['features'].values)
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(np.stack(df_test['features'].values))

    X_ae_train = []
    for d in df_train['drill'].unique():
        subset = df_train[(df_train['drill'] == d) & (df_train['hole'] <= N_NORMAL)]
        if not subset.empty:
            X_ae_train.append(np.stack(subset['features'].values))

    X_ae_train_scaled = scaler.transform(np.vstack(X_ae_train))
    autoencoder = MLPRegressor(hidden_layer_sizes=(32,16,32), activation="relu", max_iter=1000, random_state=42)
    autoencoder.fit(X_ae_train_scaled, X_ae_train_scaled)

    train_recon = autoencoder.predict(X_ae_train_scaled)
    train_err = np.mean((X_ae_train_scaled - train_recon)**2, axis=1)
    thresh_ae = np.percentile(train_err, 99.5)

    xgb = XGBClassifier(n_estimators=100, random_state=42)
    xgb.fit(X_train_scaled, df_train['label'].values)

    ae_errors = np.mean((X_test_scaled - autoencoder.predict(X_test_scaled))**2, axis=1)
    xgb_probs = xgb.predict_proba(X_test_scaled)[:, 1]
    furos = df_test['hole'].values

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.titlesize": 11,
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "text.usetex": False
    })

    fig, ax1 = plt.subplots(figsize=(3.5, 3.0))

    idx_falha = int(len(furos) * 0.8)
    ax1.axvspan(furos[idx_falha], furos[-1], color='red', alpha=0.1, label='Critical (20%)')

    ax1.plot(furos, ae_errors, color='tab:green', marker='o', markersize=3,
             linewidth=1.0, label='AE Error')
    ax1.axhline(thresh_ae, color='tab:green', linestyle='--', linewidth=1.2,
                label=f'AE Thresh ({thresh_ae:.3f})')

    ax1.set_xlabel('Hole Sequence', labelpad=2)
    ax1.set_ylabel('AE MSE ($\mathcal{L}$)', color='tab:green', labelpad=2)
    ax1.tick_params(axis='y', labelcolor='tab:green')

    ax1.set_ylim(0, max(max(ae_errors)*1.1, thresh_ae*2))

    ax2 = ax1.twinx()
    ax2.plot(furos, xgb_probs, color='tab:orange', linestyle='-.', marker='s',
             markersize=3, markerfacecolor='white', alpha=0.7, label='XGB Conf.')
    ax2.axhline(0.5, color='tab:orange', linestyle=':', linewidth=1.2)

    ax2.set_ylabel('XGB Prob. ($P_f$)', color='tab:orange', labelpad=2)
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.set_ylim(-0.05, 1.05)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', frameon=True, edgecolor='inherit')

    plt.grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout()

    nome_arquivo = f"drill_{TEST_DRILL:02d}_temporal_monitoring.png"
    plt.savefig(os.path.join(SAVE_PATH, nome_arquivo), dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.close()

print(f"\n[SUCESSO] Processamento finalizado! Gráficos salvos em: {SAVE_PATH}")