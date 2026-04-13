import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os


ROOT_DATA = r"C:\Users\beatr\OneDrive\Desktop\Projeto_Brocas_AE\data\segmented"
DRILL_TARGET = "drill_4mm_06"
CANAL = "Tr1"

def buscar_arquivo_manual(raiz, drill_folder, num_furo):
    for item in os.listdir(raiz):
        if drill_folder in item.lower():
            caminho_drill = os.path.join(raiz, item)
            for root, dirs, files in os.walk(caminho_drill):
                for f in files:
                    nome_furo = f"hole{str(num_furo).zfill(5)}"
                    if nome_furo in f.lower() and CANAL.lower() in f.lower() and f.endswith(".wav"):
                        return os.path.join(root, f)
    return None

file_normal = buscar_arquivo_manual(ROOT_DATA, DRILL_TARGET, 1)
file_failure = buscar_arquivo_manual(ROOT_DATA, DRILL_TARGET, 24)

if not file_normal or not file_failure:
    print(f"Raiz verificada: {ROOT_DATA}")
    if os.path.exists(ROOT_DATA):
        print(f"Pastas na raiz: {os.listdir(ROOT_DATA)[:3]}...")
else:
    print(f"Furo 01: {os.path.basename(file_normal)}")
    print(f"Furo 24: {os.path.basename(file_failure)}")

    plt.style.use('seaborn-v0_8-muted')
    fig, ax = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    files = [file_normal, file_failure]
    titles = ['(a) Stable Operating Condition (Hole 1)', '(b) Critical Wear Phase (Hole 24)']

    for i, path in enumerate(files):
        y, sr = librosa.load(path, sr=None)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=4096)), ref=np.max)

        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear',
                                       ax=ax[i], cmap='magma', vmin=-80, vmax=0)

        ax[i].set_title(titles[i], fontsize=14, pad=15, fontweight='bold')
        ax[i].set_xlabel('Time (s)', fontsize=12)
        if i == 0: ax[i].set_ylabel('Frequency (Hz)', fontsize=12)
        ax[i].set_ylim(0, 48000)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(img, cax=cbar_ax, format="%+2.0f dB")
    cbar_ax.set_ylabel('Intensity (dB)', rotation=270, labelpad=15)

    plt.savefig("comparativo_espectral_furo1_vs_24_v2.png", dpi=300, bbox_inches='tight')
    plt.show()