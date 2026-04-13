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
    print("Arquivos não encontrados.")
else:
    plt.rcParams.update({'font.size': 14})

    fig, ax = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

    files = [file_normal, file_failure]
    labels = ['(a)', '(b)']

    for i, path in enumerate(files):
        y, sr = librosa.load(path, sr=None)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=4096)), ref=np.max)

        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear',
                                       ax=ax[i], cmap='magma', vmin=-80, vmax=0)

        ax[i].set_xlabel(f'Time (s)\n\n{labels[i]}', fontsize=18, fontweight='bold', labelpad=10)

        ax[i].tick_params(axis='both', labelsize=14)

        if i == 0:
            ax[i].set_ylabel('Frequency (Hz)', fontsize=18, fontweight='bold')

        ax[i].set_ylim(0, 48000)

        ax[i].set_yticklabels(['0', '10k', '20k', '30k', '40k'])

    plt.subplots_adjust(right=0.85, bottom=0.25, wspace=0.1)
    cbar_ax = fig.add_axes([0.88, 0.3, 0.02, 0.5])
    cbar = fig.colorbar(img, cax=cbar_ax, format="%+2.0f dB")
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Intensity (dB)', rotation=270, labelpad=20, fontsize=16, fontweight='bold')

    plt.savefig("comparativo_espectral_IEEE_FINAL.pdf", dpi=600, bbox_inches='tight')
    plt.show()