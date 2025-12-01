import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile

# Evita problemas com cache do Matplotlib
os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="mplconfig_")

# ========================================
# CONFIGURAÇÕES
# ========================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

AUDIO_PATH = os.path.join(DATA_DIR, "standardized", "06", "06_common_Tr3_ext.wav")
JAMS_PATH  = os.path.join(DATA_DIR, "raw", "drill_4mm_06_batch_00_collet_1_30-01-2025","jams.txt")

ENERGY_THRESHOLD = 0.02
MIN_HOLE_DURATION = 0.5
HOP_LENGTH = 512
FRAME_LENGTH = 1024
# ============================================================
# LEITURA SIMPLES DO JAMS
# ============================================================
def load_jams_events(jams_path):
    """O JAMS tem apenas números. Vamos ler todos como floats."""
    events = []
    with open(jams_path, "r") as f:
        for line in f:
            try:
                events.append(float(line.strip()))
            except:
                pass
    return sorted(events)

# ============================================================
# DETECTOR SIMPLES DE FUROS (seu método do preview)
# ============================================================
def detect_holes(y, sr):
    rms = librosa.feature.rms(
        y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
    )[0]

    rms_norm = rms / max(1e-9, np.max(rms))

    mask = rms_norm < ENERGY_THRESHOLD  # Agora "furo" = queda de energia

    holes = []
    start = None

    for i, is_hole in enumerate(mask):
        if is_hole and start is None:
            start = i
        elif not is_hole and start is not None:
            end = i
            start_s = start * HOP_LENGTH
            end_s = end * HOP_LENGTH
            duration = (end_s - start_s) / sr

            if duration >= MIN_HOLE_DURATION:
                holes.append((start_s, end_s))

            start = None

    if start is not None:
        holes.append((start * HOP_LENGTH, len(y)))

    return holes, rms_norm

# ============================================================
# PLOT DO ESPECTROGRAMA + FUROS + JAMS
# ============================================================
def plot_all(y, sr, holes, jams):
    plt.figure(figsize=(17, 6))

    # Espectrograma
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(img, format='%+2.0f dB')

    plt.title("Espectrograma — Furos detectados (vermelho) vs Travamentos JAMS (ciano)")

    # ---------------------------------------
    # FUROS DETECTADOS — VERMELHO
    # ---------------------------------------
    for (start, end) in holes:
        start_t = start / sr
        end_t = end / sr

        # bloco em vermelho transparente
        plt.axvspan(start_t, end_t, color='red', alpha=0.35)

        # bordas do furo
        plt.axvline(start_t, color='red', linestyle='--', linewidth=1.3)
        plt.axvline(end_t, color='red', linestyle='--', linewidth=1.3)

    # ---------------------------------------
    # TRAVAMENTOS JAMS — CIANO
    # ---------------------------------------
    for t in jams:
        plt.axvline(t, color='cyan', linestyle='--', linewidth=2)
        plt.text(
            t, 1000, f"{t:.1f}s",
            color='cyan',
            ha='center',
            fontsize=12,
            fontweight='bold'
        )

    plt.tight_layout()
    plt.show()

# ============================================================
# MAIN
# ============================================================
def main():
    print(f"🎧 Carregando áudio: {AUDIO_PATH}")
    y, sr = librosa.load(AUDIO_PATH, sr=None, mono=True)

    print(f"📄 Lendo eventos do JAMS: {JAMS_PATH}")
    jams = load_jams_events(JAMS_PATH)
    print("   → Travamentos JAMS:", jams)

    print("🔍 Detectando furos...")
    holes, rms_norm = detect_holes(y, sr)
    print(f"   → {len(holes)} furos detectados")

    for i, (s, e) in enumerate(holes, 1):
        print(f"     Furo {i}: {s/sr:.2f}s → {e/sr:.2f}s  ({(e-s)/sr:.2f}s)")

    plot_all(y, sr, holes, jams)

# ============================================================
if __name__ == "__main__":
    main()