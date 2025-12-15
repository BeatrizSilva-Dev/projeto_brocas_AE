import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile

os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="mplconfig_")

# =========================
# CONFIGURAÇÕES
# =========================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

AUDIO_PATH = os.path.join(DATA_DIR, "standardized", "06", "06_common_Tr3_ext.wav")
JAMS_PATH  = os.path.join(DATA_DIR, "raw", "drill_4mm_06_batch_00_collet_1_30-01-2025","jams.txt")

ENERGY_THRESHOLD = 0.02
MIN_HOLE_DURATION = 0.5
MATCH_TOLERANCE = 0.5    # tolerância em segundos para ser considerado acerto
HOP_LENGTH = 512
FRAME_LENGTH = 1024


# =========================
# LEITURA DO JAMS
# =========================
def load_jams_events(jams_path):
    events = []
    with open(jams_path, "r") as f:
        for line in f:
            try:
                events.append(float(line.strip()))
            except:
                pass
    return sorted(events)


# =========================
# DETECTOR DE FUROS
# =========================
def detect_holes(y, sr):
    rms = librosa.feature.rms(
        y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
    )[0]

    rms_norm = rms / max(1e-9, np.max(rms))
    mask = rms_norm < ENERGY_THRESHOLD

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

    return holes


# =========================
# COMPARAÇÃO FUROS × JAMS
# =========================
def compare_holes_with_jams(holes, jams, sr):
    results = []

    for i, (start, end) in enumerate(holes, 1):
        furo_t = (start + end) / (2 * sr)  # tempo central do furo

        best_match = None
        best_dist = 999

        for t in jams:
            dist = abs(furo_t - t)
            if dist < best_dist:
                best_dist = dist
                best_match = t

        hit = best_dist <= MATCH_TOLERANCE

        results.append({
            "furo": i,
            "start": start / sr,
            "end": end / sr,
            "centro": furo_t,
            "match": best_match,
            "dist": best_dist,
            "acertou": hit
        })

    return results


# =========================
# MAIN
# =========================
def main():
    print("\n🎧 Carregando áudio…")
    y, sr = librosa.load(AUDIO_PATH, sr=None, mono=True)

    jams = load_jams_events(JAMS_PATH)
    print(f"📄 Travamentos JAMS: {jams}")

    holes = detect_holes(y, sr)
    print(f"🔍 {len(holes)} furos detectados\n")

    results = compare_holes_with_jams(holes, jams, sr)

    # --------------------------
    # RELATÓRIO FINAL
    # --------------------------
    print("==================================")
    print("     COMPARAÇÃO FUROS × JAMS")
    print("==================================")

    for r in results:
        print(
            f"Furo {r['furo']:2d} | centro = {r['centro']:.2f}s | "
            f"match = {r['match']} | dist = {r['dist']:.3f}s | "
            f"{'✔ ACERTOU' if r['acertou'] else '❌ ERROU'}"
        )

    qtd_acertos = sum(r["acertou"] for r in results)
    print("\n==================================")
    print(f"TOTAL DE ACERTOS: {qtd_acertos}/{len(holes)}")
    print("==================================\n")


if __name__ == "__main__":
    main()
