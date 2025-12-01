#!/usr/bin/env python3
"""
segment_full.py — SEGMENTAÇÃO + AUMENTAÇÃO
"""

import os
import librosa
import soundfile as sf
import pandas as pd
import numpy as np
from scipy import signal

from src.preprocessing.augment_audio import augment_sample

# ============================================
# DIRETÓRIOS
# ============================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
SEGMENTED_DIR = os.path.join(DATA_DIR, "segmented")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")

# --- CONFIGURAÇÃO ESPECÍFICA  ---
AUGMENTED_DIR = os.path.join(DATA_DIR, "augmented_shift")
METADATA_SEG_CSV = os.path.join(METADATA_DIR, "segmented_metadata.csv") # Opcional: pode sobrescrever ou não
METADATA_AUG_CSV = os.path.join(METADATA_DIR, "augmented_shift_metadata.csv")

# Mapeamento de mics comuns
MIC_MAPPING = {
    "Tr1": ("common", "ext"), "Tr2": ("common", "ext"), "Tr3": ("common", "ext"),
    "Tr4": ("common", "int"), "Tr5": ("common", "int"), "Tr6": ("common", "int"),
}

# Cria pastas
for d in [SEGMENTED_DIR, METADATA_DIR, AUGMENTED_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================
# PARÂMETROS DE DETECÇÃO
# ============================================
MIN_HOLE_DURATION = 2.0
SMOOTH_WINDOW = 9
FRAME_LENGTH = 1024
HOP_LENGTH = 512
MIN_PROMINENCE_VALLEY = 0.01

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_jam_list(raw_folder_path):
    if raw_folder_path is None: return []
    jams_file = os.path.join(raw_folder_path, "jams.txt")
    if not os.path.exists(jams_file): return []
    try:
        with open(jams_file, 'r') as f:
            parts = f.read().replace("\n", "").split(",")
        return [int(p.strip()) for p in parts if p.strip().isdigit()]
    except:
        return []

# --- DETECTOR DE FUROS ---
def detect_holes_by_deep_valleys(y, sr):
    y_filt = librosa.effects.preemphasis(y)
    rms = librosa.feature.rms(y=y_filt, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    rms_norm = rms / (np.max(rms) + 1e-9)

    holes = []
    thresh = 0.25
    in_hole = False
    start = 0
    for i, val in enumerate(rms_norm):
        if val < thresh and not in_hole:
            in_hole = True; start = i * HOP_LENGTH
        elif val >= thresh and in_hole:
            in_hole = False
            if (i * HOP_LENGTH - start) > 2.0 * sr:
                holes.append((start, i * HOP_LENGTH))
    if in_hole and (len(rms_norm)*HOP_LENGTH - start) > 2.0 * sr:
        holes.append((start, len(rms_norm)*HOP_LENGTH))

    return holes, rms_norm

def refine_long_holes(y, sr, holes):
    return holes

# =========================================================
# MAIN
# =========================================================
def main():
    print(f"--- INICIANDO GERAÇÃO DATASET SHIFT (23 PASTAS) ---")
    print(f"📂 Lendo de: {RAW_DIR}")
    print(f"📂 Salvando aumentados em: {AUGMENTED_DIR}")

    segmented_metadata = []
    augmented_metadata = [] # Lista para os novos dados

    pastas = sorted(os.listdir(RAW_DIR))

    for folder_name in pastas:
        folder_path = os.path.join(RAW_DIR, folder_name)
        if not os.path.isdir(folder_path): continue

        parts = folder_name.split('_')
        if len(parts) < 3: continue
        drill_id = parts[2]

        print(f"\n➡️  Processando Broca: {drill_id}")

        jam_holes_list = load_jam_list(folder_path)

        for root, _, files in os.walk(folder_path):
            for file in files:
                if not file.lower().endswith(".wav") or file.startswith("._"): continue

                filepath = os.path.join(root, file)

                # --- IDENTIFICAÇÃO ---
                mic_type = "unknown"
                mic_id = 0
                position = "unknown"
                filename_lower = file.lower()

                if "ultrasonic" in filename_lower or "_ult_" in filename_lower or "ch1" in filename_lower:
                    mic_type = "ult"
                    position = "ext" if "ext" in filename_lower else "int"
                    try: mic_id = int("".join(filter(str.isdigit, file))[-1])
                    except: mic_id = 1
                else:
                    mic_name_found = [k for k in MIC_MAPPING.keys() if k in file]
                    if mic_name_found:
                        k = mic_name_found[0]
                        mic_type = "com"
                        _, position = MIC_MAPPING[k]
                        try: mic_id = int(k.replace("Tr", ""))
                        except: mic_id = 0
                    else:
                        continue

                # --- PROCESSAMENTO ---
                try:
                    y, sr = librosa.load(filepath, sr=None, mono=True)
                    holes, _ = detect_holes_by_deep_valleys(y, sr)

                    for idx_local, (start, end) in enumerate(holes, 1):
                        is_jam = 1 if idx_local in jam_holes_list else 0

                        fname = f"{drill_id}_h{idx_local:02d}_{mic_type}_{mic_id}_{position}"
                        if is_jam: fname += "_jam"
                        fname += ".wav"

                        out_dir = os.path.join(SEGMENTED_DIR, drill_id)
                        ensure_dir(out_dir)
                        out_path = os.path.join(out_dir, fname)

                        # Salva o original (opcional se já tiver rodado, mas mal não faz)
                        sf.write(out_path, y[start:end], sr)

                        meta = {
                            "drill_id": drill_id,
                            "hole_idx": idx_local,
                            "mic_type": mic_type,
                            "mic_id": mic_id,
                            "position": position,
                            "fail": is_jam,
                            "filepath_wav": out_path,
                            "start_sample": start,
                            "end_sample": end,
                            "duration_s": (end-start)/sr,
                            "sample_rate": sr,
                            "source_file": file
                        }
                        segmented_metadata.append(meta)


                        if is_jam == 1:
                            y_seg = y[start:end]
                            # Chama a função de augment
                            augmented = augment_sample(
                                y_seg,
                                sr,
                                drill_id,
                                idx_local,
                                meta,
                                AUGMENTED_DIR # <--- Salva em augmented
                            )
                            augmented_metadata.extend(augmented)

                except Exception as e:
                    print(f"   ❌ Erro em {file}: {e}")

    # Salva os CSVs
    print("\n" + "="*50)

    # Salva o segmentado original (atualizado)
    pd.DataFrame(segmented_metadata).to_csv(METADATA_SEG_CSV, index=False)

    # Salva o aumentado
    if augmented_metadata:
        pd.DataFrame(augmented_metadata).to_csv(METADATA_AUG_CSV, index=False)
        print(f"✅ SUCESSO! {len(augmented_metadata)} arquivos de SHIFT gerados.")
        print(f"📄 Metadata salvo em: {METADATA_AUG_CSV}")
    else:
        print("⚠️ Nenhum dado aumentado foi gerado. Verifique se 'jams.txt' foi lido.")

    print("="*50)

if __name__ == "__main__":
    main()