#!/usr/bin/env python3
"""
segment_full.py
... (descrição) ...
"""

import os
import librosa
import librosa.display
import soundfile as sf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# =========================================================
# DIRETÓRIOS
# =========================================================
STANDARDIZED_DIR = "data/standardized"
RAW_DIR = "data/raw" # <--- Vamos usar este
SEGMENTED_DIR = "data/segmented"
DOCS_IMG_DIR = "docs/img"
METADATA_CSV = "data/metadata/segmented_metadata.csv"

# =========================================================
# PARÂMETROS DETECTOR DE FUROS
# =========================================================
# ... (Seus parâmetros MIN_HOLE_DURATION, ETC. continuam os mesmos) ...
MIN_HOLE_DURATION = 2.0
SMOOTH_WINDOW = 9
VALLEY_WINDOW_SEC = 1.0
DEPTH_THRESH = 0.15
MIN_PROMINENCE_VALLEY = 0.01
HOP_LENGTH = 512
FRAME_LENGTH = 1024
GROUP_GAP_SEC = 3.0
MERGE_GAP_SEC = 2.0
DROP_PROMINENCE = 0.02
DROP_SEARCH_SEC = 1.5
START_FACTOR = 0.10
START_ABS_THRESH = 0.02
START_SEARCH_SEC = 20.0

# =========================================================
# UTILITÁRIOS
# =========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# <--- NOVA FUNÇÃO HELPER ---
def find_raw_folder_by_id(raw_dir, drill_id):
    """
    Scaneia o RAW_DIR e encontra o nome da pasta original 
    correspondente ao drill_id (ex: "01" -> "drill_4mm_01_batch...").
    """
    if not os.path.isdir(raw_dir):
        return None
        
    for folder_name in os.listdir(raw_dir):
        full_path = os.path.join(raw_dir, folder_name)
        if not os.path.isdir(full_path):
            continue
        
        parts = folder_name.split('_')
        # Baseado na sua estrutura: drill_4mm_01_batch... -> "01" é o 3º item
        if len(parts) > 2 and parts[2] == drill_id:
            return full_path # Retorna ex: "data/raw/drill_4mm_01..."
    
    return None # Não encontrou

# <--- FUNÇÃO HELPER MODIFICADA ---
def load_jam_list(raw_folder_path):
    """
    Procura por um 'jams.txt' no diretório RAW da broca e retorna
    uma lista de inteiros (números dos furos) que são falhas.
    """
    # Se a pasta raw não for encontrada, retorna lista vazia
    if raw_folder_path is None:
        print(f"  [AVISO] Pasta 'raw' correspondente não encontrada. Assumindo 0 falhas.")
        return []

    jams_file = os.path.join(raw_folder_path, "jams.txt")
    jam_holes = []
    
    if os.path.exists(jams_file):
        try:
            with open(jams_file, 'r') as f:
                content = f.read()
                parts = content.strip().replace('\n', '').split(',')
                jam_holes = [int(p.strip()) for p in parts if p.strip().isdigit()]
            print(f"  [INFO] 'jams.txt' encontrado em '{raw_folder_path}'. Falhas nos furos: {jam_holes}")
        except Exception as e:
            print(f"  [AVISO] Falha ao ler 'jams.txt': {e}. Assumindo sem falhas.")
    else:
        print(f"  [INFO] 'jams.txt' não encontrado em '{raw_folder_path}'. Assumindo 0 falhas.")
    return jam_holes
# <--- FIM DAS MUDANÇAS ---


# =========================================================
# DETECTOR DE FUROS AVANÇADO (versão completa)
# =========================================================
def detect_holes_by_deep_valleys(y, sr):
    hop_length = HOP_LENGTH
    frame_length = FRAME_LENGTH
    END_MARGIN_SEC = 4.0

    y = librosa.effects.preemphasis(y)
    y_max = np.max(np.abs(y))
    if y_max > 0:
        y = y / y_max

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_smooth = np.convolve(rms, np.ones(SMOOTH_WINDOW)/SMOOTH_WINDOW, mode='same')
    rms_max = np.max(rms_smooth)
    rms_norm = rms_smooth / rms_max if rms_max > 0 else rms_smooth

    diff = np.abs(np.diff(rms_norm, prepend=rms_norm[0]))
    diff_max = np.max(diff)
    if diff_max > 0:
        diff = diff / diff_max

    valleys_all, props = signal.find_peaks(-rms_norm, prominence=MIN_PROMINENCE_VALLEY)
    window_frames = max(1, int((VALLEY_WINDOW_SEC * sr) / hop_length))

    valleys_kept = []
    depths = []
    local_peaks = {}
    for v in valleys_all:
        left_idx = max(0, v - window_frames)
        right_idx = min(len(rms_norm) - 1, v + window_frames)
        max_before = np.max(rms_norm[left_idx:v+1]) if v - left_idx > 0 else rms_norm[v]
        max_after = np.max(rms_norm[v:right_idx+1]) if right_idx - v > 0 else rms_norm[v]
        local_peak = max(max_before, max_after)
        depth = local_peak - rms_norm[v]
        depths.append(depth)
        local_peaks[v] = local_peak
        if depth >= DEPTH_THRESH:
            valleys_kept.append(v)
    if len(valleys_kept) < 2 and len(valleys_all) >= 2:
        N = min(10, len(valleys_all))
        sorted_idx = np.argsort(depths)[-N:]
        valleys_kept = list(np.array(valleys_all)[sorted_idx])
        valleys_kept.sort()

    drop_peaks, _ = signal.find_peaks(-diff, prominence=DROP_PROMINENCE)

    # --- Agrupamento ---
    holes = []
    if valleys_kept:
        group_start = valleys_kept[0]
        group_end = valleys_kept[0]
        max_gap = int((GROUP_GAP_SEC * sr) / hop_length)
        for i in range(1, len(valleys_kept)):
            if valleys_kept[i] - valleys_kept[i - 1] > max_gap:
                dur = (group_end - group_start) * hop_length / sr
                if dur >= MIN_HOLE_DURATION:
                    holes.append((group_start * hop_length, group_end * hop_length))
                group_start = valleys_kept[i]
            group_end = valleys_kept[i]
        dur = (group_end - group_start) * hop_length / sr
        if dur >= MIN_HOLE_DURATION:
            holes.append((group_start * hop_length, group_end * hop_length))

    # --- Mesclar furos próximos ---
    merged_holes = []
    if holes:
        merged_holes.append(holes[0])
        for s, e in holes[1:]:
            last_s, last_e = merged_holes[-1]
            if (s/sr) - (last_e/sr) < MERGE_GAP_SEC:
                merged_holes[-1] = (last_s, e)
            else:
                merged_holes.append((s, e))
    holes = merged_holes

    # --- Ajuste dos inícios/fins ---
    refined_holes = []
    search_window_drop = int(DROP_SEARCH_SEC * sr / hop_length)
    search_window_start = int(START_SEARCH_SEC * sr / hop_length)
    end_margin_frames = int(END_MARGIN_SEC * sr / hop_length)
    for idx, (s, e) in enumerate(holes):
        s_frame = int(s / hop_length)
        e_frame = int(e / hop_length)
        ref_valley = None
        if len(valleys_kept) > 0:
            candidates = [v for v in valleys_kept if v <= s_frame + 2]
            if len(candidates) == 0:
                ref_valley = valleys_kept[np.argmin(np.abs(np.array(valleys_kept) - s_frame))]
            else:
                ref_valley = candidates[-1]
        local_peak = local_peaks.get(ref_valley, None)
        s_new_frame = None
        if local_peak is not None:
            start_threshold = max(START_ABS_THRESH, local_peak * START_FACTOR)
            start_search_left = max(0, s_frame - search_window_start)
            segment = rms_norm[start_search_left:s_frame]
            hits = np.where(segment >= start_threshold)[0]
            if hits.size > 0:
                s_new_frame = start_search_left + hits[0]
        if s_new_frame is None:
            candidates_drop = [p for p in drop_peaks if s_frame - search_window_drop <= p < s_frame]
            if candidates_drop:
                s_new_frame = candidates_drop[-1]
        if idx == 0:
            s_new_frame = 0
        if s_new_frame is not None:
            s = int(s_new_frame * hop_length)
        if len(refined_holes) > 0:
            s = max(s, refined_holes[-1][1])
        e_frame_extended = e_frame + end_margin_frames
        e = min(len(y), int(e_frame_extended * hop_length))
        if idx == len(holes) - 1:
            e = len(y)
        refined_holes.append((s, e))
    return refined_holes, rms_norm, diff, valleys_all, valleys_kept, depths, drop_peaks, hop_length


# ... (Funções detect_holes_by_deep_valleys_custom, refine_long_holes,
#      save_rms_histogram, save_spectrogram, gerar_overview_drill continuam IGUAIS) ...

def detect_holes_by_deep_valleys_custom(y, sr, DEPTH_THRESH=0.15, GROUP_GAP_SEC=3.0, MERGE_GAP_SEC=2.0):
    orig = (globals()['DEPTH_THRESH'], globals()['GROUP_GAP_SEC'], globals()['MERGE_GAP_SEC'])
    globals()['DEPTH_THRESH'], globals()['GROUP_GAP_SEC'], globals()['MERGE_GAP_SEC'] = DEPTH_THRESH, GROUP_GAP_SEC, MERGE_GAP_SEC
    result = detect_holes_by_deep_valleys(y, sr)
    globals()['DEPTH_THRESH'], globals()['GROUP_GAP_SEC'], globals()['MERGE_GAP_SEC'] = orig
    return result

def refine_long_holes(y, sr, holes, duration_factor=1.5, aggressive_factor=2.5):
    if not holes: 
        return []
    durations = np.array([(e - s) / sr for s, e in holes])
    median_dur = np.median(durations)
    refined_holes = []
    for (s, e), dur in zip(holes, durations):
        if dur <= duration_factor * median_dur:
            refined_holes.append((s, e))
            continue
        print(f"→ Refinando furo longo ({dur:.2f}s)...")
        y_seg = y[s:e]
        if dur > aggressive_factor * median_dur:
            params = dict(DEPTH_THRESH=0.05, GROUP_GAP_SEC=0.5, MERGE_GAP_SEC=0.25)
        else:
            params = dict(DEPTH_THRESH=0.08, GROUP_GAP_SEC=1.0, MERGE_GAP_SEC=0.5)
        sub_holes, *_ = detect_holes_by_deep_valleys_custom(y_seg, sr, **params)
        sub_holes_global = [(s + s2, s + e2) for (s2, e2) in sub_holes]
        refined_holes.extend(sub_holes_global if sub_holes_global else [(s, e)])
    return refined_holes

def save_rms_histogram(y_segment, sr, out_path):
    rms = librosa.feature.rms(y=y_segment, frame_length=1024, hop_length=512)[0]
    plt.figure(figsize=(6,4))
    plt.hist(rms, bins=30, color='steelblue', edgecolor='black', alpha=0.8)
    plt.xlabel("RMS"); plt.ylabel("Contagem de frames")
    plt.title("Histograma RMS do segmento")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_spectrogram(y_segment, sr, out_path):
    try:
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y_segment)), ref=np.max)
        plt.figure(figsize=(8,4))
        librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log', cmap='magma')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Espectrograma do segmento")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    except Exception as e:
        print(f"  [AVISO] Não foi possível gerar espectrograma: {e}")
        plt.close()

def gerar_overview_drill(drill_id, metadata_list):
    drill_metadata = [m for m in metadata_list if m["drill_id"] == drill_id]
    if not drill_metadata: return
    plt.figure(figsize=(12,2))
    for m in drill_metadata:
        sr = m.get("sample_rate", 44100) 
        fail = m.get("fail", 0) 
        
        start_sec = m["start_sample"]/sr
        end_sec = m["end_sample"]/sr
        color = 'red' if fail else 'blue'
        plt.barh(0, width=end_sec-start_sec, left=start_sec, height=0.5, color=color, edgecolor='black', alpha=0.8)
    plt.yticks([]); plt.xlabel("Tempo total (s)")
    plt.title(f"Overview Drill {drill_id}")
    out_dir = os.path.join(DOCS_IMG_DIR, drill_id)
    ensure_dir(out_dir)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{drill_id}_overview.png"))
    plt.close()

# =========================================================
# SEGMENTAÇÃO PRINCIPAL
# =========================================================
def main():
    metadata_list = [] 

    # Loop principal focado no STANDARDIZED_DIR
    for drill_folder_name in os.listdir(STANDARDIZED_DIR): # ex: "01"
        drill_folder_standardized = os.path.join(STANDARDIZED_DIR, drill_folder_name)
        if not os.path.isdir(drill_folder_standardized):
            continue
        
        drill_id = drill_folder_name # ex: "01"
        print(f"\nProcessando broca: {drill_id}...")

        # <--- MUDANÇA PRINCIPAL AQUI ---
        # 1. Encontra a pasta RAW original correspondente ao drill_id
        raw_folder_path = find_raw_folder_by_id(RAW_DIR, drill_id)
        
        # 2. Carrega a lista de falhas (jams) a partir da pasta RAW
        jam_holes_list = load_jam_list(raw_folder_path)
        # <--- FIM DA MUDANÇA ---

        # Este loop continua lendo os .wav da pasta STANDARDIZED
        for file in os.listdir(drill_folder_standardized): 
            if not file.lower().endswith(".wav") or file.startswith("._"):
                continue
            
            filepath = os.path.join(drill_folder_standardized, file)
            
            mic_type = "ult" if "ult" in file.lower() or "ultrasonic" in file.lower() else "com"
            mic_id_str = ''.join(filter(str.isdigit, file))
            mic_id = int(mic_id_str) if mic_id_str else 0
            position = "ext" if mic_id in [1, 2, 3] else "int"

            # furos e refinamento
            try:
                y, sr = librosa.load(filepath, sr=None, mono=True)
            except Exception as e:
                print(f"[ERRO] Falha ao carregar {filepath}: {e}")
                continue
                
            holes, *_ = detect_holes_by_deep_valleys(y, sr)
            holes_refined = refine_long_holes(y, sr, holes)

            print(f"  → {file}: {len(holes_refined)} furos detectados.")

            for idx_local, (start, end) in enumerate(holes_refined, 1):
                if (end - start) < (sr * 0.1): 
                    print(f"    ... Pulando furo {idx_local} (muito curto).")
                    continue
                
                # Verifica se o índice do furo ATUAL está na lista de falhas
                is_jam = 1 if idx_local in jam_holes_list else 0
                    
                filename_base = f"{drill_id}_hole{idx_local:02d}_{mic_type}_{mic_id}_{position}"
                
                if is_jam:
                    filename_base += "_jam" # Adiciona _jam ao nome do arquivo
                
                filename = filename_base + ".wav"
                out_dir = os.path.join(SEGMENTED_DIR, drill_id)
                ensure_dir(out_dir)
                out_path = os.path.join(out_dir, filename)
                
                sf.write(out_path, y[start:end], sr)

                duration_s = (end - start) / sr
                
                metadata_list.append({
                    "drill_id": drill_id,
                    "hole_idx": idx_local,
                    "mic_type": mic_type,
                    "mic_id": mic_id,
                    "position": position,
                    "fail": is_jam, # <--- Salva 0 ou 1 corretamente
                    "filepath_wav": out_path, 
                    "start_sample": start,
                    "end_sample": end,
                    "duration_s": duration_s,
                    "sample_rate": sr,
                    "source_file": file
                })

    print("\n✅ Segmentação concluída com sucesso.")

    if not metadata_list:
        print("\n⚠️ Nenhum furo foi detectado ou salvo. O CSV não será criado.")
        return 

    print(f"\n📊 Salvando metadados de {len(metadata_list)} furos em {METADATA_CSV}...")
    
    ensure_dir(os.path.dirname(METADATA_CSV))
    
    df_metadata = pd.DataFrame(metadata_list)
    
    # Salva o CSV (sobrescreve por padrão)
    df_metadata.to_csv(METADATA_CSV, index=False)
    
    print(f"✅ Metadados salvos com sucesso!")


if __name__ == "__main__":
    main()