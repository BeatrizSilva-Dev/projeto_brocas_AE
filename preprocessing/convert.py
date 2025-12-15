# src/preprocessing/convert.py

import os
import pandas as pd
import soundfile as sf
import librosa
import numpy as np

# =========================================================
# CONFIGURAÇÃO DE DIRETÓRIOS (CORRIGIDA)
# =========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
OUTPUT_DIR = os.path.join(DATA_DIR, "standardized")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")
METADATA_CSV = os.path.join(METADATA_DIR, "initial_metadata.csv")

# Garante que as pastas existem
for d in [DATA_DIR, RAW_DIR, OUTPUT_DIR, METADATA_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"📂 Lendo RAW de: {RAW_DIR}")
print(f"📂 Salvando em:  {OUTPUT_DIR}")
# =========================================================

# Mapeamento de mics comuns (Seu código original)
MIC_MAPPING = {
    "Tr1": ("common", "ext"),
    "Tr2": ("common", "ext"),
    "Tr3": ("common", "ext"),
    "Tr4": ("common", "int"),
    "Tr5": ("common", "int"),
    "Tr6": ("common", "int"),
}

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def process_channel(y_channel, sr, out_path):
    sf.write(out_path, y_channel, sr)

def process_multichannel(filepath, drill_id, mic_type, position, metadata_list):
    """Separa canais multicanais e salva individualmente"""
    try:
        y, sr = sf.read(filepath, always_2d=True)
        num_channels = y.shape[1]

        # Se for mono mas entrou aqui, garante formato correto
        if num_channels == 1:
            y = y.reshape(-1, 1)

        for ch in range(num_channels):
            y_ch = y[:, ch]
            out_name = f"{drill_id}_{mic_type}_ch{ch+1}_{position}.wav"

            out_dir = os.path.join(OUTPUT_DIR, drill_id)
            ensure_dir(out_dir)
            out_path = os.path.join(out_dir, out_name)

            process_channel(y_ch, sr, out_path)

            metadata_list.append({
                "drill_id": drill_id,
                "mic_type": mic_type,
                "mic_id": f"ch{ch+1}",
                "position": position,
                "sr": sr,
                "filepath_wav": out_path
            })
            print(f"  ✅ [ULT] Canal {ch+1} salvo: {out_name}")
    except Exception as e:
        print(f"  ❌ Erro ao processar multicanal {filepath}: {e}")

def process_wav(filepath, drill_id, mic_name, mic_type, position, mic_id, metadata_list):
    try:
        y, sr = librosa.load(filepath, sr=None, mono=True)

        out_name = f"{drill_id}_{mic_type}_{mic_id}_{position}.wav"
        out_dir = os.path.join(OUTPUT_DIR, drill_id)
        ensure_dir(out_dir)

        out_path = os.path.join(out_dir, out_name)
        sf.write(out_path, y, sr)

        metadata_list.append({
            "drill_id": drill_id,
            "mic_type": mic_type,
            "mic_id": mic_id,
            "position": position,
            "sr": sr,
            "filepath_wav": out_path
        })
        print(f"  ✅ [COM] Processado: {out_name}")
    except Exception as e:
        print(f"  ❌ Erro ao processar {filepath}: {e}")

def main():
    metadata_list = []

    if not os.path.exists(RAW_DIR):
        print(f"❌ ERRO: A pasta RAW não existe: {RAW_DIR}")
        return

    for drill_folder in os.listdir(RAW_DIR):
        drill_path = os.path.join(RAW_DIR, drill_folder)
        if not os.path.isdir(drill_path):
            continue

        # Tenta extrair ID. Ex: drill_4mm_01_batch... → drill_id = 01
        parts = drill_folder.split("_")
        if len(parts) >= 3:
            drill_id = parts[2]
        else:
            print(f"⚠️ Pasta ignorada (nome fora do padrão): {drill_folder}")
            continue

        print(f"\n🔄 Processando Broca ID: {drill_id}")

        for root, _, files in os.walk(drill_path):
            for file in files:
                if not file.lower().endswith(".wav") or file.startswith("._"):
                    continue

                filepath = os.path.join(root, file)

                # --- ORDEM ALTERADA AQUI (CRUCIAL PARA FUNCIONAR) ---

                # 1. Verifica SE É ULTRASSÔNICO PRIMEIRO
                # (Isso evita que arquivos ultrassônicos com nome "Tr1" sejam pegos errados)
                if "ultrasonic" in root.lower() or "ultrasonic" in file.lower() or "ult" in file.lower():
                    position = "ext" if "ext" in file.lower() else "int"
                    mic_type = "ultrasonic"
                    # Usa o nome do arquivo sem extensão como ID
                    mic_id = os.path.splitext(file)[0]
                    process_multichannel(filepath, drill_id, mic_type, position, metadata_list)
                    continue # Pula para o próximo arquivo se achou ultrassom

                # 2. Se não for ultrassônico, verifica se é COMUM
                mic_name_found = [k for k in MIC_MAPPING.keys() if k in file]
                if mic_name_found:
                    mic_name = mic_name_found[0]
                    mic_type, position = MIC_MAPPING[mic_name]
                    mic_id = mic_name
                    process_wav(filepath, drill_id, mic_name, mic_type, position, mic_id, metadata_list)
                    continue

                # print(f"⚠️ Arquivo não identificado: {file}")

    ensure_dir(os.path.dirname(METADATA_CSV))
    pd.DataFrame(metadata_list).to_csv(METADATA_CSV, index=False)
    print(f"\n📊 Metadata inicial salva em: {METADATA_CSV}")
    print(f"✅ Total de arquivos processados: {len(metadata_list)}")

if __name__ == "__main__":
    main()