import os
import numpy as np
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

def augment_audio(y, sr):
    """
    Configuração DATASET A: Apenas SHIFT (Deslocamento).
    Gera 2 áudios por falha.
    """
    augment = Compose([
        # ATIVADO: Shift (Deslocamento no tempo)
        # 'p=1.0' garante que sempre vai acontecer
        Shift(min_shift=-0.1, max_shift=0.1, p=1.0),

        # DESLIGADOS (p=0.0):
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.0),
        TimeStretch(min_rate=0.9, max_rate=1.1, p=0.0),
        PitchShift(min_semitones=-1, max_semitones=1, p=0.0)
    ])

    augmented = []
    # Gera 2 variações (para ser justo com o teste do Noise)
    for i in range(2):
        y_aug = augment(samples=np.copy(y), sample_rate=sr)
        augmented.append(y_aug)

    return augmented

# A função augment_sample continua igual, pois ela recebe o diretório base como argumento
def augment_sample(y_seg, sr, drill_id, hole_idx, meta, base_aug_dir):
    versions = augment_audio(y_seg, sr)
    results = []

    # Usa o diretório base passado como argumento (será o novo augmented_shift)
    out_dir = os.path.join(base_aug_dir, drill_id)
    os.makedirs(out_dir, exist_ok=True)

    for i, y_aug in enumerate(versions):
        fname = f"{drill_id}_h{hole_idx:02d}_aug{i+1}.wav"
        out_path = os.path.join(out_dir, fname)

        sf.write(out_path, y_aug, sr)

        meta_aug = meta.copy()
        meta_aug["filepath_wav"] = out_path
        meta_aug["aug_version"] = i + 1
        results.append(meta_aug)

    return results
'''import os
import numpy as np
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

def augment_audio(y, sr):
    """
    Experimento Final: Foca APENAS no 'AddGaussianNoise' (mas 10x mais fraco)
    e desliga o 'Shift' (que foi o culpado).
    """

    augment = Compose([
        # --- Aumentação "Segura" (Ativada) ---
        # Ruído 10x mais fraco que o nosso primeiro teste (0.001 vs 0.01)
        AddGaussianNoise(min_amplitude=0.0001, max_amplitude=0.001, p=1.0),

        # --- Aumentações "Perigosas" (Desligadas) ---
        Shift(min_shift=-0.1, max_shift=0.1, p=0.0), # p=0.0 (DESLIGADO)
        TimeStretch(min_rate=0.9, max_rate=1.1, p=0.0),
        PitchShift(min_semitones=-1, max_semitones=1, p=0.0)
    ])

    augmented = []

    # Vamos criar 2 áudios "seguros" por falha
    for i in range(2):
        y_aug = augment(samples=np.copy(y), sample_rate=sr)
        augmented.append(y_aug)

    return augmented'''
def augment_sample(y_seg, sr, drill_id, hole_idx, meta, base_aug_dir):
    """
    Gera arquivos aumentados + metadados.
    """

    versions = augment_audio(y_seg, sr)
    results = []

    out_dir = os.path.join(base_aug_dir, drill_id)
    os.makedirs(out_dir, exist_ok=True)

    for i, y_aug in enumerate(versions):
        fname = f"{drill_id}_h{hole_idx:02d}_aug{i+1}.wav"
        out_path = os.path.join(out_dir, fname)

        sf.write(out_path, y_aug, sr)

        meta_aug = meta.copy()
        meta_aug["filepath_wav"] = out_path
        meta_aug["aug_version"] = i + 1
        results.append(meta_aug)

    return results
