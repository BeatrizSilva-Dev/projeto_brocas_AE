import numpy as np
import pandas as pd
import librosa
import warnings
from sklearn.preprocessing import StandardScaler # Importa o Scaler aqui

warnings.filterwarnings("ignore")

# =====================================================
# SAFE HELPERS
# =====================================================

def safe_load_audio(filepath, sr):
    """Carrega o áudio de forma segura"""
    try:
        audio, sr = librosa.load(filepath, sr=sr)
        return audio, sr
    except Exception as e:
        print(f"[WARN] Falha ao carregar áudio {filepath}: {e}")
        return None, None

def extract_audio_features(filepath, sample_rate):
    """Extrai um conjunto ESTATÍSTICO RICO de features do arquivo de áudio."""
    audio, sr = safe_load_audio(filepath, sample_rate)
    if audio is None: return None
    if len(audio) < 100: return None 
    
    feats = {}
    
    # 1. Calcule as features frame-a-frame
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=1024, hop_length=512)
    sc = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=1024, hop_length=512)[0]
    sb = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=1024, hop_length=512)[0]
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=1024, hop_length=512)[0]
    rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=512)[0]
    
    # 2. Calcule estatísticas ricas
    feature_arrays = {
        'sc': sc,
        'sb': sb,
        'zcr': zcr,
        'rms': rms
    }

    for name, arr in feature_arrays.items():
        feats[f"{name}_mean"] = float(np.mean(arr))
        feats[f"{name}_std"] = float(np.std(arr))
        feats[f"{name}_median"] = float(np.median(arr))
        feats[f"{name}_min"] = float(np.min(arr))
        feats[f"{name}_max"] = float(np.max(arr))
        
    # 3. Para cada um dos 13 MFCCs
    for i in range(13):
        mfcc_band = mfccs[i]
        feats[f"mfcc_{i}_mean"] = float(np.mean(mfcc_band))
        feats[f"mfcc_{i}_std"] = float(np.std(mfcc_band))
        feats[f"mfcc_{i}_median"] = float(np.median(mfcc_band))
        feats[f"mfcc_{i}_max"] = float(np.max(mfcc_band))
        feats[f"mfcc_{i}_min"] = float(np.min(mfcc_band))
        
    # 4. Features Delta (taxa de mudança)
    mfcc_delta = librosa.feature.delta(mfccs)
    for i in range(13):
        feats[f"mfcc_delta_{i}_mean"] = float(np.mean(mfcc_delta[i]))
        feats[f"mfcc_delta_{i}_std"] = float(np.std(mfcc_delta[i]))

    return feats

# =====================================================
# LOAD DATASET (MODIFICADO)
# =====================================================
def load_dataset(segmented_csv):
    """
    Carrega o dataset, normaliza as features, e retorna
    X, y, e groups.
    """
    print("Carregando dataset...")

    try:
        df = pd.read_csv(segmented_csv)
    except FileNotFoundError:
        print(f"[ERROR] Arquivo não encontrado: {segmented_csv}")
        return pd.DataFrame(), np.array([]), np.array([])
    
    all_features = []
    all_labels = []
    all_groups = [] 

    required_cols = ["filepath_wav", "sample_rate", "fail", "drill_id"]
    if not all(col in df.columns for col in required_cols):
        print(f"[ERROR] O CSV deve conter as colunas: {required_cols}")
        return pd.DataFrame(), np.array([]), np.array([]) 

    print(f"Processando {len(df)} linhas do CSV...")
    
    for idx, row in df.iterrows():
        features = extract_audio_features(
            filepath=row["filepath_wav"],
            sample_rate=int(row["sample_rate"])
        )

        if features is None:
            # print(f"[WARN] Pulando linha {idx}, não foi possível extrair features.")
            continue

        all_features.append(features)
        all_labels.append(row["fail"])
        all_groups.append(row["drill_id"]) 

    if not all_features:
        print("[ERROR] Nenhuma feature foi extraída.")
        return pd.DataFrame(), np.array([]), np.array([])
        
    X_df = pd.DataFrame(all_features).fillna(0) # .fillna(0) é uma segurança
    y = np.array(all_labels)
    groups = np.array(all_groups) 

    # --- INÍCIO DA MUDANÇA QUE VOCÊ PEDIU ---
    print(f"✅ Total de amostras carregadas: {len(X_df)}")
    print(f"✅ Total de grupos únicos (drill_id): {len(np.unique(groups))}")
    
    counts = np.bincount(y)
    if len(counts) < 2:
         print(f"✅ BALANÇO DE CLASSES: {counts[0]} amostras (fail=0) e 0 amostras (fail=1)")
    else:
         print(f"✅ BALANÇO DE CLASSES: {counts[0]} amostras (fail=0) e {counts[1]} amostras (fail=1)")

    print("\n✅ Resumo de Amostras por Grupo (Broca):")
    unique_groups, group_counts = np.unique(groups, return_counts=True)
    for group, count in zip(unique_groups, group_counts):
        print(f"  - Grupo (Broca) '{group}': {count} amostras")
    # --- FIM DA MUDANÇA ---
    
    # --- NORMALIZAÇÃO (Movida de evo_opt_xgb para cá) ---
    # É melhor normalizar o dataset INTEIRO uma vez aqui,
    # do que normalizar em cada fold (mais rápido).
    print("\nNormalizando features (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    
    # Retorna o DataFrame normalizado
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_df.columns, index=X_df.index)
    print("✅ Normalização concluída.")
    
    return X_scaled_df, y, groups