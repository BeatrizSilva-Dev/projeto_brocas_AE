import os
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# CONFIGURAÇÃO DE FONTE
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.weight': 'normal',
    'font.size': 14,             # Base padrão

    'figure.titlesize': 24,      # Título da Figura
    'axes.titlesize': 24,        # Título do Eixo
    'axes.labelsize': 24,        # Nome dos Eixos

    'xtick.labelsize': 16,       # Números do eixo X
    'ytick.labelsize': 16,       # Números do eixo Y

    'legend.fontsize': 14,       # Legenda
    'figure.figsize': (8, 6),   # Tamanho físico
    'lines.linewidth': 2.5,
    'lines.markersize': 8
})

# CONFIGURAÇÃO DE CAMINHOS
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

SEGMENTED_CSV = os.path.join(METADATA_DIR, "segmented_metadata.csv")

# OPÇÃO A: SHIFT (Deslocamento)
AUGMENTED_CSV = os.path.join(METADATA_DIR, "augmented_shift_metadata.csv")
FEATURES_CACHE_CSV = os.path.join(METADATA_DIR, "all_features_cache_shift.csv")
TSNE_OUTPUT_PNG = os.path.join(PLOTS_DIR, "tsne_audio_features_compare_shift.png")

# OPÇÃO B: NOISE (Ruído)
# AUGMENTED_CSV = os.path.join(METADATA_DIR, "augmented_noise_metadata.csv")
# FEATURES_CACHE_CSV = os.path.join(METADATA_DIR, "all_features_cache_noise.csv")
# TSNE_OUTPUT_PNG = os.path.join(PLOTS_DIR, "tsne_audio_features_noise_compare.png")

def extract_features(y, sr, n_mfcc=20):
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        features = np.hstack([mfcc_mean, mfcc_std, rms_mean, rms_std, zcr_mean, zcr_std])
        return features
    except Exception as e:
        print(f"  [ERRO] Falha ao extrair: {e}")
        return None

def get_or_create_features():
    if os.path.exists(FEATURES_CACHE_CSV):
        print(f"\n🔄 Carregando cache: {FEATURES_CACHE_CSV}")
        df_features = pd.read_csv(FEATURES_CACHE_CSV)
        if 'source' not in df_features.columns:
            print("[ERRO] Cache inválido. Recriando...")
            os.remove(FEATURES_CACHE_CSV)
            return get_or_create_features()
        print("✅ Cache carregado.")
        return df_features

    print(f"\n🧊 Criando novo cache em {FEATURES_CACHE_CSV}...")
    try:
        df_seg = pd.read_csv(SEGMENTED_CSV)
        df_aug = pd.read_csv(AUGMENTED_CSV)
    except FileNotFoundError as e:
        print(f"❌ ERRO: Arquivo não encontrado: {e}")
        return None

    df_seg_fail = df_seg[df_seg['fail'] == 1].copy()
    df_seg_fail['source'] = 'segmented'

    df_aug_fail = df_aug[df_aug['fail'] == 1].copy()
    df_aug_fail['source'] = 'augmented'

    df_combined = pd.concat([df_seg_fail, df_aug_fail], ignore_index=True)
    print(f"  Total de falhas para processar: {len(df_combined)}")

    feature_list = []
    source_list = []

    print("\n⏳ Extraindo features...")
    for index, row in tqdm(df_combined.iterrows(), total=len(df_combined)):
        filepath = os.path.join(PROJECT_ROOT, os.path.normpath(row['filepath_wav']))
        if not os.path.exists(filepath): continue
        try:
            y, sr = librosa.load(filepath, sr=None, mono=True)
            features = extract_features(y, sr)
            if features is not None:
                feature_list.append(features)
                source_list.append(row['source'])
        except: pass

    df_features = pd.DataFrame(feature_list)
    df_features['source'] = source_list

    n_mfcc = 20
    base_cols = [f'mfcc_mean_{i}' for i in range(n_mfcc)] + \
                [f'mfcc_std_{i}' for i in range(n_mfcc)] + \
                ['rms_mean', 'rms_std', 'zcr_mean', 'zcr_std']
    df_features.columns = base_cols + ['source']

    df_features.to_csv(FEATURES_CACHE_CSV, index=False)
    print(f"✅ Cache salvo em {FEATURES_CACHE_CSV}")
    return df_features

def compare_statistics(df_features):
    print("\n" + "="*50)
    print("📊 ESTATÍSTICAS (ORIGINAL vs. AUMENTADO)")
    print("="*50)
    df_orig = df_features[df_features['source'] == 'segmented'].drop(columns=['source'])
    df_aug = df_features[df_features['source'] == 'augmented'].drop(columns=['source'])
    key_features = ['mfcc_mean_0', 'mfcc_std_0', 'rms_mean', 'rms_std', 'zcr_mean', 'zcr_std']
    print("\n--- ORIGINAIS ---")
    print(df_orig[key_features].describe())
    print("\n--- AUMENTADOS ---")
    print(df_aug[key_features].describe())
    print("="*50)

def generate_tsne(df):
    print("\n🔮 Rodando t-SNE...")
    labels = df['source']
    df_numeric = df.drop(columns=['source'])

    df_scaled = StandardScaler().fit_transform(df_numeric)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_jobs=-1)
    tsne_results = tsne.fit_transform(df_scaled)

    # Configuração da Figura
    plt.figure(figsize=(8, 6), dpi=300)

    # Augmented (Laranja)
    idx_aug = labels == 'augmented'
    plt.scatter(tsne_results[idx_aug, 0], tsne_results[idx_aug, 1],
                label='Augmented (Novos)', s=12, alpha=0.5, c='orange')

    # Segmented (Azul)
    idx_seg = labels == 'segmented'
    plt.scatter(tsne_results[idx_seg, 0], tsne_results[idx_seg, 1],
                label='Segmented (Originais)', s=15, alpha=1.0, c='blue',
                edgecolors='w', linewidths=0.5)

    plt.title("t-SNE: Features de Áudio (Originais vs. Aumentados)")
    plt.xlabel("Componente t-SNE 1")
    plt.ylabel("Componente t-SNE 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # SALVAMENTO SEGURO
    pdf_output = TSNE_OUTPUT_PNG.replace(".png", ".pdf")

    plt.savefig(pdf_output, format='pdf', dpi=300, bbox_inches='tight')
    print(f"✅ PDF Salvo: {pdf_output}")

    plt.savefig(TSNE_OUTPUT_PNG, format='png', dpi=300, bbox_inches='tight')
    print(f"✅ PNG Salvo: {TSNE_OUTPUT_PNG}")

    plt.show()

if __name__ == "__main__":
    df = get_or_create_features()
    if df is not None:
        compare_statistics(df)
        generate_tsne(df)
        print("\n✨ Concluído!")