import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

# ------------------------------------------------------
# CONFIGURAÇÃO
# ------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

SEGMENTED = os.path.join(DATA_DIR, "metadata", "segmented_metadata.csv")
AUGMENTED = os.path.join(DATA_DIR, "metadata", "augmented_metadata.csv")
#SEGMENTED = "data/metadata/segmented_metadata.csv"
#AUGMENTED = "data/metadata/augmented_metadata.csv"
OUTPUT_COMBINED = os.path.join(DATA_DIR, "metadata", "combined_metadata.csv")
#OUTPUT_COMBINED = "data/metadata/combined_metadata.csv"
OUTPUT_PLOTS_DIR = "plots"
TSNE_OUTPUT = os.path.join(OUTPUT_PLOTS_DIR, "tsne_compare.png")


# ------------------------------------------------------
# FUNÇÃO 1 — COMBINAR SEGMENTED + AUGMENTED
# ------------------------------------------------------
def combine_metadata():
    print("\n🔄 Carregando arquivos...")

    df_segmented = pd.read_csv(SEGMENTED)
    df_augmented = pd.read_csv(AUGMENTED)

    print(f"  ✔ Segmentado: {len(df_segmented)} linhas")
    print(f"  ✔ Augmented: {len(df_augmented)} linhas")

    # Determinar colunas em comum
    common_cols = list(set(df_segmented.columns) & set(df_augmented.columns))
    if not common_cols:
        raise ValueError("❌ ERRO: Não existem colunas em comum entre os arquivos!")

    df_segmented = df_segmented[common_cols].copy()
    df_segmented["source"] = "segmented"

    df_augmented = df_augmented[common_cols].copy()
    df_augmented["source"] = "augmented"

    df_combined = pd.concat([df_segmented, df_augmented], ignore_index=True)

    os.makedirs(os.path.dirname(OUTPUT_COMBINED), exist_ok=True)
    df_combined.to_csv(OUTPUT_COMBINED, index=False)

    print(f"✅ Arquivo COMBINADO salvo em: {OUTPUT_COMBINED}")
    print(f"   Total final: {len(df_combined)} linhas")

    return df_combined


# ------------------------------------------------------
# FUNÇÃO 2 — GERAR t-SNE
# ------------------------------------------------------
def generate_tsne(df):
    print("\n🔮 Rodando t-SNE para comparação Segmentado vs Augmented...")

    # Remover colunas não numéricas para t-SNE
    df_numeric = df.select_dtypes(include=[np.number])

    if df_numeric.empty:
        raise ValueError("❌ ERRO: Não existem colunas numéricas para rodar t-SNE!")

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(30, len(df_numeric) - 1),
        learning_rate='auto',
        init='random'
    )
    tsne_results = tsne.fit_transform(df_numeric)

    os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)

    plt.figure(figsize=(12, 8))

    for src in df["source"].unique():
        idx = df["source"] == src
        plt.scatter(
            tsne_results[idx, 0],
            tsne_results[idx, 1],
            label=src,
            s=12,
            alpha=0.6
        )

    plt.title("t-SNE — Segmentado vs Augmented")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.legend()
    plt.grid(True)

    plt.savefig(TSNE_OUTPUT)
    plt.show()

    print(f"📊 Gráfico t-SNE salvo em: {TSNE_OUTPUT}")


# ------------------------------------------------------
# EXECUÇÃO PRINCIPAL
# ------------------------------------------------------
if __name__ == "__main__":
    df_combined = combine_metadata()
    generate_tsne(df_combined)
    print("\n✨ Processo concluído com sucesso!")
