import pandas as pd
import os

# CONFIGURAÇÃO
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")

# Arquivo Original
FILE_ORIGINAL = os.path.join(METADATA_DIR, "segmented_metadata.csv")

# Arquivo Aumentado (Ajuste o nome se necessário, ex: augmented_shift_metadata.csv)
FILE_AUGMENTED = os.path.join(METADATA_DIR, "augmented_shift_metadata.csv")

def print_stats(df, title):
    """Função auxiliar para imprimir as estatísticas de um DataFrame"""
    print(f"\n=== {title} ===")

    if df is None or df.empty:
        print("  [Vazio ou Não Encontrado]")
        return

    # Conta valores
    counts = df["fail"].value_counts()
    count_normal = counts.get(0, 0)
    count_fail = counts.get(1, 0)
    total = count_normal + count_fail

    print(f"  Total de Amostras: {total}")
    print(f"  - Normais (0): {count_normal}")
    print(f"  - Falhas  (1): {count_fail}")

    if total > 0:
        pct_fail = (count_fail / total) * 100
        print(f"  - Porcentagem de Falhas: {pct_fail:.2f}%")

        if count_fail > 0:
            ratio = count_normal / count_fail
            print(f"  - Ratio (Proporção): 1 falha para cada {ratio:.2f} normais")
        else:
            print("  - Ratio: Infinito (sem falhas)")

def main():
    print(f"📂 Verificando diretório: {METADATA_DIR}")

    # 1. Carregar Original
    df_orig = None
    if os.path.exists(FILE_ORIGINAL):
        df_orig = pd.read_csv(FILE_ORIGINAL)
        print_stats(df_orig, "DATASET ORIGINAL (Segmentado)")
    else:
        print(f"❌ Arquivo Original não encontrado: {FILE_ORIGINAL}")

    # 2. Carregar Aumentado
    df_aug = None
    if os.path.exists(FILE_AUGMENTED):
        df_aug = pd.read_csv(FILE_AUGMENTED)
        print_stats(df_aug, "DADOS AUMENTADOS (Novos)")
    else:
        print(f"\n⚠️ Arquivo Aumentado não encontrado: {FILE_AUGMENTED}")
        print("   (Verifique se o nome do arquivo no código está igual ao gerado pelo segment_full.py)")

    # 3. Simular o Treinamento (Combinado)
    if df_orig is not None and df_aug is not None:
        # No treino, usamos: Normais Originais + Falhas Originais + Falhas Aumentadas
        # Como o df_orig já tem normais e falhas, e o df_aug tem falhas extras:
        df_combined = pd.concat([df_orig, df_aug], ignore_index=True)

        print_stats(df_combined, "CENÁRIO FINAL DE TREINAMENTO (Original + Aumentado)")

        # Comparação de ganho
        ratio_orig = df_orig["fail"].value_counts().get(0,0) / df_orig["fail"].value_counts().get(1,1)
        ratio_final = df_combined["fail"].value_counts().get(0,0) / df_combined["fail"].value_counts().get(1,1)

        print("\n🚀 MELHORIA DO BALANCEAMENTO:")
        print(f"  Antes: 1 falha a cada {ratio_orig:.1f} normais")
        print(f"  Agora: 1 falha a cada {ratio_final:.1f} normais")

if __name__ == "__main__":
    main()