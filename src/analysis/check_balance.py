'''import pandas as pd
import os

# --- Configuração ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

METADATA_CSV = os.path.join(DATA_DIR, "metadata", "segmented_metadata.csv")
#METADATA_CSV = os.path.join(DATA_DIR, "metadata", "augmented_metadata.csv")

# --------------------

def analyze_balance(csv_path):
    """
    Carrega o CSV de metadados e imprime a contagem 
    de classes (balanceamento) da coluna 'fail'.
    """
    
    # 1. Verifica se o arquivo existe
    if not os.path.exists(csv_path):
        print(f"Erro: Arquivo não encontrado em:")
        print(f"{os.path.abspath(csv_path)}")
        return

    print(f"Carregando CSV: {csv_path}...")
    
    # 2. Carrega o CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Erro ao ler o CSV: {e}")
        return

    # 3. Verifica se a coluna 'fail' existe
    if "fail" not in df.columns:
        print(f"Erro: O CSV não contém uma coluna chamada 'fail'.")
        print(f"Colunas encontradas: {df.columns.tolist()}")
        return
        
    print("\n--- Análise de Balanceamento de Classe ---")
    
    # 4. Conta os valores na coluna 'fail'
    balance_counts = df["fail"].value_counts()
    
    # 5. Pega as contagens individuais
    #    .get(key, 0) é uma forma segura que retorna 0 se a chave (0 ou 1) não existir
    count_normal = balance_counts.get(0, 0)
    count_fail = balance_counts.get(1, 0)
    
    total = count_normal + count_fail
    
    if total == 0:
        print("O CSV está vazio ou a coluna 'fail' não tem dados.")
        return
        
    # 6. Calcula e imprime os resultados
    percent_fail = (count_fail / total) * 100
    
    print(f"Total de Amostras: {total}")
    print("-----------------------------------------")
    print(f"Amostras NORMAIS (fail = 0): {count_normal}")
    print(f"Amostras de FALHA  (fail = 1): {count_fail}")
    print("-----------------------------------------")
    print(f"Proporção: {percent_fail:.2f}% das amostras são falhas.")
    
    if count_fail > 0:
        ratio = count_normal / count_fail
        print(f"Ratio (scale_pos_weight): {ratio:.2f}")
        print(f"(Isso significa que você tem {ratio:.2f} amostras normais para cada 1 amostra de falha)")
    else:
        print("\nAVISO: Nenhuma amostra de falha (fail=1) foi encontrada!")

if __name__ == "__main__":
    analyze_balance(METADATA_CSV)'''
import pandas as pd
import os

# =====================================================
# CONFIGURAÇÃO
# =====================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")

# Arquivo Original
FILE_ORIGINAL = os.path.join(METADATA_DIR, "segmented_metadata.csv")

# Arquivo Aumentado (Ajuste o nome se necessário, ex: augmented_shift_metadata.csv)
FILE_AUGMENTED = os.path.join(METADATA_DIR, "augmented_shift_metadata.csv")

# =====================================================

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