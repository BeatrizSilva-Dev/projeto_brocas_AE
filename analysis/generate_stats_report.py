import pandas as pd
import os

# CONFIGURAÇÃO
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
METADATA_DIR = os.path.join(PROJECT_ROOT, "data", "metadata")

# OPÇÃO A: Gerar relatório do SHIFT
#INPUT_CSV = "all_features_cache_shift.csv"
#OUTPUT_NAME = "relatorio_estatistico_shift.csv"

# OPÇÃO B: Gerar relatório do NOISE (Ruído)
INPUT_CSV = "all_features_cache_noise.csv"
OUTPUT_NAME = "relatorio_estatistico_noise.csv"

# =====================================================

def main():
    cache_path = os.path.join(METADATA_DIR, INPUT_CSV)
    output_path = os.path.join(PROJECT_ROOT, OUTPUT_NAME)

    if not os.path.exists(cache_path):
        print(f"❌ Erro: Arquivo de entrada não encontrado: {cache_path}")
        print("Verifique se você rodou o 'evaluate_augmentation.py' para este cenário antes.")
        return

    print(f"📖 Lendo dados de: {INPUT_CSV}")
    df = pd.read_csv(cache_path)

    # Separa Originais e Aumentados
    df_orig = df[df['source'] == 'segmented'].drop(columns=['source'])
    df_aug = df[df['source'] == 'augmented'].drop(columns=['source'])

    print(f"  - Amostras Originais: {len(df_orig)}")
    print(f"  - Amostras Aumentadas: {len(df_aug)}")

    # Seleciona as colunas principais para o relatório
    key_cols = ['mfcc_mean_0', 'mfcc_std_0', 'rms_mean', 'rms_std', 'zcr_mean', 'zcr_std']

    # Gera as estatísticas (describe)
    stats_orig = df_orig[key_cols].describe()
    stats_aug = df_aug[key_cols].describe()

    # Adiciona prefixos para ficar claro
    stats_orig.columns = [f"ORIG_{c}" for c in stats_orig.columns]
    stats_aug.columns = [f"AUG_{c}" for c in stats_aug.columns]

    # Junta tudo lado a lado
    stats_final = pd.concat([stats_orig, stats_aug], axis=1)

    # Reordena as colunas para facilitar a comparação visual (Orig vs Aug lado a lado)
    cols_ordered = []
    for col in key_cols:
        cols_ordered.append(f"ORIG_{col}")
        cols_ordered.append(f"AUG_{col}")

    stats_final = stats_final[cols_ordered]

    # Salva
    stats_final.to_csv(output_path)

    print("\n✅ RELATÓRIO GERADO COM SUCESSO!")
    print(f"Arquivo salvo na raiz como: {OUTPUT_NAME}")
    print("-" * 30)
    print("DICA: Abra este CSV no Excel. Compare as linhas 'mean' (Média) e 'std' (Desvio Padrão).")

if __name__ == "__main__":
    main()