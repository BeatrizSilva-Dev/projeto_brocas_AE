import pandas as pd
import os

# CONFIGURAÇÃO
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
AUDIT_DIR = os.path.join(DATA_DIR, "auditoria_humana_detalhada")

# Arquivos
FILE_QUEST = os.path.join(AUDIT_DIR, "QUESTIONARIO_DETALHADO.csv")
FILE_KEY = os.path.join(AUDIT_DIR, "GABARITO.csv")

def main():
    print("📊 PROCESSANDO RESULTADOS DA AUDITORIA...")

    if not os.path.exists(FILE_QUEST) or not os.path.exists(FILE_KEY):
        print("❌ Erro: Arquivos não encontrados na pasta 'auditoria_humana_final'.")
        return

    # Carrega
    df_respostas = pd.read_csv(FILE_QUEST)
    df_gabarito = pd.read_csv(FILE_KEY)

    # Junta (Merge) para saber quem é quem
    # Supondo que a coluna chave seja 'Arquivo' ou 'Arquivo_Teste'
    # Vamos padronizar os nomes das colunas de merge se necessário
    col_chave_resp = 'Arquivo' if 'Arquivo' in df_respostas.columns else 'Arquivo_Teste'
    col_chave_gab = 'Arquivo' if 'Arquivo' in df_gabarito.columns else 'Arquivo_Teste'

    df_final = pd.merge(df_respostas, df_gabarito, left_on=col_chave_resp, right_on=col_chave_gab)

    # Agrupa por Tipo Real (Original, Noise, Shift)
    # Calcula a média das notas (Q5) e a média dos binários (Q1-Q4)

    # Mapeie os nomes exatos das suas colunas do Excel aqui:
    cols_binarias = [c for c in df_respostas.columns if "Q1" in c or "Q2" in c or "Q3" in c or "Q4" in c]
    col_mos = [c for c in df_respostas.columns if "Q5" in c][0]
    col_tipo = 'TIPO_REAL' if 'TIPO_REAL' in df_final.columns else 'Real'

    print(f"\n--- RESULTADOS POR TIPO ---")
    grouped = df_final.groupby(col_tipo)

    summary = grouped[[*cols_binarias, col_mos]].mean()

    # Formata para porcentagem os binários e mantém nota no MOS
    print(summary)

    print("\n--- TABELA FORMATADA (Copie os valores para o LaTeX) ---")
    for index, row in summary.iterrows():
        print(f"\n>> TIPO: {index}")
        for col in cols_binarias:
            print(f"   {col}: {row[col]*100:.1f}%")
        print(f"   MOS (Naturalidade): {row[col_mos]:.2f}")

if __name__ == "__main__":
    main()