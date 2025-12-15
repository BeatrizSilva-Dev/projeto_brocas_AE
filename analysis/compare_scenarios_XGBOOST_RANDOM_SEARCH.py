import pandas as pd
import os
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.weight': 'normal',     # Volta para normal
    'font.size': 14,             # Base padrão

    'figure.titlesize': 24,      # Título da Figura
    'axes.titlesize': 28,        # Título do Gráfico

    'axes.labelsize': 20,        # Nome dos Eixos X e Y
    'xtick.labelsize': 12,       # Números do eixo X
    'ytick.labelsize': 12,       # Números do eixo Y

    'legend.fontsize': 14,
    'legend.fontsize': 16,       # Legenda

    'figure.figsize': (12, 7),   # Tamanho físico bom
    'lines.linewidth': 2.5,      # Linhas visíveis mas não grosseiras
    'lines.markersize': 8        # Bolinhas visíveis
})
# CONFIGURAÇÃO DOS ARQUIVOS
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "src", "optimization")

#APENAS O ARQUIVO DO RANDOM SEARCH
FILE_RS = os.path.join(DATA_DIR, "results_Random_Search_XGBoost_on_NOISE.csv")

def get_data(filepath, label):
    if not os.path.exists(filepath):
        print(f"⚠️ Arquivo não encontrado: {os.path.basename(filepath)}")
        return None, 0.0
    df = pd.read_csv(filepath)
    best_f1 = df['f1'].max()
    print(f"✅ {label}: Melhor F1 = {best_f1:.4f} (em {len(df)} avaliações)")
    return df, best_f1

def main():
    print("Analisando Random Search (XGBoost)")

    # Carregar dados
    df_rs, f1_rs = get_data(FILE_RS, "Random Search Noise")

    if df_rs is None:
        return

    # 1. TABELA DE RESULTADOS
    data = {
        "Cenário": ["Random Search (XGBoost + Noise)"],
        "Melhor F1-Score": [f1_rs],
        "Avaliações Totais": [len(df_rs)]
    }

    df_table = pd.DataFrame(data)
    print("\nRESUMO FINAL")
    print(df_table)

    # Salvar CSV da tabela
    df_table.to_csv("resumo_random_search_xgboost.csv", index=False)

    # 2. GRÁFICO DE CONVERGÊNCIA
    plt.figure(figsize=(12, 7), dpi=300)

    # O Random Search é ruidoso, então plotamos o "Melhor Até Agora" (cummax)
    y_values = df_rs['f1'].cummax()
    x_values = range(1, len(y_values) + 1)

    plt.plot(x_values, y_values, label=f'Random Search (Max: {f1_rs:.4f})',
             color='#1f77b4', linewidth=2)

    plt.title('Convergência: Random Search (XGBoost)', fontsize=14)
    plt.xlabel('Número de Avaliações (Trials)', fontsize=12)
    plt.ylabel('Melhor F1-Score Acumulado', fontsize=12)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Salva em PDF e PNG
    plt.savefig("analise_random_search_xgboost.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig("analise_random_search_xgboost.png", dpi=300, bbox_inches='tight')
    print("\n✅ Gráfico de Convergência salvo (PDF e PNG)")

    plt.show()

if __name__ == "__main__":
    main()