import pandas as pd
import matplotlib.pyplot as plt
import os

# CONFIGURAÇÃO DE FONTE
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.weight': 'normal',
    'font.size': 14,             # Tamanho base

    'figure.titlesize': 24,      # Título da Figura
    'axes.titlesize': 24,        # Título do Eixo
    'axes.labelsize': 24,        # Nome dos Eixos X e Y

    'xtick.labelsize': 16,       # Números do eixo X
    'ytick.labelsize': 16,       # Números do eixo Y

    'legend.fontsize': 14,       # Legenda

    'figure.figsize': (8, 6),   # Tamanho físico um pouco maior para caber as curvas
    'lines.linewidth': 2.5,      # Linhas visíveis
    'lines.markersize': 8        # Bolinhas visíveis
})

# CONFIGURAÇÃO DOS ARQUIVOS
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "src", "optimization")

FILE_DEAP = os.path.join(DATA_DIR, "results_XGBOOST_on_NOISE.csv")
FILE_RS = os.path.join(DATA_DIR, "results_Random_Search_XGBoost_on_NOISE.csv")

# Configuração da População do DEAP (Para alinhar o eixo X)
POP_SIZE = 50

def get_best_f1(df):
    if df is None or df.empty: return 0.0
    return df['f1'].max()

def main():
    print("Gerando Comparativo: DEAP vs Random Search")

    # Carregar dados
    df_deap = pd.read_csv(FILE_DEAP) if os.path.exists(FILE_DEAP) else None
    df_rs = pd.read_csv(FILE_RS) if os.path.exists(FILE_RS) else None

    if df_deap is None: print(f"⚠️ Arquivo DEAP não encontrado: {os.path.basename(FILE_DEAP)}"); return
    if df_rs is None: print(f"⚠️ Arquivo RS não encontrado: {os.path.basename(FILE_RS)}"); return

    # Melhores F1
    f1_deap = get_best_f1(df_deap)
    f1_rs = get_best_f1(df_rs)

    print(f"🧬 DEAP Melhor F1: {f1_deap:.4f}")
    print(f"🎲 RS Melhor F1:   {f1_rs:.4f}")

    # GRÁFICO 1: BARRAS (QUEM VENCEU?)
    methods = ['Busca Aleatória', 'Algoritmo Genético']
    scores = [f1_rs, f1_deap]
    colors = ['blue', 'purple']

    plt.figure(dpi=300)
    bars = plt.bar(methods, scores, color=colors, width=0.6, alpha=0.9)

    plt.title('Eficácia da Otimização', fontsize=18)
    plt.ylabel('Melhor F1-Score', fontsize=18)

    # Ajuste de Zoom no Eixo Y
    if max(scores) > 0:
        min_val = min(scores)
        plt.ylim(min_val - 0.02, max(scores) + 0.02)
    else:
        plt.ylim(0, 1)

    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.0005, f"{yval:.4f}",
                 ha='center', va='bottom', fontweight='bold', fontsize=14)

    plt.tight_layout()
    plt.savefig("comparativo_otimizadores_barras.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig("comparativo_otimizadores_barras.png", dpi=300, bbox_inches='tight')
    print("✅ Gráfico de Barras salvo.")

    # GRÁFICO 2: CURVAS DE CONVERGÊNCIA (EVOLUÇÃO)
    plt.figure(dpi=300)

    # Plot Random Search (Acumulado Máximo)
    rs_x = df_rs.index + 1
    rs_y = df_rs['f1'].cummax()
    plt.plot(rs_x, rs_y, label=f'Busca Aleatória (Max: {f1_rs:.4f})', color='blue', linestyle='--', linewidth=1.5)

    # Plot DEAP (Escalonado)
    deap_x = (df_deap.index + 1) * POP_SIZE
    deap_y = df_deap['f1']
    plt.plot(deap_x, deap_y, label=f'Algoritmo Genético (Max: {f1_deap:.4f})', color='purple', linewidth=2.5)

    plt.title('Dinâmica de Convergência: Evolução vs. Acaso', fontsize=18)
    plt.xlabel('Número de Modelos Avaliados (Custo)', fontsize=18)
    plt.ylabel('Melhor F1-Score Encontrado', fontsize=18)
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("comparativo_otimizadores_curvas.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig("comparativo_otimizadores_curvas.png", dpi=300, bbox_inches='tight')
    print("✅ Gráfico de Curvas salvo.")

    plt.show()

if __name__ == "__main__":
    main()