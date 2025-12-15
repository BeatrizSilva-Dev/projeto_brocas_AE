
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

    'figure.figsize': (8, 6),   # Tamanho físico bom
    'lines.linewidth': 2.5,      # Linhas visíveis
    'lines.markersize': 8        # Bolinhas visíveis
})

# CONFIGURAÇÃO DE CAMINHOS
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

# Pasta onde estão os CSVs (src/optimization)
DATA_DIR = os.path.join(PROJECT_ROOT, "src", "optimization")

# NOMES DOS ARQUIVOS
FILE_XGBOOST = os.path.join(DATA_DIR, "results_XGBOOST_on_NOISE.csv")
FILE_RF = os.path.join(DATA_DIR, "results_RANDOM_FOREST_on_NOISE.csv")

def get_best_f1(filepath, label):
    if not os.path.exists(filepath):
        print(f"⚠️ Arquivo não encontrado: {os.path.basename(filepath)}")
        return 0.0
    df = pd.read_csv(filepath)
    best = df['f1'].max()
    print(f"✅ {label}: {best:.4f}")
    return best

def main():
    print("--- Gerando Confronto Final (XGBoost vs Random Forest) ---")

    f1_xgb = get_best_f1(FILE_XGBOOST, "XGBoost")
    f1_rf = get_best_f1(FILE_RF, "Random Forest")

    models = ['XGBoost (Noise)', 'Random Forest (Noise)']
    scores = [f1_xgb, f1_rf]

    # Cores: Azul (XGB) e Verde (RF)
    colors = ['#1f77b4', '#2ca02c']

    plt.figure(dpi=300) # Tamanho já definido no rcParams acima
    bars = plt.bar(models, scores, color=colors, width=0.6, alpha=0.9)

    plt.title('Comparação de Qual é o Melhor Modelo', fontsize=18) # Força tamanho específico aqui
    plt.ylabel('Melhor F1-Score Alcançado', fontsize=18)

    # Zoom no eixo Y
    if max(scores) > 0:
        min_score = min([s for s in scores if s > 0])
        plt.ylim(min_score - 0.05, max(scores) + 0.05)
    else:
        plt.ylim(0, 1)

    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Coloca os valores
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.001, f"{yval:.4f}",
                 ha='center', va='bottom', fontweight='bold', fontsize=14)

    plt.tight_layout()

    # Salva em PDF e PNG
    plt.savefig("confronto_final_modelos.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig("confronto_final_modelos.png", dpi=300, bbox_inches='tight')

    print("\n✅ Gráfico salvo: confronto_final_modelos.pdf")

if __name__ == "__main__":
    main()