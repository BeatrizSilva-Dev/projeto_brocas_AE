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
# CONFIGURAÇÃO
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "runs")

def find_best_f1(model_name):
    # Procura a pasta mais recente que contenha o nome do experimento
    target_name = f"results_SYN2REAL_{model_name}"

    candidates = []
    for item in os.listdir(DATA_DIR):
        if target_name in item and os.path.isdir(os.path.join(DATA_DIR, item)):
            candidates.append(item)

    if not candidates:
        return 0.0

    # Pega a pasta mais nova
    candidates.sort(reverse=True)
    latest_folder = candidates[0]
    csv_path = os.path.join(DATA_DIR, latest_folder, f"{target_name}.csv")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df['f1'].max()
    return 0.0

def main():
    print("Gerando Tabela Syn2Real")

    f1_xgb = find_best_f1("XGBOOST")
    f1_rf = find_best_f1("RANDOM_FOREST")

    print(f"XGBoost: {f1_xgb:.4f}")
    print(f"Random Forest: {f1_rf:.4f}")

    # Monta a Tabela
    data = {
        "Modelo": ["XGBoost", "Random Forest"],
        "Treinamento": ["Sintético (Noise)", "Sintético (Noise)"],
        "Teste": ["Real (Original)", "Real (Original)"],
        "F1-Score": [f1_xgb, f1_rf],
        "Resultado": ["Falha de Generalização", "Falha de Generalização"]
    }

    df = pd.DataFrame(data)
    print("\nTABELA PARA O OVERLEAF")
    print(df)

    # Gera o código LaTeX pronto para copiar
    latex_code = df.to_latex(index=False, float_format="%.4f",
                             caption="Resultados do Protocolo Syn2Real",
                             label="tab:syn2real")
    print("\n--- CÓDIGO LATEX ---")
    print(latex_code)

if __name__ == "__main__":
    main()