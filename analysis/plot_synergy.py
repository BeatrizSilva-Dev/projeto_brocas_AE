import matplotlib.pyplot as plt

# CONFIGURAÇÃO
SCORES = {
    "Apenas Real\n(Baseline)": 0.36,
    "Apenas Sintético\n(Syn2Real)": 0.01,
    "Híbrido\n(Real + Sintético)": 0.81
}

def main():
    print("Gerando Gráfico de Sinergia")

    scenarios = list(SCORES.keys())
    values = list(SCORES.values())

    colors = ['#808080', '#FF4444', '#2CA02C']

    plt.figure(figsize=(8, 6), dpi=300)
    bars = plt.bar(scenarios, values, color=colors, alpha=0.9, width=0.6)

    plt.title('Impacto da Fonte de Dados no Desempenho (XGBoost)', fontsize=14)
    plt.ylabel('F1-Score (Generalização)', fontsize=12)
    plt.ylim(0.0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Adiciona os valores e setas
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                 f"{height:.2f}", ha='center', va='bottom', fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.savefig("grafico_sinergia.pdf", format='pdf', bbox_inches='tight')
    plt.savefig("grafico_sinergia.png", dpi=300, bbox_inches='tight')
    print("✅ Gráfico salvo: grafico_sinergia.pdf")

if __name__ == "__main__":
    main()