import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# CONFIGURAÇÃO DOS ARQUIVOS (XGBOOST)
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "src", "optimization")

# --- NOMES DOS ARQUIVOS (Padrão XGBoost) ---
FILE_ORIGINAL = os.path.join(DATA_DIR, "results_original_XGBoost.csv")
FILE_SHIFT = os.path.join(DATA_DIR, "results_XGBOOST_on_SHIFT.csv")
FILE_NOISE = os.path.join(DATA_DIR, "results_XGBOOST_on_NOISE.csv")

def get_data(filepath, label):
    if not os.path.exists(filepath):
        print(f"⚠️ Arquivo não encontrado (pulando): {os.path.basename(filepath)}")
        return None, 0.0
    df = pd.read_csv(filepath)
    best_f1 = df['f1'].max()
    print(f"✅ {label}: {best_f1:.4f}")
    return df, best_f1

def main():
    print("--- Gerando Comparativo XGBoost ---")

    # Carregar dados
    df_orig, f1_orig = get_data(FILE_ORIGINAL, "XGB Original")
    df_shift, f1_shift = get_data(FILE_SHIFT, "XGB Shift")
    df_noise, f1_noise = get_data(FILE_NOISE, "XGB Noise")

    # --- 1. TABELA DE RESULTADOS ---
    data = {
        "Cenário": ["Baseline (Original)", "Aumentado A (Shift)", "Aumentado B (Noise)"],
        "Melhor F1-Score": [f1_orig, f1_shift, f1_noise],
    }

    # Calcular Ganho Relativo
    ganhos = ["-"]
    if f1_orig > 0:
        if f1_shift > 0: ganhos.append(f"+{((f1_shift - f1_orig)/f1_orig)*100:.1f}%")
        else: ganhos.append("-")

        if f1_noise > 0: ganhos.append(f"+{((f1_noise - f1_orig)/f1_orig)*100:.1f}%")
        else: ganhos.append("-")
    else:
        ganhos = ["-", "-", "-"]

    data["Ganho Relativo"] = ganhos

    df_table = pd.DataFrame(data)
    print("\n=== TABELA FINAL (XGBOOST) ===")
    print(df_table)

    # Salvar CSV da tabela
    df_table.to_csv("tabela_comparativa_XGB.csv", index=False)

    # --- PREPARAÇÃO PARA GRÁFICOS ---
    scenarios = []
    scores = []
    colors = []

    if df_orig is not None:
        scenarios.append("Original")
        scores.append(f1_orig)
        colors.append('#FF4444') # Vermelho

    if df_shift is not None:
        scenarios.append("Shift (A)")
        scores.append(f1_shift)
        colors.append('#3333FF') # Azul

    if df_noise is not None:
        scenarios.append("Noise (B)")
        scores.append(f1_noise)
        colors.append('#2CA02C') # Verde

    # --- GRÁFICO 1: BARRAS ---
    plt.figure(figsize=(8, 5), dpi=300)
    bars = plt.bar(scenarios, scores, color=colors, alpha=0.85)

    plt.title('XGBoost: Comparação de Estratégias', fontsize=14)
    plt.ylabel('Melhor F1-Score', fontsize=12)
    plt.ylim(0.0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.4f}",
                 ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.savefig("comparativo_XGBOOST_barras.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig("comparativo_XGBOOST_barras.png", dpi=300, bbox_inches='tight')
    print("\n✅ Gráfico de Barras salvo (PDF e PNG)")

    # --- GRÁFICO 2: CURVAS ---
    plt.figure(figsize=(12, 7), dpi=300)

    if df_orig is not None:
        plt.plot(df_orig.index, df_orig['f1'], label=f'Padrão (Max: {f1_orig:.4f})',
                 color='#FF4444', linestyle='--', linewidth=1.5)

    if df_shift is not None:
        plt.plot(df_shift.index, df_shift['f1'], label=f'Shift (Max: {f1_shift:.4f})',
                 color='#3333FF', linewidth=2)

    if df_noise is not None:
        plt.plot(df_noise.index, df_noise['f1'], label=f'Noise (Max: {f1_noise:.4f})',
                 color='#2CA02C', linewidth=2)

    plt.title('Evolução da Otimização (XGBoost)', fontsize=14)
    plt.xlabel('Geração (Algoritmo Genético)', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("comparativo_XGBOOST_curvas.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig("comparativo_XGBOOST_curvas.png", dpi=300, bbox_inches='tight')
    print("✅ Gráfico de Curvas salvo (PDF e PNG)")

    plt.show()

if __name__ == "__main__":
    main()