import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import wilcoxon

def gerar_boxplot_ieee():
    try:
        caminho_script = os.path.dirname(os.path.abspath(__file__))

        path_ae = os.path.join(caminho_script, "resultados_autoencoder.csv")
        path_xgb = os.path.join(caminho_script, "resultados_xgboost_individual.csv")

        df_ae_raw = pd.read_csv(path_ae)
        df_xgb_raw = pd.read_csv(path_xgb)

        def calcular_f1_por_broca(df, col_score, threshold_col, col_label, modelo):
            resultados_f1 = []
            for drill in sorted(df['drill'].unique()):
                df_drill = df[df['drill'] == drill].copy()
                y_true = df_drill[col_label].values
                furos_acima = (df_drill[col_score] > df_drill[threshold_col]).astype(int).values

                janela = 10 if modelo == 'Autoencoder' else 3
                preds = np.zeros(len(furos_acima))
                for i in range(janela - 1, len(furos_acima)):
                    if np.all(furos_acima[i-(janela-1) : i+1] == 1):
                        preds[i] = 1

                score = f1_score(y_true, preds)
                resultados_f1.append({'drill': drill, 'f1_score': score, 'modelo': modelo})
            return pd.DataFrame(resultados_f1)

        df_f1_ae = calcular_f1_por_broca(df_ae_raw, 'ultrasonic_mse', 'adaptive_threshold', 'ground_truth', 'Autoencoder')
        df_f1_xgb = calcular_f1_por_broca(df_xgb_raw, 'prob_xgb', 'threshold_xgb', 'label_real', 'XGBoost')

        stat, p_value = wilcoxon(df_f1_ae['f1_score'], df_f1_xgb['f1_score'])
        print(f"p-value: {p_value:.6f}")

        df_plot = pd.concat([df_f1_ae, df_f1_xgb], axis=0).reset_index(drop=True)

        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 10,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'pdf.fonttype': 42,
            'ps.fonttype': 42
        })

        plt.figure(figsize=(3.5, 3.0))
        sns.set_style("white")

        ax = sns.boxplot(x='modelo', y='f1_score', data=df_plot,
                         palette=['#3498db', '#e67e22'],
                         width=0.5, linewidth=1.2, fliersize=0)

        sns.stripplot(x='modelo', y='f1_score', data=df_plot,
                      color='black', alpha=0.4, jitter=True, size=3.5)

        plt.ylabel('F1-score per Drill Unit')
        plt.xlabel('Detection Architecture')
        plt.ylim(-0.05, 1.05)
        plt.grid(axis='y', linestyle=':', alpha=0.5)
        sns.despine()

        nome_arquivo = "boxplot_f1_IEEE.pdf"
        caminho_final = os.path.join(caminho_script, nome_arquivo)

        plt.savefig(caminho_final, bbox_inches='tight', pad_inches=0.01)

        plt.show()

    except Exception as e:
        print(f"Ocorreu um erro: {e}")

if __name__ == "__main__":
    gerar_boxplot_ieee()