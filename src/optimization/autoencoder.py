import os
import re
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import f1_score, confusion_matrix, classification_report, recall_score, roc_auc_score


ROOT_DATASET = r"C:\...\Projeto_Brocas_AE\data\segmented"
CANAL_ALVO = "4"
N_NORMAL = 5
N_MFCC = 20

def extract_hole_number(filename):
    match = re.search(r"hole(\d+)", filename)
    return int(match.group(1)) if match else None

def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    rms = librosa.feature.rms(y=y)[0]
    return np.hstack([
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
        np.mean(rms), np.std(rms)
    ])

# 2. PROCESSAMENTO LEAVE-ONE-DRILL-OUT (LODO) 
regional_results = []
processed_drills = {}

# Listas globais para acumular os dados e calcular a AUC 
all_global_scores = []
all_global_labels = []

for drill_folder in os.listdir(ROOT_DATASET):
    path = os.path.join(ROOT_DATASET, drill_folder)
    if not os.path.isdir(path): continue

    print(f"Analisando {drill_folder}...")

    # Coleta e ordenação de arquivos do Canal 4 (Ultrassom)
    files = []
    for root, _, fs in os.walk(path):
        if "ultrasonic" not in root.lower(): continue
        for f in fs:
            if f.lower().endswith(".wav") and (f"ch{CANAL_ALVO}" in f.lower() or f"tr{CANAL_ALVO}" in f.lower()):
                hole = extract_hole_number(f)
                if hole is not None:
                    files.append((hole, os.path.join(root, f)))

    files.sort(key=lambda x: x[0])

    # Remove o último furo 
    if len(files) > N_NORMAL + 2:
        files = files[:-1]
    else:
        continue

    X = []
    for hole, fpath in files:
        try:
            y, sr = librosa.load(fpath, sr=None)
            X.append(extract_features(y, sr))
        except:
            continue

    if len(X) <= N_NORMAL: continue

    X = np.array(X)
    n_holes = len(X)

    # Normalização baseada apenas nos furos saudáveis (5 iniciais)
    scaler = StandardScaler()
    scaler.fit(X[:N_NORMAL])
    X_scaled = scaler.transform(X)

    # Arquitetura Autoencoder (32, 16, 32)
    ae = MLPRegressor(hidden_layer_sizes=(32, 16, 32), max_iter=1000, random_state=42)
    ae.fit(X_scaled[:N_NORMAL], X_scaled[:N_NORMAL])

    recon = ae.predict(X_scaled)
    errors = np.mean((X_scaled - recon)**2, axis=1)

    # Threshold Estatístico (Percentil 99.5)
    threshold = np.percentile(errors[:N_NORMAL], 99.5)
    furos_acima = (errors > threshold).astype(int)

    # Lógica de Persistência Temporal
    preds_persistentes = np.zeros(n_holes)
    janela_persistencia = 10

    for i in range(janela_persistencia - 1, n_holes):
        if np.all(furos_acima[i-(janela_persistencia-1) : i+1] == 1):
            preds_persistentes[i] = 1

    # Divisão de Regiões para Avaliação (50/20)
    idx_50 = int(n_holes * 0.5)
    idx_80 = int(n_holes * 0.8)

    alerta_na_normal = 1 if np.any(preds_persistentes[:idx_50]) else 0
    alerta_na_anomalia = 1 if np.any(preds_persistentes[idx_80:]) else 0

    regional_results.append({'y_true': 0, 'y_pred': alerta_na_normal})
    regional_results.append({'y_true': 1, 'y_pred': alerta_na_anomalia})

    # Acumula exclusivamente os scores correspondentes às regiões limpas (Normal 50% e Anomalia 20%)
    for i in range(n_holes):
        if i < idx_50:
            all_global_scores.append(errors[i])
            all_global_labels.append(0)
        elif i >= idx_80:
            all_global_scores.append(errors[i])
            all_global_labels.append(1)

    processed_drills[drill_folder] = {
        'errors_ae': errors,
        'labels': [1 if (i/n_holes) >= 0.8 else 0 for i in range(n_holes)],
        'thresh': threshold
    }

# 3. MÉTRICAS E PLOTAGEM
if len(regional_results) > 0:
    df_res = pd.DataFrame(regional_results)
    
    # Cálculo consolidado da AUC 
    consolidated_auc = roc_auc_score(all_global_labels, all_global_scores)
    
    print(f"Consolidated AUC: {consolidated_auc:.4f}")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 10
    })

    fig, ax = plt.subplots(figsize=(3.3, 2.8))

    cm = confusion_matrix(df_res['y_true'], df_res['y_pred'])

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                annot_kws={"size": 14, "weight": "bold"},
                xticklabels=['No Alert', 'Alert'],
                yticklabels=['Normal (50%)', 'Anomaly (20%)'],
                ax=ax)

    ax.set_ylabel('Ground Truth', fontweight='bold', fontsize=9)
    ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=9)

    plt.setp(ax.get_yticklabels(), rotation=90, va="center")

    plt.tight_layout(pad=0.1)

    plt.savefig("matriz_autoencoder.pdf", bbox_inches='tight', pad_inches=0.01)
    plt.show()

    export_data = []
    for drill_name, data in processed_drills.items():
        for i, (err, label) in enumerate(zip(data['errors_ae'], data['labels'])):
            export_data.append({
                'drill': drill_name,
                'hole': i + 1,
                'ultrasonic_mse': err,
                'ground_truth': label,
                'adaptive_threshold': data['thresh']
            })

    df_export = pd.DataFrame(export_data)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    opt_path = os.path.join(base_dir, "src", "optimization")
    if not os.path.exists(opt_path): os.makedirs(opt_path)

    csv_path = os.path.join(opt_path, "resultados_autoencoder.csv")
    df_export.to_csv(csv_path, index=False)

    print(f"\n Dados exportados para: {csv_path}")

else:
    print("\n Nenhum dado foi processado. Verifique os caminhos do dataset.")