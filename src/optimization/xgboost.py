import os
import re
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report, recall_score


ROOT_DATASET = r"C:\...\Projeto_Brocas_AE\data\segmented"
CANAL_ALVO = "4"
N_NORMAL = 5
N_MFCC = 20

def extract_hole_number(filename):
    match = re.search(r"hole(\d+)", filename)
    return int(match.group(1)) if match else None

def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=2048)
    rms = librosa.feature.rms(y=y)[0]
    return np.hstack([np.mean(mfcc, axis=1), np.std(mfcc, axis=1), np.mean(rms), np.std(rms)])

# 2. CARREGAMENTO DOS DADOS
drills_data = {}

print("Extraindo características das brocas...")
for drill_folder in os.listdir(ROOT_DATASET):
    path = os.path.join(ROOT_DATASET, drill_folder)
    if not os.path.isdir(path): continue

    files = []
    for root, _, fs in os.walk(path):
        if "ultrasonic" not in root.lower(): continue
        for f in fs:
            if f.lower().endswith(".wav") and (f"ch{CANAL_ALVO}" in f.lower() or f"tr{CANAL_ALVO}" in f.lower()):
                hole = extract_hole_number(f)
                if hole is not None: files.append((hole, os.path.join(root, f)))

    files.sort(key=lambda x: x[0])
    if len(files) < 12: continue

    files = files[:-1]

    X_drill, y_drill = [], []
    n = len(files)
    for i, (hole, fpath) in enumerate(files):
        try:
            y, sr = librosa.load(fpath, sr=None)
            X_drill.append(extract_features(y, sr))
            y_drill.append(1 if (i/n) >= 0.8 else 0)
        except: continue

    if len(X_drill) > 0:
        drills_data[drill_folder] = {"X": np.array(X_drill), "y": np.array(y_drill)}

# 3. VALIDAÇÃO LEAVE-ONE-OUT + LÓGICA DE PERSISTÊNCIA
regional_results = []
export_data = []

print(f"Iniciando validação cruzada em {len(drills_data)} brocas...")

for drill_test in drills_data:
    X_train, y_train = [], []
    for drill_name, data in drills_data.items():
        if drill_name != drill_test:
            X_train.append(data["X"])
            y_train.append(data["y"])

    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    X_test = drills_data[drill_test]["X"]
    y_test = drills_data[drill_test]["y"]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        objective='binary:logistic',
        scale_pos_weight=10
    )
    model.fit(X_train_sc, y_train)

    probs = model.predict_proba(X_test_sc)[:, 1]
    n_holes = len(probs)

    # 5. Threshold Adaptativo (Percentil 99.5 nos primeiros 5 furos da broca de teste)
    thresh_final = max(np.percentile(probs[:N_NORMAL], 99.5), 0.1)
    furos_acima = (probs > thresh_final).astype(int)

    # 6. Persistência reduzida para o XGBoost
    preds_persistentes = np.zeros(n_holes)
    janela = 3
    for i in range(janela - 1, n_holes):
        if np.all(furos_acima[i-(janela-1) : i+1] == 1):
            preds_persistentes[i] = 1

    # 7. Avaliação Regional
    idx_50 = int(n_holes * 0.5)
    idx_80 = int(n_holes * 0.8)

    alerta_na_normal = 1 if np.any(preds_persistentes[:idx_50]) else 0
    alerta_na_anomalia = 1 if np.any(preds_persistentes[idx_80:]) else 0

    regional_results.append({'y_true': 0, 'y_pred': alerta_na_normal})
    regional_results.append({'y_true': 1, 'y_pred': alerta_na_anomalia})

    for i in range(n_holes):
        export_data.append({
            'drill': drill_test,
            'hole': i + 1,
            'prob_xgb': probs[i],
            'label_real': y_test[i],
            'threshold_xgb': thresh_final
        })

if regional_results:
    df_res = pd.DataFrame(regional_results)
    f1_global = f1_score(df_res['y_true'], df_res['y_pred'])
    recall_global = recall_score(df_res['y_true'], df_res['y_pred'])

    print(f"F1-Score Global: {f1_global:.2f}")
    print(f"Recall Global (Sensibilidade): {recall_global:.2f}")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 10
    })

    fig, ax = plt.subplots(figsize=(3.3, 2.8))

    cm = confusion_matrix(df_res['y_true'], df_res['y_pred'])

    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
                annot_kws={"size": 14, "weight": "bold"},
                xticklabels=['No Alert', 'Alert'],
                yticklabels=['Normal (50%)', 'Anomaly (20%)'],
                ax=ax)

    ax.set_ylabel('Ground Truth', fontweight='bold', fontsize=9)
    ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=9)

    plt.setp(ax.get_yticklabels(), rotation=90, va="center")

    plt.tight_layout(pad=0.1)

    plt.savefig("matriz_xgboost.pdf", bbox_inches='tight', pad_inches=0.01)
    plt.show()

    df_export = pd.DataFrame(export_data)
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_xgboost_individual.csv")
    df_export.to_csv(csv_path, index=False)
    print(f"\nDados exportados para: {csv_path}")
