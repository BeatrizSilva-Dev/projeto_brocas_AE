import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from functools import partial
import time
import os
import librosa
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import csv
import gc

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# CONFIGURAÇÃO
# 1. Escolher o Modelo
MODEL_TYPE = "RANDOM_FOREST"

# 2. dataset
TARGET_AUGMENTED_CSV = "augmented_noise_metadata.csv"

# CAMINHOS
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")
SEGMENTED_METADATA_CSV = os.path.join(METADATA_DIR, "segmented_metadata.csv")
AUGMENTED_METADATA_CSV = os.path.join(METADATA_DIR, "augmented_noise_metadata.csv") # Ajuste se necessário

# SEED
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# DEAP SETUP
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=5)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def check_bounds(min_val, max_val):
    def decorator(func):
        def wrapper(*args, **kwargs):
            offspring = func(*args, **kwargs)
            for ind in offspring:
                for i in range(len(ind)):
                    if ind[i] > max_val: ind[i] = max_val
                    elif ind[i] < min_val: ind[i] = min_val
            return offspring
        return wrapper
    return decorator

# EXTRAÇÃO
def extract_features(y, sr, n_mfcc=20):
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        stats = [np.mean(mfcc, axis=1), np.std(mfcc, axis=1)]
        rms = librosa.feature.rms(y=y)[0]
        stats.extend([np.mean(rms), np.std(rms)])
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        stats.extend([np.mean(zcr), np.std(zcr)])
        return np.hstack(stats)
    except: return None

# CARREGADOR INTELIGENTE (Identifica a Origem)
def load_dataset_with_source(segmented_csv, augmented_csv):
    print("🔄 Carregando dados para Teste Sintético -> Real...")
    try:
        df_seg = pd.read_csv(segmented_csv)
        df_aug = pd.read_csv(augmented_csv)
    except: return pd.DataFrame(), None, None, None

    # Marca a origem dos dados
    # 'normal' = 0
    # 'real_fail' = 1
    # 'synthetic_fail' = 2

    df_normal = df_seg[df_seg['fail'] == 0].copy()
    df_normal['data_type'] = 'normal'

    df_fail_real = df_seg[df_seg['fail'] == 1].copy()
    df_fail_real['data_type'] = 'real_fail'

    df_fail_syn = df_aug.copy()
    df_fail_syn['data_type'] = 'synthetic_fail'

    # Concatena tudo
    df_all = pd.concat([df_normal, df_fail_real, df_fail_syn], ignore_index=True)
    print(f"📊 Total: {len(df_all)} | Reais: {len(df_fail_real)} | Sintéticas: {len(df_fail_syn)}")

    feature_list = []
    y_list = []
    groups_list = []
    data_type_list = [] # Nova lista para guardar a origem

    print("⏳ Extraindo features...")
    for _, row in tqdm(df_all.iterrows(), total=len(df_all)):
        path = os.path.join(PROJECT_ROOT, os.path.normpath(row['filepath_wav']))
        if os.path.exists(path):
            try:
                y, sr = librosa.load(path, sr=None, mono=True)
                f = extract_features(y, sr)
                if f is not None:
                    feature_list.append(f)
                    y_list.append(row['fail'])
                    groups_list.append(str(row['drill_id']))
                    data_type_list.append(row['data_type']) # Guarda se é real ou fake
            except: pass

    X = pd.DataFrame(feature_list)
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

    return X, pd.Series(y_list), pd.Series(groups_list), np.array(data_type_list)

# AVALIAÇÃO COM FILTRO
def evaluate_synthetic_to_real(individual, X, y, groups, data_types, scale_pos_weight, n_splits=5):
    model = None

    # Configuração dos Modelos
    if MODEL_TYPE == "XGBOOST":
        model = XGBClassifier(
            n_estimators=int(50 + individual[0]*450),
            max_depth=int(3 + individual[1]*12),
            learning_rate=0.001 + individual[2]*0.299,
            subsample=0.5 + individual[3]*0.5,
            colsample_bytree=0.5 + individual[4]*0.5,
            n_jobs=-1, random_state=SEED, scale_pos_weight=scale_pos_weight,
            eval_metric="logloss"
        )
    elif MODEL_TYPE == "RANDOM_FOREST":
        model = RandomForestClassifier(
            n_estimators=int(50 + individual[0]*450),
            max_depth=int(3 + individual[1]*27),
            min_samples_split=int(2 + individual[2]*18),
            max_features=0.1 + individual[4]*0.9,
            n_jobs=-1, random_state=SEED, class_weight="balanced"
        )
    gkf = GroupKFold(n_splits=n_splits)
    scores = []

    for train_idx, val_idx in gkf.split(X, y, groups):

        # TREINO: Usa TUDO que estiver nas pastas de treino (Original + Sintético)
        # Não filtra nada aqui. O modelo vê 'normal', 'real_fail' e 'synthetic_fail'.
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        # TESTE: Usa APENAS DADOS REAIS das pastas de teste
        # Filtra para remover qualquer 'synthetic_fail' que tenha caído no teste
        mask_val_real = np.isin(data_types[val_idx], ['normal', 'real_fail'])

        final_val_idx = val_idx[mask_val_real]

        if len(final_val_idx) > 0:
            model.fit(X_train, y_train)
            pred = model.predict(X.iloc[final_val_idx])

            # Calcula F1 (Sempre comparando com dados reais)
            scores.append(f1_score(y.iloc[final_val_idx], pred, zero_division=0))
    gc.collect()
    if len(scores) == 0: return (0.0,)
    return (np.mean(scores),)

# DEAP Registration
toolbox.register("mate", tools.cxBlend, alpha=0.2)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.5)
toolbox.decorate("mate", check_bounds(0.0, 1.0))
toolbox.decorate("mutate", check_bounds(0.0, 1.0))
toolbox.register("select", tools.selTournament, tournsize=3)

# MAIN
def main():
    # CONFIGURAÇÃO DE RETOMADA
    # nome da pasta aqui (ex: "2025-12-05_...")
    RESUME_FOLDER_NAME = None

    # Nome base do experimento
    experiment_name = f"results_SYN2REAL1_{MODEL_TYPE}"

    # LÓGICA INTELIGENTE DE CAMINHOS
    if RESUME_FOLDER_NAME:
        # MODO RETOMADA
        run_dir = os.path.join(PROJECT_ROOT, "runs", RESUME_FOLDER_NAME)

        if not os.path.exists(run_dir):
            print(f"❌ ERRO CRÍTICO: A pasta não existe: {run_dir}")
            return

        # Tenta achar o CSV automaticamente
        csv_files = [f for f in os.listdir(run_dir) if f.endswith('.csv')]
        if not csv_files:
            print(f"❌ ERRO: Nenhum CSV encontrado em {run_dir}")
            return

        csv_path = os.path.join(run_dir, csv_files[0])
        print(f"\n🔄 MODO RETOMADA ATIVADO!")
        print(f"   📂 Pasta: {RESUME_FOLDER_NAME}")

    else:
        # MODO NOVO (Começa do zero)
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join(PROJECT_ROOT, "runs", f"{run_id}_{experiment_name}")
        os.makedirs(run_dir, exist_ok=True)
        csv_path = os.path.join(run_dir, f"{experiment_name}.csv")
        print(f"\n🚀 INICIANDO NOVO EXPERIMENTO (DO ZERO)")
        print(f"   📁 Pasta: {run_dir}")

    start_time_total = time.time()

    # Carrega com os tipos de dados
    X, y, groups, d_types = load_dataset_with_source(SEGMENTED_METADATA_CSV, AUGMENTED_METADATA_CSV)
    if X.empty: return

    # Configuração
    pop_size = 50; generations = 100; n_splits = 5
    scale_pos_weight = (y==0).sum() / (y==1).sum()

    # Registra a função de avaliação ESPECIAL
    toolbox.register("evaluate", evaluate_synthetic_to_real,
                     X=X, y=y, groups=groups, data_types=d_types,
                     scale_pos_weight=scale_pos_weight, n_splits=n_splits)

    population = toolbox.population(n=pop_size)
    best_scores = []
    start_gen = 1

    # RECUPERAÇÃO DO HISTÓRICO
    if RESUME_FOLDER_NAME and os.path.exists(csv_path):
        try:
            df_hist = pd.read_csv(csv_path)
            if not df_hist.empty:
                start_gen = len(df_hist) + 1
                best_scores = df_hist['f1'].tolist()

                print(f"   📊 Histórico carregado: {len(df_hist)} gerações feitas.")
                print(f"   ⏭️  Retomando da Geração {start_gen}...")

                # INJEÇÃO GENÉTICA: Recupera o melhor indivíduo para não começar burro
                last_best_genes = df_hist.iloc[-1][['g1','g2','g3','g4','g5']].values.tolist()
                population[0][:] = last_best_genes
            else:
                print("   ⚠️ CSV vazio. Começando do zero.")
        except Exception as e:
            print(f"   ❌ Erro ao ler CSV: {e}")
            return

    # Cria cabeçalho se for novo
    if start_gen == 1:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["g1","g2","g3","g4","g5","f1"])

    print(f"\nIniciando Otimização Syn2Real...")

    for gen in range(start_gen, generations + 1):
        print(f"Gen {gen}/{generations}...", end="\r")

        # Avaliação
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.3)
        invalids = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalids: ind.fitness.values = toolbox.evaluate(ind)
        population[:] = toolbox.select(offspring, len(population))

        # Melhor da geração
        best = tools.selBest(population, 1)[0]
        f1 = best.fitness.values[0]
        best_scores.append(f1)

        # Salva (Modo Append)
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([*best, f1])

        gc.collect() # Limpeza de memória

    print(f"\n✅ {MODEL_TYPE} (Syn->Real) Finalizado! Melhor F1: {max(best_scores):.4f}")

    # Gráfico
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    plt.plot(best_scores)
    plt.title(f"Syn2Real {MODEL_TYPE}")
    plt.ylabel("F1-Score em Dados Reais")
    plt.xlabel("Geração")
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, "sim_to_real.png"))

if __name__ == "__main__":
    main()