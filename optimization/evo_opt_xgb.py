import csv
import os
import random
import time
from datetime import datetime
import gc
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm
from xgboost import XGBClassifier

# Tenta importar bibliotecas (DEAP e OPTUNA)
try:
    from deap import base, creator, tools, algorithms
except ImportError:
    print("⚠️ Biblioteca DEAP não instalada. O método 'DEAP' não funcionará.")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING) # Limpa o log do console
except ImportError:
    print("⚠️ Biblioteca OPTUNA não instalada. O método 'OPTUNA' não funcionará.")

# CONFIGURAÇÕES GERAIS
# ESCOLHA O MÉTODO AQUI: "DEAP", "RANDOM_SEARCH", "OPTUNA"
OPTIMIZATION_METHOD = "RANDOM_SEARCH"

# ESCOLHA O MODELO: "XGBOOST", "RANDOM_FOREST", "MLP", "SVM"
MODEL_TYPE = "RANDOM_FOREST"

# CSVs
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")
SEGMENTED_METADATA_CSV = os.path.join(METADATA_DIR, "segmented_metadata.csv")
AUGMENTED_METADATA_CSV = os.path.join(METADATA_DIR, "augmented_noise_metadata.csv") # Ajuste se necessário

# PARÂMETROS DE EXECUÇÃO
SEED = 42
POP_SIZE = 50
GENERATIONS = 100
TOTAL_TRIALS = POP_SIZE * GENERATIONS # 5000 avaliações para manter a isonomia

random.seed(SEED)
np.random.seed(SEED)

# SETUP DO DEAP
if OPTIMIZATION_METHOD == "DEAP" and 'creator' in globals():
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

    toolbox.register("mate", tools.cxBlend, alpha=0.2)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.5)
    toolbox.decorate("mate", check_bounds(0.0, 1.0))
    toolbox.decorate("mutate", check_bounds(0.0, 1.0))
    toolbox.register("select", tools.selTournament, tournsize=3)

# FUNÇÕES AUXILIARES (Dados e Features)
def extract_features(y, sr, n_mfcc=20):
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        stats = np.hstack([
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
            np.mean(librosa.feature.rms(y=y)[0]), np.std(librosa.feature.rms(y=y)[0]),
            np.mean(librosa.feature.zero_crossing_rate(y=y)[0]), np.std(librosa.feature.zero_crossing_rate(y=y)[0])
        ])
        return stats
    except: return None

def load_combined_dataset(segmented_csv, augmented_csv):
    print(f"🔄 Carregando dataset...")
    try:
        df_seg = pd.read_csv(segmented_csv)
        df_aug = pd.read_csv(augmented_csv) # Descomente se for usar dados aumentados
        #df_all = df_seg[df_seg['fail'].isin([0, 1])].copy() # Apenas originais neste exemplo

        # Para usar aumentado (NOISE ou SHIFT), descomente abaixo e comente a linha acima:
        df_normal = df_seg[df_seg['fail'] == 0]
        df_fail_orig = df_seg[df_seg['fail'] == 1]
        df_fail_aug = pd.read_csv(augmented_csv)
        df_all = pd.concat([df_normal, df_fail_orig, df_fail_aug], ignore_index=True)

    except Exception as e:
        print(f"❌ Erro ao ler CSV: {e}")
        return pd.DataFrame(), None, None

    print(f"📊 Total de amostras: {len(df_all)}")

    feats, ys, grps = [], [], []
    print("Extraindo features...")
    for _, row in tqdm(df_all.iterrows(), total=len(df_all)):
        path = os.path.join(PROJECT_ROOT, row['filepath_wav'])
        if os.path.exists(path):
            y, sr = librosa.load(path, sr=None, mono=True)
            f = extract_features(y, sr)
            if f is not None:
                feats.append(f)
                ys.append(row['fail'])
                grps.append(str(row['drill_id']))

    X = pd.DataFrame(feats)
    X = pd.DataFrame(StandardScaler().fit_transform(X))
    return X, pd.Series(ys), pd.Series(grps)

# FUNÇÃO DE AVALIAÇÃO
def evaluate_model(genes, X, y, groups, scale_pos_weight, n_splits=10):
    # genes: lista de 5 floats entre 0 e 1

    model = None
    if MODEL_TYPE == "XGBOOST":
        model = XGBClassifier(
            n_estimators=int(50 + genes[0] * 450),
            max_depth=int(3 + genes[1] * 12),
            learning_rate=0.001 + genes[2] * 0.299,
            subsample=0.5 + genes[3] * 0.5,
            colsample_bytree=0.5 + genes[4] * 0.5,
            eval_metric="logloss", random_state=SEED, n_jobs=-1, scale_pos_weight=scale_pos_weight
        )
    elif MODEL_TYPE == "RANDOM_FOREST":
        model = RandomForestClassifier(
            n_estimators=int(50 + genes[0] * 450),
            max_depth=int(3 + genes[1] * 27),
            min_samples_split=int(2 + genes[2] * 18),
            min_samples_leaf=int(1 + genes[3] * 9),
            max_features=0.1 + genes[4] * 0.9,
            random_state=SEED, n_jobs=-1, class_weight="balanced"
        )
    elif MODEL_TYPE == "MLP":
        # Mapeamento Logarítmico mantido igual ao do AG
        model = MLPClassifier(
            hidden_layer_sizes=(int(10 + genes[0] * 190),),
            alpha=10 ** (genes[1] * 4 - 5),
            learning_rate_init=10 ** (genes[2] * 3 - 4),
            activation='relu', solver='adam', max_iter=300, random_state=SEED, early_stopping=True
        )
    elif MODEL_TYPE == "SVM":
        model = SVC(
            C=10 ** (genes[0] * 4 - 2),
            gamma=10 ** (genes[1] * 4 - 3),
            kernel='rbf', random_state=SEED, class_weight="balanced", cache_size=1000
        )

    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    for train_idx, val_idx in gkf.split(X, y, groups):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[val_idx])
        scores.append(f1_score(y.iloc[val_idx], pred, zero_division=0))

    return np.mean(scores)

# Wrapper para o DEAP
def evaluate_deap(individual, **kwargs):
    return (evaluate_model(individual, **kwargs),)

# EXECUÇÃO PRINCIPAL
def main():
    # CONFIGURAÇÃO DE RETOMADA
    RESUME_FOLDER = "2025-11-29_23-09-26_results_RANDOM_SEARCH_RANDOM_FOREST_on_NOISE"

    exp_name = f"results_{OPTIMIZATION_METHOD}_{MODEL_TYPE}_on_NOISE"

    # --- Lógica de Pastas (Novo ou Retomada) ---
    if RESUME_FOLDER:
        run_dir = os.path.join(PROJECT_ROOT, "runs", RESUME_FOLDER)
        if not os.path.exists(run_dir):
            print(f"❌ ERRO: A pasta de retomada não existe: {run_dir}")
            return
        print(f"\n🔄 MODO RETOMADA: Continuando na pasta {RESUME_FOLDER}")
    else:
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_name = f"results_{OPTIMIZATION_METHOD}_{MODEL_TYPE}_on_NOISE"
        run_dir = os.path.join(PROJECT_ROOT, "runs", f"{run_id}_{exp_name}")
        os.makedirs(run_dir, exist_ok=True)
        csv_path = os.path.join(run_dir, "results_Random_Search_Random_Forest_on_NOISE.csv")

        print(f"\n🚀 INICIANDO: {exp_name}")
        print(f"\n🚀 MODO NOVO: Criando pasta: {run_dir}")
    # Tenta achar o CSV correto dentro da pasta
    csv_name = f"results_Random_Search_Random_Forest_on_NOISE.csv" # Nome que estava no seu código
    if RESUME_FOLDER:
        files = [f for f in os.listdir(run_dir) if f.endswith('.csv')]
        if files: csv_name = files[0]

    csv_path = os.path.join(run_dir, csv_name)
    print(f"Arquivo CSV: {csv_path}")

    # Carregar Dados
    X, y, groups = load_combined_dataset(SEGMENTED_METADATA_CSV, AUGMENTED_METADATA_CSV)
    if X.empty: return

    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    # Correção para evitar erro se groups for None ou vazio
    if groups is not None and len(groups) > 0:
        n_splits = min(10, len(np.unique(groups)))
    else:
        n_splits = 5 # Fallback

    best_scores = []
    best_config = None
    best_f1_global = -1.0
    start_iter = 1

    # LÓGICA DE RECUPERAÇÃO (Lê o que já foi feito)
    if RESUME_FOLDER and os.path.exists(csv_path):
        try:
            df_hist = pd.read_csv(csv_path)
            if not df_hist.empty:
                start_iter = len(df_hist) + 1
                print(f"   ✅ Histórico encontrado: {len(df_hist)} iterações já feitas.")
                print(f"   ⏭️  Retomando da iteração {start_iter}...")

                # Recupera o melhor F1 para o gráfico não começar do zero
                if 'f1' in df_hist.columns:
                    best_f1_global = df_hist['f1'].max()
                    # Reconstrói o histórico visual (best so far)
                    curr_max = 0
                    for val in df_hist['f1']:
                        if val > curr_max: curr_max = val
                        best_scores.append(curr_max)
                    print(f"   🏆 Melhor F1 recuperado: {best_f1_global:.4f}")
        except Exception as e:
            print(f"⚠️ Erro ao ler CSV: {e}. Começando do zero.")

    # Se for novo, cria o header
    if start_iter == 1:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["trial", "g1", "g2", "g3", "g4", "g5", "f1"])

    start_time = time.time()

    # CASO 1: DEAP (Algoritmo Genético)
    if OPTIMIZATION_METHOD == "DEAP":
        toolbox.register("evaluate", evaluate_deap, X=X, y=y, groups=groups,
                         scale_pos_weight=scale_pos_weight, n_splits=n_splits)

        population = toolbox.population(n=POP_SIZE)

        print(f"🧬 Rodando GA: Pop {POP_SIZE} x Gen {GENERATIONS}...")
        for gen in range(1, GENERATIONS + 1):
            # Avaliação
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            for ind in invalid_ind:
                ind.fitness.values = toolbox.evaluate(ind)

            # Log do melhor da geração
            best_ind = tools.selBest(population, k=1)[0]
            f1 = best_ind.fitness.values[0]
            best_scores.append(f1)

            if f1 > best_f1_global:
                best_f1_global = f1
                best_config = list(best_ind)

            print(f"   Gen {gen}/{GENERATIONS} - Best F1: {f1:.4f}")

            # Salva no CSV
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([gen, *best_ind, f1])

            # Evolução
            offspring = toolbox.select(population, len(population))
            offspring = algorithms.varAnd(offspring, toolbox, cxpb=0.7, mutpb=0.3)
            for ind in offspring: del ind.fitness.values
            population[:] = offspring
            gc.collect()
    # CASO 2: RANDOM SEARCH
    elif OPTIMIZATION_METHOD == "RANDOM_SEARCH":
        print(f"🎲 Rodando Random Search: {TOTAL_TRIALS} iterações...")
        start_iter = 1
        # LÓGICA DE RETOMADA
        if os.path.exists(csv_path):
            print(f"🔄 Encontrado arquivo existente: {csv_path}")
            try:
                df_hist = pd.read_csv(csv_path)
                if not df_hist.empty:
                    start_iter = len(df_hist) + 1
                    print(f"   ⏭️ Retomando da iteração {start_iter}...")

                    # Recupera o melhor F1 até agora para continuar o gráfico/histórico
                    if 'f1' in df_hist.columns:
                        best_f1_global = df_hist['f1'].max()
                        # Reconstrói o histórico de melhores para o gráfico não ficar estranho
                        current_best = 0
                        for val in df_hist['f1']:
                            if val > current_best: current_best = val
                            best_scores.append(current_best)
                        print(f"   🏆 Melhor F1 anterior recuperado: {best_f1_global:.4f}")
            except Exception as e:
                print(f"⚠️ Erro ao ler CSV para retomada: {e}. Começando do zero.")

        # Abre o arquivo (Modo 'a' se retomada, 'w' se novo)
        mode = 'a' if start_iter > 1 else 'w'
        with open(csv_path, mode, newline="") as f:
            writer = csv.writer(f)
            if start_iter == 1: # Só escreve header se for novo
                writer.writerow(["trial", "g1", "g2", "g3", "g4", "g5", "f1"])

        for i in range(start_iter, TOTAL_TRIALS + 1):
            # Gera genes aleatórios [0, 1]
            genes = [random.random() for _ in range(5)]

            f1 = evaluate_model(genes, X, y, groups, scale_pos_weight, n_splits)

            # Atualiza histórico de melhor global para o gráfico de convergência
            if f1 > best_f1_global:
                best_f1_global = f1
                best_config = genes
                print(f"   Iter {i}: Novo Recorde! F1: {f1:.4f}")

            # Para o gráfico ficar bonito, repetimos o melhor valor atual
            best_scores.append(best_f1_global)

            if i % 50 == 0: # Log a cada 50 iterações
                print(f"   Iter {i}/{TOTAL_TRIALS} - Atual: {f1:.4f} | Melhor: {best_f1_global:.4f}")

            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([i, *genes, f1])
            gc.collect()

    # CASO 3: OPTUNA (Bayesian Optimization)
    elif OPTIMIZATION_METHOD == "OPTUNA":
        print(f"🔮 Rodando Optuna (TPE): {TOTAL_TRIALS} trials...")

        def objective(trial):
            # O Optuna sugere valores entre 0 e 1, igual ao GA
            g1 = trial.suggest_float("g1", 0.0, 1.0)
            g2 = trial.suggest_float("g2", 0.0, 1.0)
            g3 = trial.suggest_float("g3", 0.0, 1.0)
            g4 = trial.suggest_float("g4", 0.0, 1.0)
            g5 = trial.suggest_float("g5", 0.0, 1.0)
            genes = [g1, g2, g3, g4, g5]

            return evaluate_model(genes, X, y, groups, scale_pos_weight, n_splits)

        study = optuna.create_study(direction="maximize")

        # Loop manual para poder salvar CSV e plotar progresso passo a passo
        for i in range(1, TOTAL_TRIALS + 1):
            study.optimize(objective, n_trials=1)

            trial = study.best_trial
            current_best_f1 = trial.value
            best_scores.append(current_best_f1) # Histórico de convergência

            last_trial = study.trials[-1]
            last_f1 = last_trial.value
            params = [last_trial.params[f"g{k}"] for k in range(1,6)]

            if i % 50 == 0:
                print(f"   Trial {i}/{TOTAL_TRIALS} - Atual: {last_f1:.4f} | Melhor Global: {current_best_f1:.4f}")

            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([i, *params, last_f1])
            gc.collect()

        best_f1_global = study.best_value
        best_config = [study.best_params[f"g{k}"] for k in range(1,6)]

    # FINALIZAÇÃO E PLOT
    print(f"\n🏆 MELHOR RESULTADO ({OPTIMIZATION_METHOD}): F1 = {best_f1_global:.4f}")
    print(f"⚙️ Configuração: {best_config}")

    plt.figure(figsize=(10, 6))
    plt.plot(best_scores, label=f'{OPTIMIZATION_METHOD}')
    plt.title(f'Convergência - {MODEL_TYPE}')
    plt.xlabel('Avaliações (Trials/Indivíduos)')
    plt.ylabel('Melhor F1-Score Acumulado')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, "convergence_plot.png"))

    # Salva tempo total
    duration = (time.time() - start_time) / 60
    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        f.write(f"Metodo: {OPTIMIZATION_METHOD}\nModelo: {MODEL_TYPE}\nTempo: {duration:.2f} min\nMelhor F1: {best_f1_global}")

if __name__ == "__main__":
    main()