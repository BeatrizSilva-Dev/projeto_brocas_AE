import os
import shutil
import pandas as pd
import random

# =====================================================
# CONFIGURAÇÃO
# =====================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")
AUDIT_DIR = os.path.join(DATA_DIR, "auditoria_humana_detalhada") # Pasta nova

# Arquivos de entrada
FILE_ORIG = os.path.join(METADATA_DIR, "segmented_metadata.csv")
FILE_NOISE = os.path.join(METADATA_DIR, "augmented_noise_metadata.csv")
FILE_SHIFT = os.path.join(METADATA_DIR, "augmented_shift_metadata.csv")

# 20 Trios = 60 Áudios no total (Amostra Estatística Robusta)
QTD_TRIOS = 20

def main():
    print("🎧 PREPARANDO AUDITORIA DETALHADA (5 PERGUNTAS)...")

    # Limpa e recria a pasta
    if os.path.exists(AUDIT_DIR):
        shutil.rmtree(AUDIT_DIR)
    os.makedirs(AUDIT_DIR)

    # 1. Carregar Dataframes
    try:
        df_orig = pd.read_csv(FILE_ORIG)
        df_noise = pd.read_csv(FILE_NOISE)
        df_shift = pd.read_csv(FILE_SHIFT)
        # Filtra apenas as falhas no original para garantir comparação justa
        df_orig = df_orig[df_orig['fail'] == 1]
    except FileNotFoundError as e:
        print(f"❌ Erro: Faltando arquivo CSV ({e}).")
        return

    samples_final = []

    # 2. Selecionar Trios (Maçã com Maçã)
    # Embaralha os originais
    orig_candidates = df_orig.sample(frac=1, random_state=42).reset_index(drop=True)

    count = 0
    for _, row_orig in orig_candidates.iterrows():
        if count >= QTD_TRIOS: break

        d_id = row_orig['drill_id']
        h_idx = row_orig['hole_idx']

        # Busca correspondentes nas tabelas de aumento
        match_noise = df_noise[(df_noise['drill_id'] == d_id) & (df_noise['hole_idx'] == h_idx)]
        match_shift = df_shift[(df_shift['drill_id'] == d_id) & (df_shift['hole_idx'] == h_idx)]

        # Se o trio estiver completo (Original + Noise + Shift existem)
        if not match_noise.empty and not match_shift.empty:
            # Adiciona os 3 tipos à lista
            for tipo, path in [('ORIGINAL', row_orig['filepath_wav']),
                               ('NOISE', match_noise.iloc[0]['filepath_wav']),
                               ('SHIFT', match_shift.iloc[0]['filepath_wav'])]:
                samples_final.append({
                    'path': path,
                    'type': tipo,
                    'group_id': f"{d_id}_{h_idx}"
                })
            count += 1

    print(f"✅ {count} Trios selecionados (Total: {len(samples_final)} áudios).")

    # 3. Embaralhar (Teste Cego)
    random.shuffle(samples_final)

    gabarito_data = []
    questionario_data = []

    print(f"\nCopiando arquivos para: {AUDIT_DIR}")

    for i, item in enumerate(samples_final):
        fake_name = f"teste_{i+1:02d}.wav"

        src_path = item['path']
        if not os.path.isabs(src_path): src_path = os.path.join(PROJECT_ROOT, src_path)
        dst_path = os.path.join(AUDIT_DIR, fake_name)

        try:
            shutil.copy2(src_path, dst_path)

            # Preenche o Gabarito (para você ver DEPOIS)
            gabarito_data.append({
                'Arquivo_Teste': fake_name,
                'TIPO_REAL': item['type'],
                'GRUPO': item['group_id']
            })

            # Preenche o Questionário (para você preencher AGORA)
            # Exatamente as colunas que você pediu
            questionario_data.append({
                'Arquivo': fake_name,
                'Q1_Fundo_Integro (1=Sim, 0=Nao)': '',
                'Q2_Falha_Audivel (1=Sim, 0=Nao)': '',
                'Q3_Tem_Artefatos (0=Nao, 1=Sim)': '', # Cuidado: aqui 1 é ruim!
                'Q4_Tempo_Inteiro (1=Sim, 0=Nao)': '',
                'Q5_Naturalidade (1 a 5)': '',
                'OBS': ''
            })
        except Exception as e:
            print(f"  ❌ Erro ao copiar {src_path}")

    # Salva os CSVs
    pd.DataFrame(gabarito_data).to_csv(os.path.join(AUDIT_DIR, "GABARITO.csv"), index=False)
    pd.DataFrame(questionario_data).to_csv(os.path.join(AUDIT_DIR, "QUESTIONARIO_DETALHADO.csv"), index=False)

    print("\n✅ SUCESSO! Vá em 'data/auditoria_humana_detalhada'.")
    print("👉 Abra o 'QUESTIONARIO_DETALHADO.csv'.")
    print("👉 Preencha as colunas Q1 a Q5 enquanto ouve.")
    print("🔒 Não abra o GABARITO antes de terminar!")

if __name__ == "__main__":
    main()