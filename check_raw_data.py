import os

# CONFIGURAÇÃO AUTOMÁTICA DE CAMINHO
# 1. Descobre onde o arquivo está
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Sobe um nível para chegar na raiz (Projeto_Brocas_AE)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# 3. Define o caminho da pasta RAW
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

print(f"--- AUDITORIA DA PASTA RAW ---")
print(f"📂 Buscando em: {RAW_DIR}")

if not os.path.exists(RAW_DIR):
    print("❌ ERRO CRÍTICO: Pasta data/raw ainda não encontrada!")
    print("   Verifique se a pasta 'data' está na raiz do projeto.")
    exit()

pastas = sorted(os.listdir(RAW_DIR))
total_wavs_geral = 0

print("-" * 80)
print(f"{'PASTA':<55} | {'ID LIDO':<8} | {'WAVs'}")
print("-" * 80)

for pasta in pastas:
    caminho_completo = os.path.join(RAW_DIR, pasta)
    if not os.path.isdir(caminho_completo):
        continue

    # Conta arquivos .wav recursivamente
    wav_count = 0
    for root, dirs, files in os.walk(caminho_completo):
        for f in files:
            if f.lower().endswith(".wav") and not f.startswith("._"):
                wav_count += 1

    total_wavs_geral += wav_count

    # Simula a lógica de leitura do ID
    parts = pasta.split('_')
    id_detectado = "???"

    # Tenta pegar o 3º elemento (índice 2)
    if len(parts) > 2:
        id_detectado = parts[2]

    # ALERTA VISUAL SE TIVER POUCOS ARQUIVOS
    aviso = ""
    if wav_count < 10:
        aviso = "⚠️ VAZIA/ERRO"

    print(f"{pasta:<55} | {id_detectado:<8} | {wav_count} {aviso}")

print("-" * 80)
print(f"TOTAL DE ARQUIVOS .WAV NA PASTA RAW: {total_wavs_geral}")