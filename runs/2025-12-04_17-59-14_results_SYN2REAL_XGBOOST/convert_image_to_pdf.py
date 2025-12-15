import os
from PIL import Image
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 14,          # Tamanho geral
    'axes.titlesize': 16,     # Tamanho do título
    'axes.labelsize': 14,     # Tamanho dos eixos X e Y
    'xtick.labelsize': 12,    # Tamanho dos números no eixo X
    'ytick.labelsize': 12,    # Tamanho dos números no eixo Y
    'legend.fontsize': 12,    # Tamanho da legenda
    'figure.titlesize': 18    # Tamanho do título da figura
})
# ==========================================
# CONFIGURAÇÃO (MUDE AQUI)
# ==========================================
# Coloque o nome exato do arquivo da sua imagem (ex: "sim_to_real.png")
NOME_DA_IMAGEM = "sim_to_real.png"

# ==========================================

def converter_para_pdf():
    # Pega o caminho atual onde o script está
    current_dir = os.getcwd()
    input_path = os.path.join(current_dir, NOME_DA_IMAGEM)

    # Define o nome de saída (mesmo nome, mas com .pdf)
    output_pdf = os.path.splitext(input_path)[0] + ".pdf"

    if not os.path.exists(input_path):
        print(f"❌ Erro: Não encontrei o arquivo '{NOME_DA_IMAGEM}' nesta pasta.")
        print(f"   Caminho procurado: {input_path}")
        return

    try:
        print(f"🔄 Convertendo '{NOME_DA_IMAGEM}' para PDF...")

        # Abre a imagem
        image = Image.open(input_path)

        # Se a imagem tiver transparência (RGBA), o PDF exige conversão para fundo branco (RGB)
        if image.mode == 'RGBA':
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3]) # 3 é o canal alpha
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')

        # Salva com alta resolução (300 DPI é padrão de impressão)
        image.save(output_pdf, "PDF", resolution=300.0)

        print(f"✅ Sucesso! PDF criado: {os.path.basename(output_pdf)}")

    except Exception as e:
        print(f"❌ Falha na conversão: {e}")

if __name__ == "__main__":
    converter_para_pdf()