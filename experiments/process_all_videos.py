import os
import pandas as pd
from pathlib import Path
from src.preprocessing.keypoint_extractor import extrair_dados_video

# Lista de pontos do corpo definidos pelo MediaPipe (na ordem correta!)
landmark_names = [
    'nariz', 'olho_esquerdo_interno', 'olho_esquerdo', 'olho_esquerdo_externo',
    'olho_direito_interno', 'olho_direito', 'olho_direito_externo', 'orelha_esquerda',
    'orelha_direita', 'boca_esquerda', 'boca_direita', 'ombro_esquerdo', 'ombro_direito',
    'cotovelo_esquerdo', 'cotovelo_direito', 'pulso_esquerdo', 'pulso_direito',
    'dedo_polegar_esquerdo', 'dedo_polegar_direito', 'dedo_indicador_esquerdo',
    'dedo_indicador_direito', 'dedo_medio_esquerdo', 'dedo_medio_direito',
    'quadril_esquerdo', 'quadril_direito', 'joelho_esquerdo', 'joelho_direito',
    'tornozelo_esquerdo', 'tornozelo_direito', 'calcanhar_esquerdo', 'calcanhar_direito',
    'dedo_pe_esquerdo', 'dedo_pe_direito'
]

# Geração dos nomes de coluna no formato: 'ombro_esquerdo_x', 'ombro_esquerdo_y', ...
colunas = [f"{nome}_{coord}" for nome in landmark_names for coord in ['x', 'y', 'z', 'vis']]
colunas.append("rotulo")

# Caminho da pasta contendo os vídeos
video_folder = Path("data/raw")
dados_completos = []

# Itera sobre todos os vídeos na pasta
for nome_arquivo in os.listdir(video_folder):
    if nome_arquivo.endswith(".mp4"):
        caminho_completo = video_folder / nome_arquivo

        # Define o rótulo com base no nome do arquivo
        if "excerto" in nome_arquivo:
            rotulo = 1
        elif "exerrado" in nome_arquivo:
            rotulo = 0
        else:
            continue  # Ignora arquivos sem nome esperado

        print(f"Processando: {nome_arquivo} | Rótulo: {rotulo}")
        dados = extrair_dados_video(str(caminho_completo), rotulo)
        dados_completos.extend(dados)

# Cria o DataFrame e salva
df = pd.DataFrame(dados_completos, columns=colunas)

# Cria pasta processed se não existir
os.makedirs("data/processed", exist_ok=True)

# Salva o CSV
df.to_csv("data/processed/dados_rotulados.csv", index=False)
print("✅ Dataset salvo em data/processed/dados_rotulados.csv")
