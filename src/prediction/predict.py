import joblib
import pandas as pd
import numpy as np
import sys

# Caminho padrão do modelo salvo
MODEL_PATH = "models/modelo.pkl"

def carregar_modelo(caminho_modelo=MODEL_PATH):
    try:
        modelo = joblib.load(caminho_modelo)
        print("✅ Modelo carregado com sucesso.")
        return modelo
    except Exception as e:
        print(f"❌ Erro ao carregar o modelo: {e}")
        sys.exit(1)

def carregar_amostra_csv(caminho_csv):
    try:
        df = pd.read_csv(caminho_csv)
        print("✅ Amostra carregada com sucesso.")
        return df
    except Exception as e:
        print(f"❌ Erro ao carregar a amostra: {e}")
        sys.exit(1)

def fazer_predicao(modelo, dados):
    try:
        predicoes = modelo.predict(dados)
        return predicoes
    except Exception as e:
        print(f"❌ Erro ao fazer predição: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Exemplo de uso: python src/prediction/predict.py data/processed/alguma_amostra.csv
    if len(sys.argv) != 2:
        print("Uso: python src/prediction/predict.py <caminho_para_amostra_csv>")
        sys.exit(1)

    caminho_amostra = sys.argv[1]
    modelo = carregar_modelo()
    df_amostra = carregar_amostra_csv(caminho_amostra)

    # Se o CSV tiver uma coluna 'rotulo', remove antes da predição
    if 'rotulo' in df_amostra.columns:
        df_amostra = df_amostra.drop(columns=['rotulo'])

    predicoes = fazer_predicao(modelo, df_amostra)
    print("🔮 Predições:", predicoes)
