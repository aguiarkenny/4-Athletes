# 4-Athletes

Files and development related to 4 Athletes Project

# Classificador de Execução de Exercícios usando MediaPipe + Machine Learning

Este projeto classifica execuções de exercícios físicos como corretas ou incorretas com base nos keypoints extraídos usando o MediaPipe Pose.

Estrutura do Projeto:

- data/
  - raw/ → Vídeos brutos (.mp4)
  - processed/ → Datasets processados (.csv)
- src/
  - preprocessing/ → Extração dos keypoints dos vídeos
  - training/ → Scripts para treinar o modelo
  - prediction/ → Scripts para rodar predições com o modelo treinado
  - utils/ → Funções auxiliares (opcional)
- models/ → Modelos treinados salvos (.pkl)
- experiments/ → Scripts para execuções manuais
- notebooks/ → Notebooks Jupyter (opcional)
- README.md → Este arquivo
- requirements.txt → Dependências do projeto

Como Usar:

1. Instalar as dependências:

É recomendado usar um ambiente virtual (venv, conda, etc). Para instalar as bibliotecas necessárias:

pip install -r requirements.txt

2. Processar os vídeos e gerar o CSV:

Coloque seus vídeos .mp4 dentro da pasta data/raw/. Os nomes dos arquivos devem conter "excerto" (execução correta) ou "exerrado" (execução incorreta), e então execute:

python experiments/process_all_videos.py

Esse script vai gerar o arquivo dados_rotulados.csv dentro da pasta data/processed/.

3. Treinar o modelo:

Depois de gerar o CSV, execute:

python src/training/train_model.py

Isso vai treinar um modelo de Árvore de Decisão e salvá-lo em models/modelo.pkl.

4. Rodar predições em novos dados:

Para rodar predições com base em um arquivo CSV (pode usar o mesmo dados_rotulados.csv):

python src/prediction/predict.py data/processed/dados_rotulados.csv

O script mostrará as predições no terminal e, se houver rótulo no arquivo, mostrará a acurácia.

Requisitos:

- Python 3.8+
- MediaPipe
- OpenCV
- Pandas
- Scikit-learn
- Joblib

Observações:

- Os vídeos devem estar no formato .mp4 e conter no nome "excerto" ou "exerrado".
- Os keypoints são extraídos com MediaPipe Pose e salvos com nomes de colunas baseados nos pontos anatômicos.
- O modelo atual utiliza uma árvore de decisão com profundidade máxima de 5.
