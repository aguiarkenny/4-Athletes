import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import joblib
import os

# Carrega o dataset
df = pd.read_csv("data/processed/dados_rotulados.csv")

# Divide os dados em X e y
X = df.drop(columns=["rotulo"])
y = df["rotulo"]

# Separa em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treina o modelo
modelo = DecisionTreeClassifier(max_depth=5, random_state=42)
modelo.fit(X_train, y_train)

# Avalia o modelo
y_pred = modelo.predict(X_test)
print("Relatório de classificação:")
print(classification_report(y_test, y_pred))

# Garante que a pasta models existe
os.makedirs("models", exist_ok=True)

# Salva o modelo
joblib.dump(modelo, "models/modelo.pkl")
print("✅ Modelo salvo em models/modelo.pkl")
