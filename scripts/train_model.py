
import pickle
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Générer des données aléatoires
X, y = make_classification(n_samples=100, n_features=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle
scaler = StandardScaler()

# Apprentissage du scaler sur les données d'entraînement
X_train_scaled = scaler.fit_transform(X_train)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Enregistrer le modèle avec pickle
with open(r"C:\Users\joeto\projetfina_ML\artifacts\model.pickle", "wb") as f:
    pickle.dump(model, f)
with open(r"C:\Users\joeto\projetfina_ML\artifacts\scaler.pickle", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)