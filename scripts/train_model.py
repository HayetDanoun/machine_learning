
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


import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Charger le jeu de données de référence (data_ref)
data_ref = pd.read_csv(r"C:\Users\joeto\projetfina_ML\data\ref_data.csv")  # Remplace par le bon chemin de ton fichier

# Séparer les variables indépendantes (X) et la variable cible (y)
X = data_ref.drop(columns=['label'])  # Remplace 'target' par le nom de la colonne cible
y = data_ref['label']  # Remplace 'target' par le nom de la colonne cible

# Si nécessaire, normaliser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Diviser les données en jeu d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialiser ton modèle (ici, un RandomForestClassifier comme exemple)
model = RandomForestClassifier()

# Entraîner le modèle avec le jeu de données de référence
model.fit(X_train, y_train)

with open(r"C:\Users\joeto\projetfina_ML\artifacts\model.pickle", "wb") as f:
    pickle.dump(model, f)
with open(r"C:\Users\joeto\projetfina_ML\artifacts\scaler.pickle", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Le modèle a été entraîné et sauvegardé avec succès !")
